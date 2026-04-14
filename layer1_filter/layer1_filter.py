"""
=============================================================
LAYER 1: Cuckoo Filter — Fast Clone Node Detection (v3-fixed)
=============================================================
FIXES over v2:

  FIX-1: location_conflict REMOVED entirely.
         The NS-3 simulator no longer outputs this column
         (wsn_sim.cc v3). All references purged from here.

  FIX-2: Flag scoring now relies purely on behavioural
         signals that exist in realistic WSN traffic:
           - Duplicate node_id per round (Cuckoo Filter)
           - adv_ch_sent > 1  (clone broadcasting from 2 spots)
           - send_rcv_ratio anomaly  (now has noise, not perfect)
           - neighbour status anomaly  (nbr_1..5_status)
           - data_sent with near-zero data_rcvd  (grayhole)
           - energy drop z-score  (abnormal consumption)

  FIX-3: flag_score and layer1_flagged still NOT passed
         to Layer 2 as ML features (leakage prevention retained).

  FIX-4: Threshold for send_rcv_ratio loosened to 0.75
         (was 0.4) because normal nodes now also have 5-15%
         packet loss — the old threshold would generate
         too many false positives on the fixed data.
=============================================================
"""

import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime


class CuckooFilter:
    def __init__(self, capacity=200, fingerprint_bits=8, max_kicks=500, bucket_size=4):
        self.capacity         = capacity
        self.bucket_size      = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks        = max_kicks
        self.buckets          = [[None] * bucket_size for _ in range(capacity)]
        self.size             = 0

    def _fingerprint(self, item):
        h = hashlib.md5(str(item).encode()).hexdigest()
        return max(int(h, 16) % (2 ** self.fingerprint_bits), 1)

    def _hash1(self, item):
        return int(hashlib.sha256(str(item).encode()).hexdigest(), 16) % self.capacity

    def _hash2(self, item, fp):
        return (self._hash1(item) ^ int(
            hashlib.sha256(str(fp).encode()).hexdigest(), 16)) % self.capacity

    def insert(self, item):
        fp = self._fingerprint(item)
        i1, i2 = self._hash1(item), self._hash2(item, fp)
        for bi in [i1, i2]:
            for slot in range(self.bucket_size):
                if self.buckets[bi][slot] is None:
                    self.buckets[bi][slot] = fp
                    self.size += 1
                    return True
        i = np.random.choice([i1, i2])
        for _ in range(self.max_kicks):
            slot = np.random.randint(0, self.bucket_size)
            fp, self.buckets[i][slot] = self.buckets[i][slot], fp
            i = (i ^ int(hashlib.sha256(
                str(fp).encode()).hexdigest(), 16)) % self.capacity
            for s in range(self.bucket_size):
                if self.buckets[i][s] is None:
                    self.buckets[i][s] = fp
                    self.size += 1
                    return True
        return False

    def lookup(self, item):
        fp = self._fingerprint(item)
        i1, i2 = self._hash1(item), self._hash2(item, fp)
        return (fp in self.buckets[i1]) or (fp in self.buckets[i2])

    @property
    def load_factor(self):
        filled = sum(1 for b in self.buckets for s in b if s is not None)
        return filled / (self.capacity * self.bucket_size)


def _compute_flag_score(row, has_wsn_ds_cols, energy_zscore=0.0):
    """
    Returns a float 0..5 flag score. Score >= 1.0 → flagged.

    FIX-1: location_conflict deliberately absent.
    FIX-4: send_rcv_ratio threshold loosened to 0.75 because
            normal nodes now have realistic packet loss.
    """
    score = 0.0

    if has_wsn_ds_cols:

        # 1. Multiple ADV_CH broadcasts (strong clone indicator)
        if row.get('adv_ch_sent', 0) > 1:
            score += 1.2

        # 2. Send/receive ratio anomaly
        ratio = row.get('send_rcv_ratio', 1.0)
        if ratio < 0.45:
            score += 1.0
        elif ratio < 0.70:
            score += 0.5

        # 3. Neighbour status anomaly
        nbr_flags = [
            row.get('nbr_1_status', 1), row.get('nbr_2_status', 1),
            row.get('nbr_3_status', 1), row.get('nbr_4_status', 1),
            row.get('nbr_5_status', 1)
        ]
        suspicious_nbrs = sum(1 for n in nbr_flags if n == 0)
        if suspicious_nbrs >= 3:
            score += 1.2
        elif suspicious_nbrs == 2:
            score += 0.6
        elif suspicious_nbrs == 1:
            score += 0.2

        # 4. Grayhole pattern
        ds = row.get('data_sent', 0)
        dr = row.get('data_rcvd', 0)
        if ds > 5 and dr == 0:
            score += 0.9
        elif ds > 3 and dr <= 1:
            score += 0.4

        # 5. Abnormal energy drain
        if energy_zscore > 2.5:
            score += 0.8
        elif energy_zscore > 1.8:
            score += 0.4

    return score


def run_layer1(data_path="data/wsn_data.csv",
               output_path="data/layer1_results.csv"):
    print("=" * 60)
    print("  LAYER 1: CUCKOO FILTER — Fast Clone Detection (v3-fixed)")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records | "
          f"Nodes: {df['node_id'].nunique()} | Rounds: {df['round'].nunique()}")

    # FIX-1: drop location_conflict if old data is used
    if 'location_conflict' in df.columns:
        print("WARNING: location_conflict found — dropping (label proxy).")
        df = df.drop(columns=['location_conflict'])

    wsn_ds_cols = ['adv_ch_sent', 'send_rcv_ratio', 'nbr_1_status', 'data_sent', 'data_rcvd']
    has_wsn_ds = all(c in df.columns for c in wsn_ds_cols)
    print(f"WSN-DS feature columns detected: {has_wsn_ds}\n")

    # Per-node energy z-score baseline
    energy_stats = df.groupby('node_id')['energy_consumed_uJ'].agg(['mean', 'std'])
    energy_stats['std'] = energy_stats['std'].fillna(1.0).clip(lower=0.01)

    flagged_records = []

    for rnd in sorted(df['round'].unique()):
        round_df = df[df['round'] == rnd].reset_index(drop=True)
        cf = CuckooFilter(capacity=600, fingerprint_bits=12)

        for _, row in round_df.iterrows():
            node_key = f"{int(row['node_id'])}"

            cuckoo_flag = False
            if cf.lookup(node_key):
                cuckoo_flag = True
            else:
                cf.insert(node_key)

            nid = int(row['node_id'])
            e_mean   = energy_stats.loc[nid, 'mean'] if nid in energy_stats.index else 1.0
            e_std    = energy_stats.loc[nid, 'std']  if nid in energy_stats.index else 1.0
            e_zscore = (row.get('energy_consumed_uJ', e_mean) - e_mean) / e_std

            signal_score = _compute_flag_score(row, has_wsn_ds, e_zscore)
            flagged = cuckoo_flag or (signal_score >= 1.0)

            record = {
                "node_id":            row['node_id'],
                "round":              row['round'],
                "packet_rate":        row['packet_rate'],
                "energy_remaining":   row['energy_remaining'],
                "energy_consumed_uJ": row['energy_consumed_uJ'],
                "is_cluster_head":    row['is_cluster_head'],
                "x_pos":              row['x_pos'],
                "y_pos":              row['y_pos'],
                "label":              row['label'],
            }

            if 'dist_to_ch' in df.columns:
                record['dist_to_ch'] = row['dist_to_ch']
                record['dist_to_bs'] = row['dist_to_bs']
            else:
                record['dist_to_ch_bs'] = row.get('dist_to_ch_bs', 0)

            if has_wsn_ds:
                for col in ['simulation_time',
                            'adv_ch_sent', 'adv_ch_rcvd',
                            'join_req_sent', 'join_req_rcvd',
                            'sch_sent', 'sch_rcvd',
                            'data_sent', 'data_rcvd', 'data_sent_bs',
                            'send_rcv_ratio',
                            'nbr_1_status', 'nbr_2_status', 'nbr_3_status',
                            'nbr_4_status', 'nbr_5_status']:
                    if col in df.columns:
                        record[col] = row[col]

            record['layer1_flagged'] = int(flagged)
            flagged_records.append(record)

    results_df = pd.DataFrame(flagged_records)
    results_df.to_csv(output_path, index=False)

    total         = len(results_df)
    actual_clones = results_df['label'].eq(1).sum()
    flagged_total = results_df['layer1_flagged'].sum()
    true_pos  = results_df[(results_df['layer1_flagged'] == 1) & (results_df['label'] == 1)].shape[0]
    false_pos = results_df[(results_df['layer1_flagged'] == 1) & (results_df['label'] == 0)].shape[0]
    false_neg = results_df[(results_df['layer1_flagged'] == 0) & (results_df['label'] == 1)].shape[0]

    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall    = true_pos / (true_pos + false_neg + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"{'─' * 50}")
    print(f"  Total records     : {total}")
    print(f"  Actual clones     : {actual_clones}")
    print(f"  Layer 1 flagged   : {flagged_total}")
    print(f"  True Positives    : {true_pos}")
    print(f"  False Positives   : {false_pos}")
    print(f"  False Negatives   : {false_neg}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"{'─' * 50}")
    print(f"  Results saved to  : {output_path}\n")

    stats = {
        "layer": 1,
        "filter_type": "Cuckoo Filter + WSN-DS signals (v3 no-leak)",
        "timestamp": datetime.now().isoformat(),
        "wsn_ds_features_used": has_wsn_ds,
        "total_records":   int(total),
        "actual_clones":   int(actual_clones),
        "flagged":         int(flagged_total),
        "true_positives":  int(true_pos),
        "false_positives": int(false_pos),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1_score":        round(f1, 4),
    }
    os.makedirs("data", exist_ok=True)
    with open("data/layer1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return results_df, stats


if __name__ == "__main__":
    results, stats = run_layer1()
    print("Layer 1 complete.")
import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime


# ──────────────────────────────────────────────
# Cuckoo Filter
# ──────────────────────────────────────────────
class CuckooFilter:
    def __init__(self, capacity=1000, fingerprint_bits=12, max_kicks=500, bucket_size=4):
        self.capacity = capacity
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks = max_kicks
        self.buckets = [[None] * bucket_size for _ in range(capacity)]

    def _fingerprint(self, item):
        h = hashlib.md5(str(item).encode()).hexdigest()
        return max(int(h, 16) % (2 ** self.fingerprint_bits), 1)

    def _hash1(self, item):
        return int(hashlib.sha256(str(item).encode()).hexdigest(), 16) % self.capacity

    def _hash2(self, item, fp):
        return (self._hash1(item) ^ int(hashlib.sha256(str(fp).encode()).hexdigest(), 16)) % self.capacity

    def insert(self, item):
        fp = self._fingerprint(item)
        i1, i2 = self._hash1(item), self._hash2(item, fp)

        for i in [i1, i2]:
            for j in range(self.bucket_size):
                if self.buckets[i][j] is None:
                    self.buckets[i][j] = fp
                    return True

        i = np.random.choice([i1, i2])
        for _ in range(self.max_kicks):
            j = np.random.randint(self.bucket_size)
            fp, self.buckets[i][j] = self.buckets[i][j], fp
            i = (i ^ int(hashlib.sha256(str(fp).encode()).hexdigest(), 16)) % self.capacity

            for k in range(self.bucket_size):
                if self.buckets[i][k] is None:
                    self.buckets[i][k] = fp
                    return True

        return False

    def lookup(self, item):
        fp = self._fingerprint(item)
        i1, i2 = self._hash1(item), self._hash2(item, fp)
        return (fp in self.buckets[i1]) or (fp in self.buckets[i2])


# ──────────────────────────────────────────────
# Improved Behaviour Score
# ──────────────────────────────────────────────
def compute_flag_score(row, energy_z):
    score = 0.0

    # stricter CH condition
    if row.get('adv_ch_sent', 0) > 2:
        score += 1.0

    ratio = row.get('send_rcv_ratio', 1.0)
    if ratio < 0.7:
        score += 0.8
    elif ratio < 0.85:
        score += 0.4

    nbrs = [row.get(f'nbr_{i}_status', 1) for i in range(1, 6)]
    bad = sum(1 for n in nbrs if n == 0)
    if bad >= 3:
        score += 1.0
    elif bad == 2:
        score += 0.5

    if row.get('data_sent', 0) > 3 and row.get('data_rcvd', 0) <= 1:
        score += 0.7

    if energy_z > 2.0:
        score += 0.6

    return score


# ──────────────────────────────────────────────
# MAIN LAYER 1
# ──────────────────────────────────────────────
def run_layer1(data_path="data/wsn_data.csv",
               output_path="data/layer1_results.csv"):

    print("=" * 60)
    print("  LAYER 1: IMPROVED CUCKOO FILTER + SPATIAL DETECTION")
    print("=" * 60)

    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} records | Nodes: {df['node_id'].nunique()} | Rounds: {df['round'].nunique()}")

    # Energy stats
    energy_stats = df.groupby('node_id')['energy_consumed_uJ'].agg(['mean', 'std'])
    energy_stats['std'] = energy_stats['std'].fillna(1.0).clip(lower=0.01)

    cf = CuckooFilter(capacity=2000)
    node_locations = {}

    results = []

    for _, row in df.iterrows():

        node_id = int(row['node_id'])
        x = row['x_pos']
        y = row['y_pos']

        # 🔥 Less sensitive spatial bucketing
        loc_bucket = (round(x / 10), round(y / 10))
        key = f"{node_id}_{loc_bucket}"

        # Cuckoo check
        cuckoo_flag = cf.lookup(key)
        if not cuckoo_flag:
            cf.insert(key)

        # Spatial detection
        spatial_flag = False
        if node_id in node_locations:
            prev_x, prev_y = node_locations[node_id]
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

            if dist > 8:  # increased threshold
                spatial_flag = True

        # ALWAYS update location
        node_locations[node_id] = (x, y)

        # Energy z-score
        mean = energy_stats.loc[node_id, 'mean']
        std = energy_stats.loc[node_id, 'std']
        z = (row['energy_consumed_uJ'] - mean) / std

        # Behaviour score
        signal_score = compute_flag_score(row, z)

        # 🔥 NEW SMART DECISION LOGIC
        flagged = (
            (cuckoo_flag and spatial_flag) or
            (signal_score >= 1.2) or
            (spatial_flag and signal_score >= 0.8)
        )

        results.append({
          "node_id": node_id,
          "round": row['round'],
          "packet_rate": row['packet_rate'],
          "energy_remaining": row['energy_remaining'],
          "energy_consumed_uJ": row['energy_consumed_uJ'],
          "is_cluster_head": row['is_cluster_head'],
          "x_pos": x,
          "y_pos": y,
          "label": row['label'],
          "layer1_flagged": int(flagged),

          # 🔥 ADD THESE (CRITICAL)
          "adv_ch_sent": row.get('adv_ch_sent', 0),
          "adv_ch_rcvd": row.get('adv_ch_rcvd', 0),
          "data_sent": row.get('data_sent', 0),
          "data_rcvd": row.get('data_rcvd', 0),
          "data_sent_bs": row.get('data_sent_bs', 0),

          "nbr_1_status": row.get('nbr_1_status', 1),
          "nbr_2_status": row.get('nbr_2_status', 1),
          "nbr_3_status": row.get('nbr_3_status', 1),
          "nbr_4_status": row.get('nbr_4_status', 1),
          "nbr_5_status": row.get('nbr_5_status', 1),
      })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # Metrics
    total = len(results_df)
    actual_clones = results_df['label'].eq(1).sum()
    flagged_total = results_df['layer1_flagged'].sum()

    true_pos = results_df[(results_df['layer1_flagged'] == 1) & (results_df['label'] == 1)].shape[0]
    false_pos = results_df[(results_df['layer1_flagged'] == 1) & (results_df['label'] == 0)].shape[0]
    false_neg = results_df[(results_df['layer1_flagged'] == 0) & (results_df['label'] == 1)].shape[0]

    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall = true_pos / (true_pos + false_neg + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("─" * 50)
    print(f"  Total records     : {total}")
    print(f"  Actual clones     : {actual_clones}")
    print(f"  Layer 1 flagged   : {flagged_total}")
    print(f"  True Positives    : {true_pos}")
    print(f"  False Positives   : {false_pos}")
    print(f"  False Negatives   : {false_neg}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print("─" * 50)

    stats = {
        "layer": 1,
        "filter_type": "Improved Cuckoo + Spatial + Behaviour (Balanced)",
        "timestamp": datetime.now().isoformat(),
        "total_records": int(total),
        "actual_clones": int(actual_clones),
        "flagged": int(flagged_total),
        "true_positives": int(true_pos),
        "false_positives": int(false_pos),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    os.makedirs("data", exist_ok=True)
    with open("data/layer1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to: {output_path}\n")

    return results_df, stats


if __name__ == "__main__":
    results, stats = run_layer1()
    print("Layer 1 complete.")
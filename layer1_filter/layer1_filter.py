"""
=============================================================
LAYER 1: Cuckoo Filter — Fast Clone Node Detection
=============================================================
Reads WSN data from MATLAB CSV export.
Detects duplicate Node IDs (clone nodes) using Cuckoo Filter.
Flags suspected clones → passes to Layer 2.
=============================================================
"""

import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime

# ──────────────────────────────────────────────
# Cuckoo Filter Implementation (Pure Python)
# ──────────────────────────────────────────────
class CuckooFilter:
    def __init__(self, capacity=200, fingerprint_bits=8, max_kicks=500, bucket_size=4):
        self.capacity       = capacity
        self.bucket_size    = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks      = max_kicks
        self.buckets        = [[None]*bucket_size for _ in range(capacity)]
        self.size           = 0
        self.false_positives = 0
        self.true_detections = 0

    def _fingerprint(self, item):
        h = hashlib.md5(str(item).encode()).hexdigest()
        fp = int(h, 16) % (2**self.fingerprint_bits)
        return max(fp, 1)  # fingerprint != 0

    def _hash1(self, item):
        return int(hashlib.sha256(str(item).encode()).hexdigest(), 16) % self.capacity

    def _hash2(self, item, fp):
        return (self._hash1(item) ^ int(hashlib.sha256(str(fp).encode()).hexdigest(), 16)) % self.capacity

    def insert(self, item):
        fp = self._fingerprint(item)
        i1 = self._hash1(item)
        i2 = self._hash2(item, fp)

        for bucket_idx in [i1, i2]:
            for slot in range(self.bucket_size):
                if self.buckets[bucket_idx][slot] is None:
                    self.buckets[bucket_idx][slot] = fp
                    self.size += 1
                    return True

        # Evict randomly
        i = np.random.choice([i1, i2])
        for _ in range(self.max_kicks):
            slot = np.random.randint(0, self.bucket_size)
            fp, self.buckets[i][slot] = self.buckets[i][slot], fp
            i = (i ^ int(hashlib.sha256(str(fp).encode()).hexdigest(), 16)) % self.capacity
            for s in range(self.bucket_size):
                if self.buckets[i][s] is None:
                    self.buckets[i][s] = fp
                    self.size += 1
                    return True
        return False  # Filter full

    def lookup(self, item):
        fp = self._fingerprint(item)
        i1 = self._hash1(item)
        i2 = self._hash2(item, fp)
        return (fp in self.buckets[i1]) or (fp in self.buckets[i2])

    def delete(self, item):
        fp = self._fingerprint(item)
        i1 = self._hash1(item)
        i2 = self._hash2(item, fp)
        for bucket_idx in [i1, i2]:
            if fp in self.buckets[bucket_idx]:
                self.buckets[bucket_idx][self.buckets[bucket_idx].index(fp)] = None
                self.size -= 1
                return True
        return False

    @property
    def load_factor(self):
        filled = sum(1 for b in self.buckets for s in b if s is not None)
        return filled / (self.capacity * self.bucket_size)


# ──────────────────────────────────────────────
# Bloom Filter (Alternative / Comparison)
# ──────────────────────────────────────────────
class BloomFilter:
    def __init__(self, capacity=1000, error_rate=0.01):
        self.capacity   = capacity
        self.error_rate = error_rate
        # Optimal bit array size
        self.bit_size = int(-capacity * np.log(error_rate) / (np.log(2)**2))
        self.hash_count = int(self.bit_size / capacity * np.log(2))
        self.bit_array  = np.zeros(self.bit_size, dtype=bool)

    def _hashes(self, item):
        hashes = []
        for seed in range(self.hash_count):
            h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.bit_size)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = True

    def __contains__(self, item):
        return all(self.bit_array[idx] for idx in self._hashes(item))


# ──────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ──────────────────────────────────────────────
def run_layer1(data_path="data/wsn_data.csv", output_path="data/layer1_results.csv"):
    print("=" * 60)
    print("  LAYER 1: CUCKOO FILTER — Fast Clone Detection")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} node records from {data_path}")
    print(f"Unique nodes: {df['node_id'].nunique()} | Rounds: {df['round'].nunique()}\n")

    # Cuckoo filter — one per round (fresh per round like real WSN)
    flagged_records = []
    detection_stats = []

    for rnd in sorted(df['round'].unique()):
        round_df = df[df['round'] == rnd].reset_index(drop=True)
        cf = CuckooFilter(capacity=600, fingerprint_bits=12)

        for _, row in round_df.iterrows():
            node_key = f"{int(row['node_id'])}"

            if cf.lookup(node_key):
                # Already seen this node ID this round → CLONE SUSPECT
                flagged = True
                cf.true_detections += (row['label'] == 1)
                cf.false_positives += (row['label'] == 0)
            else:
                cf.insert(node_key)

                # spatial anomaly detection
                dist_origin = np.sqrt(row['x_pos']**2 + row['y_pos']**2)

                flagged = dist_origin > 95

            flagged_records.append({
                "node_id":           row['node_id'],
                "round":             row['round'],
                "packet_rate":       row['packet_rate'],
                "energy_remaining":  row['energy_remaining'],
                "energy_consumed_uJ":row['energy_consumed_uJ'],
                "dist_to_ch_bs":     row['dist_to_ch_bs'],
                "is_cluster_head":   row['is_cluster_head'],
                "x_pos":             row['x_pos'],
                "y_pos":             row['y_pos'],
                "label":             row['label'],
                "layer1_flagged":    int(flagged),
            })

        detection_stats.append({
            "round":          rnd,
            "nodes_alive":    len(round_df),
            "flagged":        sum(1 for r in flagged_records if r['round']==rnd and r['layer1_flagged']==1),
            "true_clones":    round_df['label'].eq(1).sum(),
            "true_mal":       round_df['label'].eq(2).sum(),
            "load_factor":    round(cf.load_factor, 3),
        })

    results_df = pd.DataFrame(flagged_records)
    results_df.to_csv(output_path, index=False)

    # ── Metrics ──
    total       = len(results_df)
    actual_clones = results_df['label'].eq(1).sum()
    flagged_total = results_df['layer1_flagged'].sum()
    true_pos    = results_df[(results_df['layer1_flagged']==1) & (results_df['label']==1)].shape[0]
    false_pos   = results_df[(results_df['layer1_flagged']==1) & (results_df['label']==0)].shape[0]
    false_neg   = results_df[(results_df['layer1_flagged']==0) & (results_df['label']==1)].shape[0]

    precision = true_pos / (true_pos + false_pos + 1e-9)
    recall    = true_pos / (true_pos + false_neg + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"{'─'*50}")
    print(f"  Total records processed : {total}")
    print(f"  Actual clone events     : {actual_clones}")
    print(f"  Layer 1 flagged         : {flagged_total}")
    print(f"  True Positives          : {true_pos}")
    print(f"  False Positives         : {false_pos}")
    print(f"  False Negatives         : {false_neg}")
    print(f"  Precision               : {precision:.4f}")
    print(f"  Recall                  : {recall:.4f}")
    print(f"  F1 Score                : {f1:.4f}")
    print(f"{'─'*50}")
    print(f"  Results saved to: {output_path}")
    print(f"  {flagged_total} records forwarded to Layer 2\n")

    # Save stats
    stats = {
        "layer": 1,
        "filter_type": "Cuckoo Filter",
        "timestamp": datetime.now().isoformat(),
        "total_records": int(total),
        "actual_clones": int(actual_clones),
        "flagged": int(flagged_total),
        "true_positives": int(true_pos),
        "false_positives": int(false_pos),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
    with open("data/layer1_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return results_df, stats


if __name__ == "__main__":
    results, stats = run_layer1()
    print("Layer 1 complete. Run layer2_ml.py next.")

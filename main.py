"""
=============================================================
MAIN PIPELINE — WSN FL-HybridClone v2
=============================================================
Supports:
  NS-3 dataset (23-column WSN-DS-style)  → data/ns3/
  Legacy NS-3 dataset (9-column)         → backward compatible
=============================================================
"""

import sys
import os
import json
import time
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, 'layer1_filter'))
sys.path.insert(0, os.path.join(BASE_DIR, 'layer2_ml'))
sys.path.insert(0, os.path.join(BASE_DIR, 'layer3_blockchain'))

from layer1_filter   import run_layer1
from layer2_ml       import run_layer2
from layer3_blockchain import run_layer3


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
DATA_SOURCE  = "ns3"                         # "ns3" or "matlab"
DATA_PATH    = f"data/{DATA_SOURCE}/wsn_data.csv"
OUTPUT_DIR   = f"data/outputs_{DATA_SOURCE}/"
FL_ROUNDS    = 10                            # federated learning rounds
SEQ_LEN      = 10                            # CNN-BiLSTM sequence window

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

L1_OUT           = os.path.join(OUTPUT_DIR, "layer1_results.csv")
L2_OUT           = os.path.join(OUTPUT_DIR, "layer2_results.csv")
L3_OUT           = os.path.join(OUTPUT_DIR, "layer3_results.csv")
FINAL_REPORT     = os.path.join(OUTPUT_DIR, "final_report.json")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║   WSN FL-HybridClone Detection System  v2               ║
║   Cuckoo Filter + FL CNN-BiLSTM + Blockchain            ║
╚══════════════════════════════════════════════════════════╝
    """)


def detect_feature_version(path):
    """Return 'v2' if CSV has WSN-DS columns, else 'v1'."""
    try:
        import pandas as pd
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        wsn_ds_cols = ['adv_ch_sent', 'nbr_1_status', 'location_conflict', 'dist_to_ch']
        found = sum(1 for c in wsn_ds_cols if c in cols)
        return 'v2' if found >= 3 else 'v1'
    except Exception:
        return 'unknown'


def print_final_report(l1, l2, l3):
    ver = detect_feature_version(DATA_PATH)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                 FINAL SYSTEM REPORT  ({ver})              ║
╚══════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────┐
  │  LAYER 1 — Cuckoo Filter  {'(+ WSN-DS signals)' if ver=='v2' else ''}
  │  Flagged   : {l1['flagged']:<6}  True+ : {l1['true_positives']:<6}
  │  Precision : {l1['precision']:<6}  Recall: {l1['recall']:<6}
  │  F1 Score  : {l1['f1_score']:<6}
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  LAYER 2 — {l2['model']}
  │  Accuracy  : {l2['accuracy']:<6}  AUC   : {l2['roc_auc']:<6}
  │  Features  : {l2['num_features']:<6}  → L3  : {l2['forwarded_to_layer3']:<6}
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  LAYER 3 — Blockchain  (FL rounds on chain: {l3.get('fl_rounds_on_chain',0)})
  │  Detection : {l3['detection_rate']:<6}  Revoked : {l3['nodes_revoked']:<6}
  │  Clone DR  : {l3.get('clone_detection_rate', 'n/a'):<6}  Mal DR  : {l3.get('malicious_detection_rate', 'n/a'):<6}
  │  FP Block  : {l3['false_blocked']:<6}  Chain OK: {str(l3['chain_valid']):<6}
  └─────────────────────────────────────────────────────┘
    """)

    report = {
        "system":     "WSN FL-HybridClone v2",
        "timestamp":  datetime.now().isoformat(),
        "data_source": DATA_SOURCE,
        "feature_version": ver,
        "layer1": l1,
        "layer2": l2,
        "layer3": l3,
    }
    with open(FINAL_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report → {FINAL_REPORT}")
    print("  Pipeline complete!\n")


def main():
    print_banner()

    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        print("Run NS-3 simulation first (wsn_sim.cc)")
        return

    ver = detect_feature_version(DATA_PATH)
    print(f"Dataset  : {DATA_PATH}")
    print(f"Feature version detected: {ver}  "
          f"({'23-col WSN-DS style' if ver=='v2' else '9-col legacy'})")
    print("─" * 60)

    start = time.time()

    # ── LAYER 1 ──
    t1 = time.time()
    print("\n[1/3] Layer 1 — Cuckoo Filter\n")
    _, l1_stats = run_layer1(DATA_PATH, L1_OUT)
    print(f"Layer 1 done in {time.time()-t1:.1f}s")

    # ── LAYER 2 ──
    t2 = time.time()
    print("\n[2/3] Layer 2 — FL CNN-BiLSTM + Attention\n")
    _, l2_stats = run_layer2(
        L1_OUT, L2_OUT,
        fl_rounds=FL_ROUNDS, seq_len=SEQ_LEN)
    print(f"Layer 2 done in {time.time()-t2:.1f}s")

    # ── LAYER 3 ──
    t3 = time.time()
    print("\n[3/3] Layer 3 — Blockchain Verification\n")
    _, l3_stats, _ = run_layer3(L2_OUT, L3_OUT)
    print(f"Layer 3 done in {time.time()-t3:.1f}s")

    print(f"\nTotal time: {time.time()-start:.1f}s")
    print_final_report(l1_stats, l2_stats, l3_stats)


if __name__ == "__main__":
    main()
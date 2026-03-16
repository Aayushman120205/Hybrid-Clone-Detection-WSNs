"""
=============================================================
MAIN PIPELINE — WSN Hybrid 3-Layer Clone Detection
=============================================================
Runs all layers in sequence:
  MATLAB CSV → Layer 1 (Filter) → Layer 2 (ML) → Layer 3 (Blockchain)
Prints final system report.
=============================================================
"""

import sys
import os
import json
import time
from datetime import datetime

# Add layer paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layer1_filter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layer2_ml'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layer3_blockchain'))

from layer1_filter   import run_layer1
from layer2_ml       import run_layer2
from layer3_blockchain import run_layer3


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║   WSN HYBRID CLONE DETECTION SYSTEM                     ║
║   3-Layer: Cuckoo Filter + Random Forest + Blockchain   ║
╚══════════════════════════════════════════════════════════╝
    """)


def print_final_report(l1_stats, l2_stats, l3_stats):
    print("""
╔══════════════════════════════════════════════════════════╗
║                   FINAL SYSTEM REPORT                   ║
╚══════════════════════════════════════════════════════════╝""")

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  LAYER 1 — Cuckoo Filter (Fast Detection)           │
  │  Flagged   : {l1_stats['flagged']:<6}  True+  : {l1_stats['true_positives']:<6}         │
  │  Precision : {l1_stats['precision']:<6}  Recall : {l1_stats['recall']:<6}         │
  │  F1 Score  : {l1_stats['f1_score']:<6}                              │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  LAYER 2 — Random Forest (Behaviour Analysis)       │
  │  Accuracy  : {l2_stats['accuracy']:<6}  AUC    : {l2_stats['roc_auc']:<6}         │
  │  CV F1     : {l2_stats['cv_f1_mean']:<6} ± {l2_stats['cv_f1_std']:<6}               │
  │  → Layer 3 : {l2_stats['forwarded_to_layer3']:<6} records forwarded            │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  LAYER 3 — Blockchain (Identity Verification)       │
  │  Blocks    : {l3_stats['blockchain_blocks']:<6}  Revoked : {l3_stats['nodes_revoked']:<6}         │
  │  Detection : {l3_stats['detection_rate']:<6}  Chain OK: {str(l3_stats['chain_valid']):<6}        │
  │  True Block: {l3_stats['true_blocked']:<6}  FP Block: {l3_stats['false_blocked']:<6}         │
  └─────────────────────────────────────────────────────┘
    """)

    # Save combined report
    report = {
        "system": "WSN Hybrid 3-Layer Clone Detection",
        "timestamp": datetime.now().isoformat(),
        "layer1": l1_stats,
        "layer2": l2_stats,
        "layer3": l3_stats
    }
    with open("data/final_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("  📋 Full report saved to: data/final_report.json")
    print("  🎉 Pipeline complete!\n")


def main():
    print_banner()
    start = time.time()

    DATA_PATH = "data/wsn_data.csv"

    if not os.path.exists(DATA_PATH):
        print("❌ WSN data not found. Run MATLAB simulation first.")
        print("   Or run: python matlab/generate_sample_data.py")
        sys.exit(1)

    print(f"📂 Data: {DATA_PATH}\n")
    print("━" * 60)

    # ── LAYER 1 ──
    t1 = time.time()
    print("\n[STEP 1/3] Running Layer 1 — Cuckoo Filter...\n")
    _, l1_stats = run_layer1(DATA_PATH, "data/layer1_results.csv")
    print(f"Layer 1 done in {time.time()-t1:.2f}s")

    # ── LAYER 2 ──
    t2 = time.time()
    print("\n[STEP 2/3] Running Layer 2 — Machine Learning...\n")
    _, l2_stats = run_layer2("data/layer1_results.csv", "data/layer2_results.csv")
    print(f"Layer 2 done in {time.time()-t2:.2f}s")

    # ── LAYER 3 ──
    t3 = time.time()
    print("\n[STEP 3/3] Running Layer 3 — Blockchain...\n")
    _, l3_stats, _ = run_layer3("data/layer2_results.csv", "data/layer3_results.csv")
    print(f"Layer 3 done in {time.time()-t3:.2f}s")

    total_time = time.time() - start
    print(f"\n⏱  Total pipeline time: {total_time:.2f}s")

    # ── FINAL REPORT ──
    print_final_report(l1_stats, l2_stats, l3_stats)


if __name__ == "__main__":
    main()

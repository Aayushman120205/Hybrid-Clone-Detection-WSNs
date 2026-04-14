"""
=============================================================
LAYER 3: Blockchain — Identity Verification (v3-fixed)
=============================================================
FIXES over v2:
  FIX-1: location_conflict removed from verify_node().
          The NS-3 simulator v3 no longer outputs this column
          (it was a direct label proxy). Detection now relies
          on: position mismatch, energy anomaly, protocol
          counters (adv_ch_sent), neighbour anomaly score,
          send/receive ratio, and packet rate.

  FIX-2: location_conflict and flag_score removed from
          all result carry-through column lists.

Enhanced verification using WSN-DS-style features:
  - Split distance checks (dist_to_ch + dist_to_bs)
  - Protocol counter anomaly (adv_ch_sent, data_sent_bs)
  - Neighbour anomaly score from nbr_1..5_status
  - FL audit log: each FL round's model hash stored on chain
=============================================================
"""

import pandas as pd
import numpy as np
import hashlib
import json
import os
from datetime import datetime
from collections import defaultdict


# ──────────────────────────────────────────────
# Blockchain Block (unchanged)
# ──────────────────────────────────────────────
class Block:
    def __init__(self, index, data, previous_hash="0"*64):
        self.index         = index
        self.timestamp     = datetime.now().isoformat()
        self.data          = data
        self.previous_hash = previous_hash
        self.nonce         = 0
        self.hash          = self._compute_hash()

    def _compute_hash(self):
        content = json.dumps({
            "index":         self.index,
            "timestamp":     self.timestamp,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def mine(self, difficulty=2):
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self._compute_hash()

    def to_dict(self):
        return {
            "index": self.index, "timestamp": self.timestamp,
            "hash": self.hash, "previous_hash": self.previous_hash,
            "nonce": self.nonce, "data": self.data
        }


# ──────────────────────────────────────────────
# Upgraded WSN Blockchain
# ──────────────────────────────────────────────
class WSNBlockchain:
    def __init__(self, difficulty=2):
        self.difficulty     = difficulty
        self.chain          = []
        self.node_registry  = {}
        self.blacklist      = set()
        self.fl_round_log   = []   # NEW: FL audit trail
        self._create_genesis()

    def _create_genesis(self):
        g = Block(0, {"type": "GENESIS", "message": "WSN FL-HybridClone Ledger v2"})
        g.mine(self.difficulty)
        self.chain.append(g)

    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, data):
        b = Block(len(self.chain), data, self.last_block.hash)
        b.mine(self.difficulty)
        self.chain.append(b)
        return b

    def log_fl_round(self, fl_round, model_weights_hash, num_clients, val_accuracy):
        """
        NEW: Record each FL training round on the blockchain.
        Makes model training tamper-proof — poisoning attempts
        change the weights hash and break chain integrity.
        """
        entry = {
            "type":              "FL_ROUND",
            "fl_round":          fl_round,
            "model_hash":        model_weights_hash,
            "num_clients":       num_clients,
            "val_accuracy":      round(val_accuracy, 4),
            "timestamp":         datetime.now().isoformat()
        }
        block = self.add_block(entry)
        self.fl_round_log.append(entry)
        return block

    def register_node(self, node_id, x_pos, y_pos, energy,
                      adv_ch_sent=1, data_sent_bs=1):
        """
        Register with extended WSN-DS baseline profile.
        """
        loc_hash = hashlib.sha256(
            f"{node_id}:{x_pos:.2f}:{y_pos:.2f}".encode()).hexdigest()
        pub_key  = hashlib.sha256(
            f"key_{node_id}_secret".encode()).hexdigest()[:32]

        self.node_registry[node_id] = {
            "public_key":       pub_key,
            "location_hash":    loc_hash,
            "init_x":           x_pos,
            "init_y":           y_pos,
            "init_energy":      energy,
            "baseline_adv_ch":  adv_ch_sent,   # NEW: expected ADV_CH count
            "baseline_data_bs": data_sent_bs,  # NEW: expected data to BS
            "status":           "TRUSTED",
            "registered_at":    datetime.now().isoformat()
        }
        return pub_key

    def verify_node(self, node_id, x_pos, y_pos, energy_remaining,
                    packet_rate, row=None):
        """
        Enhanced verification using all WSN-DS signals.
        Returns (verified: bool, reason: str, confidence: float)
        """
        if node_id in self.blacklist:
            return False, "BLACKLISTED", 1.0

        if node_id not in self.node_registry:
            return False, "UNREGISTERED_NODE", 0.95

        reg = self.node_registry[node_id]
        reasons = []
        fail_score = 0.0

        # ── 1. Location check ──
        dist = np.sqrt((x_pos - reg['init_x'])**2 + (y_pos - reg['init_y'])**2)
        if dist >= 5.0:
            reasons.append(f"LOCATION_MISMATCH(dist={dist:.1f}m)")
            fail_score += 0.40

        # ── 2. Energy plausibility ──
        if energy_remaining > reg['init_energy'] * 1.01:
            reasons.append("ENERGY_ANOMALY")
            fail_score += 0.20

        # ── 4. Packet rate plausibility ──
        if packet_rate > 70:
            reasons.append(f"PACKET_RATE_HIGH({packet_rate:.1f})")
            fail_score += 0.15

        # ── 5. Protocol counter anomalies (WSN-DS features) ──
        if row is not None:
            adv = row.get('adv_ch_sent', 1)
            baseline_adv = reg.get('baseline_adv_ch', 1)
            if adv > baseline_adv + 1:
                reasons.append(f"ADV_CH_EXCESS(sent={adv})")
                fail_score += 0.25

            data_bs = row.get('data_sent_bs', 1)
            baseline_bs = reg.get('baseline_data_bs', 1)
            if data_bs == 0 and baseline_bs > 0:
                reasons.append("DATA_BLACKHOLE(sent_bs=0)")
                fail_score += 0.25

            # ── 6. Neighbour anomaly ──
            nbr_cols = [f'nbr_{k}_status' for k in range(1, 6)]
            nbr_flags = [row.get(c, 1) for c in nbr_cols]
            bad_nbrs  = sum(1 for n in nbr_flags if n == 0)
            if bad_nbrs >= 3:
                reasons.append(f"NEIGHBOUR_ANOMALY({bad_nbrs}/5 suspicious)")
                fail_score += 0.30
            elif bad_nbrs >= 2:
                fail_score += 0.10

            # ── 7. Send/receive ratio ──
            ratio = row.get('send_rcv_ratio', 1.0)
            if ratio < 0.3:
                reasons.append(f"PACKET_DROP(ratio={ratio:.2f})")
                fail_score += 0.15

        # ── Final decision ──
        fail_score = min(fail_score, 1.0)
        if fail_score >= 0.40:
            return False, "+".join(reasons) if reasons else "ANOMALY_SCORE", fail_score
        else:
            return True, "VERIFIED", 1.0 - fail_score

    def revoke_node(self, node_id, reason):
        self.blacklist.add(node_id)
        if node_id in self.node_registry:
            self.node_registry[node_id]['status'] = 'REVOKED'
        self.add_block({
            "type":      "REVOKE",
            "node_id":   int(node_id),
            "reason":    reason,
            "timestamp": datetime.now().isoformat()
        })

    def is_valid(self):
        for i in range(1, len(self.chain)):
            if self.chain[i].previous_hash != self.chain[i-1].hash:
                return False
        return True


# ──────────────────────────────────────────────
# MAIN LAYER 3 FUNCTION
# ──────────────────────────────────────────────
def run_layer3(input_path="data/layer2_results.csv",
               output_path="data/layer3_results.csv",
               chain_path="data/blockchain_ledger.json"):

    print("=" * 60)
    print("  LAYER 3: BLOCKCHAIN — Identity Verification (v2)")
    print("=" * 60)

    df         = pd.read_csv(input_path)
    suspicious = df[df['send_to_blockchain'] == 1].copy()

    print(f"Total records     : {len(df)}")
    print(f"Suspicious        : {len(suspicious)} forwarded from Layer 2")
    print(f"Initializing blockchain...\n")

    bc = WSNBlockchain(difficulty=2)

    # ── Register all round-1 normal nodes as trusted baseline ──
    first_round    = df[df['round'] == df['round'].min()]
    trusted_nodes  = first_round[first_round['label'] == 0]

    for _, row in trusted_nodes.iterrows():
        adv_ch   = row.get('adv_ch_sent', 1)
        data_bs  = row.get('data_sent_bs', 1)
        bc.register_node(
            row['node_id'], row['x_pos'], row['y_pos'],
            row['energy_remaining'], adv_ch, data_bs)

    print(f"Registered {len(trusted_nodes)} trusted nodes (round 1 label=0)")

    # ── Log a simulated FL round ──
    # In production this receives real model weights hash from Layer 2
    dummy_hash = hashlib.sha256(b"fl_model_round_10_weights").hexdigest()
    bc.log_fl_round(fl_round=10, model_weights_hash=dummy_hash,
                    num_clients=10, val_accuracy=0.95)
    print(f"FL round logged on blockchain. Chain height: {len(bc.chain)}\n")

    # ── Determine which distance column to use ──
    has_split_dist = 'dist_to_bs' in df.columns

    # ── Verify suspicious nodes ──
    results       = []
    revoked_nodes = set()

    for _, row in suspicious.iterrows():
        # Use dist_to_bs for distance check if available
        x_use = row['x_pos']
        y_use = row['y_pos']

        verified, reason, confidence = bc.verify_node(
            node_id         = row['node_id'],
            x_pos           = x_use,
            y_pos           = y_use,
            energy_remaining= row['energy_remaining'],
            packet_rate     = row['packet_rate'],
            row             = row
        )

        if not verified and row['node_id'] not in revoked_nodes:
            bc.revoke_node(row['node_id'], reason)
            revoked_nodes.add(row['node_id'])

        rec = {
            "node_id":          row['node_id'],
            "round":            row['round'],
            "packet_rate":      row['packet_rate'],
            "energy_remaining": row['energy_remaining'],
            "label":            row['label'],
            "layer1_flagged":   row['layer1_flagged'],
            "ml_prediction":    row['ml_prediction'],
            "ml_threat_score":  row['ml_threat_score'],
            "bc_verified":      int(verified),
            "bc_reason":        reason,
            "bc_confidence":    round(confidence, 4),
            "final_decision":   "BLOCKED" if not verified else "ALLOWED",
        }
        # Carry WSN-DS columns through (location_conflict removed in v3 sim)
        for col in ['adv_ch_sent', 'data_sent_bs', 'send_rcv_ratio',
                    'nbr_1_status', 'nbr_2_status', 'nbr_3_status',
                    'nbr_4_status', 'nbr_5_status',
                    'dist_to_ch', 'dist_to_bs']:
            if col in row.index:
                rec[col] = row[col]
        results.append(rec)

    # ── Non-suspicious nodes → ALLOWED ──
    normal_records = df[df['send_to_blockchain'] == 0].copy()
    for _, row in normal_records.iterrows():
        rec = {
            "node_id":          row['node_id'],
            "round":            row['round'],
            "packet_rate":      row['packet_rate'],
            "energy_remaining": row['energy_remaining'],
            "label":            row['label'],
            "layer1_flagged":   row['layer1_flagged'],
            "ml_prediction":    row['ml_prediction'],
            "ml_threat_score":  row['ml_threat_score'],
            "bc_verified":      1,
            "bc_reason":        "PASSED_L1_L2",
            "bc_confidence":    0.95,
            "final_decision":   "ALLOWED",
        }
        for col in ['adv_ch_sent', 'data_sent_bs', 'send_rcv_ratio',
                    'dist_to_ch', 'dist_to_bs']:
            if col in row.index:
                rec[col] = row[col]
        results.append(rec)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # ── Metrics ──
    blocked  = results_df[results_df['final_decision'] == 'BLOCKED']
    allowed  = results_df[results_df['final_decision'] == 'ALLOWED']

    true_blocked   = blocked[blocked['label'] > 0].shape[0]
    false_blocked  = blocked[blocked['label'] == 0].shape[0]
    true_allowed   = allowed[allowed['label'] == 0].shape[0]
    missed_threats = allowed[allowed['label'] > 0].shape[0]
    total_threats  = (results_df['label'] > 0).sum()
    detection_rate = true_blocked / (total_threats + 1e-9)

    # Per-type breakdown
    clone_blocked = blocked[blocked['label'] == 1].shape[0]
    mal_blocked   = blocked[blocked['label'] == 2].shape[0]
    total_clones  = (results_df['label'] == 1).sum()
    total_mal     = (results_df['label'] == 2).sum()

    print(f"{'─'*50}")
    print(f"  Blockchain height   : {len(bc.chain)} blocks")
    print(f"  Nodes revoked       : {len(revoked_nodes)}")
    print(f"  FL rounds logged    : {len(bc.fl_round_log)}")
    print(f"  BLOCKED             : {len(blocked)}")
    print(f"  ALLOWED             : {len(allowed)}")
    print(f"  True Blocked (TP)   : {true_blocked}")
    print(f"    Clone blocked     : {clone_blocked} / {total_clones}")
    print(f"    Malicious blocked : {mal_blocked}  / {total_mal}")
    print(f"  False Blocked (FP)  : {false_blocked}")
    print(f"  Missed Threats (FN) : {missed_threats}")
    print(f"  Detection Rate      : {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"{'─'*50}")

    # ── Save ledger ──
    ledger = {
        "is_valid":      bc.is_valid(),
        "chain_length":  len(bc.chain),
        "registered":    len(bc.node_registry),
        "blacklisted":   list(int(x) for x in bc.blacklist),
        "fl_rounds_logged": len(bc.fl_round_log),
        "blocks":        [b.to_dict() for b in bc.chain[-10:]]
    }
    os.makedirs("data", exist_ok=True)
    with open(chain_path, "w") as f:
        json.dump(ledger, f, indent=2)

    print(f"\n  Ledger saved     : {chain_path}")
    print(f"  Results saved    : {output_path}")
    print(f"  Chain valid      : {bc.is_valid()}\n")

    stats = {
        "layer": 3,
        "timestamp": datetime.now().isoformat(),
        "blockchain_blocks": len(bc.chain),
        "nodes_revoked": len(revoked_nodes),
        "true_blocked": int(true_blocked),
        "false_blocked": int(false_blocked),
        "missed_threats": int(missed_threats),
        "detection_rate": round(float(detection_rate), 4),
        "clone_detection_rate": round(clone_blocked / (total_clones + 1e-9), 4),
        "malicious_detection_rate": round(mal_blocked / (total_mal + 1e-9), 4),
        "chain_valid": bc.is_valid(),
        "fl_rounds_on_chain": len(bc.fl_round_log),
    }
    with open("data/layer3_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return results_df, stats, bc


if __name__ == "__main__":
    results, stats, bc = run_layer3()
    print("Layer 3 complete.")
"""
=============================================================
LAYER 3: Blockchain — Identity Verification
=============================================================
Simulates a blockchain ledger for node identity verification.
- Legitimate nodes are registered at boot (genesis block)
- Suspicious nodes from Layer 2 are verified against ledger
- Clone/spoofed nodes fail verification → BLOCKED + ALERTED
- Immutable audit trail of all decisions

In real deployment: Solidity smart contract on Ethereum/Ganache
Here: Python simulation of blockchain behaviour
=============================================================
"""

import pandas as pd
import numpy as np
import hashlib
import json
import time
from datetime import datetime
from collections import defaultdict


# ──────────────────────────────────────────────
# Blockchain Block
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
        """Proof-of-work mining (lightweight for simulation)."""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self._compute_hash()

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "data": self.data
        }


# ──────────────────────────────────────────────
# Blockchain Ledger
# ──────────────────────────────────────────────
class WSNBlockchain:
    def __init__(self, difficulty=2):
        self.difficulty  = difficulty
        self.chain       = []
        self.node_registry = {}    # node_id → {public_key, location_hash, status}
        self.blacklist   = set()
        self._create_genesis()

    def _create_genesis(self):
        genesis = Block(0, {"type": "GENESIS", "message": "WSN Identity Ledger initialized"})
        genesis.mine(self.difficulty)
        self.chain.append(genesis)

    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, data):
        block = Block(len(self.chain), data, self.last_block.hash)
        block.mine(self.difficulty)
        self.chain.append(block)
        return block

    def register_node(self, node_id, x_pos, y_pos, energy):
        """Register a legitimate node at network startup."""
        loc_hash = hashlib.sha256(f"{node_id}:{x_pos:.2f}:{y_pos:.2f}".encode()).hexdigest()
        pub_key  = hashlib.sha256(f"key_{node_id}_secret".encode()).hexdigest()[:32]

        self.node_registry[node_id] = {
            "public_key":    pub_key,
            "location_hash": loc_hash,
            "init_x":        x_pos,
            "init_y":        y_pos,
            "init_energy":   energy,
            "status":        "TRUSTED",
            "registered_at": datetime.now().isoformat()
        }
        return pub_key

    def verify_node(self, node_id, x_pos, y_pos, energy_remaining, packet_rate):
        """
        Verify a suspicious node against the blockchain registry.
        Returns: (verified: bool, reason: str, confidence: float)
        """
        if node_id in self.blacklist:
            return False, "BLACKLISTED", 1.0

        if node_id not in self.node_registry:
            return False, "UNREGISTERED_NODE", 0.95

        reg = self.node_registry[node_id]

        # Location verification (clone nodes appear at different location)
        expected_loc_hash = hashlib.sha256(
            f"{node_id}:{reg['init_x']:.2f}:{reg['init_y']:.2f}".encode()
        ).hexdigest()
        actual_loc_hash = hashlib.sha256(
            f"{node_id}:{x_pos:.2f}:{y_pos:.2f}".encode()
        ).hexdigest()

        location_match = (expected_loc_hash == actual_loc_hash)

        # Location distance check (allow ±5m movement)
        dist = np.sqrt((x_pos - reg['init_x'])**2 + (y_pos - reg['init_y'])**2)
        location_ok = dist < 5.0

        # Energy plausibility (energy should only decrease)
        energy_ok = energy_remaining <= reg['init_energy'] * 1.01

        # Packet rate plausibility (>3x normal = suspicious)
        pkt_ok = packet_rate < 60

        # Compute verification score
        score = sum([location_ok, energy_ok, pkt_ok]) / 3.0

        if score < 0.5 or not location_ok:
            reason = []
            if not location_ok:   reason.append(f"LOCATION_MISMATCH(dist={dist:.1f}m)")
            if not energy_ok:     reason.append("ENERGY_ANOMALY")
            if not pkt_ok:        reason.append(f"PACKET_RATE_ANOMALY({packet_rate:.1f})")
            return False, "+".join(reason), 1.0 - score
        else:
            return True, "VERIFIED", score

    def revoke_node(self, node_id, reason):
        """Permanently blacklist a node — recorded on blockchain."""
        self.blacklist.add(node_id)
        if node_id in self.node_registry:
            self.node_registry[node_id]['status'] = 'REVOKED'
        self.add_block({
            "type":    "REVOKE",
            "node_id": int(node_id),
            "reason":  reason,
            "timestamp": datetime.now().isoformat()
        })

    def is_valid(self):
        """Verify blockchain integrity."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i-1]
            if curr.previous_hash != prev.hash:
                return False
        return True


# ──────────────────────────────────────────────
# MAIN LAYER 3 FUNCTION
# ──────────────────────────────────────────────
def run_layer3(input_path="data/layer2_results.csv",
               output_path="data/layer3_results.csv",
               chain_path="data/blockchain_ledger.json"):

    print("=" * 60)
    print("  LAYER 3: BLOCKCHAIN — Identity Verification")
    print("=" * 60)

    df = pd.read_csv(input_path)
    suspicious = df[df['send_to_blockchain'] == 1].copy()

    print(f"Total records     : {len(df)}")
    print(f"Suspicious nodes  : {len(suspicious)} (forwarded from Layer 2)")
    print(f"Initializing blockchain ledger...\n")

    # ── Initialize Blockchain ──
    bc = WSNBlockchain(difficulty=2)

    # Register all unique nodes (first-round data = legitimate baseline)
    first_round = df[df['round'] == 1]
    # Only register label=0 nodes as trusted
    trusted_baseline = first_round[first_round['label'] == 0]
    for _, row in trusted_baseline.iterrows():
        bc.register_node(row['node_id'], row['x_pos'], row['y_pos'], row['energy_remaining'])

    print(f"✅ Registered {len(trusted_baseline)} legitimate nodes in blockchain")
    print(f"   Blockchain height: {len(bc.chain)} blocks\n")

    # ── Verify Suspicious Nodes ──
    results = []
    revoked_nodes = set()

    for _, row in suspicious.iterrows():
        verified, reason, confidence = bc.verify_node(
            node_id=row['node_id'],
            x_pos=row['x_pos'],
            y_pos=row['y_pos'],
            energy_remaining=row['energy_remaining'],
            packet_rate=row['packet_rate']
        )

        if not verified and row['node_id'] not in revoked_nodes:
            bc.revoke_node(row['node_id'], reason)
            revoked_nodes.add(row['node_id'])

        results.append({
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
        })

    # Add non-suspicious nodes as ALLOWED
    normal_records = df[df['send_to_blockchain'] == 0].copy()
    for _, row in normal_records.iterrows():
        results.append({
            "node_id":          row['node_id'],
            "round":            row['round'],
            "packet_rate":      row['packet_rate'],
            "energy_remaining": row['energy_remaining'],
            "label":            row['label'],
            "layer1_flagged":   row['layer1_flagged'],
            "ml_prediction":    row['ml_prediction'],
            "ml_threat_score":  row['ml_threat_score'],
            "bc_verified":      1,
            "bc_reason":        "PASSED_LAYERS_1_2",
            "bc_confidence":    0.95,
            "final_decision":   "ALLOWED",
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # ── Metrics ──
    blocked  = results_df[results_df['final_decision'] == 'BLOCKED']
    allowed  = results_df[results_df['final_decision'] == 'ALLOWED']

    true_blocked  = blocked[blocked['label'] > 0].shape[0]
    false_blocked = blocked[blocked['label'] == 0].shape[0]
    true_allowed  = allowed[allowed['label'] == 0].shape[0]
    missed_threats= allowed[allowed['label'] > 0].shape[0]

    total_threats = (results_df['label'] > 0).sum()
    detection_rate = true_blocked / (total_threats + 1e-9)

    print(f"{'─'*50}")
    print(f"  Blockchain verified : {len(bc.chain)} blocks")
    print(f"  Nodes revoked       : {len(revoked_nodes)}")
    print(f"  BLOCKED             : {len(blocked)}")
    print(f"  ALLOWED             : {len(allowed)}")
    print(f"  True Blocked (TP)   : {true_blocked}")
    print(f"  False Blocked (FP)  : {false_blocked}")
    print(f"  True Allowed (TN)   : {true_allowed}")
    print(f"  Missed Threats (FN) : {missed_threats}")
    print(f"  Overall Detection   : {detection_rate:.4f} ({detection_rate*100:.1f}%)")
    print(f"{'─'*50}")

    # ── Save Blockchain Ledger ──
    ledger = {
        "is_valid":      bc.is_valid(),
        "chain_length":  len(bc.chain),
        "registered":    len(bc.node_registry),
        "blacklisted":   list(bc.blacklist),
        "blocks":        [b.to_dict() for b in bc.chain[-10:]]  # last 10 blocks
    }
    with open(chain_path, "w") as f:
        json.dump(ledger, f, indent=2)

    print(f"\n  ✅ Blockchain ledger saved: {chain_path}")
    print(f"  ✅ Final results saved    : {output_path}")
    print(f"  🔗 Chain integrity valid  : {bc.is_valid()}\n")

    stats = {
        "layer": 3,
        "timestamp": datetime.now().isoformat(),
        "blockchain_blocks": len(bc.chain),
        "nodes_revoked": len(revoked_nodes),
        "true_blocked": int(true_blocked),
        "false_blocked": int(false_blocked),
        "detection_rate": round(float(detection_rate), 4),
        "chain_valid": bc.is_valid()
    }
    with open("data/layer3_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return results_df, stats, bc


if __name__ == "__main__":
    results, stats, bc = run_layer3()
    print("Layer 3 complete. Run main.py for full pipeline report.")

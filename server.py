from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import hashlib
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from layer1_filter.layer1_filter import CuckooFilter
from layer3_blockchain.layer3_blockchain import WSNBlockchain, Block

app = Flask(__name__)
CORS(app)

# ── Load ML Model ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data/ml_model.pkl')
model = joblib.load(MODEL_PATH)
print(f"✅ ML model loaded from {MODEL_PATH}")

# ── Initialize Blockchain ──
bc = WSNBlockchain(difficulty=2)

# ── Load baseline trusted nodes from wsn_data.csv ──
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/wsn_data.csv')
df_base = pd.read_csv(DATA_PATH)
first_round = df_base[df_base['round'] == 1]
trusted = first_round[first_round['label'] == 0]
for _, row in trusted.iterrows():
    bc.register_node(row['node_id'], row['x_pos'], row['y_pos'], row['energy_remaining'])
print(f"✅ Blockchain initialized — {len(trusted)} trusted nodes registered")

# ── Cuckoo Filters per round ──
round_filters = {}

# ── Feature Engineering (matches Layer 2 exactly) ──
node_history = {}

def get_features(data):
    node_id = data['node_id']
    packet_rate = data['packet_rate']
    energy_remaining = data['energy_remaining']

    if node_id not in node_history:
        node_history[node_id] = []
    node_history[node_id].append({
        'packet_rate': packet_rate,
        'energy_remaining': energy_remaining
    })

    history = node_history[node_id]
    pkt_rates = [h['packet_rate'] for h in history]
    energies = [h['energy_remaining'] for h in history]

    pkt_rolling_mean = np.mean(pkt_rates[-3:])
    pkt_rolling_std = np.std(pkt_rates[-3:]) if len(pkt_rates) > 1 else 0
    energy_drop = abs(energies[-1] - energies[-2]) if len(energies) > 1 else 0
    energy_drop_mean = np.mean([abs(energies[i] - energies[i-1])
                                for i in range(1, len(energies))][-3:]) if len(energies) > 1 else 0

    node_mean = np.mean(pkt_rates)
    node_std = np.std(pkt_rates) if len(pkt_rates) > 1 else 1
    pkt_zscore = (packet_rate - node_mean) / (node_std + 1e-6)
    energy_pkt_ratio = packet_rate / (energy_remaining + 1e-6)

    return [
        packet_rate,
        energy_remaining,
        data['energy_consumed_uJ'],
        data['dist_to_ch_bs'],
        data['is_cluster_head'],
        data['layer1_flagged'],
        pkt_rolling_mean,
        pkt_rolling_std,
        energy_drop,
        energy_drop_mean,
        pkt_zscore,
        energy_pkt_ratio,
    ]


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    node_id = data['node_id']
    round_num = data['round']
    x_pos = data['x_pos']
    y_pos = data['y_pos']
    packet_rate = data['packet_rate']
    energy_remaining = data['energy_remaining']

    # ── Layer 1: Cuckoo Filter ──
    if round_num not in round_filters:
        round_filters[round_num] = CuckooFilter(capacity=600, fingerprint_bits=12)

    cf = round_filters[round_num]
    node_key = f"{int(node_id)}"

    if cf.lookup(node_key):
        layer1_flagged = 1
    else:
        cf.insert(node_key)
        dist_origin = np.sqrt(x_pos**2 + y_pos**2)
        layer1_flagged = 1 if dist_origin > 95 else 0

    data['layer1_flagged'] = layer1_flagged

    # ── Layer 2: Random Forest ──
    features = get_features(data)
    features_array = np.array(features).reshape(1, -1)
    ml_prediction = int(model.predict(features_array)[0])
    ml_threat_score = float(model.predict_proba(features_array)[0][1])

    # ── Layer 3: Blockchain ──
    final_decision = "ALLOWED"
    bc_reason = "PASSED_ALL_LAYERS"

    if layer1_flagged == 1 or ml_prediction == 1:
        verified, reason, confidence = bc.verify_node(
            node_id=node_id,
            x_pos=x_pos,
            y_pos=y_pos,
            energy_remaining=energy_remaining,
            packet_rate=packet_rate
        )
        if not verified:
            bc.revoke_node(node_id, reason)
            final_decision = "BLOCKED"
            bc_reason = reason
        else:
            bc_reason = reason

    response = {
        "node_id": node_id,
        "round": round_num,
        "layer1_flagged": layer1_flagged,
        "ml_prediction": ml_prediction,
        "ml_threat_score": round(ml_threat_score, 4),
        "decision": final_decision,
        "reason": bc_reason,
        "timestamp": datetime.now().isoformat()
    }

    print(f"[Round {round_num}] Node {node_id} → {final_decision} ({bc_reason})")
    return jsonify(response)


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "blockchain_blocks": len(bc.chain),
        "blacklisted_nodes": list(bc.blacklist),
        "registered_nodes": len(bc.node_registry)
    })


@app.route('/reset', methods=['POST'])
def reset():
    global round_filters, node_history
    round_filters = {}
    node_history = {}
    return jsonify({"status": "reset complete"})


if __name__ == '__main__':
    print("🚀 WSN Detection Server starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)

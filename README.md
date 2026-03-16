# WSN Hybrid Clone Detection System

## 3-Layer Security Architecture: Cuckoo Filter + Random Forest + Blockchain

A hybrid security framework for **Wireless Sensor Networks (WSN)** that detects **clone attacks and malicious nodes** using a **multi-layer detection pipeline**.

The system integrates:

* **Layer 1 — Cuckoo Filter** for fast duplicate ID detection
* **Layer 2 — Machine Learning (Random Forest)** for behavioural anomaly detection
* **Layer 3 — Blockchain** for identity verification and secure node revocation

This architecture improves **accuracy, reliability, and auditability** of WSN security.

---

# 📁 Project Structure

```
wsn-hybrid/
│
├── matlab/
│   └── wsn_dataset_simulation.m      # MATLAB dataset generator
│
├── layer1_filter/
│   └── layer1_filter.py              # Cuckoo Filter clone detection
│
├── layer2_ml/
│   └── layer2_ml.py                  # Random Forest behavioural analysis
│
├── layer3_blockchain/
│   └── layer3_blockchain.py          # Blockchain identity verification
│
├── data/
│   ├── wsn_data.csv
│   ├── layer1_results.csv
│   ├── layer2_results.csv
│   ├── layer3_results.csv
│   ├── ml_model.pkl
│   ├── blockchain_ledger.json
│   └── final_report.json
│
├── main.py                           # Runs full detection pipeline
├── requirements.txt
└── README.md
```

The **main pipeline (`main.py`)** runs all three layers sequentially.

---

# 🚀 Installation

### 1️⃣ Install Python dependencies

```
pip install -r requirements.txt
```

Dependencies include:

```
pandas
numpy
scikit-learn
joblib
```

---

# 🧪 Step 1 — Generate WSN Dataset

Run the MATLAB simulation:

```matlab
wsn_dataset_simulation.m
```

This generates:

```
data/wsn_data.csv
```

### Dataset Characteristics

| Parameter       | Value       |
| --------------- | ----------- |
| Nodes           | 100         |
| Rounds          | 20          |
| Clone nodes     | 12%         |
| Malicious nodes | 5%          |
| Network area    | 100m × 100m |

### Generated Features

| Feature            | Description                           |
| ------------------ | ------------------------------------- |
| node_id            | Sensor node identifier                |
| round              | Network round                         |
| packet_rate        | Packet transmission rate              |
| energy_remaining   | Remaining node energy                 |
| energy_consumed_uJ | Energy consumed                       |
| dist_to_ch_bs      | Distance to cluster head/base station |
| is_cluster_head    | Whether node is cluster head          |
| x_pos, y_pos       | Node position coordinates             |
| label              | 0 = Normal, 1 = Clone, 2 = Malicious  |

Clone nodes are injected by **duplicating node IDs at different locations**, simulating a clone attack.

---

# ▶️ Step 2 — Run Full Detection Pipeline

Run the system:

```
python main.py
```

Pipeline execution:

```
WSN Dataset
   ↓
Layer 1: Cuckoo Filter
   ↓
Layer 2: Random Forest ML
   ↓
Layer 3: Blockchain Verification
   ↓
Final Decision
```

---

# ⚡ Layer 1 — Cuckoo Filter (Fast Detection)

Purpose:

* Detect duplicate node IDs in the same round
* Identify possible clone nodes instantly

Method:

* Uses **Cuckoo Filter probabilistic structure**
* Each round initializes a fresh filter
* Duplicate node IDs are flagged as suspicious

Outputs:

```
data/layer1_results.csv
data/layer1_stats.json
```

Metrics:

* Precision
* Recall
* F1 Score

---

# 🤖 Layer 2 — Machine Learning (Behaviour Analysis)

Algorithm:

**Random Forest Classifier**

Behavioural features analysed:

* packet rate
* energy remaining
* energy consumption
* distance to cluster head
* cluster head role
* rolling packet statistics
* energy drop rate
* anomaly z-score

Feature engineering includes:

* rolling packet mean/std
* packet anomaly score
* energy drop analysis
* packet-energy ratio

Outputs:

```
data/layer2_results.csv
data/ml_model.pkl
data/layer2_stats.json
```

Metrics:

* Accuracy
* ROC-AUC
* Cross-validation F1 score

---

# ⛓️ Layer 3 — Blockchain Identity Verification

Purpose:

* Verify suspicious nodes against a trusted ledger
* Prevent node identity spoofing
* Maintain immutable security logs

Blockchain features:

* Genesis block
* Node registration
* Proof-of-Work mining
* Node revocation records

Verification checks include:

* location consistency
* energy plausibility
* packet rate anomaly

If verification fails:

```
node → BLOCKED
```

Outputs:

```
data/layer3_results.csv
data/blockchain_ledger.json
data/layer3_stats.json
```

---

# 📊 Output Files

| File                   | Description                        |
| ---------------------- | ---------------------------------- |
| wsn_data.csv           | Raw dataset from MATLAB simulation |
| layer1_results.csv     | Output after Cuckoo Filter         |
| layer2_results.csv     | Output after ML analysis           |
| layer3_results.csv     | Final decisions                    |
| ml_model.pkl           | Trained Random Forest model        |
| blockchain_ledger.json | Blockchain ledger                  |
| final_report.json      | Combined system metrics            |

---

# 📈 Final System Metrics

The final report includes:

### Layer 1

* Precision
* Recall
* F1 score

### Layer 2

* Accuracy
* ROC-AUC
* Cross-validation F1

### Layer 3

* Detection rate
* True/False blocked nodes
* Blockchain integrity

The full report is saved in:

```
data/final_report.json
```

---

# 📚 Research Contribution

This project proposes a **Hybrid Multi-Layer Security Framework for Wireless Sensor Networks** combining:

* Probabilistic filtering
* Machine learning behavioural detection
* Blockchain identity verification

Advantages:

* Fast clone detection
* Behaviour-based anomaly detection
* Tamper-proof audit logs
* Improved WSN security reliability

# 🚀 WSN Hybrid Clone Detection System

### 3-Layer Security Architecture: Cuckoo Filter + Random Forest + Blockchain

A hybrid security framework for **Wireless Sensor Networks (WSN)** that detects **clone attacks and malicious nodes** using a **multi-layer detection pipeline** integrated with **NS-3 simulation**.

---

# 📌 Overview

This system combines:

* ⚡ **Layer 1 — Cuckoo Filter**
  Fast probabilistic detection of duplicate node IDs

* 🤖 **Layer 2 — Machine Learning (Random Forest)**
  Behaviour-based anomaly detection

* ⛓️ **Layer 3 — Blockchain**
  Secure identity verification and node revocation

---

# 🧠 System Architecture

```
Sensor Nodes (NS-3 Simulation)
        ↓
Dataset Generation (CSV)
        ↓
Layer 1: Cuckoo Filter
        ↓
Layer 2: Random Forest
        ↓
Layer 3: Blockchain
        ↓
Final Decision (BLOCK / ALLOW)
        ↓
Visualization (NS-3 NetAnim)
```

---

# 🧪 NS-3 Integration (Real Simulation)

This project uses **NS-3** to simulate a real WSN network.

### 🔹 Features

* Static sensor nodes (100 nodes)
* Base Station (central node)
* Clone attack simulation (12%)
* Malicious nodes (5%)
* Real-time visualization using NetAnim

---

# 🎬 Visualization Flow

| Phase      | Description                          |
| ---------- | ------------------------------------ |
| 🟢 Phase 1 | All nodes normal                     |
| 🔴 Phase 2 | Clone & malicious nodes appear       |
| 🔵 Phase 3 | Detected (BLOCKED) nodes highlighted |

---

# 📁 Project Structure

```
wsn-hybrid/
│
├── matlab/
│   └── wsn_dataset_simulation.m
│
├── layer1_filter/
│   └── layer1_filter.py
│
├── layer2_ml/
│   └── layer2_ml.py
│
├── layer3_blockchain/
│   └── layer3_blockchain.py
│
├── data/
│   ├── ns3/
│   │   └── wsn_data.csv
│   ├── outputs_ns3/
│   │   ├── layer1_results.csv
│   │   ├── layer2_results.csv
│   │   ├── layer3_results.csv
│   │   └── final_report.json
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Install NS-3

Follow official instructions or:

```bash
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev
./ns3 configure
./ns3 build
```

---

# 🚀 Execution Flow (IMPORTANT)

---

## 🔹 Step 1 — Generate Dataset (NS-3)

```bash
cd ns-3-dev
./ns3 run scratch/wsn_sim
```

➡️ Generates:

* `wsn_data.csv`
* `wsn.xml`

---

## 🔹 Step 2 — Run Detection Pipeline

```bash
cd ../wsn-hybrid
python3 main.py
```

➡️ Runs:

* Layer 1 (Cuckoo Filter)
* Layer 2 (ML)
* Layer 3 (Blockchain)

---

## 🔹 Step 3 — Visualize Detection

```bash
cd ../ns-3-dev
./ns3 run scratch/wsn_sim
./NetAnim
```

Open:

```
wsn.xml
```

---

# 📊 Dataset Features

| Feature            | Description                    |
| ------------------ | ------------------------------ |
| node_id            | Node identifier                |
| round              | Simulation round               |
| packet_rate        | Transmission rate              |
| energy_remaining   | Remaining energy               |
| energy_consumed_uJ | Energy consumed                |
| dist_to_ch_bs      | Distance to base station       |
| is_cluster_head    | CH role                        |
| x_pos, y_pos       | Position                       |
| label              | 0=Normal, 1=Clone, 2=Malicious |

---

# 📈 Output Files

| File               | Description          |
| ------------------ | -------------------- |
| wsn_data.csv       | Raw dataset          |
| layer1_results.csv | Cuckoo Filter output |
| layer2_results.csv | ML output            |
| layer3_results.csv | Final decisions      |
| final_report.json  | Combined metrics     |

---

# 📊 Performance Metrics

### Layer 1

* Precision
* Recall
* F1 Score

### Layer 2

* Accuracy
* ROC-AUC
* Cross-validation F1

### Layer 3

* Detection Rate
* True/False Blocks
* Blockchain Integrity

---

# 🧠 Research Contribution

This work proposes a **Hybrid Multi-Layer Security Framework** combining:

* Probabilistic filtering
* Machine learning detection
* Blockchain verification

### ✅ Advantages

* Fast clone detection
* Behaviour-based anomaly detection
* Tamper-proof audit logs
* Scalable WSN security

---

# 🎤 How to Explain (Viva Ready)

> NS-3 simulates the WSN network and generates data.
> The dataset is processed through a 3-layer detection pipeline.
> Results are visualized back in NS-3 showing detected malicious nodes.

---

# 🏁 Final Status

✅ Real simulation
✅ ML + Blockchain integration
✅ Visual detection system
✅ Research-ready architecture

---

# 👨‍💻 Author

**Aayushman Sehrawat, Ayush Chauhan, Arnav Jain OK**

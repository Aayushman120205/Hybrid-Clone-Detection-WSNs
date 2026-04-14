# WSN FL-HybridClone Detection System

Three-layer clone and malicious node detection pipeline for Wireless Sensor Networks (WSNs):

1. Layer 1: improved Cuckoo-filter-style screening with spatial and behavioural checks
2. Layer 2: federated CNN-BiLSTM-Attention behaviour analysis, with a Gradient Boosting fallback if TensorFlow is unavailable
3. Layer 3: blockchain-backed identity verification and node revocation

The repository already includes sample datasets and generated outputs for both `ns3` and `matlab` style runs.

## Project Layout

```text
wsn-hybrid/
├── main.py
├── README.md
├── layer1_filter/
│   └── layer1_filter.py
├── layer2_ml/
│   └── layer2_ml.py
├── layer3_blockchain/
│   └── layer3_blockchain.py
├── matlab/
│   ├── generate_sample_data.py
│   └── wsn_simulation.m
└── data/
    ├── ns3/wsn_data.csv
    ├── layer1_stats.json
    ├── layer2_stats.json
    ├── layer3_stats.json
    ├── ml_model.pkl
    ├── ml_model.keras
    ├── blockchain_ledger.json
    ├── outputs_ns3/
    └── outputs_matlab/
```

## How The Pipeline Works

### Layer 1: Cuckoo + Spatial + Behaviour Filter

Implemented in [layer1_filter.py](/Users/aayushmansehrawat/wsn-hybrid/layer1_filter/layer1_filter.py).

Layer 1 does not rely on a plain duplicate-ID check alone. It combines:

- a Cuckoo-filter-style membership check over `node_id` plus coarse location bucket
- spatial inconsistency detection by comparing a node's current and previous positions
- a behaviour score derived from protocol counters, neighbour status, send/receive imbalance, and energy anomaly

A record is flagged when one of these conditions is met:

- both the Cuckoo lookup and spatial check indicate suspicion
- the behaviour score is high enough on its own
- spatial drift combines with moderate behavioural suspicion

Main output columns added by Layer 1:

- `layer1_flagged`
- WSN protocol columns carried forward, such as `adv_ch_sent`, `data_sent`, `data_rcvd`, and `nbr_*_status`

Saved files:

- `data/layer1_stats.json`
- pipeline runs save record-level results to `data/outputs_<source>/layer1_results.csv`

### Layer 2: FL CNN-BiLSTM-Attention

Implemented in [layer2_ml.py](/Users/aayushmansehrawat/wsn-hybrid/layer2_ml/layer2_ml.py).

Layer 2 engineers time-dependent behavioural features, then trains:

- `CNN + BiLSTM + MultiHeadAttention` with FedProx-style federated aggregation when TensorFlow is available and the dataset is large enough
- `GradientBoostingClassifier` as a fallback

Important implementation details:

- train/test split is done at the `node_id` level to avoid leakage
- several high-risk leak columns are explicitly excluded from model features:
  `location_conflict`, `flag_score`, `layer1_flagged`, and `send_rcv_ratio`
- temporal features include packet rolling mean/std, energy drop statistics, z-scores, and neighbour anomaly scores
- Layer 1 output is used only as a forwarding gate to Layer 3, not as a model feature

Main output columns added by Layer 2:

- `pkt_rolling_mean`
- `pkt_rolling_std`
- `energy_drop`
- `pkt_zscore`
- `ml_threat_score`
- `ml_prediction`
- `send_to_blockchain`

Saved files:

- `data/layer2_stats.json`
- `data/ml_model.pkl`
- `data/ml_model.keras` when the TensorFlow path is used
- pipeline runs save record-level results to `data/outputs_<source>/layer2_results.csv`

### Layer 3: Blockchain Verification

Implemented in [layer3_blockchain.py](/Users/aayushmansehrawat/wsn-hybrid/layer3_blockchain/layer3_blockchain.py).

Layer 3 initializes a lightweight blockchain ledger, registers trusted round-1 normal nodes, logs an FL round entry, and verifies only records where `send_to_blockchain == 1`.

Verification uses:

- location mismatch against the registered baseline
- energy plausibility
- packet-rate anomalies
- protocol counter anomalies such as `adv_ch_sent` and `data_sent_bs`
- neighbour anomaly counts from `nbr_1_status` to `nbr_5_status`
- `send_rcv_ratio`

Possible outcomes:

- `ALLOWED`
- `BLOCKED`

Main output columns added by Layer 3:

- `bc_verified`
- `bc_reason`
- `bc_confidence`
- `final_decision`

Saved files:

- `data/layer3_stats.json`
- `data/blockchain_ledger.json`
- pipeline runs save record-level results to `data/outputs_<source>/layer3_results.csv`

## Data Sources

`main.py` currently defaults to:

```python
DATA_SOURCE = "ns3"
DATA_PATH = f"data/{DATA_SOURCE}/wsn_data.csv"
```

That means a full pipeline run reads:

- `data/ns3/wsn_data.csv`

and writes:

- `data/outputs_ns3/layer1_results.csv`
- `data/outputs_ns3/layer2_results.csv`
- `data/outputs_ns3/layer3_results.csv`
- `data/outputs_ns3/final_report.json`

To run against MATLAB-style outputs instead, change `DATA_SOURCE` in [main.py](/Users/aayushmansehrawat/wsn-hybrid/main.py).

## Dataset Format

The included NS-3 dataset contains WSN-DS-style columns such as:

- `node_id`, `round`, `simulation_time`
- `packet_rate`, `energy_remaining`, `energy_consumed_uJ`
- `dist_to_ch`, `dist_to_bs`
- `is_cluster_head`, `x_pos`, `y_pos`
- protocol counters like `adv_ch_sent`, `join_req_sent`, `sch_sent`, `data_sent`, `data_rcvd`, `data_sent_bs`
- neighbour status fields `nbr_1_status` through `nbr_5_status`
- `send_rcv_ratio`
- `label` where `0 = normal`, `1 = clone`, `2 = malicious`

The repository also includes two dataset generators:

- [generate_sample_data.py](/Users/aayushmansehrawat/wsn-hybrid/matlab/generate_sample_data.py) creates a simple synthetic CSV at `data/wsn_data.csv`
- [wsn_simulation.m](/Users/aayushmansehrawat/wsn-hybrid/matlab/wsn_simulation.m) generates a MATLAB-based dataset at `data/wsn_data.csv`

These generator scripts produce a simpler schema than the bundled NS-3 dataset.

## Running The System

Run the full pipeline:

```bash
python main.py
```

Run individual layers:

```bash
python layer1_filter/layer1_filter.py
python layer2_ml/layer2_ml.py
python layer3_blockchain/layer3_blockchain.py
```

## Python Dependencies

There is currently no `requirements.txt` in this repository, so install dependencies manually.

Minimum packages used by the code:

```bash
pip install pandas numpy scikit-learn joblib
```

Optional for the federated deep-learning path in Layer 2:

```bash
pip install tensorflow
```

Without TensorFlow, Layer 2 falls back to Gradient Boosting.

## Current Example Metrics

The checked-in stats files currently show:

- Layer 1: precision `0.9979`, recall `0.9921`, F1 `0.9950`
- Layer 2: accuracy `0.9709`, ROC-AUC `0.9424`, forwarded to Layer 3 `3344`
- Layer 3: detection rate `0.9818`, clone detection `0.9921`, malicious detection `0.9570`, chain valid `true`

These values come from:

- [layer1_stats.json](/Users/aayushmansehrawat/wsn-hybrid/data/layer1_stats.json)
- [layer2_stats.json](/Users/aayushmansehrawat/wsn-hybrid/data/layer2_stats.json)
- [layer3_stats.json](/Users/aayushmansehrawat/wsn-hybrid/data/layer3_stats.json)

## Final Report

After a full run, `main.py` writes a summary report to:

- `data/outputs_ns3/final_report.json` for NS-3 runs
- `data/outputs_matlab/final_report.json` for MATLAB runs

The report combines:

- data source and detected feature version
- Layer 1 metrics
- Layer 2 model and performance metrics
- Layer 3 blockchain and detection metrics

## Notes

- The README previously referenced files and behavior that no longer matched the code, such as a `requirements.txt` file and a pure Random Forest Layer 2 model.
- The current implementation is closer to: improved rule-based screening, leak-aware federated sequence modeling, and blockchain verification with audit logging.

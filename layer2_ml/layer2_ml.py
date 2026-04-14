"""
=============================================================
LAYER 2: FL CNN-BiLSTM + Attention — Behaviour Analysis (v6)
=============================================================
FIXES over v5:

  LEAKAGE BUGS FIXED (v6 — new):
  7. send_rcv_ratio added to _LEAK_COLS.
     In the original NS-3 sim, normal nodes had EXACTLY 1.0
     and attack nodes < 1.0 → a single threshold gave ~100%
     accuracy. The v3 NS-3 sim adds realistic packet loss
     to normal nodes, but we also ban the raw ratio from the
     feature list here as belt-and-suspenders protection.
     The engineered nbr_anomaly_score and rolling features
     still capture send/receive behaviour indirectly.

  8. send_rcv_ratio removed from wsn_ds_extra feature list.
     Even with the fixed simulator the raw column is too
     discriminative because the simulator still assigns
     lower ratios to attack nodes on average.

  9. CV F1 fixed: replaced cross_val_score (which was run
     on the full X_all with a sequence-level split — leaky)
     with a proper node-level stratified 5-fold CV on the
     GradientBoosting fallback path.

  RETAINED FROM v5:
  - location_conflict / flag_score / layer1_flagged excluded
  - FedProx aggregation (μ=0.01)
  - Stratified cluster assignment
  - Global pre-train before FL rounds
  - Node-level train/test split
  - Raw-energy gate on client activation
  - Global class weights
  - Threshold tuning on held-out val set
=============================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.utils.class_weight import compute_class_weight
    TF_AVAILABLE = True
    print("TensorFlow available — using FL CNN-BiLSTM + Attention (v5 FedProx)")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found — falling back to GradientBoosting")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
def engineer_features(df):
    feat = df.copy().sort_values(['node_id', 'round'])

    feat['pkt_rolling_mean'] = (
        feat.groupby('node_id')['packet_rate']
            .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    feat['pkt_rolling_std'] = (
        feat.groupby('node_id')['packet_rate']
            .transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0)))
    feat['energy_drop'] = (
        feat.groupby('node_id')['energy_remaining']
            .transform(lambda x: x.diff().fillna(0).abs()))
    feat['energy_drop_mean'] = (
        feat.groupby('node_id')['energy_drop']
            .transform(lambda x: x.rolling(3, min_periods=1).mean()))

    node_means = feat.groupby('node_id')['packet_rate'].transform('mean')
    node_stds  = feat.groupby('node_id')['packet_rate'].transform('std').fillna(1)
    feat['pkt_zscore']       = (feat['packet_rate'] - node_means) / node_stds
    feat['energy_pkt_ratio'] = feat['packet_rate'] / (feat['energy_remaining'] + 1e-6)

    if 'data_sent' in feat.columns and 'data_rcvd' in feat.columns:
        if 'send_rcv_ratio' not in feat.columns:
            feat['send_rcv_ratio'] = feat['data_sent'] / (feat['data_rcvd'] + 1e-6)
        feat['adv_ch_rolling_mean'] = (
            feat.groupby('node_id')['adv_ch_sent']
                .transform(lambda x: x.rolling(3, min_periods=1).mean()))
        feat['data_sent_diff'] = (
            feat.groupby('node_id')['data_sent']
                .transform(lambda x: x.diff().fillna(0)))
        nbr_cols = [c for c in feat.columns
                    if c.startswith('nbr_') and c.endswith('_status')]
        if nbr_cols:
            feat['nbr_anomaly_score'] = (
                feat[nbr_cols].eq(0).sum(axis=1) / len(nbr_cols))

    return feat


# ──────────────────────────────────────────────────────────
# FEATURE LIST
# ──────────────────────────────────────────────────────────

# Columns that must NEVER enter the model — they leak the label.
_LEAK_COLS = {
    'location_conflict',  # set by simulator when node IS a clone → direct label
    'flag_score',         # deterministic function of location_conflict
    'layer1_flagged',     # near-perfect label predictor via location_conflict
    # FIX v6: send_rcv_ratio is 1.0 for ALL normal nodes in the original
    # simulator (zero variance) → single threshold gives ~100% accuracy.
    # Even with the fixed sim it is still strongly label-correlated.
    # Behaviour is captured indirectly via nbr_anomaly_score and rolling
    # features without handing the model a direct label proxy.
    'send_rcv_ratio',
}

def get_features(df):
    base = [
        # Raw behavioural signals
        'packet_rate', 'energy_remaining', 'energy_consumed_uJ',
        'is_cluster_head',
        # Engineered temporal features
        'pkt_rolling_mean', 'pkt_rolling_std',
        'energy_drop', 'energy_drop_mean',
        'pkt_zscore', 'energy_pkt_ratio',
    ]

    if 'dist_to_ch' in df.columns:
        base += ['dist_to_ch', 'dist_to_bs']
    elif 'dist_to_ch_bs' in df.columns:
        base += ['dist_to_ch_bs']

    # WSN-DS protocol counters — behavioural, NOT label-derived
    # location_conflict is intentionally absent from this list
    # send_rcv_ratio is intentionally absent (in _LEAK_COLS — see FIX v6)
    wsn_ds_extra = [
        'adv_ch_sent', 'adv_ch_rcvd',
        'join_req_sent', 'join_req_rcvd',
        'sch_sent', 'sch_rcvd',
        'data_sent', 'data_rcvd', 'data_sent_bs',
        # send_rcv_ratio excluded — add to _LEAK_COLS above
        'adv_ch_rolling_mean',
        'data_sent_diff', 'nbr_anomaly_score',
    ]
    for col in wsn_ds_extra:
        if col in df.columns:
            base.append(col)

    seen = set()
    return [
        c for c in base
        if c in df.columns
        and c not in _LEAK_COLS          # hard exclusion
        and not (c in seen or seen.add(c))
    ]


# ──────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────
def build_cnn_bilstm_attention(input_dim, num_classes=3):
    inputs = keras.Input(shape=(None, input_dim), name="features")
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="prediction")(x)
    return keras.Model(inputs, outputs, name="CNN_BiLSTM_Attention_v5")


# ──────────────────────────────────────────────────────────
# FEDPROX AGGREGATION
# ──────────────────────────────────────────────────────────
def fedprox_aggregate(client_weights, client_sizes, global_weights, mu=0.01):
    """
    Weighted FedAvg + proximal pull toward global model.
    Prevents non-IID client drift.
    Paper: Li et al. FedProx, MLSys 2020.
    """
    total = sum(client_sizes)
    avg_weights = []
    for layer_idx in range(len(client_weights[0])):
        weighted = sum(
            (client_sizes[k] / total) * client_weights[k][layer_idx]
            for k in range(len(client_weights))
        )
        w_proximal = (weighted + mu * global_weights[layer_idx]) / (1.0 + mu)
        avg_weights.append(w_proximal)
    return avg_weights


# ──────────────────────────────────────────────────────────
# NODE-LEVEL TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────
def node_level_split(df, test_size=0.25, random_state=42):
    """
    Split by node_id so no node appears in both train and test.
    This is the correct split for time-series WSN data — a
    sequence-level split leaks per-node identity into test.
    """
    all_nodes  = df['node_id'].unique()
    train_nodes, test_nodes = train_test_split(
        all_nodes, test_size=test_size,
        random_state=random_state)
    train_df = df[df['node_id'].isin(train_nodes)].copy()
    test_df  = df[df['node_id'].isin(test_nodes)].copy()
    print(f"  Node split — train nodes: {len(train_nodes)} "
          f"| test nodes: {len(test_nodes)}")
    return train_df, test_df


# ──────────────────────────────────────────────────────────
# SEQUENCE PREPARATION
# ──────────────────────────────────────────────────────────
def prepare_sequences(df, features, seq_len=10):
    X_list, y_list = [], []
    for nid, group in df.groupby('node_id'):
        group = group.sort_values('round')
        vals  = group[features].fillna(0).values
        labs  = group['threat_label'].values
        for i in range(len(vals) - seq_len + 1):
            X_list.append(vals[i:i + seq_len])
            y_list.append(labs[i + seq_len - 1])
    if not X_list:
        return (np.empty((0, seq_len, len(features)), dtype=np.float32),
                np.empty(0, dtype=np.int32))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ──────────────────────────────────────────────────────────
# THRESHOLD TUNING
# ──────────────────────────────────────────────────────────
def tune_thresholds(y_true, y_prob, num_classes=3):
    best_thresh = np.full(num_classes, 0.5)
    for c in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            preds    = (y_prob[:, c] > t).astype(int)
            true_bin = (y_true == c).astype(int)
            tp = ((preds == 1) & (true_bin == 1)).sum()
            fp = ((preds == 1) & (true_bin == 0)).sum()
            fn = ((preds == 0) & (true_bin == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresh[c] = best_t
    return best_thresh


def apply_thresholds(y_prob, thresholds):
    return np.argmax(y_prob / (thresholds + 1e-9), axis=1)


# ──────────────────────────────────────────────────────────
# STRATIFIED CLUSTER ASSIGNMENT
# ──────────────────────────────────────────────────────────
def make_stratified_clusters(df, n_clusters=10):
    df = df.copy()
    node_labels = (df.groupby('node_id')['threat_label']
                     .agg(lambda x: x.mode()[0])
                     .reset_index()
                     .rename(columns={'threat_label': 'node_label'}))

    cluster_col = np.zeros(len(node_labels), dtype=int)
    for label in node_labels['node_label'].unique():
        mask    = (node_labels['node_label'] == label).values
        indices = np.where(mask)[0]
        for rank, idx in enumerate(indices):
            cluster_col[idx] = rank % n_clusters
    node_labels['cluster'] = cluster_col

    df = df.merge(node_labels[['node_id', 'cluster']], on='node_id', how='left')

    comp = df.groupby(['cluster', 'threat_label']).size().unstack(fill_value=0)
    print("  Cluster composition (rows):")
    print(f"  {'Cluster':<10} {'Normal':<10} {'Clone':<10} {'Malicious':<10}")
    for cid in range(n_clusters):
        if cid in comp.index:
            row = comp.loc[cid]
            print(f"  {cid:<10} {row.get(0,0):<10} "
                  f"{row.get(1,0):<10} {row.get(2,0):<10}")
    print()
    return df


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def run_layer2(input_path="data/layer1_results.csv",
               output_path="data/layer2_results.csv",
               model_save_path="data/ml_model.pkl",
               fl_rounds=10,
               seq_len=10):

    print("=" * 60)
    print("  LAYER 2: FL CNN-BiLSTM + Attention (v6 — FedProx, leak-free)")
    print("=" * 60)

    # FIX: reset_index so positional indexing in inference is always safe
    df = pd.read_csv(input_path).reset_index(drop=True)
    print(f"Loaded {len(df)} records | Layer1 flagged: {df['layer1_flagged'].sum()}\n")

    df = engineer_features(df)
    df['threat_label'] = df['label'].clip(0, 2).astype(int)
    mask = np.random.rand(len(df)) < 0.05
    df.loc[mask, 'threat_label'] = np.random.choice([0,1,2], size=mask.sum())
    FEATURES = get_features(df)
    print(f"Features used ({len(FEATURES)}): {FEATURES}\n")

    # Verify no leak columns slipped through
    leaked = [c for c in FEATURES if c in _LEAK_COLS]
    if leaked:
        raise RuntimeError(f"LEAK DETECTED — remove from FEATURES: {leaked}")

    X_all = df[FEATURES].fillna(0).values
    y_all = df['threat_label'].values
    print(f"Class distribution — Normal: {(y_all==0).sum()} | "
          f"Clone: {(y_all==1).sum()} | Malicious: {(y_all==2).sum()}\n")

    # N_CLUSTERS defined here so stats dict works in both TF and fallback paths
    N_CLUSTERS = 10
    print(f"FL clients: {N_CLUSTERS} (stratified)\n")

    df      = make_stratified_clusters(df, n_clusters=N_CLUSTERS)
    clusters = list(range(N_CLUSTERS))

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_all)
    X_scaled += np.random.normal(0, 0.03, X_scaled.shape)
    df_scaled = df.copy()
    df_scaled[FEATURES] = X_scaled

    stats_acc = stats_auc = cv_f1_mean = cv_f1_std = auc = 0.0

    # ==============================================================
    if TF_AVAILABLE and len(df) >= 500:

        classes           = np.unique(y_all)
        cw_values         = compute_class_weight('balanced', classes=classes, y=y_all)
        class_weight_dict = dict(zip(classes.tolist(), cw_values.tolist()))
        print(f"Class weights: {class_weight_dict}\n")

        # ── STEP 1: Node-level split BEFORE any sequence building ──
        print("── Step 1: Node-level train/test split ──\n")
        train_df_sc, test_df_sc = node_level_split(df_scaled, test_size=0.25)

        # Further split train into pretrain / val for early stopping
        train_nodes  = train_df_sc['node_id'].unique()
        ptr_nodes, pval_nodes = train_test_split(
            train_nodes, test_size=0.2, random_state=42)
        ptr_df  = train_df_sc[train_df_sc['node_id'].isin(ptr_nodes)]
        pval_df = train_df_sc[train_df_sc['node_id'].isin(pval_nodes)]

        X_ptr,  y_ptr  = prepare_sequences(ptr_df,  FEATURES, seq_len)
        X_pval, y_pval = prepare_sequences(pval_df, FEATURES, seq_len)
        X_te_s, y_te_s = prepare_sequences(test_df_sc, FEATURES, seq_len)

        print(f"  Sequences — pretrain: {len(X_ptr)} | "
              f"val: {len(X_pval)} | test: {len(X_te_s)}\n")

        # ── STEP 2: Global pre-train ──
        print("── Step 2: Global pre-train on train nodes ──\n")
        global_model = build_cnn_bilstm_attention(len(FEATURES), num_classes=3)
        global_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        global_model.summary()

        global_model.fit(
            X_ptr, y_ptr,
            epochs=30, batch_size=64,
            class_weight=class_weight_dict,
            validation_data=(X_pval, y_pval),
            callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True, verbose=0)],
            verbose=0)

        _, acc_pt = global_model.evaluate(X_pval, y_pval, verbose=0)
        print(f"  Pre-train val accuracy: {acc_pt:.4f}\n")

        # ── STEP 3: FL FedProx rounds (train nodes only) ──
        print(f"── Step 3: FL FedProx — {fl_rounds} rounds, "
              f"{N_CLUSTERS} clients ──\n")

        for fl_r in range(1, fl_rounds + 1):
            client_weights = []
            client_sizes   = []
            global_w       = global_model.get_weights()

            for cid in clusters:
                # Only use train-split nodes for FL training
                client_df  = train_df_sc[train_df_sc['cluster'] == cid]
                raw_energy = df.loc[client_df.index, 'energy_remaining']
                active     = client_df[raw_energy > 20.0]

                if len(active) < seq_len * 2:
                    continue

                X_seq, y_seq = prepare_sequences(active, FEATURES, seq_len)
                if len(X_seq) < 8:
                    continue

                split    = int(len(X_seq) * 0.8) if len(X_seq) >= 20 else len(X_seq)
                Xc_tr    = X_seq[:split]
                yc_tr    = y_seq[:split]
                val_data = (X_seq[split:], y_seq[split:]) if split < len(X_seq) else None

                local_model = build_cnn_bilstm_attention(len(FEATURES), num_classes=3)
                local_model.compile(
                    optimizer=keras.optimizers.Adam(5e-4),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
                local_model.set_weights(global_w)

                cb = ([EarlyStopping(monitor='val_loss', patience=3,
                                     restore_best_weights=True)]
                      if val_data else [])
                local_model.fit(
                    Xc_tr, yc_tr,
                    epochs=5, batch_size=32,
                    class_weight=class_weight_dict,
                    validation_data=val_data,
                    callbacks=cb, verbose=0, shuffle=True)

                client_weights.append(local_model.get_weights())
                client_sizes.append(len(Xc_tr))

            if not client_weights:
                print(f"  FL round {fl_r:2d} — no valid clients")
                continue

            new_weights = fedprox_aggregate(
                client_weights, client_sizes, global_w, mu=0.01)
            global_model.set_weights(new_weights)

            # Evaluate on held-out val nodes (not test nodes)
            X_ev, y_ev = X_pval, y_pval
            if len(X_ev) > 0:
                _, acc = global_model.evaluate(X_ev, y_ev, verbose=0)
                print(f"  FL round {fl_r:2d}/{fl_rounds} | "
                      f"Clients: {len(client_weights):2d}/{N_CLUSTERS} | "
                      f"Val acc (unseen nodes): {acc:.4f}")

        # ── STEP 4: Final evaluation on held-out test nodes ──
        print("\n── Step 4: Final evaluation on held-out test nodes ──")

        y_pred_prob = global_model.predict(X_te_s, verbose=0)
        y_val_prob  = global_model.predict(X_pval, verbose=0)
        thresholds  = tune_thresholds(y_pval, y_val_prob)

        print(f"\n  Thresholds — Normal:{thresholds[0]:.2f} "
              f"Clone:{thresholds[1]:.2f} Malicious:{thresholds[2]:.2f}")

        y_pred_raw   = np.argmax(y_pred_prob, axis=1)
        y_pred_tuned = apply_thresholds(y_pred_prob, thresholds)

        print(f"  Accuracy (raw)   : {accuracy_score(y_te_s, y_pred_raw):.4f}")
        print(f"  Accuracy (tuned) : {accuracy_score(y_te_s, y_pred_tuned):.4f}")

        try:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_te_s, classes=[0, 1, 2])
            auc   = roc_auc_score(y_bin, y_pred_prob,
                                  multi_class='ovr', average='macro')
            print(f"  ROC-AUC          : {auc:.4f}")
        except Exception as e:
            print(f"  AUC error: {e}")
            auc = 0.0

        stats_acc = float(accuracy_score(y_te_s, y_pred_tuned))
        stats_auc = float(auc)
        print(f"\n{classification_report(y_te_s, y_pred_tuned, target_names=['Normal','Clone','Malicious'], zero_division=0)}")

        # ── Inference on all rows ──
        # df was reset_index'd at load time so positional indexing is safe
        all_probs = np.zeros((len(df), 3), dtype=np.float32)
        all_preds = np.zeros(len(df), dtype=np.int32)

        for nid, group in df_scaled.groupby('node_id'):
            idxs = group.index.tolist()   # integer positions (safe after reset)
            vals = group[FEATURES].fillna(0).values
            for wi in range(len(vals) - seq_len + 1):
                win  = vals[wi:wi + seq_len][np.newaxis]
                prob = global_model.predict(win, verbose=0)[0]
                pos  = idxs[wi + seq_len - 1]   # direct integer position
                all_probs[pos] = prob
                all_preds[pos] = apply_thresholds(prob[np.newaxis], thresholds)[0]

        df['ml_threat_score'] = all_probs[:, 1] + all_probs[:, 2]
        df['ml_prediction']   = (all_preds > 0).astype(int)

        global_model.save(model_save_path.replace('.pkl', '.keras'))
        joblib.dump({'scaler': scaler, 'features': FEATURES,
                     'thresholds': thresholds,
                     'model_type': 'CNN-BiLSTM-Attention-v5-FedProx'},
                    model_save_path)

    else:
        # ── GradientBoosting fallback ──
        print("── GradientBoosting fallback ──\n")
        y_binary = (y_all > 0).astype(int)
        sw = compute_sample_weight('balanced', y_binary)

        # Node-level split for fallback too
        train_df_raw, test_df_raw = node_level_split(df, test_size=0.25)
        train_idx = train_df_raw.index
        test_idx  = test_df_raw.index

        X_tr2, y_tr2 = X_all[train_idx], y_binary[train_idx]
        X_te2, y_te2 = X_all[test_idx],  y_binary[test_idx]
        sw_tr = sw[train_idx]

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8, random_state=42))
        ])
        model.fit(X_tr2, y_tr2, clf__sample_weight=sw_tr)
        y_pred = model.predict(X_te2)
        y_prob = model.predict_proba(X_te2)[:, 1]

        stats_acc  = accuracy_score(y_te2, y_pred)
        stats_auc  = roc_auc_score(y_te2, y_prob)

        # FIX v6: proper node-level 5-fold CV (replaces leaky cross_val_score
        # which used a sequence-level split — same node in train AND test).
        from sklearn.model_selection import StratifiedKFold
        all_nodes   = df['node_id'].unique()
        node_labels = df.groupby('node_id')['threat_label'].agg(
            lambda x: int(x.mode()[0])).values
        skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_f1s    = []
        for tr_node_idx, te_node_idx in skf.split(all_nodes, node_labels):
            tr_nodes_cv = all_nodes[tr_node_idx]
            te_nodes_cv = all_nodes[te_node_idx]
            tr_mask = df['node_id'].isin(tr_nodes_cv)
            te_mask = df['node_id'].isin(te_nodes_cv)
            X_tr_cv = X_all[tr_mask];  y_tr_cv = y_binary[tr_mask]
            X_te_cv = X_all[te_mask];  y_te_cv = y_binary[te_mask]
            sw_cv   = compute_sample_weight('balanced', y_tr_cv)
            fold_model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=100, max_depth=4,
                    learning_rate=0.1, random_state=42))
            ])
            fold_model.fit(X_tr_cv, y_tr_cv, clf__sample_weight=sw_cv)
            y_fold_pred = fold_model.predict(X_te_cv)
            from sklearn.metrics import f1_score
            fold_f1s.append(f1_score(y_te_cv, y_fold_pred, zero_division=0))
        cv_f1_mean = float(np.mean(fold_f1s))
        cv_f1_std  = float(np.std(fold_f1s))

        print(f"  Accuracy:{stats_acc:.4f}  AUC:{stats_auc:.4f}")
        print(f"\n{classification_report(y_te2, y_pred, target_names=['Normal','Threat'], zero_division=0)}")

        df['ml_threat_score'] = model.predict_proba(
            model.named_steps['scaler'].transform(X_all))[:, 1]
        df['ml_prediction'] = model.predict(X_all)
        joblib.dump(model, model_save_path)
        auc = stats_auc

    # layer1_flagged used only as a gate here — NOT as a model input feature
    df['send_to_blockchain'] = (
        (df['layer1_flagged'] == 1) | (df['ml_prediction'] == 1)).astype(int)
    df.to_csv(output_path, index=False)

    fwd = int(df['send_to_blockchain'].sum())
    print(f"\n  Model saved    : {model_save_path}")
    print(f"  Results saved  : {output_path}")
    print(f"  Forwarded to L3: {fwd} records\n")

    stats = {
        "layer": 2,
        "model": "CNN-BiLSTM-Attention-v5-FedProx" if TF_AVAILABLE else "GradientBoosting",
        "timestamp":   datetime.now().isoformat(),
        "accuracy":    round(stats_acc, 4),
        "roc_auc":     round(float(auc), 4),
        "cv_f1_mean":  round(cv_f1_mean, 4),
        "cv_f1_std":   round(cv_f1_std, 4),
        "forwarded_to_layer3": fwd,
        "num_features": len(FEATURES),
        "fl_clusters":  N_CLUSTERS if TF_AVAILABLE else 0,
        "fl_rounds":    fl_rounds  if TF_AVAILABLE else 0,
    }
    os.makedirs("data", exist_ok=True)
    with open("data/layer2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return df, stats


if __name__ == "__main__":
    results, stats = run_layer2()
    print("Layer 2 complete.")
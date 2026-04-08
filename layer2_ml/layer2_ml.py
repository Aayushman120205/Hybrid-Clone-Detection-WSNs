"""
=============================================================
LAYER 2: Machine Learning — Behaviour Analysis
=============================================================
Reads Layer 1 output (flagged + all nodes).
Trains Random Forest on behavioural features.
Detects abnormal packet rate, energy patterns.
Outputs ML predictions → passes suspicious to Layer 3.
=============================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
def engineer_features(df):
    """Create behavioural features for ML model."""
    feature_df = df.copy()

    # Per-node rolling statistics (behaviour over time)
    feature_df = feature_df.sort_values(['node_id', 'round'])

    feature_df['pkt_rate_rolling_mean'] = (
        feature_df.groupby('node_id')['packet_rate']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    feature_df['pkt_rate_rolling_std'] = (
        feature_df.groupby('node_id')['packet_rate']
        .transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    )
    feature_df['energy_drop_rate'] = (
        feature_df.groupby('node_id')['energy_remaining']
        .transform(lambda x: x.diff().fillna(0).abs())
    )
    feature_df['energy_drop_rate_mean'] = (
        feature_df.groupby('node_id')['energy_drop_rate']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Anomaly score: deviation from node's own mean
    node_means = feature_df.groupby('node_id')['packet_rate'].transform('mean')
    node_stds  = feature_df.groupby('node_id')['packet_rate'].transform('std').fillna(1)
    feature_df['pkt_zscore'] = (feature_df['packet_rate'] - node_means) / node_stds

    # Energy vs packet ratio (high packets with high energy = suspicious)
    feature_df['energy_pkt_ratio'] = (
        feature_df['packet_rate'] / (feature_df['energy_remaining'] + 1e-6)
    )

    return feature_df


# ──────────────────────────────────────────────
# MAIN ML FUNCTION
# ──────────────────────────────────────────────
def run_layer2(input_path="data/layer1_results.csv",
               output_path="data/layer2_results.csv",
               model_save_path="data/ml_model.pkl"):

    print("=" * 60)
    print("  LAYER 2: RANDOM FOREST — Behaviour Analysis")
    print("=" * 60)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records from Layer 1")
    print(f"Layer 1 flagged: {df['layer1_flagged'].sum()} records\n")

    # ── Feature Engineering ──
    df = engineer_features(df)

    # Binary labels: 0=normal, 1=threat (clone OR malicious)
    df['threat_label'] = (df['label'] > 0).astype(int)

    FEATURES = [
        'packet_rate',
        'energy_remaining',
        'energy_consumed_uJ',
        'dist_to_ch_bs',
        'is_cluster_head',
        'layer1_flagged',
        'pkt_rate_rolling_mean',
        'pkt_rate_rolling_std',
        'energy_drop_rate',
        'energy_drop_rate_mean',
        'pkt_zscore',
        'energy_pkt_ratio',
    ]

    X = df[FEATURES].fillna(0)
    y = df['threat_label']

    print(f"Features used: {len(FEATURES)}")
    print(f"Class distribution — Normal: {(y==0).sum()} | Threat: {(y==1).sum()}\n")

    # ── Train / Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── Model: Random Forest ──
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    # ── Cross-Validation ──
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"Cross-Validation F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Test Evaluation ──
    y_pred     = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=['Normal','Threat'])
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"{'─'*50}")

    # ── Feature Importance ──
    rf = model.named_steps['clf']
    importances = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
    print("\nTop Feature Importances:")
    for feat, imp in importances[:6]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30} {bar} {imp:.4f}")

    # ── Predict on ALL data ──
    df['ml_threat_score']  = model.predict_proba(X.fillna(0))[:, 1]
    df['ml_prediction']    = model.predict(X.fillna(0))

    # Forward to Layer 3: ML flagged as threat
    df['send_to_blockchain'] = (
        (df['layer1_flagged'] == 1) | (df['ml_prediction'] == 1)
    ).astype(int)

    df.to_csv(output_path, index=False)
    joblib.dump(model, model_save_path)

    flagged_for_l3 = df['send_to_blockchain'].sum()
    print(f"\n  Model saved to  : {model_save_path}")
    print(f"  Results saved to: {output_path}")
    print(f"  {flagged_for_l3} records forwarded to Layer 3 (Blockchain)\n")

    stats = {
        "layer": 2,
        "model": "RandomForest",
        "timestamp": datetime.now().isoformat(),
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "forwarded_to_layer3": int(flagged_for_l3),
        "feature_importances": {f: round(float(i), 4) for f, i in importances}
    }
    with open("data/layer2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return df, stats


if __name__ == "__main__":
    results, stats = run_layer2()
    print("Layer 2 complete. Run layer3_blockchain.py next.")

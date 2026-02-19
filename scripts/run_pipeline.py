"""
Full Inference Pipeline
========================
Loads trained models and runs the complete inference pipeline:
data → anomaly detection → risk scoring → MILP optimization → recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data.preprocess import DataPreprocessor
from src.models.autoencoder import load_autoencoder
from src.models.lstm_predictor import load_predictor
from src.models.xgboost_rul import XGBoostRUL
from src.models.bayesian_survival import BayesianSurvival
from src.optimization.milp_scheduler import MaintenanceScheduler


def run_pipeline(test_mode=False):
    """
    Run the full inference pipeline.

    Parameters
    ----------
    test_mode : bool
        If True, uses a small subset for quick testing.
    """
    print("=" * 70)
    print("  MAINTENANCE DECISION SUPPORT — INFERENCE PIPELINE")
    print("=" * 70)

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n[PIPELINE] Loading trained models...")

    autoencoder = load_autoencoder()
    predictor = load_predictor()
    xgb_model = XGBoostRUL.load()
    survival_model = BayesianSurvival.load()

    # =========================================================================
    # Load Test Data
    # =========================================================================
    print("\n[PIPELINE] Loading test data...")

    test_data = np.load(os.path.join(config.PROCESSED_DATA_DIR, "test_data.npz"))
    X_test = test_data["X"]
    y_rul = test_data["y_rul"]
    unit_ids = test_data["unit_ids"]

    if test_mode:
        # Use only first 50 samples
        X_test = X_test[:50]
        y_rul = y_rul[:50]
        unit_ids = unit_ids[:50]

    print(f"[PIPELINE] Test data: {X_test.shape[0]} sequences, "
          f"{len(np.unique(unit_ids))} unique units")

    # =========================================================================
    # Stage 1: Anomaly Detection
    # =========================================================================
    print("\n[PIPELINE] Stage 1: Anomaly Detection")

    anomaly_scores, is_anomaly = autoencoder.detect_anomalies(
        torch.FloatTensor(X_test)
    )
    print(f"  Anomalies detected: {is_anomaly.sum()}/{len(is_anomaly)} "
          f"({is_anomaly.mean():.1%})")

    # =========================================================================
    # Stage 2: Failure Risk Prediction
    # =========================================================================
    print("\n[PIPELINE] Stage 2: Failure Risk Prediction")

    failure_proba, attention_weights = predictor.predict_proba(
        torch.FloatTensor(X_test)
    )
    print(f"  High risk (>0.7): {(failure_proba > 0.7).sum()}")
    print(f"  Medium risk (0.4-0.7): {((failure_proba > 0.4) & (failure_proba <= 0.7)).sum()}")
    print(f"  Low risk (<0.4): {(failure_proba <= 0.4).sum()}")

    # =========================================================================
    # Stage 3: Per-Unit Risk Aggregation
    # =========================================================================
    print("\n[PIPELINE] Stage 3: Per-Unit Risk Aggregation")

    # For each unique unit, take the latest (most recent) prediction
    unit_risks = {}
    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        latest_idx = np.where(mask)[0][-1]  # Last sequence for this unit
        unit_risks[int(uid)] = float(failure_proba[latest_idx])

    print(f"  Units assessed: {len(unit_risks)}")

    # =========================================================================
    # Stage 4: MILP Optimization
    # =========================================================================
    print("\n[PIPELINE] Stage 4: Maintenance Scheduling (MILP)")

    scheduler = MaintenanceScheduler()
    result = scheduler.create_schedule(
        machine_risks=unit_risks,
        machine_names={uid: f"Engine-{uid:03d}" for uid in unit_risks},
    )

    # =========================================================================
    # Stage 5: Generate Recommendations
    # =========================================================================
    print("\n[PIPELINE] Stage 5: Generating Recommendations")

    schedule = result["schedule"]

    recommendations = []
    for _, row in schedule.iterrows():
        rec = {
            "machine": row["machine_name"],
            "risk_score": row["failure_risk"],
            "risk_level": row["risk_level"],
            "action": "Immediate maintenance" if row["risk_level"] == "Service Immediately"
                      else "Schedule maintenance" if row["risk_level"] == "Schedule Soon"
                      else "Continue monitoring",
            "scheduled_slot": row["scheduled_slot"] if row["is_scheduled"] else "N/A",
            "is_anomalous": bool(is_anomaly[
                np.where(unit_ids == row["machine_id"])[0][-1]
            ]) if row["machine_id"] in unit_ids else False,
        }
        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)

    print("\n" + "=" * 70)
    print("MAINTENANCE RECOMMENDATIONS")
    print("=" * 70)
    print(rec_df.to_string(index=False))

    # Save recommendations
    rec_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "recommendations.csv"), index=False)
    print(f"\n[PIPELINE] Recommendations saved to {config.PROCESSED_DATA_DIR}/recommendations.csv")

    return {
        "anomaly_scores": anomaly_scores,
        "failure_proba": failure_proba,
        "attention_weights": attention_weights,
        "schedule": schedule,
        "recommendations": rec_df,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true",
                        help="Run with a small subset for testing")
    args = parser.parse_args()

    results = run_pipeline(test_mode=args.test_mode)

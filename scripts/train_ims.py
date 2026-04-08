"""
IMS Bearing Data Training Script
===================================
End-to-end training pipeline for IMS bearing vibration data.
Parallel to train_all.py (C-MAPSS), uses the same model architectures.
"""

import os
import sys
import time
import numpy as np
import torch

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data.ims_download import download_ims, load_ims_experiment
from src.data.ims_preprocess import IMSPreprocessor
from src.models.autoencoder import LSTMAutoencoder, AutoencoderTrainer
from src.models.lstm_predictor import LSTMPredictor, PredictorTrainer
from src.models.xgboost_rul import XGBoostRUL
from src.models.bayesian_survival import BayesianSurvival
from src.evaluation.simulation import MaintenanceSimulator


def main(experiment=2):
    """
    Train all models on IMS bearing data.

    Parameters
    ----------
    experiment : int
        Which IMS experiment to use (1, 2, or 3).
        Default 2 (single outer race failure, clearest degradation).
    """
    start_time = time.time()
    print("=" * 70)
    print("  SMART INDUSTRIAL MAINTENANCE SYSTEM — IMS TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Device: {config.DEVICE}")
    print(f"  IMS Experiment: {experiment}")
    print(f"  Random Seed: {config.RANDOM_SEED}")
    print()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # =========================================================================
    # Step 1: Download IMS Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: IMS DATA DOWNLOAD")
    print("=" * 70)

    download_ims()
    snapshots, channel_names, exp_info = load_ims_experiment(experiment)

    # =========================================================================
    # Step 2: Feature Extraction & Preprocessing
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE EXTRACTION & PREPROCESSING")
    print("=" * 70)

    preprocessor = IMSPreprocessor()
    data, df_features = preprocessor.fit_transform(snapshots, channel_names, exp_info)
    preprocessor.save()

    # Save processed data
    for split_name, split_data in data.items():
        np.savez_compressed(
            os.path.join(config.IMS_PROCESSED_DIR, f"ims_{split_name}_data.npz"),
            X=split_data["X"],
            y_rul=split_data["y_rul"],
            y_binary=split_data["y_binary"],
        )

    X_train = data["train"]["X"]
    y_train_rul = data["train"]["y_rul"]
    y_train_binary = data["train"]["y_binary"]
    X_val = data["val"]["X"]
    y_val_rul = data["val"]["y_rul"]
    y_val_binary = data["val"]["y_binary"]

    n_features = X_train.shape[2]
    print(f"\n[IMS TRAIN] Feature dimension: {n_features}")

    # =========================================================================
    # Step 3: Train LSTM Autoencoder (Anomaly Detection)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: LSTM AUTOENCODER TRAINING (Bearing Anomaly Detection)")
    print("=" * 70)

    autoencoder = LSTMAutoencoder(
        input_dim=n_features,
        seq_len=config.IMS_SEQUENCE_LENGTH,
    )
    ae_trainer = AutoencoderTrainer(autoencoder)

    # Train on healthy data (high RUL)
    healthy_mask = y_train_rul > config.IMS_MAX_RUL * 0.5
    X_healthy = X_train[healthy_mask]
    X_val_ae = X_val[y_val_rul > config.IMS_MAX_RUL * 0.5] if len(X_val) > 0 else None

    if len(X_healthy) > 0:
        print(f"[IMS TRAIN] Training autoencoder on {len(X_healthy)} healthy samples")
        ae_trainer.train(X_healthy, X_val_ae if X_val_ae is not None and len(X_val_ae) > 0 else None)
        ae_trainer.save_model(os.path.join(config.MODELS_DIR, "ims_autoencoder.pt"))
    else:
        print("[IMS TRAIN] Warning: No healthy samples found. Skipping autoencoder.")

    # =========================================================================
    # Step 4: Train LSTM Failure Predictor
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: LSTM FAILURE PREDICTOR (Bearing Failure)")
    print("=" * 70)

    predictor = LSTMPredictor(input_dim=n_features)
    pred_trainer = PredictorTrainer(predictor)
    pred_trainer.train(X_train, y_train_binary, X_val, y_val_binary)
    pred_trainer.save_model(os.path.join(config.MODELS_DIR, "ims_lstm_predictor.pt"))

    # =========================================================================
    # Step 5: Train XGBoost RUL (Tabular features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: XGBOOST RUL (Bearing Remaining Life)")
    print("=" * 70)

    # Use flat features from df_features
    exclude_cols = ["file_index", "unit_id", "RUL"]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    n = len(df_features)
    n_train_flat = int(n * config.TRAIN_RATIO)
    n_val_flat = int(n * config.VAL_RATIO)

    X_train_xgb = df_features.iloc[:n_train_flat][feature_cols]
    y_train_xgb = df_features.iloc[:n_train_flat]["RUL"]
    X_val_xgb = df_features.iloc[n_train_flat:n_train_flat + n_val_flat][feature_cols]
    y_val_xgb = df_features.iloc[n_train_flat:n_train_flat + n_val_flat]["RUL"]

    xgb_model = XGBoostRUL()
    xgb_model.train(X_train_xgb, y_train_xgb.values, X_val_xgb, y_val_xgb.values,
                     feature_names=feature_cols)
    xgb_model.evaluate(X_val_xgb, y_val_xgb.values)
    xgb_model.save(os.path.join(config.MODELS_DIR, "ims_xgboost_rul.pkl"))

    # =========================================================================
    # Step 6: Bayesian Survival Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: BAYESIAN SURVIVAL ANALYSIS (Bearing Lifetime)")
    print("=" * 70)

    # Use RMS features + RUL for survival analysis
    rms_cols = [c for c in df_features.columns if c.endswith("_rms")]
    if rms_cols:
        survival_cols = rms_cols + ["RUL"]
        df_survival = df_features.iloc[:n_train_flat][["unit_id"] + survival_cols].copy()

        survival_model = BayesianSurvival()
        try:
            survival_model.fit(df_survival)
            survival_model.save(os.path.join(config.MODELS_DIR, "ims_bayesian_survival.pkl"))
        except Exception as e:
            print(f"[IMS TRAIN] Survival model fitting failed: {e}")
            print("[IMS TRAIN] Skipping — survival analysis may need more data points.")

    # =========================================================================
    # Step 7: Run Simulation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: MAINTENANCE POLICY SIMULATION (Bearings)")
    print("=" * 70)

    simulator = MaintenanceSimulator(n_machines=20, n_periods=100)
    sim_df, sim_summary = simulator.run_comparison(n_simulations=50)
    sim_plot_path = os.path.join(config.MODELS_DIR, "..", "ims_simulation_comparison.png")
    simulator.plot_comparison(sim_df, save_path=sim_plot_path)

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  IMS TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Models saved to: {config.MODELS_DIR}")
    print(f"  Processed data: {config.IMS_PROCESSED_DIR}")
    print()
    print("  Saved IMS models:")
    for f in os.listdir(config.MODELS_DIR):
        if f.startswith("ims_"):
            size = os.path.getsize(os.path.join(config.MODELS_DIR, f)) / 1e6
            print(f"    - {f} ({size:.2f} MB)")
    print()
    print("  Next: Run 'streamlit run dashboard/app.py' and select IMS dataset")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train models on IMS bearing data")
    parser.add_argument("--experiment", type=int, default=2,
                       help="IMS experiment number (1, 2, or 3). Default: 2")
    args = parser.parse_args()
    main(experiment=args.experiment)

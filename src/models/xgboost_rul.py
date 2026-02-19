"""
XGBoost RUL (Remaining Useful Life) Estimator
===============================================
Gradient boosted regression for RUL prediction using engineered features.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class XGBoostRUL:
    """
    XGBoost-based Remaining Useful Life regression model.

    Uses engineered tabular features (rolling stats, trends, interactions)
    for accurate RUL prediction with built-in feature importance.
    """

    def __init__(self, params=None):
        self.params = params or config.XGB_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Train XGBoost RUL model.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame — flat feature matrix
        y_train : np.ndarray — RUL target values
        X_val, y_val : optional validation data
        feature_names : list[str], optional
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train_arr = X_train.values
        else:
            self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
            X_train_arr = X_train

        print(f"[XGBOOST] Training with {X_train_arr.shape[1]} features, "
              f"{X_train_arr.shape[0]} samples")

        # Setup evaluation
        eval_set = [(X_train_arr, y_train)]
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            eval_set.append((X_val_arr, y_val))

        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train_arr, y_train,
            eval_set=eval_set,
            verbose=10,
        )

        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        print(f"\n[XGBOOST] Top 10 features:")
        print(self.feature_importance.head(10).to_string(index=False))

        return self

    def predict(self, X):
        """Predict RUL for input features."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        predictions = self.model.predict(X_arr)
        # Clip predictions to valid range
        return np.clip(predictions, 0, config.MAX_RUL)

    def evaluate(self, X, y_true):
        """
        Evaluate model performance.

        Returns
        -------
        dict with RMSE, MAE, R², and score within tolerance metrics.
        """
        y_pred = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Custom industry metric: % predictions within ±10 and ±20 cycles
        within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "within_10_pct": within_10,
            "within_20_pct": within_20,
        }

        print(f"\n[XGBOOST] Evaluation Results:")
        print(f"  RMSE:  {rmse:.2f} cycles")
        print(f"  MAE:   {mae:.2f} cycles")
        print(f"  R²:    {r2:.4f}")
        print(f"  Within ±10 cycles: {within_10:.1f}%")
        print(f"  Within ±20 cycles: {within_20:.1f}%")

        return metrics

    def walk_forward_cv(self, X, y, n_splits=5):
        """
        Walk-forward time-series cross-validation.

        Simulates real deployment where model is trained on past data
        and tested on future data.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        n = len(X_arr)
        fold_size = n // (n_splits + 1)

        results = []
        for i in range(n_splits):
            train_end = fold_size * (i + 2)
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start:
                break

            X_tr, y_tr = X_arr[:train_end], y[:train_end]
            X_te, y_te = X_arr[test_start:test_end], y[test_start:test_end]

            model = xgb.XGBRegressor(**self.params)
            model.fit(X_tr, y_tr, verbose=0)
            y_pred = np.clip(model.predict(X_te), 0, config.MAX_RUL)

            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            mae = mean_absolute_error(y_te, y_pred)
            results.append({"fold": i+1, "rmse": rmse, "mae": mae, "n_train": len(X_tr), "n_test": len(X_te)})

        df_results = pd.DataFrame(results)
        print(f"\n[XGBOOST] Walk-Forward CV Results ({n_splits} folds):")
        print(df_results.to_string(index=False))
        print(f"\n  Mean RMSE: {df_results['rmse'].mean():.2f} ± {df_results['rmse'].std():.2f}")
        print(f"  Mean MAE:  {df_results['mae'].mean():.2f} ± {df_results['mae'].std():.2f}")

        return df_results

    def save(self, filepath=None):
        """Save trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "params": self.params,
        }, filepath)
        print(f"[XGBOOST] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        """Load a trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")
        state = joblib.load(filepath)
        instance = cls(params=state["params"])
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.feature_importance = state["feature_importance"]
        print(f"[XGBOOST] Model loaded from {filepath}")
        return instance

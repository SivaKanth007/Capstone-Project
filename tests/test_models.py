"""
Unit Tests — ML Models
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.models.autoencoder import LSTMAutoencoder
from src.models.lstm_predictor import LSTMPredictor
from src.models.xgboost_rul import XGBoostRUL


@pytest.fixture
def sample_sequences():
    """Generate sample sequences for testing."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 30
    n_features = 14
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_rul = np.random.uniform(0, 125, n_samples).astype(np.float32)
    y_binary = (y_rul <= 30).astype(np.float32)
    return X, y_rul, y_binary


@pytest.fixture
def flat_features():
    """Generate flat feature matrix for XGBoost."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.uniform(0, 125, n_samples).astype(np.float32)
    return X, y


class TestLSTMAutoencoder:
    def test_forward_pass_shape(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        x_tensor = torch.FloatTensor(X[:5])
        output = model(x_tensor)
        assert output.shape == x_tensor.shape

    def test_anomaly_score_shape(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        scores = model.compute_anomaly_score(torch.FloatTensor(X[:10]))
        assert scores.shape == (10,)
        assert np.all(scores >= 0)

    def test_threshold_setting(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        scores = model.compute_anomaly_score(torch.FloatTensor(X))
        threshold = model.set_threshold(scores)
        assert threshold > 0
        assert model.threshold == threshold

    def test_detect_anomalies(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        scores = model.compute_anomaly_score(torch.FloatTensor(X))
        model.set_threshold(scores)
        scores, is_anomaly = model.detect_anomalies(torch.FloatTensor(X[:10]))
        assert is_anomaly.shape == (10,)
        assert is_anomaly.dtype == bool


class TestLSTMPredictor:
    def test_forward_pass_output(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMPredictor(input_dim=X.shape[2])
        x_tensor = torch.FloatTensor(X[:5])
        proba, attn = model(x_tensor)

        assert proba.shape == (5,)
        assert attn.shape == (5, X.shape[1])
        assert torch.all(proba >= 0) and torch.all(proba <= 1)

    def test_predict_proba(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMPredictor(input_dim=X.shape[2])
        proba, attn = model.predict_proba(torch.FloatTensor(X[:10]))

        assert proba.shape == (10,)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert attn.shape == (10, X.shape[1])


class TestXGBoostRUL:
    def test_train_and_predict(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X[:150], y[:150])

        predictions = model.predict(X[150:])
        assert len(predictions) == 50
        assert np.all(predictions >= 0)
        assert np.all(predictions <= config.MAX_RUL)

    def test_evaluate(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X[:150], y[:150])

        metrics = model.evaluate(X[150:], y[150:])
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0

    def test_feature_importance(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X, y)

        assert model.feature_importance is not None
        assert len(model.feature_importance) == X.shape[1]

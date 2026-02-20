# 🏭 Data-Driven Anomaly Detection & Risk-Aware Maintenance Scheduling

> **FSE 570 Data Science Capstone** — Arizona State University, Ira A. Fulton Schools of Engineering

An end-to-end decision support system that transforms raw industrial sensor data into actionable maintenance recommendations using deep learning, Bayesian uncertainty quantification, and mathematical optimization.

## 👥 Team

| Name | Role |
|------|------|
| Anoushka Jaydas Dighe | Team Member |
| Deva Siva Kanth Tavvala | Team Member |
| Mohit Kumar Petla | Team Member |
| Umang Rajnikant Bid | Team Member |
| Urvansh Jignesh Shah | Team Member |

## 🏗️ Architecture

```
Raw Sensor Data → Preprocessing → Anomaly Detection → Risk Prediction → MILP Optimization → Dashboard
                  (LSTM Autoencoder)   (LSTM + XGBoost + Bayesian)    (PuLP Scheduler)     (Streamlit)
```

### Pipeline Components

| Component | Technique | Purpose |
|-----------|-----------|---------|
| **Anomaly Detection** | LSTM Temporal Autoencoder | Detect abnormal sensor patterns |
| **Failure Prediction** | LSTM Classifier + Attention | Failure probability within 30 cycles |
| **RUL Estimation** | XGBoost Regression | Remaining Useful Life estimation |
| **Uncertainty** | Bayesian Weibull Survival | Calibrated 90%/95% credible intervals |
| **Explainability** | SHAP + Attention Weights | Feature attribution & temporal importance |
| **Optimization** | MILP (PuLP CBC) | Crew-constrained maintenance scheduling |
| **Dashboard** | Streamlit + Plotly | Interactive fleet monitoring |

## 📊 Dataset

- **NASA C-MAPSS** — Turbofan engine degradation simulation
  - 21 sensors × 100+ units × 260+ cycles
  - Sensor types: temperature, pressure, vibration, speed, power
- **Synthetic maintenance logs** — Generated repair history, costs, downtime
- **Operational context** — Machine specs, crew schedules, production lines

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train All Models

```bash
python scripts/train_all.py
```

This will:
- Download the C-MAPSS dataset (NASA S3 direct download, with Kaggle API fallback)
- Generate synthetic maintenance & operational data
- Engineer features (rolling stats, trends, regimes)
- Train LSTM Autoencoder, LSTM Predictor, XGBoost, and Bayesian Survival models
- Run Monte Carlo maintenance policy simulation

### 3. Run Inference Pipeline

```bash
python scripts/run_pipeline.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run Unit Tests

```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
├── config.py                    # Global configuration
├── requirements.txt             # Dependencies
├── PROJECT_REPORT.md            # Full project report
├── .gitignore
├── data/
│   ├── raw/                     # NASA C-MAPSS dataset
│   ├── processed/               # Windowed sequences
│   └── synthetic/               # Generated maintenance logs
├── src/
│   ├── data/                    # Data engineering pipeline
│   │   ├── download.py          # Multi-source dataset download
│   │   ├── preprocess.py        # Cleaning, normalization, windowing
│   │   ├── feature_engineering.py
│   │   └── synthetic_generator.py
│   ├── models/                  # ML models
│   │   ├── autoencoder.py       # LSTM temporal autoencoder
│   │   ├── lstm_predictor.py    # Failure probability classifier
│   │   ├── xgboost_rul.py       # RUL regression
│   │   └── bayesian_survival.py # Weibull survival analysis
│   ├── explainability/          # Model interpretability
│   │   ├── shap_analysis.py     # SHAP feature attribution
│   │   └── attention_viz.py     # Temporal attention heatmaps
│   ├── optimization/            # Decision optimization
│   │   └── milp_scheduler.py    # PuLP maintenance scheduler
│   └── evaluation/              # Evaluation & simulation
│       └── simulation.py        # Monte Carlo policy comparison
├── scripts/
│   ├── train_all.py             # End-to-end training
│   └── run_pipeline.py          # Full inference pipeline
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── notebooks/
│   └── capstone_colab.ipynb     # Google Colab notebook (GPU)
└── tests/
    ├── test_preprocess.py
    ├── test_models.py
    └── test_optimizer.py
```

## ⚡ GPU Support

The system automatically detects and uses CUDA GPUs for LSTM training:

```python
# config.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For Google Colab with GPU, use the notebook at `notebooks/capstone_colab.ipynb`.

## 📈 Results

| Model | Metric | Value |
|-------|--------|-------|
| LSTM Predictor | F1-Score | **0.933** |
| LSTM Predictor | AUC-ROC | **0.997** |
| XGBoost RUL | RMSE | **10.48 cycles** |
| XGBoost RUL | R² | **0.937** |
| Bayesian Survival | C-Index | **0.992** |
| MILP Optimization | Cost Reduction | **97.4%** vs reactive |
| MILP Optimization | Downtime Reduction | **72.4%** vs reactive |

## 🛠️ Maintenance Categories

| Level | Threshold | Action |
|-------|-----------|--------|
| 🔴 Critical | Risk ≥ 70% | Service Immediately |
| 🟡 Elevated | Risk 40-70% | Schedule Soon |
| 🟢 Normal | Risk < 40% | Continue Monitoring |

## 📄 License

This project is developed for FSE 570 at Arizona State University.

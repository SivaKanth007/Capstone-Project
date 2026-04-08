# 🏭 Smart Industrial Maintenance System

> **FSE 570 Data Science Capstone** — Arizona State University

An end-to-end AI system that detects anomalies in industrial sensors, predicts machine failures, and generates optimized maintenance schedules. Supports both the **NASA C-MAPSS turbofan** dataset and the **NASA IMS Bearing** vibration dataset.

## 👥 Team

| Name | Role |
|------|------|
| Anoushka Jaydas Dighe | Team Member |
| Deva Siva Kanth Tavvala | Team Member |
| Mohit Kumar Petla | Team Member |
| Umang Rajnikant Bid | Team Member |
| Urvansh Jignesh Shah | Team Member |

---

## 🚀 Quick Start (Fresh Install)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/SivaKanth007/Capstone-Project.git
cd Capstone-Project
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Requires Python 3.9 or higher.**  
> No GPU needed — everything runs on CPU. GPU (CUDA) is auto-detected if available.

### Step 3 — Train All Models

```bash
python scripts/train_all.py
```

This single command does everything:
- Downloads the NASA C-MAPSS turbofan dataset automatically
- Generates synthetic maintenance logs
- Engineers 200+ features from sensor data
- Trains 4 ML models (LSTM Autoencoder, LSTM Predictor, XGBoost, Bayesian Survival)
- Runs Monte Carlo simulation comparing maintenance policies

**⏱ Takes ~5 minutes on CPU. Using the NVIDIA RTX 3050 Ti Laptop GPU it completes significantly faster!**

### Step 3b — Run IMS Bearing Pipeline (Notebook)

Open `notebooks/Smart_Industrial_Maintenance_Full_Pipeline.ipynb` in Jupyter and run all cells. This pipeline:
- Downloads the NASA IMS Bearing vibration dataset (~2 GB) via `kagglehub`
- Extracts time-domain and frequency-domain features from raw vibration signals
- Trains the same 4-model suite on bearing degradation data
- Runs MILP scheduling and Monte Carlo simulation

### Step 4 — Run Inference Pipeline

```bash
python scripts/run_pipeline.py
```

Loads the trained models and produces maintenance recommendations saved to `data/processed/recommendations.csv`.

### Step 5 — Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501** in your browser.

### Step 6 — Run Tests

```bash
python -m pytest
```

All 27 unit tests should pass.

---

## 📊 What This System Does

```
Raw Sensor Data → Preprocessing → ML Models → MILP Optimizer → Dashboard
```

| Component | What It Does |
|-----------|-------------|
| **Anomaly Detection** | LSTM Autoencoder flags unusual sensor patterns |
| **Failure Prediction** | LSTM Classifier estimates failure probability (next 30 cycles) |
| **RUL Estimation** | XGBoost predicts Remaining Useful Life in cycles |
| **Uncertainty** | Bayesian Weibull model gives 90%/95% confidence intervals |
| **Scheduling** | MILP optimizer assigns maintenance to crews optimally |
| **Dashboard** | Streamlit app shows live fleet health and schedule |

---

## 📈 Results

| Model | Metric | Value |
|-------|--------|-------|
| LSTM Failure Predictor | F1-Score | **0.933** |
| LSTM Failure Predictor | AUC-ROC | **0.997** |
| XGBoost RUL | RMSE | **10.48 cycles** |
| XGBoost RUL | R² | **0.937** |
| Bayesian Survival | C-Index | **0.992** |
| MILP Optimization | Cost Reduction | **97.4%** vs reactive |
| MILP Optimization | Downtime Reduction | **72.4%** vs reactive |

---

## 📁 Project Structure

```
Capstone-Project/
├── config.py                    # All settings (paths, hyperparameters)
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
├── PROJECT_REPORT.md            # Full project report
│
├── scripts/
│   ├── train_all.py             # ← Run this first to train C-MAPSS models
│   ├── train_ims.py             # IMS bearing model training
│   └── run_pipeline.py          # ← Run this to get predictions
│
├── dashboard/
│   └── app.py                   # ← Streamlit dashboard
│
├── src/
│   ├── data/                    # Data download, preprocessing, features
│   │   ├── download.py          # C-MAPSS dataset download
│   │   ├── ims_download.py      # IMS bearing dataset download (kagglehub)
│   │   ├── ims_preprocess.py    # IMS vibration signal preprocessing
│   │   ├── preprocess.py        # C-MAPSS preprocessing
│   │   ├── feature_engineering.py
│   │   └── synthetic_generator.py
│   ├── models/                  # LSTM, XGBoost, Bayesian Survival models
│   ├── explainability/          # SHAP + attention visualization
│   ├── optimization/            # MILP maintenance scheduler
│   └── evaluation/              # Monte Carlo simulation
│
├── tests/                       # Unit tests (4 modules)
│
├── notebooks/
│   └── Smart_Industrial_Maintenance_Full_Pipeline.ipynb  # IMS bearing pipeline
│
├── data/                        # Created automatically after training
│   ├── raw/                     # NASA C-MAPSS dataset
│   ├── raw_ims/                 # NASA IMS Bearing dataset (3 experiments)
│   ├── processed/               # Preprocessed sequences
│   └── synthetic/               # Generated maintenance logs
│
├── models/saved/                # Trained model files
│   ├── autoencoder.pt           # C-MAPSS LSTM Autoencoder
│   ├── lstm_predictor.pt        # C-MAPSS LSTM Predictor
│   ├── xgboost_rul.pkl          # C-MAPSS XGBoost RUL
│   ├── bayesian_survival.pkl    # C-MAPSS Weibull Survival
│   ├── preprocessor.pkl         # MinMaxScaler + features
│   ├── ims_autoencoder.pt       # IMS LSTM Autoencoder
│   ├── ims_xgboost.pkl          # IMS XGBoost RUL
│   └── ims_survival.pkl         # IMS Weibull Survival
│
└── assets/                      # Dashboard screenshots
```

---

## ⚡ GPU Support

The system automatically uses a CUDA GPU if available:

```python
# config.py — detected automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For the IMS Bearing pipeline, open `notebooks/Smart_Industrial_Maintenance_Full_Pipeline.ipynb`.

---

## 🔴🟡🟢 Risk Levels

| Level | Condition | Action |
|-------|-----------|--------|
| 🔴 Critical | Risk ≥ 70% | Service Immediately |
| 🟡 Elevated | Risk 40–70% | Schedule Soon |
| 🟢 Normal | Risk < 40% | Continue Monitoring |

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `No data found` | Run `python scripts/train_all.py` first |
| Dashboard blank | Make sure `run_pipeline.py` has been run |
| Tests not found | Run `python -m pytest` from the project root |

---

## 📄 License

Developed for FSE 570 at Arizona State University.

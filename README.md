

# 🏥 PhysioNet ICU Mortality Risk Prediction Engine
### *High-Throughput Hybrid ML/DL Clinical Decision Support System*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-228B22?style=for-the-badge&logo=xgboost&logoColor=white)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI_Portal-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-000000?style=for-the-badge&logo=python&logoColor=white)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end clinical machine learning framework and real-time decision-support system designed to predict patient-specific in-hospital mortality using sparse, irregular Electronic Health Record (EHR) time-series data collected during the first 48 hours of ICU admission.

---

## 📌 Executive Summary & Clinical Context

In intensive care units, vital signs and laboratory panels are recorded asynchronously based purely on patient acuity. This creates high-stakes predictive challenges characterized by irregular time-series, extreme data sparsity, and severe target class imbalance (~14.2% mortality baseline).

**This production framework addresses these constraints by:**
1. **Preserving Informative Missingness**: Converting diagnostic omissions into explicit structural signals (`_is_missing = 1`) that leverage clinical intent as an anchoring predictive marker.
2. **Deploying a 50/50 Hybrid Ensemble**: Merging **LightGBM** (to learn non-linear tabular threshold boundaries) with a custom **PyTorch Bidirectional LSTM** (to capture temporal trajectory velocity and momentum over 48 hours).
3. **Calibrating Clinical Risks**: Applying **Platt Scaling** (`CalibratedClassifierCV`) to map raw decision tree outputs to true empirical clinical probabilities.
4. **Delivering Explainable AI (XAI)**: Generating consensus **TreeSHAP** attributions to isolate patient-specific escalating vs. mitigating risk drivers in real time.

---

## 🏛️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │   Raw Irregular EHR Logs (PhysioNet Sets A & B)        │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │          Feature Engineering & Token Normalization     │
               └─────────────┬────────────────────────────┬─────────────┘
                             │                            │
             [2D Tabular Feature Matrix]        [3D Temporal Sequence Cube]
             (Shape: [N, 118 Features])          (Shape: [N, 48 Hours, 6 Vitals])
                             │                            │
                             ▼                            ▼
               ┌───────────────────────────┐┌───────────────────────────┐
               │ 5-Fold LightGBM + Platt   ││   PyTorch Bidirectional   │
               │ Scaling Probability Model ││        LSTM Engine        │
               └─────────────┬─────────────┘└─────────────┬─────────────┘
                             │                            │
                             └─────────────┬──────────────┘
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │       Calibrated 50/50 Blended Risk Probability        │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │    FastAPI Gateway Microservice & TreeSHAP Engine      │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │   Interactive Clinician Dashboard (Streamlit UI)       │
               └────────────────────────────────────────────────────────┘

```

---

## 📂 Project Directory Structure

```text
icu-mortality-prediction/
├── api/
│   └── main.py                     # High-throughput FastAPI inference router & SHAP extraction
├── app/
│   └── streamlit_ui.py             # Interactive clinician portal with Plotly visualizations
├── data/
│   ├── raw/                        # Raw PhysioNet ASCII patient text records (Set A & Set B)
│   └── processed/                  # Snappy-compressed Parquet cache & holdout scorecards
├── logs/
│   └── Pipeline_execution.log      # Thread-safe central execution trace log
├── models/
│   └── hybrid_ensemble_core.joblib # Deep-frozen compressed production artifact bundle
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central workspace path resolutions & model hyperparameters
│   ├── data_pipeline.py            # Streamlined ASCII parser & Parquet binary caching
│   ├── evaluate.py                 # Stratified CV, Platt Scaling, & PhysioNet Event metrics
│   ├── explain.py                  # Calibrated model unwrap gate & consensus TreeSHAP
│   ├── features.py                 # Upstream vocabulary injection & vectorized 3D array tensor builder
│   ├── model_dl.py                 # PyTorch Bidirectional LSTM neural core with gradient clipping
│   ├── test_set_b.py               # Prospective validation & scoring runner
│   ├── train_combined.py           # Unified 8,000-record master retraining orchestrator
│   └── utils.py                    # Structured logging, artifact I/O, & deterministic seeds
├── .gitignore                      # Storage, environment, and credential protection rules
├── requirements.txt                # Production dependency registry
└── README.md                       # Master architectural documentation

```

---

## 🔬 Core Technical Innovations & Defense Pillars

| Engineering Component | Technical Implementation | Core Architectural Advantage |
| --- | --- | --- |
| **Informative Missingness** | Binary indicators (`_is_missing = 1`) | Preserves clinical intent without injecting synthetic noise via mean/median imputation. |
| **Hybrid Blended Ensemble** | 50% LightGBM + 50% PyTorch BiLSTM | Combines high-dimensional static lab thresholds with time-series trajectory directionality. |
| **Bidirectional Parsing** | Forward ($t_1 \rightarrow t_{48}$) & Backward ($t_{48} \rightarrow t_1$) | Maps acute late-stage clinical decompensation directly against baseline admission status. |
| **Probability Calibration** | Platt Scaling (`CalibratedClassifierCV`) | Transforms uncalibrated tree margins into true empirical percentages, minimizing Brier score loss. |
| **Consensus TreeSHAP** | Model unwrap gate extracting base estimators | Calculates local additive feature attributions without runtime crashes from wrapper layers. |

---

## 🚀 Quickstart & Local Deployment

### 1. Prerequisites & Environment Setup

Clone the repository and set up an isolated Python 3.10+ virtual environment:

```bash
# Clone the repository
git clone [https://github.com/itsnik101/icu-mortality-prediction.git](https://github.com/itsnik101/icu-mortality-prediction.git)
cd icu-mortality-prediction

# Initialize and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install locked production dependencies
pip install -r requirements.txt

```

### 2. Execute the Master Pipeline

Retrain the hybrid ensemble across the combined dataset and generate serialized artifacts:

```bash
python -m src.train_combined

```

### 3. Launch the Serving Infrastructure

**Terminal 1: Start the FastAPI Backend Gateway**

```bash
uvicorn api.main:app --reload --port 8000

```

* Interactive Swagger API documentation will be available at: `http://127.0.0.1:8000/docs`

**Terminal 2: Launch the Streamlit Decision Portal**

```bash
streamlit run app/streamlit_ui.py

```

* Clinician Decision Support Dashboard will launch at: `http://localhost:8501`

---

## 📡 API Endpoint Reference

### Evaluate Patient Risk (`POST /predict`)

Evaluates vital observations and returns a calibrated mortality score with local SHAP attributions.

#### Sample Request Payload:

```json
{
  "Age": 68.0,
  "Gender": 1,
  "Observations": [
    {"Parameter": "HR", "Value": 118.0},
    {"Parameter": "GCS", "Value": 7.0},
    {"Parameter": "SysBP", "Value": 88.0},
    {"Parameter": "Temp", "Value": 38.9},
    {"Parameter": "BUN", "Value": 42.0},
    {"Parameter": "Creatinine", "Value": 2.4}
  ]
}

```

#### Sample JSON Response:

```json
{
  "Mortality_Risk_Probability": 0.7418,
  "Clinical_Status_Flag": "CRITICAL HIGH RISK",
  "Primary_Risk_Drivers": {
    "escalating": [
      {"feature": "Glasgow Coma Scale (Minimum Floor)", "impact": 0.2415},
      {"feature": "Blood Urea Nitrogen Labs (Peak Max)", "impact": 0.1832},
      {"feature": "Heart Rate Tracking (Average)", "impact": 0.1104}
    ],
    "mitigating": [
      {"feature": "Patient Age", "impact": -0.0421}
    ]
  }
}

```

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.

```

```


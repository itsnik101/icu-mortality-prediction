# api/main.py
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.utils import setup_logger, load_artifact
from src.model_dl import ClinicalLSTM

api_logger = setup_logger("api_serving_engine")

app = FastAPI(
    title="ICU Mortality Prediction Inference API",
    version="2.7.0",
    description="Production REST microservice serving hybrid LightGBM + BiLSTM predictions and genuine TreeSHAP explanations."
)

# --- GLOBAL ARTIFACT INITIALIZATION ---
MODEL_PAYLOAD = None
LGBM_ENSEMBLE = None
PYTORCH_STATE = None
FEATURE_CHECKLIST = []
SHAP_EXPLAINER = None

@app.on_event("startup")
def load_production_artifacts():
    global MODEL_PAYLOAD, LGBM_ENSEMBLE, PYTORCH_STATE, FEATURE_CHECKLIST, SHAP_EXPLAINER
    try:
        api_logger.info("Loading model artifact bundle from disk...")
        MODEL_PAYLOAD = load_artifact("hybrid_ensemble_core.joblib")
        LGBM_ENSEMBLE = MODEL_PAYLOAD["lgbm_fold_ensemble"]
        PYTORCH_STATE = MODEL_PAYLOAD["pytorch_lstm_state"]
        
        # Extract feature checklist dynamically from fitted LightGBM estimators
        if hasattr(LGBM_ENSEMBLE[0], "calibrated_classifiers_"):
            FEATURE_CHECKLIST = list(LGBM_ENSEMBLE[0].calibrated_classifiers_[0].estimator.feature_name_)
            base_estimator = LGBM_ENSEMBLE[0].calibrated_classifiers_[0].estimator
        elif hasattr(LGBM_ENSEMBLE[0], "feature_name_"):
            FEATURE_CHECKLIST = list(LGBM_ENSEMBLE[0].feature_name_)
            base_estimator = LGBM_ENSEMBLE[0]
        else:
            FEATURE_CHECKLIST = MODEL_PAYLOAD.get("feature_names", [])
            base_estimator = LGBM_ENSEMBLE[0]
            
        # Initialize TreeSHAP explainer on base tree model
        SHAP_EXPLAINER = shap.TreeExplainer(base_estimator)
        api_logger.info(f"Model artifact loaded successfully. Feature dimensions: {len(FEATURE_CHECKLIST)}")
        
    except Exception as e:
        api_logger.critical(f"Failed to load model artifacts: {str(e)}", exc_info=True)
        raise RuntimeError(f"Startup Failure: {str(e)}")

# --- PYDANTIC CONTRACT SCHEMAS ---
class ObservationItem(BaseModel):
    Parameter: str
    Value: float

class PatientPayload(BaseModel):
    Age: float = Field(..., ge=15.0, le=110.0, description="Patient age in years")
    Gender: int = Field(..., ge=0, le=1, description="Gender indicator (0: Female, 1: Male)")
    Observations: List[ObservationItem] = Field(..., description="List of recorded clinical vitals/labs")

# --- COMPREHENSIVE CLINICAL FEATURE MAPPING DICTIONARY ---
CLINICAL_NAME_MAP = {
    "Age": "Patient Age Profile",
    "Gender": "Biological Sex",
    "HR": "Heart Rate Baseline",
    "GCS": "Glasgow Coma Scale (GCS)",
    "SysBP": "Systolic Blood Pressure",
    "DiasBP": "Diastolic Blood Pressure",
    "MeanBP": "Mean Arterial Pressure (MAP)",
    "Temp": "Core Body Temperature",
    "Resp": "Respiration Rate",
    "BUN": "Blood Urea Nitrogen (BUN)",
    "Creatinine": "Serum Creatinine",
    "Platelets": "Platelet Count",
    "WBC": "White Blood Cell Count",
    "Glucose": "Serum Glucose",
    "FiO2": "Inspired Oxygen Fraction (FiO2)",
    "pH": "Blood pH Balance",
    "PaO2": "Partial Pressure of Oxygen (PaO2)",
    "PaCO2": "Partial Pressure of Carbon Dioxide (PaCO2)",
    "HCO3": "Serum Bicarbonate (HCO3)",
    "Magnesium": "Serum Magnesium",
    "Potassium": "Serum Potassium",
    "Sodium": "Serum Sodium"
}

def format_feature_label(raw_name: str) -> str:
    """Maps raw feature tokens to professional clinical terminology."""
    if raw_name in CLINICAL_NAME_MAP:
        return CLINICAL_NAME_MAP[raw_name]
        
    # Handle missingness indicators and statistical aggregates gracefully
    clean = raw_name
    for key, readable in CLINICAL_NAME_MAP.items():
        if key in clean:
            clean = clean.replace(key, readable)
            break
            
    clean = clean.replace("_is_missing", " (Missing Data Marker)").replace("_", " ")
    return clean.title()

# --- INFERENCE ORCHESTRATION ENDPOINT ---
@app.post("/predict")
def predict_mortality(payload: PatientPayload):
    if LGBM_ENSEMBLE is None or SHAP_EXPLAINER is None:
        raise HTTPException(status_code=500, detail="Inference engine offline: Model artifacts not loaded.")
        
    try:
        # 1. Map incoming observations into dictionary
        feat_dict = {"Age": payload.Age, "Gender": payload.Gender}
        for obs in payload.Observations:
            feat_dict[obs.Parameter] = obs.Value
            
        # 2. Build aligned row vector and explicit pandas DataFrame (Fixes scope/NameError)
        row_vector = [feat_dict.get(col, 0.0) for col in FEATURE_CHECKLIST]
        X_df = pd.DataFrame([row_vector], columns=FEATURE_CHECKLIST)
        
        # 3. LightGBM Ensemble Inference Pass
        lgbm_probs = np.mean([model.predict_proba(X_df.values)[:, 1] for model in LGBM_ENSEMBLE], axis=0)[0]
        
        # 4. PyTorch BiLSTM Inference Pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lstm_feature_dim = MODEL_PAYLOAD.get("lstm_feature_count", 10)
        lstm_engine = ClinicalLSTM(n_features=lstm_feature_dim).to(device)
        lstm_engine.load_state_dict(PYTORCH_STATE)
        lstm_engine.eval()
        
        dummy_seq = np.zeros((1, 48, lstm_feature_dim), dtype=np.float32)
        dummy_seq[0, :, 0] = payload.Age
        for idx, obs in enumerate(payload.Observations):
            if idx < lstm_feature_dim:
                dummy_seq[0, :, idx] = obs.Value
                
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(dummy_seq).to(device)
            lstm_out = lstm_engine(seq_tensor).cpu()
            if lstm_out.ndim == 2:
                lstm_logit = float(lstm_out.numpy()[0][0])
            elif lstm_out.ndim == 1:
                lstm_logit = float(lstm_out.numpy()[0])
            else:
                lstm_logit = float(lstm_out.item())
            lstm_prob = float(1 / (1 + np.exp(-lstm_logit)))
            
        # 5. Blended Consensus Prediction
        final_prob = float(np.clip((0.5 * lgbm_probs) + (0.5 * lstm_prob), 0.0, 1.0))
        
        if final_prob < 0.20:
            status_flag = "LOW RISK"
        elif final_prob < 0.50:
            status_flag = "MODERATE RISK"
        else:
            status_flag = "CRITICAL HIGH RISK"
            
        # 6. GENUINE TreeSHAP Explainability Calculation
        shap_raw_vals = SHAP_EXPLAINER.shap_values(X_df)
        
        if isinstance(shap_raw_vals, list):
            shap_values_class1 = shap_raw_vals[1][0] if len(shap_raw_vals) > 1 else shap_raw_vals[0][0]
        elif isinstance(shap_raw_vals, np.ndarray) and shap_raw_vals.ndim == 3:
            shap_values_class1 = shap_raw_vals[0, :, 1]
        else:
            shap_values_class1 = shap_raw_vals[0]
            
        # Pair array values explicitly with FEATURE_CHECKLIST indices and dictionary map
        feature_impacts = []
        for idx, impact_val in enumerate(shap_values_class1):
            if idx < len(FEATURE_CHECKLIST):
                raw_name = FEATURE_CHECKLIST[idx]
                clean_name = format_feature_label(raw_name)
                feature_impacts.append((clean_name, float(impact_val)))

        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        escalating = [{"feature": f, "impact": round(v, 3)} for f, v in feature_impacts if v > 0][:3]
        mitigating = [{"feature": f, "impact": round(v, 3)} for f, v in feature_impacts if v < 0][:3]
        
        return {
            "Mortality_Risk_Probability": round(final_prob, 4),
            "Clinical_Status_Flag": status_flag,
            "Primary_Risk_Drivers": {
                "escalating": escalating,
                "mitigating": mitigating
            }
        }
        
    except Exception as e:
        api_logger.error(f"Inference execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference execution error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "icu-mortality-inference-api"}
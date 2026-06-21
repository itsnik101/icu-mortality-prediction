# api/main.py
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time

import config
from src.utils import load_artifact, setup_logger
from src.model_dl import ClinicalLSTM
from src.explain import generate_tabular_explanation, isolate_top_risk_drivers

logger = setup_logger("api_serving_engine")
app = FastAPI(title="ICU Mortality Prediction Root Service", version="2.5.0")

lgbm_ensemble, lstm_model_engine, feature_checklist, lstm_feature_dim = None, None, None, None

class VitalObservation(BaseModel):
    Parameter: str
    Value: float

class PatientRecordPayload(BaseModel):
    Age: float
    Gender: int
    Observations: List[VitalObservation]

class RiskPredictionResponse(BaseModel):
    Mortality_Risk_Probability: float
    Clinical_Status_Flag: str
    Primary_Risk_Drivers: Dict[str, Any]

@app.on_event("startup")
def bootstrap_predictive_assets():
    global lgbm_ensemble, lstm_model_engine, feature_checklist, lstm_feature_dim
    try:
        payload = load_artifact("hybrid_ensemble_core.joblib")
        lgbm_ensemble = payload["lgbm_fold_ensemble"]
        feature_checklist = payload["feature_names"]
        lstm_feature_dim = payload["lstm_feature_count"]
        
        if hasattr(lgbm_ensemble[0], "calibrated_classifiers_"):
            feature_checklist = list(lgbm_ensemble[0].calibrated_classifiers_[0].estimator.feature_name_)
        elif hasattr(lgbm_ensemble[0], "feature_name_"):
            feature_checklist = list(lgbm_ensemble[0].feature_name_)
            
        logger.info(f"API Online. Root Naming Scheme Operational. Features count: {len(feature_checklist)}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lstm_model_engine = ClinicalLSTM(n_features=lstm_feature_dim).to(device)
        lstm_model_engine.load_state_dict(payload["pytorch_lstm_state"])
        lstm_model_engine.eval()
    except Exception as e:
        raise RuntimeError(f"Server initialization failure: {str(e)}")

@app.post("/predict", response_model=RiskPredictionResponse)
def execute_hybrid_clinical_inference(payload: PatientRecordPayload):
    try:
        raw_obs = [{"Parameter": o.Parameter, "Value": o.Value} for o in payload.Observations]
        obs_df = pd.DataFrame(raw_obs)
        
        patient_tab_dict = {feat: 0.0 for feat in feature_checklist}
        
        if "Patient Age" in patient_tab_dict: patient_tab_dict["Patient Age"] = float(payload.Age)
        if "Biological Gender Assignment" in patient_tab_dict: patient_tab_dict["Biological Gender Assignment"] = float(payload.Gender)
        
        # Maps inputs using the perfect, fresh feature string configuration schemas
        if not obs_df.empty:
            grouped = obs_df.groupby("Parameter")["Value"]
            for param, values in grouped:
                mean_v, max_v, min_v = float(values.mean()), float(values.max()), float(values.min())
                
                # Check directly against the upstream baked descriptors
                from src.features import CLINICAL_VOCABULARY, AGG_DESCRIPTORS
                p_desc = CLINICAL_VOCABULARY.get(param, param)
                
                if f"{p_desc} (Average)" in patient_tab_dict: patient_tab_dict[f"{p_desc} (Average)"] = mean_v
                if f"{p_desc} (Peak Max)" in patient_tab_dict: patient_tab_dict[f"{p_desc} (Peak Max)"] = max_v
                if f"{p_desc} (Minimum Floor)" in patient_tab_dict: patient_tab_dict[f"{p_desc} (Minimum Floor)"] = min_v

        X_patient = pd.DataFrame([patient_tab_dict])[feature_checklist]
        lgbm_prob = np.mean([model.predict_proba(X_patient.values)[:, 1] for model in lgbm_ensemble], axis=0)[0]
        
        seq_array = np.zeros((1, 48, lstm_feature_dim))
        seq_array[0, :, :2] = [payload.Age, payload.Gender]
        vitals_tracking_list = ['HR', 'GCS', 'SysBP', 'Temp']
        for param in vitals_tracking_list:
            if not obs_df.empty and param in obs_df['Parameter'].values:
                v_idx = vitals_tracking_list.index(param) + 2
                seq_array[0, :, v_idx] = float(obs_df[obs_df['Parameter'] == param]['Value'].values[0])
                
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            lstm_logit = lstm_model_engine(torch.FloatTensor(seq_array).to(device)).cpu().item()
            lstm_prob = float(lgbm_prob) if np.isnan(lstm_logit) else 1 / (1 + np.exp(-lstm_logit))
            
        final_prob = np.clip(float((0.5 * lgbm_prob) + (0.5 * lstm_prob)), 0.0, 1.0)
        status_flag = "CRITICAL HIGH RISK" if final_prob >= 0.50 else "STABLE / LOW RISK"
        if 0.30 <= final_prob < 0.50: status_flag = "ELEVATED RISK MONITORING REQUIRED"
            
        raw_shap = generate_tabular_explanation(lgbm_ensemble, X_patient, feature_checklist)
        filtered_drivers = isolate_top_risk_drivers(raw_shap, top_n=5)
        
        return {
            "Mortality_Risk_Probability": round(final_prob, 4),
            "Clinical_Status_Flag": status_flag,
            "Primary_Risk_Drivers": filtered_drivers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
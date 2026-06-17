# src/explain.py
"""
Project ICU: Clinical Explainability Layer
Leverages multi-fold TreeSHAP consensus mapping for GBDTs and GradientSHAP 
sequence attribution for PyTorch LSTMs to unlock total clinical transparency.
"""

import numpy as np
import pandas as pd
import shap
import torch
import logging
from typing import Dict, Any, List

import config
from src.utils import setup_logger, load_artifact
from src.model_dl import ClinicalLSTM

# Initialize module-level diagnostic auditor
logger = setup_logger("explainability_engine")

class ClinicalExplainerHub:
    """Manages consensus feature attributions across both ML and DL model assets."""
    def __init__(self):
        logger.info("Initializing Consensus Clinical Explainer Infrastructure...")
        
        # Load the frozen unified model bundle
        self.payload = load_artifact(config.HYBRID_ARTIFACT_NAME)
        self.models = self.payload["lgbm_fold_ensemble"]
        self.feature_names = self.payload["feature_names"]
        self.background_data = self.payload.get("shap_background")
        
        # Extract and verify the base tree components from our calibrated estimators
        base_estimators = [fold_model.estimator for fold_model in self.models]
        for idx, base_est in enumerate(base_estimators):
            assert hasattr(base_est, 'predict'), f"Fold {idx} base model failed runtime validation check."
            
        # FIXED: Initialize Conditional SHAP Explainers using stored training data backgrounds
        if self.background_data is not None:
            logger.info("Background reference framework detected. Spinning up Conditional SHAP matrix...")
            self.tree_explainers = [shap.TreeExplainer(model, data=self.background_data) for model in base_estimators]
        else:
            logger.warning("No background data found. Falling back to Interventional structural SHAP paths.")
            self.tree_explainers = [shap.TreeExplainer(model) for model in base_estimators]
            
        # Reconstruct and map weights into our active PyTorch Deep Learning instance
        self.lstm_model = None
        if "pytorch_lstm_state" in self.payload:
            logger.info("Deep learning weights recovered inside binary asset. Preparing PyTorch engine...")
            n_features = self.payload["lstm_feature_count"]
            self.lstm_model = ClinicalLSTM(n_features=n_features)
            self.lstm_model.load_state_dict(self.payload["pytorch_lstm_state"])
            self.lstm_model.eval()

    def _extract_shap_values(self, shap_raw: Any) -> np.ndarray:
        """
        Normalizes variations in SHAP output structures generated across different
        package versions, ensuring a consistent 1D array layout.
        """
        if isinstance(shap_raw, list):
            # Isolate the attribution vector for the positive mortality class (index 1)
            return shap_raw[1][0]
        elif len(shap_raw.shape) == 3:
            return shap_raw[0, :, 1]
        elif len(shap_raw.shape) == 2 and shap_raw.shape[0] == 1:
            return shap_raw[0]
        return shap_raw

    def generate_tabular_explanation(self, patient_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes an unbiased consensus feature attribution mapping for a single patient profile
        by averaging calculated SHAP forces across all 5 cross-validation fold models.
        """
        X_pure = patient_features[self.feature_names]
        
        # FIXED: Loop through all 5 fold models and extract individual SHAP values safely
        fold_shap_matrices = []
        for exp in self.tree_explainers:
            raw_val = exp.shap_values(X_pure)
            clean_vector = self._extract_shap_values(raw_val)
            fold_shap_matrices.append(clean_vector)
            
        # Compute consensus mean vector across all validation folds
        consensus_shap = np.array(fold_shap_matrices).mean(axis=0)
        
        # Compute the global reference base value across all models
        base_value = np.mean([
            exp.expected_value[1] if isinstance(exp.expected_value, list) else exp.expected_value
            for exp in self.tree_explainers
        ])
        
        explanations = {}
        for col_idx, feat_name in enumerate(self.feature_names):
            observed_val = float(X_pure.iloc[0, col_idx])
            impact_force = float(consensus_shap[col_idx])
            
            # NOTE: 1e-4 threshold filters out tiny baseline adjustments for clean UI presentation.
            # It is not a statistical significance marker.
            if abs(impact_force) > 1e-4:
                explanations[feat_name] = {
                    "observed_value": observed_val,
                    "impact_force": impact_force,
                    "direction": "RISK_INCREASE" if impact_force > 0 else "RISK_DECREASE"
                }
                
        # Sort values so that the primary clinical risk drivers bubble up to the top
        sorted_explanations = dict(
            sorted(explanations.items(), key=lambda item: abs(item[1]["impact_force"]), reverse=True)
        )
        
        return {
            "base_value": float(base_value),
            "feature_attributions": sorted_explanations
        }

    def generate_dl_sequence_explanation(
        self, 
        patient_sequence: np.ndarray, 
        background_sequences: np.ndarray
    ) -> Dict[str, float]:
        """
        NEW: Computes GradientSHAP attributions for PyTorch LSTM sequence predictions.
        Collapses time dimensions via absolute means to evaluate tracking weights.
        """
        if self.lstm_model is None:
            logger.warning("Deep Learning model weights missing from deployment payload. Skipping sequence path.")
            return {}
            
        try:
            # Cast raw data to Torch arrays to drive backpropagation hooks
            patient_tensor = torch.FloatTensor(patient_sequence)
            background_tensor = torch.FloatTensor(background_sequences)
            
            # Instantiate GradientExplainer to reverse engineer our recurrent layers
            dl_explainer = shap.GradientExplainer(self.lstm_model, background_tensor)
            
            # Compute path gradients across the patient sequence tensor shape: (1, 48, Features)
            shap_sequence_values = dl_explainer.shap_values(patient_tensor)
            
            # Isolate matrix and collapse the hourly time axis to isolate net feature force
            raw_sequence_matrix = shap_sequence_values[0] if isinstance(shap_sequence_values, list) else shap_sequence_values
            
            # Compute mean absolute attribution per parameter across the 48-hour observation window
            mean_absolute_forces = np.mean(np.abs(raw_sequence_matrix), axis=0) # Shape: (Features,)
            
            # Map values back to their corresponding sequence names defined in config.py
            sequence_explanations = {
                feature_name: float(mean_absolute_forces[idx])
                for idx, feature_name in enumerate(config.SEQUENCE_FEATURES)
            }
            
            # Sort features to identify high-impact parameters
            return dict(sorted(sequence_explanations.items(), key=lambda item: item[1], reverse=True))
            
        except Exception as e:
            logger.error(f"GradientSHAP computation sequence was interrupted: {str(e)}")
            return {}
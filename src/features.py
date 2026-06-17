# src/features.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import config
from src.utils import setup_logger

# Initialize module-level diagnostic auditor
logger = setup_logger("feature_engineering")

def extract_advanced_clinical_features(
    raw_long_df: pd.DataFrame, 
    return_sequences: bool = False
) -> Dict[str, Any]:
    """
    Transforms irregular time-series EHR patient records into a high-dimensional
    tabular summary matrix for GBDTs, with an optional 3D tensor sequence path
    to enable prospective Deep Learning ensembling.
    """
    logger.info(f"Initializing multi-phase extraction pipeline on matrix shape: {raw_long_df.shape}")
    
    # 1. Protect and compute core temporal tracking markers
    ts_data = raw_long_df.copy()
    if 'minutes' not in ts_data.columns:
        logger.info("Computing timeline minute arrays from chronological strings...")
        ts_data['minutes'] = ts_data['Time'].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else 0
        )
        
    # Isolate general demographic metrics (Fixed baseline metrics on admission)
    static_vars: List[str] = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    static_rows = ts_data[ts_data['Parameter'].isin(static_vars)]
    static_pivoted = static_rows.groupby(['RecordId', 'Parameter'])['Value'].mean().unstack().reset_index()
    
    # Filter time-series records to focus solely on clinical variables
    ts_vitals = ts_data[~ts_data['Parameter'].isin(static_vars)].copy()
    
    # Apply Vectorized Physiological Safety Net to isolate extreme sensor anomalies
    for col, limits in config.CLINICAL_SAFETY_BOUNDS.items():
        mask = (ts_vitals['Parameter'] == col) & ((ts_vitals['Value'] < limits[0]) | (ts_vitals['Value'] > limits[1]))
        ts_vitals.loc[mask, 'Value'] = np.nan
    ts_vitals = ts_vitals.dropna(subset=['Value'])

    # 2. Extract global trajectory windows (Min, Max, Mean)
    logger.info("Aggregating baseline chronological matrices...")
    summary_stats = ts_vitals.groupby(['RecordId', 'Parameter'])['Value'].agg(['min', 'max', 'mean']).unstack()
    summary_stats.columns = [f"{col[1]}_{col[0]}" for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()

    # 3. FIXED: High-Speed Vectorized Early vs Late Window Delta Analysis
    logger.info("Executing vectorized timeline trajectory comparisons...")
    first_6h = ts_vitals[ts_vitals['minutes'] <= 360]
    
    # Locate maximum elapsed window per patient profile to anchor late metrics
    max_times = ts_vitals.groupby('RecordId')['minutes'].max().reset_index().rename(columns={'minutes': 'max_time'})
    vitals_with_max = pd.merge(ts_vitals, max_times, on='RecordId')
    last_6h = vitals_with_max[vitals_with_max['minutes'] >= (vitals_with_max['max_time'] - 360)]
    
    # Pivot into matrices to subtract entirely without loops
    start_means = first_6h.groupby(['RecordId', 'Parameter'])['Value'].mean().unstack()
    end_means = last_6h.groupby(['RecordId', 'Parameter'])['Value'].mean().unstack()
    
    # Reindex to ensure identical feature shape footprints
    all_safety_keys = list(config.CLINICAL_SAFETY_BOUNDS.keys())
    active_ts_keys = [k for k in all_safety_keys if k not in static_vars]
    
    start_means = start_means.reindex(columns=active_ts_keys)
    end_means = end_means.reindex(columns=active_ts_keys)
    
    # Calculate delta differences. 0.0 delta tracks a stable physiological state.
    delta_df = (end_means - start_means).fillna(0.0)
    delta_df.columns = [f"{col}_delta" for col in delta_df.columns]
    delta_df = delta_df.reset_index()

    # 4. FIXED: High-Speed Vectorized Informative Missingness Mask Generation
    logger.info("Assembling structured missingness indicator matrix layers...")
    recorded_counts = ts_vitals.groupby(['RecordId', 'Parameter']).size().unstack(fill_value=0)
    recorded_counts = recorded_counts.reindex(columns=active_ts_keys, fill_value=0)
    
    # Invert matrix: 1 represents a missing indicator flag
    missing_df = (recorded_counts == 0).astype(int)
    missing_df.columns = [f"{col}_is_missing" for col in missing_df.columns]
    missing_df = missing_df.reset_index()

    # 5. Coordinate cross-channel data merges
    features_tabulated = pd.merge(static_pivoted, summary_stats, on='RecordId', how='outer')
    features_tabulated = pd.merge(features_tabulated, delta_df, on='RecordId', how='left')
    features_tabulated = pd.merge(features_tabulated, missing_df, on='RecordId', how='left')
    
    # 6. Execute medical ratio calculations and apply clean configuration default fills
    features_tabulated = construct_physiological_ratios(features_tabulated)
    
    output_package: Dict[str, Any] = {"tabular": features_tabulated}
    
    # 7. NEW: Deep Learning Sequence Array Processing Layer
    if return_sequences:
        logger.info("return_sequences parameter verified. Compiling 3D temporal arrays...")
        sequence_tensor = build_padded_sequences(ts_vitals)
        output_package["sequences"] = sequence_tensor
        
    logger.info("Advanced clinical feature construction sequence finalized successfully.")
    return output_package

def construct_physiological_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds non-linear medical risk ratios and handles cross-channel 
    imputation using fixed configuration defaults.
    """
    epsilon = 1e-9
    logger.info("Calculating advanced clinical metric indexes...")
    
    # Map raw parameters safely using unified configuration fallbacks
    hr = df['HR_mean'].fillna(config.CLINICAL_DEFAULTS.get('HR', 80.0))
    sys_bp = df['SysABP_mean'].fillna(df['NISysABP_mean']).fillna(120.0)
    pa_o2 = df['PaO2_mean'].fillna(config.CLINICAL_DEFAULTS.get('PaO2', 100.0))
    fi_o2 = df['FiO2_mean'].fillna(config.CLINICAL_DEFAULTS.get('FiO2', 0.21))
    bun = df['BUN_mean'].fillna(20.0)
    creat = df['Creatinine_mean'].fillna(1.0)
    gcs = df['GCS_mean'].fillna(config.CLINICAL_DEFAULTS.get('GCS', 15.0))
    gcs_delta = df['GCS_delta'].fillna(0.0)
    
    # Append structured ratio columns
    df['Shock_Index'] = hr / (sys_bp + epsilon)
    df['RPP'] = hr * sys_bp
    df['PF_Ratio'] = pa_o2 / (fi_o2 + epsilon)
    df['BUN_Creat_Ratio'] = bun / (creat + epsilon)
    df['GCS_Trend_Severity'] = gcs * gcs_delta
    
    # Calculate your notebook's multi-channel Physiological Chaos Score
    v_deltas = ['HR_delta', 'SysABP_delta', 'MAP_delta', 'RespRate_delta', 'Temp_delta']
    available_deltas = [c for c in v_deltas if c in df.columns]
    df['Total_Instability_Score'] = df[available_deltas].abs().sum(axis=1)
    
    # Fill remaining gaps using safe configuration parameters
    for col, default_val in config.CLINICAL_DEFAULTS.items():
        mean_col_name = f"{col}_mean"
        if mean_col_name in df.columns:
            df[mean_col_name] = df[mean_col_name].fillna(default_val)
            
    # FIXED: Migrated hardcoded drop items directly into configuration parameters
    df = df.drop(columns=config.HIGH_MISSING_DROP_COLS, errors='ignore')
    return df

def build_padded_sequences(ts_vitals: pd.DataFrame, max_len: int = 48) -> np.ndarray:
    """
    Pivots irregular longitudinal streams into a standardized 3D padded tensor
    of shape (n_patients, max_len, n_features) to drive deep sequence layers.
    """
    unique_patients = ts_vitals['RecordId'].unique()
    n_patients = len(unique_patients)
    sequence_features = config.SEQUENCE_FEATURES
    n_features = len(sequence_features)
    
    # Initialize zero-padded base matrix (zero-padding acts as missing data)
    tensor = np.zeros((n_patients, max_len, n_features))
    
    for idx, pid in enumerate(unique_patients):
        p_records = ts_vitals[ts_vitals['RecordId'] == pid].sort_values('minutes')
        
        # Group metrics into hourly observation steps
        p_records['hour_bucket'] = (p_records['minutes'] // 60).astype(int)
        p_records = p_records[p_records['hour_bucket'] < max_len]
        
        if p_records.empty:
            continue
            
        pivoted = p_records.pivot_table(index='hour_bucket', columns='Parameter', values='Value', aggfunc='mean')
        pivoted = pivoted.reindex(columns=sequence_features)
        
        # Extract values and handle right-aligned temporal masking padding
        seq_values = pivoted.values
        seq_len = min(len(seq_values), max_len)
        if seq_len > 0:
            tensor[idx, -seq_len:, :] = seq_values[-seq_len:]
            
    return tensor
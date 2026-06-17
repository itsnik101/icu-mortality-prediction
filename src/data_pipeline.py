# src/data_pipeline.py
"""
Project ICU: Data Ingestion & Cohort Assembly Pipeline
Handles batch parsing of raw irregular patient files, enforces the 48-hour 
prospective observation window, and integrates outcome targets.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

import config
from src.utils import setup_logger, EmptyPatientRecordError, MissingDemographicsError

# Initialize module-level diagnostic auditor
logger = setup_logger("data_pipeline")

def parse_chronological_time(time_str: str) -> int:
    """Converts HH:MM clinical text file timestamps into absolute integer minutes elapsed."""
    try:
        if not isinstance(time_str, str) or ':' not in time_str:
            return 0
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0

def process_raw_patient_file(file_path: Path) -> pd.DataFrame:
    """
    Transforms a single sparse patient text file into a clean time-series dataframe.
    Enforces strict clinical window cutoffs and safety checks.
    """
    try:
        record_id = int(file_path.stem)
        
        # Read the raw 3-column format: Time, Parameter, Value
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Fallback safeguard for empty files
        if df.empty:
            raise EmptyPatientRecordError(record_id, "Raw text file contains no operational entries.")
            
        # Convert textual hour blocks to total minutes elapsed
        df['minutes'] = df['Time'].apply(parse_chronological_time)
        df['RecordId'] = record_id
        
        # Enforce the strict 48-Hour Observation Window to prevent target data leakage
        df = df[df['minutes'] <= config.OBSERVATION_WINDOW_MINUTES]
        
        if df.empty:
            raise EmptyPatientRecordError(record_id, "No measurements found within the prospective 48-hour window.")
            
        # Verify essential demographic markers exist on ICU admission
        unique_parameters = df['Parameter'].values
        if 'Age' not in unique_parameters:
            raise MissingDemographicsError(record_id, "Age")
        if 'Gender' not in unique_parameters:
            raise MissingDemographicsError(record_id, "Gender")
            
        return df[['RecordId', 'minutes', 'Parameter', 'Value']]
        
    except EmptyPatientRecordError as e:
        # Log empty records as warnings rather than crashing the whole system run
        logger.warning(str(e))
        return pd.DataFrame()
    except MissingDemographicsError as e:
        logger.warning(str(e))
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected parsing failure on record {file_path.name}: {str(e)}")
        return pd.DataFrame()

def compile_raw_database() -> pd.DataFrame:
    """
    Aggregates all 4,000 separate patient text files into a master dataset.
    Uses binary Parquet caching to skip file scanning on consecutive runs.
    """
    # Define a clean path destination for our binary snapshot file
    cache_path = config.PROCESSED_DATA_DIR / "raw_database_cache.parquet"
    
    # SMART CHECK: If the snapshot file already exists, load it instantly and skip the rest!
    if cache_path.exists():
        logger.info(f"💾 Found cached database snapshot at {cache_path}. Loading instantly...")
        return pd.read_parquet(cache_path)
        
    # --- IF NO CACHE EXISTS, DO THE SLOW FILE SCAN ONCE ---
    target_folder = config.RAW_DATA_DIR / "set-a"
    patient_files = list(target_folder.glob("*.txt"))
    
    if not patient_files:
        raise FileNotFoundError(
            f"No raw patient text records found at: {target_folder}\n"
            f"Please ensure your downloaded files are unzipped inside 'data/raw/set-a/'"
        )
        
    logger.info(f"🔍 No cache found. Scanning {len(patient_files)} text files. This will take a moment...")
    
    master_frames = []
    for f in patient_files:
        patient_df = process_raw_patient_file(f)
        if not patient_df.empty:
            master_frames.append(patient_df)
            
    if not master_frames:
        raise RuntimeError("All discovered patient records failed validation checks.")
        
    # Combine everything into one table
    consolidated_long_df = pd.concat(master_frames, ignore_index=True)
    
    # SAVE THE CACHE SNAPSHOT: Freeze it down so next time is instantaneous
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    consolidated_long_df.to_parquet(cache_path, compression="snappy")
    logger.info(f"✨ Successfully created database cache snapshot at: {cache_path}")
    
    return consolidated_long_df

def attach_outcomes(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns and merges calculated patient features with their actual survival outcomes.
    Guarantees no row alignment mismatches during training loops.
    """
    outcomes_file = config.RAW_DATA_DIR / "Outcomes-a.txt"
    if not outcomes_file.exists():
        raise FileNotFoundError(f"Critical target file missing. Please place 'Outcomes-a.txt' in: {config.RAW_DATA_DIR}")
        
    outcomes_df = pd.read_csv(outcomes_file)
    # Standardize column naming conventions to avoid merge errors
    outcomes_df = outcomes_df.rename(columns={'RecordID': 'RecordId'})
    
    logger.info("Merging clinical summary metrics with survival outcome registries...")
    final_dataset = pd.merge(features_df, outcomes_df[['RecordId', 'In-hospital_death']], on='RecordId', how='inner')
    
    logger.info(f"Cohort assembly finalized successfully. Final cohort shape: {final_dataset.shape}")
    return final_dataset
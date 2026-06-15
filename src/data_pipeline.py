import os
import pandas as pd
import numpy as np
from typing import List

def parse_single_patient_file(file_path: str) -> dict:
    """
    Parses a single sparse 3-column PhysioNet text file 
    and converts it into a flat summary dictionary.
    """
    # Extract the RecordID from the filename
    record_id = os.path.basename(file_path).replace('.txt', '')
    
    # Read the text file safely
    df = pd.read_csv(file_path, header=0, names=['Time', 'Parameter', 'Value'])
    
    # Create our structured row starting with the unique ID
    patient_features = {'RecordID': int(record_id)}
    
    # Isolate unique variables present in this patient's data
    unique_parameters = df['Parameter'].unique()
    
    for param in unique_parameters:
        param_data = df[df['Parameter'] == param]
        
        # Scenario 1: Fixed Demographics (Age, Gender, etc.)
        if param in ['Age', 'Gender', 'Height', 'ICUType']:
            patient_features[param] = param_data['Value'].iloc[0]
            
        # Scenario 2: Time-Series Metrics (Aggregate to avoid dimension explosion)
        else:
            # Drop negative values or physiological noise anomalies
            valid_values = param_data['Value'][param_data['Value'] >= 0]
            
            if not valid_values.empty:
                # Capture the clinical statistical baseline across the first 48 hours
                patient_features[f'{param}_mean'] = valid_values.mean()
                patient_features[f'{param}_min'] = valid_values.min()
                patient_features[f'{param}_max'] = valid_values.max()
                # Informative Missingness: This feature was successfully measured!
                patient_features[f'{param}_is_missing'] = 0
            else:
                # Feature was never measured
                patient_features[f'{param}_is_missing'] = 1

    return patient_features

def build_consolidated_dataset(raw_dir_path: str) -> pd.DataFrame:
    """
    Iterates through the raw folder containing thousands of patient files,
    parses each sequentially, and stacks them into a single wide dataset.
    """
    all_patients = []
    
    if not os.path.exists(raw_dir_path):
        raise FileNotFoundError(f"Directory not found at: {raw_dir_path}")
        
    file_list = [f for f in os.listdir(raw_dir_path) if f.endswith('.txt')]
    print(f"--> Found {len(file_list)} records in {raw_dir_path}. Commencing pipeline parse...")
    
    for filename in file_list:
        full_path = os.path.join(raw_dir_path, filename)
        patient_dict = parse_single_patient_file(full_path)
        all_patients.append(patient_dict)
        
    # Convert list of dictionaries cleanly to a flat Dataframe
    dataset = pd.DataFrame(all_patients)
    
    # Fill remaining gaps where features were absent across the entire cohort
    dataset = dataset.fillna(dataset.median())
    
    return dataset

if __name__ == "__main__":
    print("PhysioNet 2012 Pipeline Module compiled successfully.")
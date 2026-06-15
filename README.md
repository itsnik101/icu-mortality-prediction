# ICU Mortality Risk Prediction Model

## Overview
This project is a machine learning pipeline that predicts in-hospital mortality for ICU patients[cite: 82]. It uses early-admission vital signs and lab results to act as an early warning system. The model processes 4,000 patient records and handles a severe class imbalance to achieve an 0.815 AUROC score.

## The Dataset
The data comes from the PhysioNet Computing in Cardiology Challenge 2012.
Volume: 4,000 patient ICU records.
Features: 44 clinical variables including vitals, labs, and demographics.
Target Variable: In-hospital death.

## Data Pipeline and Methodology
Real clinical data has missing values and sensor errors. Here is how the data was cleaned and prepared:

### 1. Data Cleaning
Replaced impossible values like negative heart rates with empty values based on medical safety boundaries.
Dropped columns that were missing over 90 percent of their data.

### 2. Missing Data Imputation
Filled missing healthy lab values with standard normal medical values. For missing invasive blood pressure readings, non-invasive readings were used instead.

### 3. Feature Engineering
Created new clinical metrics to capture patient health trends:
- Shock Index: Heart Rate divided by Systolic Blood Pressure.
- P/F Ratio: Used to measure lung failure severit.
- GCS Trend Severity: Tracked changes in consciousness levels.
- Total Instability Score: Measured overall physiological chaos.

### 4. Dimensionality Reduction
Removed redundant features and reduced the dataset to 67 highly predictive variables[cite: 98].

## Modeling and Performance
To handle the 1 to 6 survival-to-death ratio, class weights were balanced across multiple models[cite: 100].

The XGBoost classifier was selected as the final model because it provided the best balance of precision and recall.
- XGBoost AUROC: 0.8155 
- XGBoost F1-Score: 0.4506

Grid search was used to fine-tune the model parameters like max depth and learning rate. ]Precision-recall curves were plotted across different probability thresholds to allow adjustments based on ICU bed availabilit[cite: 105, 106].

## Tech Stack
- Language: Python 
- Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn

## How to Run the Project
1. Clone the repository: git clone https://github.com/itsnik101/icu-mortality-prediction.git
2. Install the required dependencies: pip install pandas numpy scikit-learn xgboost matplotlib seaborn
3. Download the PhysioNet Challenge 2012 dataset and place the files in the /data folder
4. Run the Jupyter Notebook cell by cell.

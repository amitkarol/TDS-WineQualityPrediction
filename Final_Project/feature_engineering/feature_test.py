import pandas as pd
import os
import sys

# Ensure the feature_engineering module is in the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from feature_generator import SemiAutomatedFeatureEngineering

# Load dataset
dataset_path = os.path.abspath("Data/cancer_patient_clean.csv")
df = pd.read_csv(dataset_path)

# Define target column and problem type
target_column = "Level"
task_type = "classification"

# Run feature engineering
feature_engineer = SemiAutomatedFeatureEngineering(df, target_column, task=task_type)
results = feature_engineer.run_pipeline()

# Print results
print("\n🔹 Final Model Performance:")
print(results)

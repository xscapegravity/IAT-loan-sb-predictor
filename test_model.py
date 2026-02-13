import pandas as pd
import numpy as np
import logging
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from iatd_loan_predictor import DataLoader, DataPreprocessor, ModelEvaluator, ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model(data_path, model_path, preprocessor_path):
    # 1. Load Data
    loader = DataLoader()
    try:
        if not os.path.exists(data_path):
           logging.warning(f"File {data_path} not found. Creating dummy data to match training logic.")
           data = pd.DataFrame({
               'LoanID': range(100),
               'Defaulted': np.random.choice([0, 1], 100),
               'Total_Debt': np.random.rand(100) * 100000,
               'Total_Assets': np.random.rand(100) * 200000,
               'Net_Profit': np.random.rand(100) * 50000,
               'Total_Revenue': np.random.rand(100) * 150000,
               'Loan_Amount': np.random.rand(100) * 50000,
               'Years_in_Business': np.random.randint(1, 20, 100),
               'Credit_Score': np.random.randint(500, 850, 100),
               'Business_Type': np.random.choice(['Retail', 'Tech', 'Service'], 100)
           })
        else:
           data = loader.load_data(data_path)

        # 2. Load Artifacts
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor file not found. Run train_model.py first.")
            
        logging.info(f"Loading model from {model_path}...")
        model_trainer = joblib.load(model_path) # Data is ModelTrainer instance
        
        logging.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = joblib.load(preprocessor_path)
        
        # 3. Prepare Test Data
        if 'Defaulted' in data.columns:
            X = data.drop(columns=['Defaulted', 'LoanID'], errors='ignore')
            y = data['Defaulted']
            
            # Split train/test - MUST use same seed as training to get the "Test" portion correctly
            # In a real prediction setting, we wouldn't split, we would just predict on new X.
            # But for "Testing model performance" on the hold-out set:
            _, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Transform using loaded preprocessor
            X_test = preprocessor.transform(X_test_raw, is_training=False)
            
            # 4. Evaluate
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model_trainer, X_test, y_test)
            
        else:
            logging.error("Target column 'Defaulted' not found in dataset. Cannot evaluate performance.")

    except Exception as e:
        logging.error(f"An error occurred during testing: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Loan Default Predictor.")
    parser.add_argument("--data_path", type=str, default="IATD_Loan_Data_Sample.csv", help="Path to input CSV.")
    parser.add_argument("--model_path", type=str, default="model.joblib", help="Path to saved model.")
    parser.add_argument("--preprocessor_path", type=str, default="preprocessor.joblib", help="Path to saved preprocessor.")
    
    args = parser.parse_args()
    
    test_model(args.data_path, args.model_path, args.preprocessor_path)

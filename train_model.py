import pandas as pd
import numpy as np
import logging
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from iatd_loan_predictor import DataLoader, DataPreprocessor, ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path, model_path, preprocessor_path):
    # 1. Load Data
    loader = DataLoader()
    try:
        if not os.path.exists(data_path):
           logging.warning(f"File {data_path} not found. Creating dummy data for demonstration.")
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

        # 2. Preprocess Data
        preprocessor = DataPreprocessor()
        
        if 'Defaulted' in data.columns:
            X = data.drop(columns=['Defaulted', 'LoanID'], errors='ignore')
            y = data['Defaulted']
            
            # Split train/test
            # Note: We are training on X_train. 
            # In a real scenario, we might want to save X_test to disk for consistency in Independent Testing,
            # or we assume the test script will load new data or re-split reliably if the data source is static.
            # Here we will re-split with same random_state in test script to simulate "hold-out".
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit preprocessor on training data
            preprocessor.fit(X_train_raw)
            
            # Transform
            X_train = preprocessor.transform(X_train_raw, is_training=True)
            
            # 3. Model Training
            rf_trainer = ModelTrainer()
            rf_trainer.train(X_train, y_train)
            
            # Save artifacts
            logging.info(f"Saving model to {model_path}...")
            joblib.dump(rf_trainer, model_path)
            
            logging.info(f"Saving preprocessor to {preprocessor_path}...")
            joblib.dump(preprocessor, preprocessor_path)
            
            logging.info("Training and saving complete.")
            
        else:
            logging.error("Target column 'Defaulted' not found in dataset.")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Loan Default Predictor.")
    parser.add_argument("--data_path", type=str, default="IATD_Loan_Data_Sample.csv", help="Path to input CSV.")
    parser.add_argument("--model_path", type=str, default="model.joblib", help="Path to save the trained model.")
    parser.add_argument("--preprocessor_path", type=str, default="preprocessor.joblib", help="Path to save the preprocessor.")
    
    args = parser.parse_args()
    
    train_model(args.data_path, args.model_path, args.preprocessor_path)

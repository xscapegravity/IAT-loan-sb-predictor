from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
from iatd_loan_predictor import DataLoader, DataPreprocessor, ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Log environment details to ensure correct Conda env is loaded
logging.info(f"Active Conda Env: {os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')}")
logging.info(f"Python Executable: {sys.executable}")

# Load Artifacts
MODEL_PATH = "model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

model_trainer = None
preprocessor = None

def load_artifacts():
    global model_trainer, preprocessor
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            logging.info("Loading model and preprocessor...")
            model_trainer = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logging.info("Artifacts loaded successfully.")
        else:
            logging.warning("Model or preprocessor not found. Please run train_model.py first.")
    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")

load_artifacts()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        if model_trainer is None or preprocessor is None:
            return render_template('index.html', error="Model not loaded. Please contact administrator.")
        
        try:
            # Extract data from form
            data = {
                'Total_Debt': float(request.form.get('total_debt', 0)),
                'Total_Assets': float(request.form.get('total_assets', 0)),
                'Net_Profit': float(request.form.get('net_profit', 0)),
                'Total_Revenue': float(request.form.get('total_revenue', 0)),
                'Loan_Amount': float(request.form.get('loan_amount', 0)),
                'Years_in_Business': int(request.form.get('years_in_business', 0)),
                'Credit_Score': int(request.form.get('credit_score', 0)),
                'Business_Type': request.form.get('business_type', 'Other')
            }
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Preprocess
            # Note: is_training=False ensures we use the fit from training
            df_processed = preprocessor.transform(df, is_training=False)
            
            # Predict
            pred = model_trainer.predict(df_processed)[0]
            prob = model_trainer.predict_proba(df_processed)[0][1] # Probability of Default (1)
            
            prediction = "Denied" if pred == 1 else "Approved"
            probability = f"{prob:.2%}" if pred == 1 else f"{(1-prob):.2%}" # Confidence
            
            # If Approved, confidence is 1 - prob(Default). If Denied, confidence is prob(Default).
            # Wait, usually businesses want "Risk Score". 
            # Let's say: Prediction: Default (Risk: High) vs No Default (Risk: Low).
            # User wants "Approved" vs "Denied".
            # If Defaulted (1) -> Denied.
            # If Not Defaulted (0) -> Approved.
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return render_template('index.html', error=f"Error processing request: {e}")

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True, port=8000)

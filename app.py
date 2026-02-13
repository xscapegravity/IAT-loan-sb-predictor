from flask import Flask, request, render_template, current_app
import pandas as pd
import joblib
import os
import sys
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Log environment details
    logging.info(f"Active Conda Env: {os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')}")
    logging.info(f"Python Executable: {sys.executable}")

    # Load artifacts and attach to app instance
    load_artifacts(app)

    # Register routes
    register_routes(app)

    return app

def load_artifacts(app):
    """Load model and preprocessor into app config."""
    model_path = app.config['MODEL_PATH']
    preprocessor_path = app.config['PREPROCESSOR_PATH']

    try:
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            logging.info(f"Loading artifacts from {model_path} and {preprocessor_path}...")
            app.model_trainer = joblib.load(model_path)
            app.preprocessor = joblib.load(preprocessor_path)
            logging.info("Artifacts loaded successfully.")
        else:
            logging.warning(f"Artifacts not found at {model_path} or {preprocessor_path}. Predictions will be unavailable.")
            app.model_trainer = None
            app.preprocessor = None
    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")
        app.model_trainer = None
        app.preprocessor = None

def register_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        prediction = None
        probability = None
        error_message = None

        if request.method == 'POST':
            if not current_app.model_trainer or not current_app.preprocessor:
                return render_template('index.html', error="Model service unavailable. Please contact support.")
            
            try:
                # Extract and validate data
                data = get_form_data(request.form)
                
                # Create DataFrame
                df = pd.DataFrame([data])
                
                # Preprocess
                df_processed = current_app.preprocessor.transform(df, is_training=False)
                
                # Predict
                pred = current_app.model_trainer.predict(df_processed)[0]
                prob = current_app.model_trainer.predict_proba(df_processed)[0][1] # Probability of Default (1)
                
                prediction = "Denied" if pred == 1 else "Approved"
                # If Approved (0), confidence is 1 - prob(1). If Denied (1), confidence is prob(1).
                confidence_val = prob if pred == 1 else 1 - prob
                probability = f"{confidence_val:.2%}"
                
            except ValueError as ve:
                logging.warning(f"Validation error: {ve}")
                error_message = f"Invalid input: {ve}"
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                error_message = f"An unexpected error occurred. Please try again."

        return render_template('index.html', prediction=prediction, probability=probability, error=error_message)

def get_form_data(form_data):
    """Extracts and validates form data."""
    try:
        return {
            'Total_Debt': float(form_data.get('total_debt', 0)),
            'Total_Assets': float(form_data.get('total_assets', 0)),
            'Net_Profit': float(form_data.get('net_profit', 0)),
            'Total_Revenue': float(form_data.get('total_revenue', 0)),
            'Loan_Amount': float(form_data.get('loan_amount', 0)),
            'Years_in_Business': int(form_data.get('years_in_business', 0)),
            'Credit_Score': int(form_data.get('credit_score', 0)),
            'Business_Type': form_data.get('business_type', 'Other')
        }
    except (ValueError, TypeError) as e:
        raise ValueError("Please ensure all numeric fields contain valid numbers.") from e

if __name__ == "__main__":
    app = create_app()
    app.run(debug=app.config['DEBUG'], port=app.config['PORT'])

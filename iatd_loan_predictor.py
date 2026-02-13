import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Handles loading of data from CSV files.
    """
    def __init__(self):
        pass

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            logging.info(f"Loading data from {filepath}...")
            df = pd.read_csv(filepath)
            logging.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {filepath}: {e}")
            raise e

class DataPreprocessor:
    """
    Encapsulates data cleaning, feature engineering, and encoding.
    """
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.training_columns = None # Columns after one-hot encoding (fit on training data)
        self.numerical_cols = []
        self.categorical_cols = []

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits imputers on the training data.

        Args:
            df (pd.DataFrame): The training DataFrame.
        """
        logging.info("Fitting preprocessor on training data...")
        
        # Identify numerical and categorical columns
        # We exclude 'Defaulted' (target) and 'LoanID' (identifier) if present from imputation strategy logic if needed,
        # but SimpleImputer can handle them if they are in the list. 
        # Typically we fit on X, but here we might accept whole DF. 
        # Best practice: Do not impute Target.
        
        all_numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Filter out target and ID from feature processing lists
        self.numerical_cols = [c for c in all_numerical if c not in ['Defaulted', 'LoanID']]
        self.categorical_cols = [c for c in all_categorical if c not in ['Defaulted', 'LoanID']]

        # Fit imputers
        if self.numerical_cols:
            self.num_imputer.fit(df[self.numerical_cols])
        if self.categorical_cols:
            self.cat_imputer.fit(df[self.categorical_cols])
            
        logging.info("Preprocessor fitting complete.")

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Applies imputation, feature engineering, and encoding to the data.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            is_training (bool): True if processing training data (defines the column structure), False for inference.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for model training/prediction.
        """
        logging.info("Transforming data...")
        df_processed = df.copy()

        # 1. Imputation
        if self.numerical_cols:
            df_processed[self.numerical_cols] = self.num_imputer.transform(df_processed[self.numerical_cols])
        if self.categorical_cols:
            # Check if columns exist (might be missing in new data if completely empty, but unlikely for standard cols)
            existing_cat_cols = [c for c in self.categorical_cols if c in df_processed.columns]
            if existing_cat_cols:
                df_processed[existing_cat_cols] = self.cat_imputer.transform(df_processed[existing_cat_cols])

        # 2. Feature Engineering
        df_processed = self._engineer_features(df_processed)

        # 3. Encoding (One-Hot)
        # Note: We apply get_dummies to the whole dataframe. 
        # We must identify which columns are categorical *after* feature engineering (binning adds categorical cols).
        # We assume standard pandas behaviour: get_dummies converts object/category cols.
        
        df_encoded = pd.get_dummies(df_processed, drop_first=True)

        # 4. Column Alignment
        if is_training:
            # Store columns for future alignment
            # If 'Defaulted' is in training data, it might become a column if not separated?
            # 'Defaulted' is numerical (0/1), so get_dummies won't touch it.
            # We usually want X features only for training_columns.
            # But the user might pass whole DF. 
            # Let's effectively store columns EXCLUDING target if it exists, to match X structure?
            # Or simpler: Store ALL columns produced. If target is present, it's just another column.
            # BUT, when inferencing, 'Defaulted' won't be there.
            # So alignment should align FEATURES.
            
            # Let's save columns excluding 'Defaulted' if present.
            cols = [c for c in df_encoded.columns if c != 'Defaulted']
            self.training_columns = cols
            
            return df_encoded
        else:
            if self.training_columns is None:
                raise ValueError("Preprocessor has not been fitted. Call fit() with training data first.")
            
            # Align columns: Keep Target if present? No, transform usually returns features + target or just processed DF.
            # If input had Defaulted, we keep it? 
            # Let's just reindex to self.training_columns. If 'Defaulted' was in input, it will be dropped if it's not in training_columns!
            # Wait, if I trained with 'Defaulted', it IS in `df_encoded`.
            # I excluded it from `self.training_columns` above? 
            
            # Re-thinking: In training, we usually fit on X. 
            # But the notebook passes the whole DF to preprocessing steps often.
            # Let's robustly handle: 
            # If is_training=True, we define the "Feature Schema". 
            # Features = [all cols except Defaulted, LoanID].
            # So `self.training_columns` should represent the FEATURE columns.
            
            # We align `df_encoded` to `self.training_columns`. 
            # Any column NOT in training_columns (like 'Defaulted' if we excluded it) will be dropped by reindex if we are not careful.
            # Actually `reindex` keeps index, adjusts columns.
            
            # Valid approach:
            # 1. Identify features.
            # 2. Reindex `df_encoded` to match `self.training_columns`.
            # 3. If input had 'Defaulted', we might want to preserve it?
            # But `transform` implies getting the model-ready features.
            # So retrieving X (aligned) is the primary goal.
            
            # Standardizing: This transform returns the dataframe with features aligned to training.
            # It drops extra columns and adds missing ones with 0.
            
            # Preserve target if it exists and we aren't specifically dropping it.
            # But for simplicity and safety in ML pipeline, let's align strictly to expected features.
            # The caller can handle target separation before/after if needed, but usually preprocessor prepares X.
            
            # For the purpose of this script matching the notebook:
            # The notebook applies get_dummies, THEN manually reindexes `df_application`.
            # `df_application_encoded = df_application_encoded.reindex(columns=X.columns, fill_value=0)`
            # So `X.columns` corresponds to the Feature variables.
            
            features_aligned = df_encoded.reindex(columns=self.training_columns, fill_value=0)
            return features_aligned

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features based on financial ratios and logic.
        """
        logging.info("Engineering features...")
        
        # Avoid Division by Zero by replacing 0 with NaN or interacting carefully
        # Notebook: df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # We will do calculations then clean up.

        # Debt_to_Asset_Ratio
        if 'Total_Debt' in df.columns and 'Total_Assets' in df.columns:
            df['Debt_to_Asset_Ratio'] = df['Total_Debt'] / df['Total_Assets']
        
        # Profit_Margin
        if 'Net_Profit' in df.columns and 'Total_Revenue' in df.columns:
            df['Profit_Margin'] = df['Net_Profit'] / df['Total_Revenue']
            
        # Loan_to_Revenue_Ratio
        if 'Loan_Amount' in df.columns and 'Total_Revenue' in df.columns:
            df['Loan_to_Revenue_Ratio'] = df['Loan_Amount'] / df['Total_Revenue']
            
        # Asset_to_Revenue_Ratio
        if 'Total_Assets' in df.columns and 'Total_Revenue' in df.columns:
            df['Asset_to_Revenue_Ratio'] = df['Total_Assets'] / df['Total_Revenue']
            
        # Net_Worth
        if 'Total_Assets' in df.columns and 'Total_Debt' in df.columns:
            df['Net_Worth'] = df['Total_Assets'] - df['Total_Debt']
            
        # Revenue_per_Year_in_Business
        if 'Total_Revenue' in df.columns and 'Years_in_Business' in df.columns:
            df['Revenue_per_Year_in_Business'] = df['Total_Revenue'] / df['Years_in_Business']

        # Handle infinite values created by division by zero
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # We might need to impute these new NaNs? The notebook might have done it implicitly or ignored.
        # But `fillna(0)` or similar is often good. 
        # Notebook mentions "Handle potential division by zero errors ... df.replace...". 
        # It doesn't explicitly mention filling them after. SimpleImputer runs earlier.
        # We'll leave as NaN or fill with 0? Most models (XGBoost) handle NaN. Logistic Regression doesn't.
        # Notebook uses SimpleImputer *before* feature eng.
        # So these new features might have NaNs. 
        # Let's fill them with 0 for safety as ratios typically 0 if denominator 0 (except specialized cases).
        df.fillna(0, inplace=True) 

        # Binning (Create categorical features)
        if 'Business_Age' in df.columns:
            df['Business_Age_Category'] = pd.cut(
                df['Business_Age'], 
                bins=[0, 5, 10, 15, 20, np.inf], 
                labels=['0-5', '5-10', '10-15', '15-20', '20+']
            )
            
        if 'Credit_Score' in df.columns:
            df['Credit_Score_Category'] = pd.cut(
                df['Credit_Score'], 
                bins=[0, 600, 700, 800, np.inf], 
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )
            
        if 'Loan_Amount' in df.columns:
            df['Loan_Size_Category'] = pd.cut(
                df['Loan_Amount'], 
                bins=[0, 50000, 100000, 500000, np.inf], 
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )

        return df

class ModelTrainer:
    """
    Handles model training and hyperparameter tuning.
    """
    def __init__(self, model=None, param_grid=None):
        if model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(random_state=42)
        else:
            self.model = model
            
        self.param_grid = param_grid if param_grid else {}
        self.best_params = None

    def train(self, X_train, y_train):
        """
        Trains the model.
        """
        logging.info("Training model...")
        self.model.fit(X_train, y_train)
        logging.info("Model training complete.")

    def tune_hyperparameters(self, X_train, y_train):
        """
        Performs hyperparameter tuning using GridSearchCV.
        """
        if not self.param_grid:
            logging.warning("No parameter grid provided for tuning. Skipping.")
            return

        logging.info("Tuning hyperparameters...")
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, 
                                   cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        logging.info(f"Hyperparameter tuning complete. Best params: {self.best_params}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class ModelEvaluator:
    """
    Evaluates model performance.
    """
    @staticmethod
    def evaluate(model, X_test, y_test):
        """
        Calculates and logs evaluation metrics.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
        
        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"ROC AUC: {roc_auc}")
        
        logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    # Configuration
    DATA_PATH = "IATD_Loan_Data_Sample.csv" # Update with actual path if needed
    
    # 1. Load Data
    loader = DataLoader()
    try:
        # Assuming the CSV is in the same directory or provide full path
        # For now, we'll try to load 'IATD_Loan_Data_Sample.csv'
        # If it doesn't exist, this will fail, but the structure is correct.
        import os
        if not os.path.exists(DATA_PATH):
           # Create dummy data for demonstration if file doesn't exist
           logging.warning(f"File {DATA_PATH} not found. Creating dummy data for demonstration.")
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
           data = loader.load_data(DATA_PATH)

        # 2. Preprocess Data
        preprocessor = DataPreprocessor()
        
        # Split into X and y
        if 'Defaulted' in data.columns:
            X = data.drop(columns=['Defaulted', 'LoanID'], errors='ignore')
            y = data['Defaulted']
            
            # Split train/test
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit preprocessor on training data
            preprocessor.fit(X_train_raw)
            
            # Transform both
            X_train = preprocessor.transform(X_train_raw, is_training=True)
            X_test = preprocessor.transform(X_test_raw, is_training=False)
            
            # 3. Model Training
            rf_trainer = ModelTrainer()
            # Optional: Hyperparameter tuning
            # param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
            # rf_trainer = ModelTrainer(param_grid=param_grid)
            # rf_trainer.tune_hyperparameters(X_train, y_train)
            
            rf_trainer.train(X_train, y_train)
            
            # 4. Model Evaluation
            evaluator = ModelEvaluator()
            evaluator.evaluate(rf_trainer, X_test, y_test)
            
        else:
            logging.error("Target column 'Defaulted' not found in dataset.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


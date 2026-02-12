# Project Architecture: Small Business Loan Default Risk Predictor

## Overview

This project implements a machine learning system to predict loan default risk for small businesses in Australia. The system analyzes financial data, credit history, and business metrics to assess the likelihood of loan defaults, helping financial institutions make informed lending decisions.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│  Data Sources   │
│  - APRA Stats   │
│  - ABS Data     │
│  - ASIC Data    │
│  - Synthetic    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ETL Pipeline  │
│  - Extract      │
│  - Transform    │
│  - Load         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Processing │
│  - Cleaning     │
│  - Imputation   │
│  - Feature Eng  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Pipeline   │
│  - Training     │
│  - Validation   │
│  - Testing      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Evaluation│
│  - Metrics      │
│  - Comparison   │
│  - Selection    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prediction API │
│  - Risk Scoring │
│  - Default Pred │
└─────────────────┘
```

## Data Architecture

### Data Sources

1. **Australian Prudential Regulation Authority (APRA)**
   - Aggregate data on business lending by banks
   - Breakdowns by business size, industry, and state
   - Macroeconomic context and trends

2. **Australian Bureau of Statistics (ABS)**
   - Business Characteristics Survey data
   - Microdata on business finance and performance
   - Business demographics and credit access

3. **ASIC Insolvency Statistics**
   - Business insolvency data
   - Industry-specific default patterns

4. **Synthetic Data**
   - Generated training dataset with realistic patterns
   - Merged with ASIC industry and regional data

### Data Files

- **Training Data**: `data/synthetic_au_smallbiz_loans.csv`
- **Application Data**: `test_business_loans_10rows.csv`

### Data Schema

#### Core Features

**Business Information:**
- `Business_ID`: Unique identifier
- `Business_Age_Years`: Age of business
- `Industry`: Business industry sector
- `Region`: Australian state/region

**Financial Metrics:**
- `Annual_Revenue_AUD`: Annual revenue in AUD
- `Annual_Profit_AUD`: Annual profit in AUD
- `Total_Assets_AUD`: Total asset value
- `Total_Liabilities_AUD`: Total liabilities

**Credit Information:**
- `Owner_Credit_Score`: Business owner's credit score
- `Previous_Loans`: Number of previous loans
- `Previous_Defaults`: Number of previous defaults

**Loan Details:**
- `Loan_Amount_AUD`: Requested loan amount
- `Loan_Term_Months`: Loan duration in months

**Derived Features:**
- `Debt_to_Asset_Ratio`: Leverage indicator
- `Profit_Margin`: Profitability metric
- `Loan_to_Revenue_Ratio`: Loan size relative to revenue
- `Net_Worth`: Assets minus liabilities
- `Revenue_per_Year_in_Business`: Revenue efficiency
- `Interest_Coverage_Ratio`: Debt servicing ability

**Categorical Features:**
- `Business_Age_Category`: New/Established/Mature
- `Credit_Score_Category`: Poor/Fair/Good/Excellent
- `Loan_Size_Category`: Small/Medium/Large
- `Leverage_Category`: Low/Medium/High

**Target Variable:**
- `Defaulted`: Binary (0 = No Default, 1 = Default)
- `Risk_Score`: Continuous probability score (0-1)

## ETL Pipeline

### Extract
- Configure connectors to data sources (CSV, database, API)
- Load raw data from multiple sources
- Handle different data formats

### Transform
- **Data Cleaning:**
  - Handle missing values (imputation strategies)
  - Standardize column names and data types
  - Remove duplicates
  - Detect and handle outliers

- **Data Preprocessing:**
  - Normalization: Scale features to common range
  - Standardization: Z-score normalization (mean=0, std=1)
  - Encoding: Convert categorical to numerical
    - One-Hot Encoding for nominal categories
    - Label Encoding for ordinal categories

- **Feature Engineering:**
  - Calculate financial ratios (Debt-to-Asset, Profit Margin)
  - Create derived metrics (Net Worth, Revenue efficiency)
  - Bin continuous variables into categories
  - Generate interaction features

### Load
- Store processed data in structured format
- Separate training and application datasets
- Maintain data versioning and lineage

### Tools & Technologies

**Recommended ETL Tools:**
- **Code-based**: Python (pandas, Dask, Apache Spark)
- **Orchestration**: Apache Airflow, Prefect, cron jobs
- **Dedicated ETL**: Talend, Apache NiFi, Microsoft SSIS, AWS Glue, Google Cloud Dataflow

**Current Implementation**: Python with pandas

## Machine Learning Architecture

### ML Pipeline Workflow

```
Raw Data → Preprocessing → Feature Engineering → Train/Test Split →
→ Model Training → Hyperparameter Tuning → Model Evaluation →
→ Model Selection → Deployment
```

### Data Preprocessing Steps

1. **Missing Value Handling**
   - Numerical: SimpleImputer (mean/median strategy)
   - Categorical: SimpleImputer (most frequent strategy)

2. **Feature Scaling**
   - StandardScaler for numerical features
   - Ensures consistent scale across features

3. **Encoding**
   - One-Hot Encoding: For categorical features
   - Label Encoding: For target variable
   - Drop first to avoid multicollinearity

4. **Train-Test Split**
   - Standard 80/20 or 70/30 split
   - Stratified sampling to maintain class balance
   - Random state for reproducibility

### Model Architecture

#### Models Implemented

1. **Logistic Regression**
   - **Type**: Linear classifier
   - **Use Case**: Baseline model, interpretable
   - **Hyperparameters**:
     - `max_iter`: 1000
     - `random_state`: 42
     - Grid Search: C (regularization strength), solver
   - **Advantages**: Fast, interpretable, probabilistic outputs
   - **Limitations**: Assumes linear relationships

2. **Random Forest**
   - **Type**: Ensemble (bagging)
   - **Use Case**: Handles non-linear relationships, robust
   - **Hyperparameters**:
     - `n_estimators`: 100
     - `random_state`: 42
     - Grid Search: max_depth, min_samples_split, min_samples_leaf
   - **Advantages**: Handles feature interactions, less prone to overfitting
   - **Limitations**: Can be slow, less interpretable

3. **Gradient Boosting**
   - **Type**: Ensemble (boosting)
   - **Use Case**: High performance on tabular data
   - **Hyperparameters**:
     - `random_state`: 42
     - Grid Search: learning_rate, n_estimators, max_depth
   - **Advantages**: Excellent performance, handles complex patterns
   - **Limitations**: Prone to overfitting, slower training

4. **XGBoost**
   - **Type**: Gradient boosting (optimized)
   - **Use Case**: Best performance, production-ready
   - **Hyperparameters**:
     - `random_state`: 42
     - `eval_metric`: 'logloss'
     - Grid Search: learning_rate, max_depth, n_estimators, subsample, colsample_bytree
   - **Advantages**: State-of-art performance, handles missing values, regularization
   - **Limitations**: Complex tuning, resource intensive

5. **Support Vector Machine (SVM)**
   - **Type**: Kernel-based classifier
   - **Use Case**: Alternative non-linear classifier
   - **Hyperparameters**:
     - Grid Search: C, kernel, gamma
   - **Advantages**: Effective in high-dimensional spaces
   - **Limitations**: Slow on large datasets, scaling sensitive

### Hyperparameter Tuning

**Strategy**: GridSearchCV
- **Cross-Validation**: 5-fold CV
- **Scoring Metric**: ROC-AUC
- **Parallelization**: n_jobs=-1 (all CPU cores)
- **Approach**: Exhaustive search over parameter grid

**Note**: Balanced hypertuning with training time and CPU usage constraints

### Feature Importance Analysis

- Tree-based models provide feature importances
- Identifies top predictors of default
- Used for feature selection and model interpretation
- Visualized with bar charts for top 10 features

## Evaluation Architecture

### Metrics Used

1. **Classification Metrics**
   - **Accuracy**: Overall correctness
   - **Precision**: True positives / (True positives + False positives)
   - **Recall**: True positives / (True positives + False negatives)
   - **F1 Score**: Harmonic mean of precision and recall

2. **Probabilistic Metrics**
   - **ROC-AUC**: Area under ROC curve (discrimination ability)
   - **ROC Curve**: True Positive Rate vs False Positive Rate

3. **Confusion Matrix Components**
   - **True Positive (TP)**: Correctly predicted default
   - **True Negative (TN)**: Correctly predicted no default
   - **False Positive (FP)**: Incorrectly predicted default (Type I error)
   - **False Negative (FN)**: Incorrectly predicted no default (Type II error)

### Business Impact

- **False Negatives**: Most costly (approving risky loans)
- **False Positives**: Lost opportunity (rejecting good loans)
- **Optimization**: Balance based on business priorities

### Validation Strategy

1. **Verification Phase**
   - Code review and unit testing
   - Integration testing
   - Data integrity checks

2. **Validation Phase**
   - Train-test split validation
   - Cross-validation (k-fold)
   - Hold-out test set evaluation

3. **Performance Monitoring**
   - Continuous tracking of metrics
   - Threshold adjustment based on business needs
   - A/B testing for model updates

## Visualization Architecture

### Stakeholder-Specific Dashboards

1. **Executive Leadership (CEO, CFO, Board)**
   - **Tools**: Tableau, Power BI, Streamlit, Dash
   - **Visualizations**:
     - Default rates over time (line/bar charts)
     - Approval/rejection rates (pie/bar charts)
     - Financial impact summary (KPI cards)
     - Overall model performance (AUC)

2. **Risk Managers**
   - **Tools**: Tableau, Excel, Python reports
   - **Visualizations**:
     - Risk score distributions (histograms, KDE plots)
     - Default rates by segment (bar charts)
     - Feature importance (bar charts)
     - Confusion matrices

3. **Data Scientists / Analysts**
   - **Tools**: Jupyter, Python (matplotlib, seaborn)
   - **Visualizations**:
     - Correlation heatmaps
     - Feature distributions (histograms, box plots)
     - ROC curves comparison
     - Learning curves
     - Residual plots

4. **Loan Officers / Operations**
   - **Tools**: Web dashboard, mobile app
   - **Visualizations**:
     - Individual loan risk scores (gauges)
     - Recommended actions (approve/reject/review)
     - Similar loan comparisons

### Key Visualizations Implemented

1. **Correlation Analysis**
   - Heatmap of feature correlations with default
   - Top N features ranked by correlation strength

2. **Distribution Analysis**
   - KDE plots: Risk score by default status
   - Violin plots: Credit score by default status
   - Histograms: Feature distributions

3. **Categorical Analysis**
   - Default rates by industry (pie chart)
   - Default rates by region (heatmap)
   - Stacked bar charts: Defaults by risk segments

4. **Model Performance**
   - ROC curves for all models
   - Confusion matrices (top 3 models)
   - Feature importance charts
   - Model comparison tables

## Technology Stack

### Core Libraries

**Data Processing:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scikit-learn`: Preprocessing and utilities

**Machine Learning:**
- `scikit-learn`: ML models and evaluation
  - `ensemble`: RandomForest, GradientBoosting
  - `linear_model`: LogisticRegression
  - `svm`: SVC
  - `model_selection`: train_test_split, GridSearchCV
  - `metrics`: Classification metrics
  - `preprocessing`: StandardScaler, encoders
- `xgboost`: XGBoost implementation

**Visualization:**
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualizations

**Utilities:**
- `warnings`: Suppress non-critical warnings
- `sys`: System utilities

### Development Environment

- **Platform**: Jupyter Notebook
- **Language**: Python 3.x
- **IDE**: Jupyter Lab / Notebook interface

## Model Outputs

### Primary Outputs

1. **Default Prediction**
   - Binary classification: 0 (No Default) or 1 (Default)
   - Per loan application prediction

2. **Probability Score**
   - Continuous probability (0.0 to 1.0)
   - Risk ranking and threshold adjustment
   - Enables tiered decision-making

3. **Feature Importance**
   - Ranked list of predictive features
   - Identifies key risk factors
   - Supports model interpretation

### How Outputs Address Business Problems

1. **Automated Risk Assessment**
   - Reduces manual review time
   - Consistent evaluation criteria
   - Scalable across large loan volumes

2. **Risk-Based Pricing**
   - Adjust interest rates based on risk score
   - Differentiate between risk tiers
   - Optimize portfolio risk-return profile

3. **Early Warning System**
   - Monitor existing loan portfolio
   - Identify deteriorating loans
   - Enable proactive intervention

4. **Portfolio Optimization**
   - Balance risk across loan portfolio
   - Identify concentration risks
   - Inform lending strategy

## Ethical Considerations

### Data Privacy

**Measures:**
- Limit data collection to necessary features
- Anonymize and encrypt sensitive data
- Implement strict access controls
- Maintain audit logs of data usage
- Regular compliance reviews

### Model Fairness

**Considerations:**
- Avoid discrimination based on protected attributes
- Test for disparate impact across demographic groups
- Regular bias audits
- Transparent decision criteria

### Model Explainability

**Approaches:**
- Feature importance visualization
- SHAP (SHapley Additive exPlanations) values
- Local interpretability for individual predictions
- Clear documentation of model logic

### Accuracy and Reliability

**Standards:**
- Regular model retraining with fresh data
- Performance monitoring and drift detection
- Conservative thresholds for high-stakes decisions
- Human oversight for borderline cases

## Deployment Architecture

### Recommended Deployment Strategy

```
┌─────────────────────┐
│  Loan Application   │
│      System         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   API Gateway       │
│  (REST/GraphQL)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Prediction Service │
│  - Load Model       │
│  - Preprocess Data  │
│  - Generate Score   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Decision Engine   │
│  - Apply Rules      │
│  - Risk Tiers       │
│  - Recommendations  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Response & Logging │
│  - Return Results   │
│  - Log Predictions  │
│  - Monitor          │
└─────────────────────┘
```

### Infrastructure Recommendations

**Cloud Platforms:**
- AWS: SageMaker, Lambda, API Gateway
- Google Cloud: AI Platform, Cloud Functions
- Azure: Machine Learning, Functions

**Containerization:**
- Docker for model packaging
- Kubernetes for orchestration

**Model Serving:**
- TensorFlow Serving / TorchServe
- FastAPI / Flask for REST API
- MLflow for model registry

**Monitoring:**
- Model performance tracking
- Prediction drift detection
- System health monitoring
- Alert systems for anomalies

## Testing Strategy

### Phases

1. **Unit Testing**
   - Test individual components
   - Data preprocessing functions
   - Feature engineering logic
   - Model inference

2. **Integration Testing**
   - End-to-end pipeline validation
   - API endpoint testing
   - Data flow verification

3. **Performance Testing**
   - Latency benchmarks
   - Throughput testing
   - Load testing

4. **Validation Testing**
   - Cross-validation results
   - Hold-out test set evaluation
   - Out-of-time validation

### Model Adjustment Strategy

**Based on Confusion Matrix:**

1. **High False Negatives (FN)**
   - Adjust threshold to be more conservative
   - Increase recall (catch more defaults)
   - Trade-off: More false positives

2. **High False Positives (FP)**
   - Adjust threshold to be more lenient
   - Increase precision (reduce false alarms)
   - Trade-off: More false negatives

3. **Overall Performance**
   - Retrain with additional data
   - Feature engineering improvements
   - Hyperparameter optimization
   - Ensemble methods

## Project Structure

```
IAT-loan-sb-predictor/
├── IATD_AI_Applications.ipynb          # Main notebook with complete pipeline
├── architecture.md                      # This file
├── Finance AI Case Study.md            # Documentation
├── data/
│   └── synthetic_au_smallbiz_loans.csv # Training data
├── test_business_loans_10rows.csv      # Test/application data
└── docs/
    ├── IATD_AI_Applications-Finance_case_study.docx
    ├── IATD_AI_Applications-Healthcare_case_study.docx
    └── IATD_Project_Assessment_AI_Applications.docx
```

## Future Enhancements

1. **Model Improvements**
   - Deep learning models (Neural Networks)
   - Ensemble stacking
   - AutoML for automated optimization

2. **Feature Engineering**
   - Time-series features (trends over time)
   - External data integration (economic indicators)
   - Network features (business relationships)

3. **Deployment**
   - Real-time prediction API
   - Batch prediction pipeline
   - Mobile application integration

4. **Monitoring**
   - Automated retraining pipelines
   - A/B testing framework
   - Explainability dashboard

5. **Business Integration**
   - CRM system integration
   - Automated decision workflows
   - Regulatory reporting

## References

- Australian Prudential Regulation Authority (APRA): Statistics on business lending
- Australian Bureau of Statistics (ABS): Business Characteristics Survey
- ASIC: Insolvency Statistics

## Conclusion

This architecture provides a comprehensive framework for predicting loan default risk in small businesses. The system combines robust data processing, multiple ML algorithms, thorough evaluation, and ethical considerations to deliver actionable insights for financial institutions. The modular design allows for continuous improvement and adaptation to changing business requirements.

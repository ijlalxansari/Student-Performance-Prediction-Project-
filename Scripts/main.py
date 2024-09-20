from preprocess_data import preprocess_data
from train_evaluate_model import train_and_evaluate_model_with_hyperparameter_tuning
from eda_analysis import perform_eda
from eng_feat import feature_engineering_pipeline

# File paths
raw_data_path = r'data/student-scores.csv'
preprocessed_data_path = r'data/preprocessed_data.csv'
model_results_path = r'data/model_results.csv'

# Step 1: Data Preprocessing
df_processed = preprocess_data(raw_data_path, preprocessed_data_path)

# Step 2: Feature Engineering
df_engineered = feature_engineering_pipeline(preprocessed_data_path)

# Define the target column (choose based on your needs)
target_column = 'combined_score'  # Or use a specific score like 'math_score'

# Step 3: Model Training and Hyperparameter Tuning
train_and_evaluate_model_with_hyperparameter_tuning(df_engineered, target_column, model_results_path)

# Step 4: Exploratory Data Analysis
perform_eda(model_results_path)

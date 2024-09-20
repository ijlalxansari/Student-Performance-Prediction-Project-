import pandas as pd

def feature_engineering_pipeline(data_path):
    # Load the preprocessed data
    df = pd.read_csv(data_path)
    
    # Example feature engineering: creating new features or modifying existing ones
    if 'combined_score' in df.columns:
        # Example of adding a new feature based on existing ones
        df['score_per_hour'] = df['combined_score'] / df['weekly_self_study_hours'].replace(0, 1)  # Avoid division by zero

    # Return the dataframe with engineered features
    return df

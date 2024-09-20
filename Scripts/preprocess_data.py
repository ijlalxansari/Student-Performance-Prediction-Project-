import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data_path, save_path):
    df = pd.read_csv(data_path)
    
    # Display the first few rows
    print(df.head())
    
    # Display the data types of each column
    print(df.dtypes)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values:\n", missing_values)
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    non_numeric_cols = df.select_dtypes(include=['object', 'bool']).columns
    
    # Fill missing values in numeric columns with the mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill missing values in non-numeric columns with mode
    for col in non_numeric_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Create a combined score for overall performance
    df['combined_score'] = df[['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].mean(axis=1)
    
    # Drop individual score columns if only the combined score is needed
    df = df.drop(['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score'], axis=1)
    
    # One-hot encoding categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    # Scaling features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)
    df = pd.DataFrame(scaled_features, columns=df.columns)
    
    # Save the preprocessed data to a CSV file
    df.to_csv(save_path, index=False)

    return df

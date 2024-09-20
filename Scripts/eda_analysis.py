import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(results_path):
    # Load the model results
    df_results = pd.read_csv(results_path)

    # Display basic statistics
    print(df_results.describe())
    
    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=df_results)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

    # Plot residuals
    residuals = df_results['Actual'] - df_results['Predicted']
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df_results.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    results_path = 'data/model_results.csv'
    perform_eda(results_path)

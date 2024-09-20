# hyperparameter_tuning.py

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def tune_hyperparameters(X_train, y_train):
    # Define the model
    model = Ridge()  # You can replace Ridge with another model if needed

    # Define the hyperparameters grid to search
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Return the best model
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    # Predict using the best model
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

if __name__ == "__main__":
    # Example paths
    X_train_path = r'data/X_train.csv'
    y_train_path = r'data/y_train.csv'
    X_test_path = r'data/X_test.csv'
    y_test_path = r'data/y_test.csv'

    # Load the datasets
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train, y_train)

    # Evaluate the best model
    evaluate_model(best_model, X_test, y_test)
    
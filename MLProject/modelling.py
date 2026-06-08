import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model_ci():
    print("=== Memulai Proses Pelatihan Model di GitHub Actions ===")
    experiment_name = "California_Housing_CI"
    mlflow.set_experiment(experiment_name)
    
    housing = fetch_california_housing()
    X = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Linear_Regression_CI") as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("-> Sukses: Model berhasil dilatih di server GitHub Actions.")

if __name__ == "__main__":
    train_model_ci()

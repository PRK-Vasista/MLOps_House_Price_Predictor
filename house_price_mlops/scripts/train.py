import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os
import numpy as np

# --- 1. Environment Setup & Logging ---

print("--- Starting MLOps Training Script ---")

#try:
    # Enable autologging for scikit-learn models
    #mlflow.sklearn.autolog()
    #print("MLflow Autologging enabled.")
#except Exception as e:
    #print(f"Error enabling MLflow autologging: {e}")
    # Terminate script if a critical setup fails
    #exit(1)
print("MLflow Autologging is DISABLED (using manual log_model).")

# --- 2. Data Loading and Validation ---

DATA_PATH = "../data/housing.csv"

try:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Define features and target
    FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    TARGET = "price"

    # Check if all necessary columns exist
    if not all(col in df.columns for col in FEATURES + [TARGET]):
        missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    X = df[FEATURES]
    y = df[TARGET]
    print("Features (X) and Target (y) defined.")

except (FileNotFoundError, ValueError, Exception) as e:
    print(f"Critical Error during Data Loading/Validation: {e}")
    exit(1)

# --- 3. Data Splitting ---

TEST_SIZE = 0.2
RANDOM_STATE = 42

try:
    print(f"Splitting data into train/test (Test Size: {TEST_SIZE}, Seed: {RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("Data split complete.")
    print(f"Train set size: {len(X_train)} samples.")
    print(f"Test set size: {len(X_test)} samples.")

except Exception as e:
    print(f"Error during Data Splitting: {e}")
    exit(1)

# --- 4. Model Training, Evaluation, and MLflow Logging ---

try:
    # Start a new MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print(f"\nStarting MLflow Run with ID: {run_id}")

        # Log parameters manually (Autolog handles model params, but structure params are useful)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("model_type", "LinearRegression")
        print("Logged run parameters (test_size, model_type).")

        # Initialize and Train Model
        print("Initializing Linear Regression model...")
        model = LinearRegression()

        print("Training model...")
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Prediction and Evaluation
        print("Making predictions on the test set...")
        preds = model.predict(X_test)

        # Calculate Mean Squared Error (MSE) first
        mse = mean_squared_error(y_test, preds) 
        # Calculate Root Mean Squared Error (RMSE) using numpy
        rmse = np.sqrt(mse)

        print("\nEvaluation Metric:")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Manual Logging of the primary metric
        mlflow.log_metric("rmse", rmse)
        print("Logged custom metric 'rmse' to MLflow.")

        # Log the model artifact (This is also handled by autolog, but explicit logging confirms it)
        mlflow.sklearn.log_model(model, "model")
        print("Logged model artifact 'model' to MLflow.")

        print("\n--- MLflow Run Finished ---")
        print(f"MLflow Run ID: {run_id}")
        print(f"View run details: http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}")

except Exception as e:
    print(f"\nCritical Error during Model Training/Logging: {e}")
    # Attempt to log failure status to MLflow if the run was started
    try:
        if 'run' in locals():
            mlflow.end_run(status="FAILED")
            print("MLflow run marked as FAILED.")
    except Exception as log_e:
        print(f"Could not mark MLflow run as FAILED: {log_e}")
    exit(1)


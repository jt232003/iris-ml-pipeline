import pytest
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.train import train_model
from src.evaluate import evaluate_model

# Fixture to ensure a model is trained and available for tests
@pytest.fixture(scope="module", autouse=True)
def setup_model_for_tests():
    """Ensures a model is trained and saved before any tests run."""
    model_path = "models/iris_model_for_test.joblib"
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("\nRunning train_model to set up test model...")
    train_model(model_output_path=model_path)
    yield model_path
    # Optional: Clean up the model file after tests (uncomment if desired)
    # if os.path.exists(model_path):
    #     os.remove(model_path)

def test_model_accuracy_threshold(setup_model_for_tests):
    """Test if the model's accuracy meets a minimum threshold."""
    model_path = setup_model_for_tests

    # Load the data and split it
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the trained model
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Define an acceptable accuracy threshold for the IRIS dataset (usually high)
    ACCURACY_THRESHOLD = 0.90 # Adjust as per your model's expected performance
    print(f"\nModel Accuracy in Test: {accuracy:.4f}")
    assert accuracy >= ACCURACY_THRESHOLD, f"Model accuracy ({accuracy:.4f}) is below threshold ({ACCURACY_THRESHOLD})."

def test_model_prediction_integrity(setup_model_for_tests):
    """Test if the model predictions are of expected type and range."""
    model_path = setup_model_for_tests

    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test), "Prediction count mismatch."
    assert all(np.issubdtype(p, np.integer) for p in predictions), "Predictions are not integers or NumPy integers."
    assert all(p in [0, 1, 2] for p in predictions), "Predictions contain unexpected target values."

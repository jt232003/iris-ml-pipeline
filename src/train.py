import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(model_output_path="models/iris_model.joblib"):
    """Loads IRIS data, trains a RandomForestClassifier, and saves the model."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # Optionally, save X_test, y_test for evaluation if needed by evaluate.py
    # For simplicity in this demo, evaluate.py will load data itself.
    return model, accuracy_score(y_test, model.predict(X_test))

if __name__ == "__main__":
    # Ensure data/ directory exists if train_model depends on it.
    # In this version, load_iris directly gets data, no external CSV needed for training.
    # However, if data pre-processing saved a CSV, you'd load that.
    # For this simple demo, we'll just run the training.
    # Let's ensure 'models' directory is created.
    os.makedirs('models', exist_ok=True)
    train_model()

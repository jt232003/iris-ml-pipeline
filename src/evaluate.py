import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

def evaluate_model(model_path="models/iris_model.joblib"):
    """Loads the trained model, evaluates it, and prints a report."""
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"--- Model Evaluation Report ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Format and print the classification report for readability
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"  {label}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.2f}" if isinstance(value, float) else f"    {metric}: {value}")
        else:
             print(f"  {label}: {metrics:.2f}" if isinstance(metrics, float) else f"  {label}: {metrics}")
    # This will be captured by CML later
    return accuracy, report

if __name__ == "__main__":
    # First, ensure a model exists to be evaluated
    # In a real pipeline, training would precede evaluation.
    # For this demo, let's create a dummy model if one doesn't exist
    # This is just for local testing the script. GitHub Actions will train it first.
    if not os.path.exists("models/iris_model.joblib"):
         print("Model not found, running training first (for local test).")
         from src.train import train_model
         train_model()
    evaluate_model()

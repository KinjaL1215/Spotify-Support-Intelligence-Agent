from pathlib import Path
import pandas as pd

from src.classifier.intent_classifier import (
    train_svm_classifier,
    save_classifier
)

PROJECT_ROOT = Path(__file__).resolve().parent

CLASSIFIER_FILE = (
    PROJECT_ROOT
    / "data"
    / "spotify_classifier_balanced.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "models"
)

print("Loading classifier dataset...")

classifier_data = pd.read_csv(CLASSIFIER_FILE)

print("Dataset size:", len(classifier_data))

print("\nTraining SVM classifier...")

(
    vectorizer,
    classifier,
    accuracy,
    X_test,
    y_test,
    predictions,
    X_test_tfidf
) = train_svm_classifier(classifier_data)

print(f"\nClassifier accuracy: {accuracy:.2%}")

print("\nSaving classifier...")

save_classifier(
    vectorizer,
    classifier,
    MODEL_DIR
)

print("\n===================================")
print("CLASSIFIER TRAINING COMPLETED")
print("===================================")

print("Model saved in:", MODEL_DIR)
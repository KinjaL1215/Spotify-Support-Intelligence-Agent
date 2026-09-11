import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report


def evaluate_classifier(
    classifier,
    X_test: pd.Series,
    y_test: pd.Series,
    X_test_tfidf,
    predictions: np.ndarray,
    prediction_file: Path,
    wrong_file: Path
) -> dict:
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    decision_scores = classifier.decision_function(X_test_tfidf)

    prediction_df = pd.DataFrame({
        "customer_text": X_test.values,
        "actual_intent": y_test.values,
        "predicted_intent": predictions
    })

    if decision_scores.ndim == 2:
        sorted_scores = np.sort(decision_scores, axis=1)
        prediction_df["confidence_margin"] = (
            sorted_scores[:, -1] - sorted_scores[:, -2]
        )

    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(prediction_file, index=False)

    wrong_predictions = prediction_df[
        prediction_df["actual_intent"] != prediction_df["predicted_intent"]
    ].copy()
    wrong_predictions.to_csv(wrong_file, index=False)

    return {
        "accuracy": accuracy,
        "report": report,
        "prediction_df": prediction_df,
        "wrong_predictions": wrong_predictions
    }

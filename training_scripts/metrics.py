from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Hugging Face Trainer."""
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def detailed_evaluation(predictions, true_labels, label_names=("invalid", "valid")):
    """Return and print a detailed per-class classification report."""
    report_text = classification_report(true_labels, predictions, target_names=list(label_names))
    print("=== Detailed Classification Report ===")
    print(report_text)
    return report_text



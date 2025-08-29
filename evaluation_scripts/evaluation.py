import json
import numpy as np
from datasets import load_from_disk
from training_scripts.trainer_setup import get_trainer
from sklearn.metrics import accuracy_score
from training_scripts.metrics import detailed_evaluation


def main():
    trainer = get_trainer()
    test_tokenized = load_from_disk("data/test_tokenized")

    predictions = trainer.predict(test_tokenized)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    report_text = detailed_evaluation(y_pred, y_true)

    results_summary = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
    }

    print("=== Final Test Results ===")
    for metric, value in results_summary.items():
        print(f"{metric}: {value:.4f}")

    with open("results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    with open("classification_report.txt", "w") as f:
        f.write(report_text)

    print("✅ Evaluation artifacts saved: results_summary.json, classification_report.txt")


if __name__ == "__main__":
    main()



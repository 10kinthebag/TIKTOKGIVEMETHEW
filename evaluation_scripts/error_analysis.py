def analyze_errors(texts, true_labels, predicted_labels, n_examples=10):
    """Print misclassified examples with brief categories."""
    errors_idx = [i for i in range(len(true_labels)) if true_labels[i] != predicted_labels[i]]
    print(f"=== Error Analysis ({len(errors_idx)} total errors) ===")

    false_positives = [i for i in errors_idx if true_labels[i] == 0 and predicted_labels[i] == 1]
    print(f"\n🔴 False Positives ({len(false_positives)}): Predicted Valid, Actually Invalid")
    for i, idx in enumerate(false_positives[:n_examples]):
        print(f"{i+1}. {texts[idx]}")

    false_negatives = [i for i in errors_idx if true_labels[i] == 1 and predicted_labels[i] == 0]
    print(f"\n🔴 False Negatives ({len(false_negatives)}): Predicted Invalid, Actually Valid")
    for i, idx in enumerate(false_negatives[:n_examples]):
        print(f"{i+1}. {texts[idx]}")



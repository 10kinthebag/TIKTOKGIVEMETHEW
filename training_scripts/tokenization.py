from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    """Tokenize the input texts for transformer models."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def main():
    df = pd.read_csv("data/cleanedData/reviews_with_labels.csv")

    from sklearn.model_selection import train_test_split

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["text"].tolist(),
        df["pseudo_label"].tolist(),
        test_size=0.4,
        random_state=42,
        stratify=df["pseudo_label"],
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "label"]
    train_tokenized.set_format(type="torch", columns=columns)
    val_tokenized.set_format(type="torch", columns=columns)
    test_tokenized.set_format(type="torch", columns=columns)

    # Save to disk for reuse
    train_tokenized.save_to_disk("data/train_tokenized")
    val_tokenized.save_to_disk("data/val_tokenized")
    test_tokenized.save_to_disk("data/test_tokenized")

    print("✅ Tokenization complete")


if __name__ == "__main__":
    main()

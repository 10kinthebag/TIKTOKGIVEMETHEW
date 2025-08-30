"""
Hybrid training approach: Combine ground truth data with pseudo-labeled data.
This maximizes training data while prioritizing high-quality human annotations.
"""
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np


MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    """Tokenize the input texts for transformer models."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def combine_datasets():
    """Combine ground truth and pseudo-labeled data for training."""
    
    print("🔄 Loading datasets...")
    
    # Load ground truth data (high quality, human-annotated)
    df_ground_truth = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
    df_ground_truth['data_source'] = 'ground_truth'
    df_ground_truth['label'] = df_ground_truth['true_label']
    
    # Load pseudo-labeled data (larger quantity, rule-based)
    df_pseudo = pd.read_csv("data/cleanedData/reviews_with_labels.csv")
    df_pseudo['data_source'] = 'pseudo_labeled'
    df_pseudo['label'] = df_pseudo['pseudo_label']
    
    print(f"📊 Ground truth data: {len(df_ground_truth)} samples")
    print(f"📊 Pseudo-labeled data: {len(df_pseudo)} samples")
    
    # Select only needed columns and ensure same structure
    ground_truth_subset = df_ground_truth[['text', 'label', 'data_source']].copy()
    pseudo_subset = df_pseudo[['text', 'label', 'data_source']].copy()
    
    # Combine datasets
    combined_df = pd.concat([ground_truth_subset, pseudo_subset], ignore_index=True)
    
    print(f"📊 Combined dataset: {len(combined_df)} samples")
    print(f"   - Ground truth: {len(ground_truth_subset)} ({len(ground_truth_subset)/len(combined_df)*100:.1f}%)")
    print(f"   - Pseudo-labeled: {len(pseudo_subset)} ({len(pseudo_subset)/len(combined_df)*100:.1f}%)")
    
    # Check label distribution
    label_dist = combined_df['label'].value_counts()
    print(f"📊 Label distribution:")
    print(f"   - Valid (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(combined_df)*100:.1f}%)")
    print(f"   - Invalid (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(combined_df)*100:.1f}%)")
    
    return combined_df


def main():
    # Combine datasets
    combined_df = combine_datasets()
    
    # Split combined data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        combined_df["text"].tolist(),
        combined_df["label"].tolist(),
        test_size=0.4,
        random_state=42,
        stratify=combined_df["label"],
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )
    
    print(f"📊 Final data split:")
    print(f"   - Training: {len(train_texts)} samples")
    print(f"   - Validation: {len(val_texts)} samples")
    print(f"   - Test: {len(test_texts)} samples")

    # Create datasets
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "label": test_labels})

    # Tokenize
    print("🔄 Tokenizing datasets...")
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "label"]
    train_tokenized.set_format(type="torch", columns=columns)
    val_tokenized.set_format(type="torch", columns=columns)
    test_tokenized.set_format(type="torch", columns=columns)

    # Save to disk
    print("💾 Saving tokenized datasets...")
    train_tokenized.save_to_disk("data/train_tokenized")
    val_tokenized.save_to_disk("data/val_tokenized")
    test_tokenized.save_to_disk("data/test_tokenized")

    print("✅ Hybrid tokenization complete!")
    print("🎯 Model will train on both ground truth and pseudo-labeled data")


if __name__ == "__main__":
    main()

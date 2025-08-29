"""
Weighted training approach: Give higher importance to ground truth data during training.
This uses sample weights to prioritize ground truth labels while still benefiting from pseudo data.
"""
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, Trainer
from sklearn.model_selection import train_test_split
import torch
import numpy as np


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


def prepare_weighted_data():
    """Prepare data with weights for ground truth vs pseudo-labeled samples."""
    
    print("🔄 Loading and weighting datasets...")
    
    # Load ground truth data
    df_ground_truth = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
    df_ground_truth['label'] = df_ground_truth['true_label']
    df_ground_truth['weight'] = 2.0  # Higher weight for ground truth
    df_ground_truth['data_source'] = 'ground_truth'
    
    # Load pseudo-labeled data  
    df_pseudo = pd.read_csv("data/cleanedData/reviews_with_labels.csv")
    df_pseudo['label'] = df_pseudo['pseudo_label']
    df_pseudo['weight'] = 1.0  # Lower weight for pseudo labels
    df_pseudo['data_source'] = 'pseudo_labeled'
    
    # Select columns
    ground_truth_subset = df_ground_truth[['text', 'label', 'weight', 'data_source']].copy()
    pseudo_subset = df_pseudo[['text', 'label', 'weight', 'data_source']].copy()
    
    # Combine
    combined_df = pd.concat([ground_truth_subset, pseudo_subset], ignore_index=True)
    
    print(f"📊 Weighted dataset prepared:")
    print(f"   - Ground truth: {len(ground_truth_subset)} samples (weight: 2.0)")
    print(f"   - Pseudo-labeled: {len(pseudo_subset)} samples (weight: 1.0)")
    
    return combined_df


class WeightedTrainer(Trainer):
    """Custom trainer that supports sample weights."""
    
    def __init__(self, sample_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Compute weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        if self.sample_weights is not None:
            # Apply sample weights
            weights = torch.tensor(self.sample_weights, device=loss.device, dtype=loss.dtype)
            if len(weights) == len(loss):
                loss = loss * weights
        
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


def main():
    # Prepare weighted data
    combined_df = prepare_weighted_data()
    
    # Split data (stratified)
    train_texts, temp_texts, train_labels, temp_labels, train_weights, temp_weights = train_test_split(
        combined_df["text"].tolist(),
        combined_df["label"].tolist(),
        combined_df["weight"].tolist(),
        test_size=0.4,
        random_state=42,
        stratify=combined_df["label"],
    )
    
    val_texts, test_texts, val_labels, test_labels, val_weights, test_weights = train_test_split(
        temp_texts, temp_labels, temp_weights,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )
    
    print(f"📊 Weighted split complete:")
    print(f"   - Training: {len(train_texts)} samples")
    print(f"   - Validation: {len(val_texts)} samples") 
    print(f"   - Test: {len(test_texts)} samples")

    # Create datasets with weights
    train_ds = Dataset.from_dict({
        "text": train_texts, 
        "label": train_labels,
        "weight": train_weights
    })
    val_ds = Dataset.from_dict({
        "text": val_texts, 
        "label": val_labels,
        "weight": val_weights
    })
    test_ds = Dataset.from_dict({
        "text": test_texts, 
        "label": test_labels,
        "weight": test_weights
    })

    # Tokenize
    print("🔄 Tokenizing weighted datasets...")
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "label", "weight"]
    train_tokenized.set_format(type="torch", columns=columns)
    val_tokenized.set_format(type="torch", columns=columns)
    test_tokenized.set_format(type="torch", columns=columns)

    # Save to disk with weights
    print("💾 Saving weighted tokenized datasets...")
    train_tokenized.save_to_disk("data/train_tokenized_weighted")
    val_tokenized.save_to_disk("data/val_tokenized_weighted")
    test_tokenized.save_to_disk("data/test_tokenized_weighted")

    print("✅ Weighted tokenization complete!")
    print("🎯 Use WeightedTrainer class for training with sample weights")


if __name__ == "__main__":
    main()

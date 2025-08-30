from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    """Tokenize the input texts for transformer models."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,  # DeBERTa-v3 can handle longer sequences better
    )


def load_data(data_source="hybrid"):
    """
    Load data based on specified source:
    - 'pseudo': Only pseudo-labeled data
    - 'ground_truth': Only ground truth data  
    - 'hybrid': Both datasets combined
    - 'policy': Use policy module filtered data (recommended)
    """
    if data_source == "pseudo":
        print("📊 Loading pseudo-labeled data only...")
        df = pd.read_csv("data/cleanedData/reviews_with_labels.csv")
        df['label'] = df['pseudo_label']
        
    elif data_source == "ground_truth":
        print("📊 Loading ground truth data only...")
        df = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
        df['label'] = df['true_label']
        
    elif data_source == "policy":
        print("🎯 Loading policy-based filtered data (recommended)...")
        # Import and use the policy-based loading function
        from training_scripts.policy_based_training import load_policy_based_data
        df = load_policy_based_data()
        
    elif data_source == "hybrid":
        print("📊 Loading hybrid dataset (ground truth + pseudo labels)...")
        
        # Load ground truth data
        df_gt = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
        df_gt['label'] = df_gt['true_label']
        df_gt['source'] = 'ground_truth'
        gt_subset = df_gt[['text', 'label', 'source']].copy()
        
        # Load pseudo-labeled data
        df_pseudo = pd.read_csv("data/cleanedData/reviews_with_labels.csv")
        df_pseudo['label'] = df_pseudo['pseudo_label'] 
        df_pseudo['source'] = 'pseudo_labeled'
        pseudo_subset = df_pseudo[['text', 'label', 'source']].copy()
        
        # Combine datasets
        df = pd.concat([gt_subset, pseudo_subset], ignore_index=True)
        
        print(f"   - Ground truth: {len(gt_subset)} samples")
        print(f"   - Pseudo-labeled: {len(pseudo_subset)} samples")
        print(f"   - Total: {len(df)} samples")
        
    else:
        raise ValueError("data_source must be 'pseudo', 'ground_truth', 'policy', or 'hybrid'")
    
    return df


def main(data_source="hybrid"):
    df = load_data(data_source)

    from sklearn.model_selection import train_test_split

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),  # Use unified 'label' column
        test_size=0.4,
        random_state=42,
        stratify=df["label"],  # Use unified 'label' column
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
    print(f"🎯 Training data source: {data_source}")


if __name__ == "__main__":
    import sys
    
    # Allow command line argument to specify data source
    data_source = "hybrid"  # default
    if len(sys.argv) > 1:
        data_source = sys.argv[1]
    
    main(data_source)

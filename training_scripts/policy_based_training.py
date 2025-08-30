"""
Policy-Based Training: Use the sophisticated policy module filtering for high-quality training data.

This approach uses:
1. Clean data from data/filteredData (label=1, valid reviews)
2. Flagged data from data/filteredDataWithFlags (label=0, policy violations)
3. Ground truth data for validation

The policy module provides comprehensive filtering based on:
- Advertisement detection
- Irrelevant content (rule-based and semantic)
- Rant detection (reviews without actual visits)
- Spam detection (gibberish, patterns)
- Short review detection
- Contradiction detection (sentiment vs rating mismatch)
- Image relevance detection
"""

import os
import sys
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import glob

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


def get_latest_filtered_files():
    """Get the most recent filtered data files."""
    # Get all timestamp files and find the latest
    clean_files = glob.glob("data/filteredData/cleaned_reviews_*.csv")
    flagged_files = glob.glob("data/filteredDataWithFlags/cleaned_reviews_*.csv")
    
    if not clean_files or not flagged_files:
        raise FileNotFoundError("No filtered data files found. Please run src/policy_module.py first.")
    
    # Extract timestamps and find the latest
    clean_timestamps = [int(f.split('_')[-1].split('.')[0]) for f in clean_files]
    flagged_timestamps = [int(f.split('_')[-1].split('.')[0]) for f in flagged_files]
    
    latest_timestamp = max(max(clean_timestamps), max(flagged_timestamps))
    
    latest_clean = f"data/filteredData/cleaned_reviews_{latest_timestamp}.csv"
    latest_flagged = f"data/filteredDataWithFlags/cleaned_reviews_{latest_timestamp}.csv"
    
    return latest_clean, latest_flagged


def load_policy_based_data():
    """
    Load data based on policy module filtering:
    - Clean data (filteredData): label=1 (valid reviews)
    - Flagged data (filteredDataWithFlags): label=0 (policy violations)
    """
    print("🎯 Loading Policy-Based Filtered Data...")
    
    try:
        latest_clean, latest_flagged = get_latest_filtered_files()
        print(f"📂 Using latest clean data: {latest_clean}")
        print(f"📂 Using latest flagged data: {latest_flagged}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Run: python src/policy_module.py data/cleanedData/reviews_cleaned.csv")
        raise
    
    # Load clean data (valid reviews)
    df_clean = pd.read_csv(latest_clean)
    df_clean['label'] = 1  # Valid reviews
    df_clean['data_source'] = 'policy_clean'
    
    # Load flagged data (policy violations)
    df_flagged = pd.read_csv(latest_flagged)
    df_flagged['label'] = 0  # Invalid reviews (policy violations)
    df_flagged['data_source'] = 'policy_flagged'
    
    print(f"📊 Policy-Based Data Statistics:")
    print(f"   Valid reviews (clean): {len(df_clean)}")
    print(f"   Invalid reviews (flagged): {len(df_flagged)}")
    
    # Combine the datasets
    df_combined = pd.concat([df_clean, df_flagged], ignore_index=True)
    
    # Show distribution of violation types from flagged data
    if 'ad_flag' in df_flagged.columns:
        print(f"\n🚨 Policy Violation Breakdown:")
        violation_cols = ['ad_flag', 'irrelevant_flag_rule', 'rant_flag', 'spam_flag', 
                         'short_review_flag', 'contradiction_flag']
        for col in violation_cols:
            if col in df_flagged.columns:
                count = df_flagged[col].sum()
                pct = count / len(df_flagged) * 100
                violation_name = col.replace('_flag', '').replace('_', ' ').title()
                print(f"   {violation_name}: {count} ({pct:.1f}%)")
    
    print(f"\n📊 Combined Dataset:")
    print(f"   Total samples: {len(df_combined)}")
    print(f"   Valid (1): {(df_combined['label'] == 1).sum()}")
    print(f"   Invalid (0): {(df_combined['label'] == 0).sum()}")
    print(f"   Balance: {(df_combined['label'] == 1).sum() / len(df_combined) * 100:.1f}% valid")
    
    return df_combined


def load_ground_truth_data():
    """Load ground truth data for validation."""
    ground_truth_path = "data/groundTruthData/reviews_ground_truth.csv"
    
    if not os.path.exists(ground_truth_path):
        print("⚠️ No ground truth data found. Using policy data for validation too.")
        return None
    
    df_gt = pd.read_csv(ground_truth_path)
    df_gt['label'] = df_gt['true_label']  # Use corrected labels
    df_gt['data_source'] = 'ground_truth'
    
    print(f"📊 Ground Truth Data:")
    print(f"   Total samples: {len(df_gt)}")
    print(f"   Valid (1): {(df_gt['label'] == 1).sum()}")
    print(f"   Invalid (0): {(df_gt['label'] == 0).sum()}")
    
    return df_gt


def create_policy_based_training_data(validation_strategy="policy_split"):
    """
    Create training datasets using policy-based approach.
    
    validation_strategy options:
    - "policy_split": Split policy data into train/val
    - "ground_truth": Use ground truth for validation, policy for training
    - "mixed": Mix ground truth into training, use policy split for validation
    """
    print(f"🚀 Creating Policy-Based Training Data (strategy: {validation_strategy})")
    
    # Load policy-filtered data
    df_policy = load_policy_based_data()
    
    # Load ground truth data
    df_ground_truth = load_ground_truth_data()
    
    if validation_strategy == "ground_truth" and df_ground_truth is not None:
        # Use ground truth for validation, policy data for training
        train_df = df_policy
        val_df = df_ground_truth
        print("🎯 Strategy: Policy training + Ground truth validation")
        
    elif validation_strategy == "mixed" and df_ground_truth is not None:
        # Mix ground truth into training, split policy data for validation
        train_policy, val_policy = train_test_split(
            df_policy, test_size=0.2, random_state=42, stratify=df_policy['label']
        )
        train_df = pd.concat([train_policy, df_ground_truth], ignore_index=True)
        val_df = val_policy
        print("🎯 Strategy: Mixed training (policy + ground truth) + Policy validation")
        
    else:
        # Split policy data into train/val
        train_df, val_df = train_test_split(
            df_policy, test_size=0.2, random_state=42, stratify=df_policy['label']
        )
        print("🎯 Strategy: Policy-only training and validation")
    
    print(f"\n📊 Final Training Split:")
    print(f"   Training: {len(train_df)} samples")
    print(f"     Valid (1): {(train_df['label'] == 1).sum()}")
    print(f"     Invalid (0): {(train_df['label'] == 0).sum()}")
    print(f"   Validation: {len(val_df)} samples")
    print(f"     Valid (1): {(val_df['label'] == 1).sum()}")
    print(f"     Invalid (0): {(val_df['label'] == 0).sum()}")
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    
    # Tokenize
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Save tokenized datasets
    train_tokenized.save_to_disk("data/policy_train_tokenized")
    val_tokenized.save_to_disk("data/policy_val_tokenized")
    
    print(f"\n✅ Policy-based datasets saved:")
    print(f"   Training: data/policy_train_tokenized")
    print(f"   Validation: data/policy_val_tokenized")
    
    return train_tokenized, val_tokenized


def main():
    """Main function to create policy-based training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create policy-based training data")
    parser.add_argument(
        "--strategy", 
        choices=["policy_split", "ground_truth", "mixed"],
        default="mixed",
        help="Validation strategy to use"
    )
    
    args = parser.parse_args()
    
    print("🎯 Policy-Based Training Data Creation")
    print("=" * 60)
    print("Using sophisticated policy module filtering!")
    print(f"Strategy: {args.strategy}")
    print()
    
    try:
        train_dataset, val_dataset = create_policy_based_training_data(args.strategy)
        
        print(f"\n🎯 Next Steps:")
        print("1. Train the model:")
        print("   python training_scripts/training.py --train_data data/policy_train_tokenized --val_data data/policy_val_tokenized")
        print("2. Or use the policy-based trainer:")
        print("   python training_scripts/policy_based_training.py --train")
        print("3. This uses high-quality labels from your team's policy decisions!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure to run policy filtering first:")
        print("   python src/policy_module.py data/cleanedData/reviews_cleaned.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # Quick training mode
        print("🚀 Quick Policy-Based Training Mode")
        train_dataset, val_dataset = create_policy_based_training_data("mixed")
        
        # Import and run training
        from training_scripts.model_setup import get_model
        from training_scripts.trainer_setup import get_trainer
        
        print("\n🔧 Setting up model and trainer...")
        model = get_model()
        trainer = get_trainer()
        trainer.train_dataset = train_dataset
        trainer.eval_dataset = val_dataset
        trainer.model = model
        
        print("🚀 Starting training with policy-based data...")
        trainer.train()
        
        print("✅ Policy-based training completed!")
    else:
        main()

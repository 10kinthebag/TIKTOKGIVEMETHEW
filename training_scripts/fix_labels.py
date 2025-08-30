"""
Fix label mapping - your friend labeled 1=suspicious, but model expects 1=valid
This script will flip the labels and retrain properly.
"""
import pandas as pd
from training_scripts.tokenization import main as tokenize_main
import os


def fix_ground_truth_labels():
    """Flip the labels in ground truth data to match model expectations."""
    
    print("🔄 Fixing label mapping in ground truth data...")
    
    # Load original data
    df = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
    
    print("Original label distribution:")
    print(f"  Label 0: {(df['true_label'] == 0).sum()} samples")
    print(f"  Label 1: {(df['true_label'] == 1).sum()} samples")
    
    # Your friend's labeling: 0=valid, 1=suspicious
    # Model expects: 0=invalid, 1=valid
    # So we need to flip: new_label = 1 - old_label
    
    df['corrected_label'] = 1 - df['true_label']
    
    print("\nAfter correction:")
    print(f"  Valid reviews (1): {(df['corrected_label'] == 1).sum()} samples")
    print(f"  Invalid reviews (0): {(df['corrected_label'] == 0).sum()} samples")
    
    # Save corrected version
    df_corrected = df[['text', 'corrected_label']].copy()
    df_corrected.columns = ['text', 'true_label']  # Keep same column name
    
    # Backup original and save corrected
    df.to_csv("data/groundTruthData/reviews_ground_truth_original.csv", index=False)
    df_corrected.to_csv("data/groundTruthData/reviews_ground_truth.csv", index=False)
    
    print("✅ Labels corrected and saved!")
    print("✅ Original backed up to reviews_ground_truth_original.csv")
    
    return df_corrected


def verify_label_correction():
    """Verify that the correction makes sense by checking some examples."""
    
    print("\n🔍 Verifying label correction with examples:")
    print("-" * 60)
    
    df = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
    
    # Show some examples of each class
    valid_examples = df[df['true_label'] == 1].head(3)
    invalid_examples = df[df['true_label'] == 0].head(3)
    
    print("VALID reviews (should be legitimate):")
    for i, row in valid_examples.iterrows():
        print(f"  • {row['text'][:80]}...")
    
    print("\nINVALID reviews (should be suspicious):")
    for i, row in invalid_examples.iterrows():
        print(f"  • {row['text'][:80]}...")
    
    print(f"\nTotal distribution:")
    print(f"  Valid: {(df['true_label'] == 1).sum()} ({(df['true_label'] == 1).mean():.1%})")
    print(f"  Invalid: {(df['true_label'] == 0).sum()} ({(df['true_label'] == 0).mean():.1%})")


def main():
    print("🛠️ Fixing Label Mapping Issue")
    print("=" * 50)
    print("Your friend labeled: 0=valid, 1=suspicious")
    print("Model expects: 0=invalid, 1=valid")
    print("Solution: Flip all labels!")
    print()
    
    # Fix the labels
    df_corrected = fix_ground_truth_labels()
    
    # Verify the fix
    verify_label_correction()
    
    print(f"\n🎯 Next steps:")
    print("1. Run: make tokenize_hybrid  # Re-tokenize with corrected labels")
    print("2. Run: make train           # Re-train with correct labels")
    print("3. Run: make test_model      # Test the corrected model")


if __name__ == "__main__":
    main()

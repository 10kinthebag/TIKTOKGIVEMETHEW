"""
Simplified Training Script for Restaurant Review Classification
Pure PyTorch implementation - combines policy-filtered data with ground truth labels
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import time
import json

# Disable TensorFlow warnings by setting environment variable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RestaurantReviewDataset(Dataset):
    """Dataset class for restaurant review classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data():
    """Load and combine all data sources with proper labeling"""
    
    print("📂 Loading data from multiple sources...")
    
    # 1. Load valid reviews (policy-filtered, no flags)
    valid_file = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/filteredData/cleaned_reviews_1756548901.csv"
    valid_df = pd.read_csv(valid_file)
    valid_df['label'] = 1  # These are valid reviews
    valid_df['source'] = 'policy_valid'
    print(f"✅ Loaded {len(valid_df)} valid reviews from policy filtering")
    
    # 2. Load invalid reviews (policy-flagged)
    invalid_file = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/filteredDataWithFlags/cleaned_reviews_1756548901.csv"
    invalid_df = pd.read_csv(invalid_file)
    invalid_df['label'] = 0  # These are invalid reviews
    invalid_df['source'] = 'policy_invalid'
    print(f"✅ Loaded {len(invalid_df)} invalid reviews from policy filtering")
    
    # 3. Load ground truth data (manually labeled)
    ground_truth_file = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/groundTruthData/reviews_ground_truth.csv"
    gt_df = pd.read_csv(ground_truth_file)
    gt_df['label'] = gt_df['true_label']  # Use the manually assigned labels
    gt_df['source'] = 'ground_truth'
    gt_df = gt_df.rename(columns={'true_label': 'original_label'})
    print(f"✅ Loaded {len(gt_df)} ground truth reviews with manual labels")
    
    # Combine all datasets
    # For policy data, use 'text' column; for ground truth, use 'text' column
    policy_data = pd.concat([valid_df, invalid_df])[['text', 'label', 'source']]
    ground_truth_data = gt_df[['text', 'label', 'source']]
    
    # Combine all data
    combined_df = pd.concat([policy_data, ground_truth_data], ignore_index=True)
    
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Valid (label=1): {len(combined_df[combined_df['label'] == 1])}")
    print(f"   Invalid (label=0): {len(combined_df[combined_df['label'] == 0])}")
    
    print(f"\n🏷️ Data Source Breakdown:")
    for source in combined_df['source'].unique():
        source_data = combined_df[combined_df['source'] == source]
        valid_count = len(source_data[source_data['label'] == 1])
        invalid_count = len(source_data[source_data['label'] == 0])
        print(f"   {source}: {len(source_data)} total ({valid_count} valid, {invalid_count} invalid)")
    
    return combined_df


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    """Main training function"""
    
    print("🚀 Starting Restaurant Review Classification Training")
    print("Using combined policy-filtered + ground truth data")
    print("=" * 60)
    
    # Configure paths
    output_dir = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/results/combined_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("\n📂 Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Split data while maintaining label distribution
    print("\n🔀 Splitting data...")
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.125,  # 0.1 / (1-0.2) = 0.125
        random_state=42, 
        stratify=train_val_df['label']
    )
    
    print(f"\n📊 Data Splits:")
    print(f"   Train: {len(train_df)} samples ({len(train_df[train_df['label']==1])} valid, {len(train_df[train_df['label']==0])} invalid)")
    print(f"   Val:   {len(val_df)} samples ({len(val_df[val_df['label']==1])} valid, {len(val_df[val_df['label']==0])} invalid)")
    print(f"   Test:  {len(test_df)} samples ({len(test_df[test_df['label']==1])} valid, {len(test_df[test_df['label']==0])} invalid)")
    
    # Initialize tokenizer and model
    print(f"\n🏗️ Loading RoBERTa model and tokenizer...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "invalid", 1: "valid"},
        label2id={"invalid": 0, "valid": 1}
    )
    
    # Create datasets
    print(f"\n📚 Creating datasets...")
    train_dataset = RestaurantReviewDataset(
        train_df['text'].tolist(), 
        train_df['label'].tolist(), 
        tokenizer
    )
    val_dataset = RestaurantReviewDataset(
        val_df['text'].tolist(), 
        val_df['label'].tolist(), 
        tokenizer
    )
    test_dataset = RestaurantReviewDataset(
        test_df['text'].tolist(), 
        test_df['label'].tolist(), 
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard
        seed=42,
        dataloader_pin_memory=False,  # Reduce memory usage
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Start training
    print(f"\n🎯 Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    print(f"\n✅ Training completed in {training_time:.2f} minutes!")
    
    # Save model
    model_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"💾 Model saved to {model_save_path}")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Detailed predictions for analysis
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_df['label'].tolist()
    
    # Classification report
    class_report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=['invalid', 'valid'],
        output_dict=True
    )
    
    # Analyze performance by data source
    test_df_copy = test_df.copy()
    test_df_copy['predicted'] = pred_labels
    
    source_performance = {}
    for source in test_df_copy['source'].unique():
        source_data = test_df_copy[test_df_copy['source'] == source]
        if len(source_data) > 0:
            source_acc = accuracy_score(source_data['label'], source_data['predicted'])
            source_performance[source] = {
                'samples': len(source_data),
                'accuracy': source_acc
            }
    
    # Save comprehensive results
    results = {
        'training_config': {
            'model_name': model_name,
            'num_epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
        },
        'data_statistics': {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'valid_label_ratio': len(df[df['label'] == 1]) / len(df)
        },
        'training_time_minutes': training_time,
        'test_results': test_results,
        'classification_report': class_report,
        'source_performance': source_performance
    }
    
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final results
    print(f"\n🎉 FINAL RESULTS:")
    print(f"   Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"   Test F1 Score: {test_results['eval_f1']:.4f}")
    print(f"   Test Precision: {test_results['eval_precision']:.4f}")
    print(f"   Test Recall: {test_results['eval_recall']:.4f}")
    
    print(f"\n📈 Performance by Data Source:")
    for source, perf in source_performance.items():
        print(f"   {source}: {perf['accuracy']:.4f} accuracy ({perf['samples']} samples)")
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    print(f"\n🚀 Your restaurant review classifier is ready!")
    
    return results


if __name__ == "__main__":
    # Check available device
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"🖥️ Using device: {device}")
    
    try:
        results = main()
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

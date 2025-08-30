"""
Simple Policy-Based Model Training
Uses the sophisticated pol    # Set up training arguments optimized for RoBERTa
    training_args = TrainingArguments(
        output_dir="./roberta_policy_model_results",
        num_train_epochs=3,
        per_device_train_batch_size=16,  # RoBERTa can handle good batch sizes
        per_device_eval_batch_size=16,
        learning_rate=2e-5,  # Standard learning rate for RoBERTa
        weight_decay=0.01,
        logging_dir="./roberta_policy_logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        warmup_steps=100,  # Warmup for better training
        report_to=None,  # Disable wandb/tensorboard for simplicity
    ) for high-quality labels.
"""

import os
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from training_scripts.metrics import compute_metrics


def train_policy_based_model():
    """Train a model using policy-based filtered data."""
    
    print("🎯 Policy-Based Model Training")
    print("=" * 50)
    
    # Check if policy data exists
    train_path = "data/policy_train_tokenized"
    val_path = "data/policy_val_tokenized"
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ Policy-based datasets not found.")
        print("🔧 Creating policy-based data first...")
        
        # Create the datasets
        from training_scripts.policy_based_training import create_policy_based_training_data
        create_policy_based_training_data("mixed")
    
    print("📂 Loading policy-based datasets...")
    train_data = load_from_disk(train_path)
    val_data = load_from_disk(val_path)
    
    # Ensure we have Dataset objects, not DatasetDict
    if isinstance(train_data, DatasetDict):
        train_dataset = train_data['train'] if 'train' in train_data else list(train_data.values())[0]
    else:
        train_dataset = train_data
        
    if isinstance(val_data, DatasetDict):
        val_dataset = val_data['validation'] if 'validation' in val_data else list(val_data.values())[0]
    else:
        val_dataset = val_data
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize model and tokenizer
    model_name = "roberta-base"
    print(f"🤖 Loading model: {model_name}")
    print("💡 RoBERTa-base: Much more powerful than DistilBERT, reliable and stable!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "invalid", 1: "valid"},
        label2id={"invalid": 0, "valid": 1}
    )
    
    # Set up training arguments optimized for DeBERTa-v3
    training_args = TrainingArguments(
        output_dir="./deberta_policy_model_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced for DeBERTa-v3 (larger model)
        per_device_eval_batch_size=8,
        learning_rate=1e-5,  # Lower learning rate for DeBERTa-v3
        weight_decay=0.01,
        logging_dir="./deberta_policy_logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        warmup_steps=500,  # Warmup helps DeBERTa-v3
        report_to=None,  # Disable wandb/tensorboard for simplicity
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("🚀 Starting training with policy-based data...")
    print("💡 This uses sophisticated filtering from your team's policy decisions!")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    final_model_path = "./models/roberta_policy_based_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("✅ Training completed!")
    print(f"📁 RoBERTa model saved to: {final_model_path}")
    
    # Evaluate final performance
    print("\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if key.startswith('eval_'):
            print(f"   {key}: {value:.4f}")
    
    return trainer, final_model_path


if __name__ == "__main__":
    train_policy_based_model()

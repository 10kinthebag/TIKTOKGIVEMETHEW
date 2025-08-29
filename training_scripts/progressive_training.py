"""
Progressive training strategy: 
1. First train on high-quality ground truth data
2. Then fine-tune on pseudo-labeled data with lower learning rate
"""
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_from_disk
import pandas as pd
from training_scripts.metrics import compute_metrics


MODEL_NAME = "distilbert-base-uncased"


def tokenize_function(examples):
    """Tokenize the input texts for transformer models."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


def prepare_ground_truth_only():
    """Prepare only ground truth data for initial training."""
    df = pd.read_csv("data/groundTruthData/reviews_ground_truth.csv")
    
    from sklearn.model_selection import train_test_split
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["true_label"].tolist(),
        test_size=0.2,  # Keep more for training initially
        random_state=42,
        stratify=df["true_label"],
    )
    
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    
    columns = ["input_ids", "attention_mask", "label"]
    train_tokenized.set_format(type="torch", columns=columns)
    val_tokenized.set_format(type="torch", columns=columns)
    
    return train_tokenized, val_tokenized


def prepare_pseudo_data():
    """Prepare pseudo-labeled data for fine-tuning."""
    df = pd.read_csv("data/cleanedData/reviews_with_labels.csv")
    
    from sklearn.model_selection import train_test_split
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["pseudo_label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["pseudo_label"],
    )
    
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized = val_ds.map(tokenize_function, batched=True)
    
    columns = ["input_ids", "attention_mask", "label"]
    train_tokenized.set_format(type="torch", columns=columns)
    val_tokenized.set_format(type="torch", columns=columns)
    
    return train_tokenized, val_tokenized


def progressive_training():
    """Execute progressive training strategy."""
    
    print("🚀 Starting Progressive Training Strategy")
    print("=" * 50)
    
    # Stage 1: Train on ground truth data
    print("\n📖 Stage 1: Training on Ground Truth Data")
    print("-" * 40)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "invalid", 1: "valid"},
        label2id={"invalid": 0, "valid": 1},
    )
    
    # Prepare ground truth data
    gt_train, gt_val = prepare_ground_truth_only()
    
    # Stage 1 training arguments (higher learning rate)
    stage1_args = TrainingArguments(
        output_dir="./results/stage1",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,  # Standard learning rate
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        seed=42,
    )
    
    # Stage 1 trainer
    stage1_trainer = Trainer(
        model=model,
        args=stage1_args,
        train_dataset=gt_train,
        eval_dataset=gt_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train stage 1
    start_time = time.time()
    stage1_trainer.train()
    stage1_time = time.time() - start_time
    
    print(f"✅ Stage 1 completed in {stage1_time/60:.2f} minutes")
    
    # Stage 2: Fine-tune on pseudo-labeled data
    print("\n🔄 Stage 2: Fine-tuning on Pseudo-labeled Data")
    print("-" * 40)
    
    # Prepare pseudo data
    pseudo_train, pseudo_val = prepare_pseudo_data()
    
    # Stage 2 training arguments (lower learning rate)
    stage2_args = TrainingArguments(
        output_dir="./results/stage2",
        num_train_epochs=2,  # Fewer epochs for fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-6,  # Much lower learning rate for fine-tuning
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        seed=42,
    )
    
    # Stage 2 trainer (continue from stage 1 model)
    stage2_trainer = Trainer(
        model=stage1_trainer.model,  # Use the trained model from stage 1
        args=stage2_args,
        train_dataset=pseudo_train,
        eval_dataset=pseudo_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train stage 2
    start_time = time.time()
    stage2_trainer.train()
    stage2_time = time.time() - start_time
    
    print(f"✅ Stage 2 completed in {stage2_time/60:.2f} minutes")
    
    # Save final model
    final_model_path = "./final_model_progressive"
    stage2_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\n🎉 Progressive Training Complete!")
    print(f"📊 Total training time: {(stage1_time + stage2_time)/60:.2f} minutes")
    print(f"💾 Final model saved to: {final_model_path}")
    print("\n📈 Training Summary:")
    print(f"   Stage 1 (Ground Truth): {stage1_time/60:.1f} min - High quality labels")
    print(f"   Stage 2 (Pseudo Labels): {stage2_time/60:.1f} min - Large dataset")


if __name__ == "__main__":
    progressive_training()

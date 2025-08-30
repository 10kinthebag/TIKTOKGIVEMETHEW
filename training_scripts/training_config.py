from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Good batch size for RoBERTa
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  # Standard learning rate for RoBERTa
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    seed=42,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)

print("✅ Training configuration set for RoBERTa-base")
print("🚀 Optimized parameters for better performance than DistilBERT")



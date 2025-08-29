from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Choose model
MODEL_NAME = "distilbert-base-uncased"  # Fast and efficient
# Alternative: "bert-base-uncased" (more accurate but slower)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,  # binary classification
    id2label={0: "invalid", 1: "valid"},
    label2id={"invalid": 0, "valid": 1}
)

print("✅ Model and tokenizer loaded successfully")

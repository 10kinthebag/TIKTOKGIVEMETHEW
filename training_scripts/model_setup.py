from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Choose model - RoBERTa-base for superior performance 
MODEL_NAME = "roberta-base"  # Much more accurate than DistilBERT, stable and reliable
# Previous: "distilbert-base-uncased" (faster but less accurate)
# RoBERTa advantages: Better pre-training, no NSP task, optimized training procedure

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_model():
    """Get the configured RoBERTa model for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,  # binary classification
        id2label={0: "invalid", 1: "valid"},
        label2id={"invalid": 0, "valid": 1}
    )
    return model

def get_tokenizer():
    """Get the RoBERTa tokenizer."""
    return tokenizer

# Initialize model for backward compatibility
model = get_model()

print("✅ RoBERTa-base model and tokenizer loaded successfully")
print("🚀 Upgraded to powerful RoBERTa model - much better than DistilBERT!")

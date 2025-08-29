from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_from_disk
try:
    # Try relative imports first (when run from training_scripts dir)
    from training_config import training_args
    from metrics import compute_metrics
except ImportError:
    # Fall back to absolute imports (when run from root dir)
    from training_scripts.training_config import training_args
    from training_scripts.metrics import compute_metrics


MODEL_NAME = "distilbert-base-uncased"


def get_trainer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "invalid", 1: "valid"},
        label2id={"invalid": 0, "valid": 1},
    )

    train_tokenized = load_from_disk("data/train_tokenized")
    val_tokenized = load_from_disk("data/val_tokenized")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    return trainer


if __name__ == "__main__":
    t = get_trainer()
    print("✅ Trainer initialized")


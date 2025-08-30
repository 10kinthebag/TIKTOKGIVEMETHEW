from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_from_disk, Dataset, DatasetDict
from typing import Union
try:
    # Try relative imports first (when run from training_scripts dir)
    from training_config import training_args
    from metrics import compute_metrics
except ImportError:
    # Fall back to absolute imports (when run from root dir)
    from training_scripts.training_config import training_args
    from training_scripts.metrics import compute_metrics


MODEL_NAME = "roberta-base"


def get_trainer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "invalid", 1: "valid"},
        label2id={"invalid": 0, "valid": 1},
    )

    train_data = load_from_disk("data/train_tokenized")
    val_data = load_from_disk("data/val_tokenized")
    
    # Ensure we have Dataset objects, not DatasetDict
    if isinstance(train_data, DatasetDict):
        train_tokenized = train_data['train'] if 'train' in train_data else list(train_data.values())[0]
    else:
        train_tokenized = train_data
    
    if isinstance(val_data, DatasetDict):
        val_tokenized = val_data['validation'] if 'validation' in val_data else list(val_data.values())[0]
    else:
        val_tokenized = val_data

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )

    return trainer


if __name__ == "__main__":
    print("✅ Trainer setup module ready")
    print("💡 Call get_trainer(model, train_dataset, eval_dataset) to create trainer")


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test loading a model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
print("✅ Setup successful!")

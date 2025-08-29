from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
save_path = "eval/bert_finetuned_fake_review"
tokenizer = BertTokenizer.from_pretrained(save_path)
model = BertForSequenceClassification.from_pretrained(save_path)
model.to(device)
model.eval()

# Function to predict a single review
def predict_review(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction

# Example usage
print(predict_review("this atmosphere has hackathon"))  # 0 = real, 1 = suspicious
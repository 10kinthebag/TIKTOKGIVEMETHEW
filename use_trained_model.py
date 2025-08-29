"""
Example: How to load and use your trained model directly
Run this after training to test your model on new reviews!
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_trained_model(model_path="./final_model"):
    """Load your trained model from disk."""
    print(f"🔄 Loading trained model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    print("✅ Model loaded successfully!")
    return tokenizer, model


def classify_review(text, tokenizer, model):
    """Classify a single review using your trained model."""
    
    # Tokenize the input
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='pt'
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'is_valid': predicted_class == 1,
            'confidence': confidence,
            'probabilities': {
                'invalid': probabilities[0][0].item(),
                'valid': probabilities[0][1].item()
            }
        }


def classify_batch(texts, tokenizer, model, batch_size=16):
    """Classify multiple reviews efficiently."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # Process each prediction in the batch
            for j, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
                results.append({
                    'text': batch_texts[j],
                    'is_valid': pred_class.item() == 1,
                    'confidence': probs[pred_class].item(),
                    'probabilities': {
                        'invalid': probs[0].item(),
                        'valid': probs[1].item()
                    }
                })
    
    return results


def main():
    # Load your trained model
    tokenizer, model = load_trained_model("./final_model")
    
    # Test examples
    test_reviews = [
        "Great food and excellent service! The pasta was delicious and the staff was very friendly.",
        "Visit my website www.example.com for amazing deals and discounts!",
        "Never been here but heard it's terrible from my friends.",
        "The restaurant has a cozy atmosphere and the prices are reasonable.",
        "Click here for 50% off! Limited time offer!",
        "Had a wonderful dining experience. Will definitely come back!"
    ]
    
    print("\n🔍 Testing individual reviews:")
    print("=" * 60)
    
    for i, review in enumerate(test_reviews, 1):
        result = classify_review(review, tokenizer, model)
        status = "✅ VALID" if result['is_valid'] else "❌ INVALID"
        
        print(f"\n{i}. Review: {review[:60]}...")
        print(f"   Result: {status}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Probabilities: Invalid={result['probabilities']['invalid']:.3f}, Valid={result['probabilities']['valid']:.3f}")
    
    print("\n🚀 Testing batch classification:")
    print("=" * 60)
    
    batch_results = classify_batch(test_reviews, tokenizer, model)
    valid_count = sum(1 for r in batch_results if r['is_valid'])
    
    print(f"Processed {len(batch_results)} reviews in batch")
    print(f"Valid reviews: {valid_count}/{len(batch_results)}")
    print(f"Invalid reviews: {len(batch_results) - valid_count}/{len(batch_results)}")


if __name__ == "__main__":
    main()

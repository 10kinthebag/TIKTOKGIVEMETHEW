"""
Restaurant Review Classifier Inference Script
Test your trained model on new restaurant reviews!
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

class RestaurantReviewClassifier:
    """Trained restaurant review classifier for inference"""
    
    def __init__(self, model_path="/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/results/combined_training/final_model"):
        """Initialize the classifier with trained model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define labels
        self.labels = {0: "INVALID", 1: "VALID"}
        
        print("✅ Model loaded successfully!")
    
    def predict_single(self, text, return_confidence=False):
        """Predict if a single review is valid or invalid"""
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'text': text,
            'prediction': self.labels[predicted_class],
            'confidence': confidence,
            'probability_invalid': probabilities[0][0].item(),
            'probability_valid': probabilities[0][1].item()
        }
        
        if return_confidence:
            return result
        else:
            return self.labels[predicted_class]
    
    def predict_batch(self, texts, return_details=True):
        """Predict multiple reviews at once"""
        results = []
        
        for text in texts:
            result = self.predict_single(text, return_confidence=True)
            results.append(result)
        
        if return_details:
            return results
        else:
            return [result['prediction'] for result in results]
    
    def analyze_review(self, text):
        """Detailed analysis of a review"""
        result = self.predict_single(text, return_confidence=True)
        
        print(f"\n📝 Review Analysis:")
        print(f"   Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Probabilities:")
        print(f"     - Valid: {result['probability_valid']:.4f}")
        print(f"     - Invalid: {result['probability_invalid']:.4f}")
        
        return result


def main():
    """Test the trained classifier with sample reviews"""
    
    print("🚀 Restaurant Review Classifier - Inference Test")
    print("=" * 60)
    
    # Initialize classifier
    classifier = RestaurantReviewClassifier()
    
    # Test with sample reviews
    test_reviews = [
        # Valid reviews
        "Great food and excellent service! The pasta was perfectly cooked and the staff was very friendly. Highly recommend this place for dinner.",
        "Amazing Italian restaurant with authentic flavors. The atmosphere is cozy and romantic, perfect for date night.",
        "Good value for money. The portions were generous and the food was tasty. Will definitely come back!",
        
        # Potentially invalid reviews  
        "Best restaurant ever!!!!! 5 stars!!!! 👍👍👍👍👍",
        "This place is terrible. Worst experience ever. Don't waste your time or money here.",
        "Call us now for discount! Special offer available! Visit our website for more deals!",
        
        # Edge cases
        "Ok.",
        "The food was decent but nothing special. Service was average.",
        "I went here with my family and we had a pleasant experience. The ambiance was nice and the food was good."
    ]
    
    print(f"\n🧪 Testing with {len(test_reviews)} sample reviews:")
    print("-" * 40)
    
    # Analyze each review
    for i, review in enumerate(test_reviews, 1):
        print(f"\n{i}. {classifier.analyze_review(review)['prediction']}")
    
    # Batch prediction
    print(f"\n📊 Batch Prediction Summary:")
    batch_results = classifier.predict_batch(test_reviews)
    
    valid_count = sum(1 for r in batch_results if r['prediction'] == 'VALID')
    invalid_count = len(batch_results) - valid_count
    
    print(f"   Total reviews: {len(batch_results)}")
    print(f"   Valid reviews: {valid_count}")
    print(f"   Invalid reviews: {invalid_count}")
    print(f"   Average confidence: {sum(r['confidence'] for r in batch_results) / len(batch_results):.4f}")
    
    print(f"\n🎯 Classification Performance:")
    for result in batch_results:
        status = "✅" if result['confidence'] > 0.8 else "⚠️" if result['confidence'] > 0.6 else "❌"
        print(f"   {status} {result['prediction']} ({result['confidence']:.3f}): {result['text'][:50]}...")


if __name__ == "__main__":
    try:
        main()
        print("\n✅ Inference test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()

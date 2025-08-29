import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ReviewClassificationPipeline:
    def __init__(self, model_path="./final_model"):
        """Initialize the hybrid classification pipeline with rules + ML model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            print("✅ Loaded fine-tuned model from", model_path)
        except Exception as e:
            print(f"⚠️ Could not load model from {model_path}: {e}")
            print("⚠️ Using rule-based filtering only")
            self.tokenizer = None
            self.model = None

    def rule_based_filter(self, text):
        """First layer: Rule-based filtering for obvious violations."""
        text_lower = text.lower()
        
        # High-confidence invalid patterns
        spam_patterns = [
            (r'http[s]?://\S+', "Contains URL"),
            (r'www\.\S+', "Contains website"),
            (r'visit\s+(my|our)\s+\w+', "Promotional content"),
            (r'never\s+been\s+(here|there)', "Never visited"),
            (r'click\s+here|call\s+now', "Call-to-action spam"),
            (r'discount|coupon|promo|deal|offer|sale', "Promotional content"),
            (r'follow\s+me|subscribe|like\s+and\s+share', "Social media promotion"),
        ]
        
        for pattern, reason in spam_patterns:
            if re.search(pattern, text_lower):
                return False, 0.95, reason
        
        # Length-based filtering
        word_count = len(text.split())
        if word_count < 3:
            return False, 0.9, "Too short"
        
        # If no clear spam indicators, pass to ML model
        return None, None, "Needs ML classification"
    
    def ml_classify(self, text):
        """Second layer: ML classification for nuanced cases."""
        if not self.model or not self.tokenizer:
            # Fallback to rule-based if model not loaded
            return True, 0.7, "ML model unavailable, defaulting to valid"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return predicted_class == 1, confidence
    
    def classify(self, text):
        """Full pipeline classification combining rules and ML."""
        if not text or not text.strip():
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'rule_based',
                'reason': 'Empty text'
            }
        
        # First: Rule-based filtering
        rule_result, rule_confidence, reason = self.rule_based_filter(text)
        
        if rule_result is not None:
            return {
                'is_valid': rule_result,
                'confidence': rule_confidence,
                'method': 'rule_based',
                'reason': reason
            }
        
        # Second: ML classification
        ml_result, ml_confidence = self.ml_classify(text)
        
        return {
            'is_valid': ml_result,
            'confidence': ml_confidence,
            'method': 'ml_model',
            'reason': 'ML classification'
        }
    
    def batch_classify(self, texts):
        """Classify multiple texts efficiently."""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results


if __name__ == "__main__":
    pipeline = ReviewClassificationPipeline()
    print("✅ Hybrid pipeline created")

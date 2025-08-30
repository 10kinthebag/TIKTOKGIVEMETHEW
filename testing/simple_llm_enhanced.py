"""
LLM-Enhanced Review Validator - Simple Integration
Combines your trained model with OpenAI LLM for low-confidence predictions
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file (override existing ones)
load_dotenv(override=True)

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Try to import LLM components
try:
    from llm_validator import LLMReviewJudge, HybridReviewValidator, ReviewContext
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LLM components not available: {e}")
    LLM_AVAILABLE = False
    
    # Create dummy classes for fallback
    class ReviewContext:
        def __init__(self, business_name, business_category="Restaurant", city="Unknown", country="Unknown"):
            self.business_name = business_name
            self.business_category = business_category
            self.city = city
            self.country = country


class SimpleEnhancedClassifier:
    """
    Simple enhanced classifier that can work with or without LLM
    """
    
    def __init__(self, model_path="./models/roberta_policy_based_model", confidence_threshold=0.70):
        """Initialize the classifier"""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load the trained model
        print(f"🤖 Loading trained model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize LLM validator if available
        self.llm_validator = None
        self.use_llm = False
        
        if LLM_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                llm_judge = LLMReviewJudge()
                self.llm_validator = HybridReviewValidator(llm_judge, confidence_threshold)
                self.use_llm = True
                print(f"✅ LLM validation enabled (threshold: {confidence_threshold})")
            except Exception as e:
                print(f"⚠️ LLM validation disabled: {e}")
        else:
            print("⚠️ LLM validation not available (missing API key or components)")
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "model_confident": 0,
            "llm_validated": 0,
            "agreements": 0,
            "disagreements": 0
        }
    
    def predict_single(self, text, business_name="Unknown Restaurant", business_category="Restaurant"):
        """Predict a single review"""
        
        # Get model prediction
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to labels
        labels = {0: "INVALID", 1: "VALID"}
        model_prediction = labels[int(predicted_class)]
        
        # Update stats
        self.stats["total_processed"] += 1
        
        # Base result
        result = {
            "text": text,
            "business_name": business_name,
            "model_prediction": model_prediction,
            "model_confidence": confidence,
            "probability_valid": probabilities[0][1].item(),
            "probability_invalid": probabilities[0][0].item(),
            "used_llm": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use LLM validation if enabled and confidence is low
        if self.use_llm and confidence <= self.confidence_threshold:
            self.stats["llm_validated"] += 1
            
            try:
                context = ReviewContext(business_name, business_category, "Istanbul", "Turkey")
                validation_result = self.llm_validator.validate_review(
                    text, model_prediction, confidence, context
                )
                
                # Check agreement
                llm_prediction = validation_result["final_decision"].upper()
                if model_prediction == llm_prediction:
                    self.stats["agreements"] += 1
                    agreement = "agreement"
                else:
                    self.stats["disagreements"] += 1
                    agreement = "disagreement"
                
                # Update result with LLM data
                result.update({
                    "final_prediction": llm_prediction,
                    "final_action": validation_result["final_action"],
                    "used_llm": True,
                    "llm_confidence": validation_result.get("llm_confidence"),
                    "agreement_status": agreement,
                    "llm_rationale": validation_result.get("rationale"),
                    "validation_source": "hybrid"
                })
                
            except Exception as e:
                print(f"⚠️ LLM validation failed: {e}")
                # Fall back to model prediction
                result.update({
                    "final_prediction": model_prediction,
                    "final_action": "keep" if model_prediction == "VALID" else "remove",
                    "validation_source": "model-only",
                    "llm_error": str(e)
                })
        else:
            # Use model prediction as final
            self.stats["model_confident"] += 1
            result.update({
                "final_prediction": model_prediction,
                "final_action": "keep" if model_prediction == "VALID" else "remove",
                "validation_source": "model-only"
            })
        
        return result
    
    def analyze_review(self, text, business_name="Unknown Restaurant"):
        """Analyze a review with detailed output"""
        result = self.predict_single(text, business_name)
        
        print(f"\n📝 Review Analysis:")
        print(f"   Text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        print(f"   Business: {business_name}")
        
        print(f"\n🤖 Model Results:")
        print(f"   Prediction: {result['model_prediction']}")
        print(f"   Confidence: {result['model_confidence']:.4f}")
        print(f"   Probabilities: Valid={result['probability_valid']:.4f}, Invalid={result['probability_invalid']:.4f}")
        
        if result['used_llm']:
            print(f"\n🧠 LLM Validation (Low Confidence ≤ {self.confidence_threshold}):")
            print(f"   LLM Prediction: {result['final_prediction']}")
            print(f"   LLM Confidence: {result.get('llm_confidence', 'N/A')}")
            print(f"   Agreement: {result.get('agreement_status', 'N/A')}")
            print(f"   LLM Rationale: {result.get('llm_rationale', 'N/A')}")
            print(f"   Final Action: {result['final_action']}")
        else:
            print(f"\n✅ High Confidence - Using Model Prediction")
            print(f"   Final Action: {result['final_action']}")
        
        return result
    
    def process_batch(self, reviews_and_businesses, save_to_file=True):
        """Process multiple reviews"""
        results = []
        
        print(f"\n🔄 Processing {len(reviews_and_businesses)} reviews...")
        
        for i, (review, business) in enumerate(reviews_and_businesses, 1):
            if i % 10 == 0:
                print(f"   Processed {i}/{len(reviews_and_businesses)}...")
            
            result = self.predict_single(review, business)
            results.append(result)
        
        # Save results
        if save_to_file:
            output_file = f"llm_enhanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
        # Print summary
        self.print_summary()
        
        return results
    
    def print_summary(self):
        """Print processing summary"""
        stats = self.stats
        total = stats["total_processed"]
        
        if total == 0:
            print("No reviews processed yet.")
            return
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total reviews: {total}")
        print(f"   Model confident: {stats['model_confident']} ({stats['model_confident']/total:.1%})")
        print(f"   LLM validated: {stats['llm_validated']} ({stats['llm_validated']/total:.1%})")
        
        if stats['llm_validated'] > 0:
            agreements = stats['agreements']
            disagreements = stats['disagreements']
            llm_total = stats['llm_validated']
            print(f"   Model-LLM agreements: {agreements} ({agreements/llm_total:.1%})")
            print(f"   Model-LLM disagreements: {disagreements} ({disagreements/llm_total:.1%})")


def test_simple_integration():
    """Test the simple integration"""
    
    print("🚀 Testing Simple LLM-Enhanced Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = SimpleEnhancedClassifier(model_path="./models/roberta_policy_based_model", confidence_threshold=0.70)
    
    # Test cases
    test_cases = [
        ("Great food and excellent service! The pasta was perfectly cooked.", "Mario's Italian"),
        ("Ok.", "Local Cafe"), 
        ("Call us now for discount! Special offer available!", "Taste Bistro"),
        ("Folk hero is a business every time amazing.", "Heritage Restaurant"),
        ("Food was decent but nothing special. Service average.", "Downtown Diner"),
        ("This restaurant is absolutely terrible! Worst experience ever!", "Bad Experience Bistro"),
        ("Amazing authentic Turkish cuisine, highly recommend the kebabs!", "Turkish Delight")
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} reviews:")
    
    # Test individual reviews
    for i, (review, business) in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"Test Case {i}")
        classifier.analyze_review(review, business)
    
    # Test batch processing
    print(f"\n{'='*50}")
    print("🔄 Testing Batch Processing")
    
    batch_results = classifier.process_batch(test_cases, save_to_file=True)
    
    print("\n✅ Testing completed!")
    return classifier, batch_results


if __name__ == "__main__":
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY environment variable not set")
        print("LLM validation will be disabled. To enable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nContinuing with model-only mode...")
    
    test_simple_integration()

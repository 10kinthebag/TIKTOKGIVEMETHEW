"""
Enhanced Restaurant Review Classifier with LLM Validation
Integrates your trained model with OpenAI LLM for low-confidence predictions
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Import our LLM validator
try:
    from llm_validator import LLMReviewJudge, HybridReviewValidator, ReviewContext
except ImportError:
    # Fallback if import fails
    print("⚠️ LLM validator not available - using model-only mode")
    LLMReviewJudge = None
    HybridReviewValidator = None
    ReviewContext = None


class EnhancedReviewClassifier:
    """
    Enhanced classifier that combines trained model with LLM validation
    """
    
    def __init__(
        self, 
        model_path: str = "../models/combined_training_model",
        use_llm: bool = True,
        confidence_threshold: float = 0.70,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the enhanced classifier
        
        Args:
            model_path: Path to the trained model
            use_llm: Whether to use LLM validation for low-confidence predictions
            confidence_threshold: Threshold below which to use LLM
            openai_api_key: OpenAI API key (optional, can use env var)
        """
        self.model_path = model_path
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold
        
        # Load the trained model
        print(f"🤖 Loading trained model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize LLM validator if requested
        self.llm_validator = None
        if use_llm and LLMReviewJudge is not None:
            try:
                llm_judge = LLMReviewJudge(api_key=openai_api_key)
                self.llm_validator = HybridReviewValidator(llm_judge, confidence_threshold)
                print(f"✅ LLM validation enabled (threshold: {confidence_threshold})")
            except Exception as e:
                print(f"⚠️ LLM validation disabled: {e}")
                self.use_llm = False
        else:
            self.use_llm = False
        
        print(f"✅ Enhanced classifier ready! (LLM: {'enabled' if self.use_llm else 'disabled'})")
    
    def predict_single(self, text: str, context: Optional[ReviewContext] = None) -> Dict[str, Any]:
        """
        Predict a single review with optional LLM validation
        
        Args:
            text: Review text
            context: Business context for LLM validation
            
        Returns:
            Comprehensive prediction result
        """
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
        
        # Base result
        result = {
            "text": text,
            "model_prediction": model_prediction,
            "model_confidence": confidence,
            "probability_valid": probabilities[0][1].item(),
            "probability_invalid": probabilities[0][0].item(),
            "used_llm": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use LLM validation if enabled and confidence is low
        if self.use_llm and confidence <= self.confidence_threshold and context:
            validation_result = self.llm_validator.validate_review(
                text, model_prediction, confidence, context
            )
            
            # Merge LLM results
            result.update({
                "final_prediction": validation_result["final_decision"].upper(),
                "final_action": validation_result["final_action"],
                "used_llm": True,
                "llm_confidence": validation_result.get("llm_confidence"),
                "agreement_status": validation_result.get("agreement_status"),
                "llm_rationale": validation_result.get("rationale"),
                "validation_source": validation_result["source"]
            })
        else:
            # Use model prediction as final
            result.update({
                "final_prediction": model_prediction,
                "final_action": "keep" if model_prediction == "VALID" else "remove",
                "validation_source": "trained-model-only"
            })
        
        return result
    
    def predict_batch(
        self, 
        reviews: List[str], 
        contexts: Optional[List[ReviewContext]] = None,
        save_results: bool = True,
        output_file: str = "enhanced_predictions.json"
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple reviews with batch processing
        
        Args:
            reviews: List of review texts
            contexts: List of business contexts (optional)
            save_results: Whether to save results to file
            output_file: Output filename
            
        Returns:
            List of prediction results
        """
        if contexts is None:
            contexts = [None] * len(reviews)
        
        results = []
        print(f"\n🔄 Processing {len(reviews)} reviews...")
        
        for i, (review, context) in enumerate(zip(reviews, contexts), 1):
            if i % 10 == 0:
                print(f"   Processed {i}/{len(reviews)} reviews...")
            
            result = self.predict_single(review, context)
            results.append(result)
        
        # Save results if requested
        if save_results:
            output_path = Path("results") / output_file
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {output_path}")
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict[str, Any]]):
        """Print summary of batch predictions"""
        total = len(results)
        model_confident = sum(1 for r in results if not r["used_llm"])
        llm_used = sum(1 for r in results if r["used_llm"])
        
        valid_predictions = sum(1 for r in results if r["final_prediction"] == "VALID")
        invalid_predictions = total - valid_predictions
        
        agreements = sum(1 for r in results if r.get("agreement_status") == "agreement")
        disagreements = sum(1 for r in results if r.get("agreement_status") == "disagreement")
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total reviews: {total}")
        print(f"   Model confident: {model_confident} ({model_confident/total:.1%})")
        print(f"   LLM validated: {llm_used} ({llm_used/total:.1%})")
        print(f"   Final VALID: {valid_predictions} ({valid_predictions/total:.1%})")
        print(f"   Final INVALID: {invalid_predictions} ({invalid_predictions/total:.1%})")
        
        if llm_used > 0:
            print(f"   Model-LLM agreements: {agreements} ({agreements/llm_used:.1%})")
            print(f"   Model-LLM disagreements: {disagreements} ({disagreements/llm_used:.1%})")
    
    def analyze_review_detailed(self, text: str, context: ReviewContext) -> Dict[str, Any]:
        """
        Detailed analysis of a single review with full explanations
        
        Args:
            text: Review text
            context: Business context
            
        Returns:
            Detailed analysis result
        """
        result = self.predict_single(text, context)
        
        print(f"\n📝 Detailed Review Analysis:")
        print(f"   Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"   Business: {context.business_name} ({context.business_category})")
        print(f"   Location: {context.city}, {context.country}")
        print(f"\n🤖 Model Results:")
        print(f"   Prediction: {result['model_prediction']}")
        print(f"   Confidence: {result['model_confidence']:.4f}")
        print(f"   Probabilities: Valid={result['probability_valid']:.4f}, Invalid={result['probability_invalid']:.4f}")
        
        if result['used_llm']:
            print(f"\n🧠 LLM Validation (Low Confidence):")
            print(f"   LLM Prediction: {result['final_prediction']}")
            print(f"   LLM Confidence: {result['llm_confidence']}")
            print(f"   Agreement: {result['agreement_status']}")
            print(f"   LLM Rationale: {result['llm_rationale']}")
            print(f"   Final Action: {result['final_action']}")
        else:
            print(f"\n✅ High Confidence - Using Model Prediction")
            print(f"   Final Action: {result['final_action']}")
        
        return result


def test_enhanced_classifier():
    """Test the enhanced classifier with sample data"""
    
    print("🚀 Testing Enhanced Restaurant Review Classifier")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not set - LLM validation will be disabled")
        use_llm = False
    else:
        use_llm = True
    
    # Initialize classifier
    classifier = EnhancedReviewClassifier(
        model_path="../models/combined_training_model",
        use_llm=use_llm,
        confidence_threshold=0.70
    )
    
    # Test cases with business context
    test_cases = [
        {
            "review": "Great food and excellent service! The pasta was perfectly cooked and the staff was very friendly.",
            "context": ReviewContext("Mario's Italian", "Italian Restaurant", "Istanbul", "Turkey")
        },
        {
            "review": "Ok.",
            "context": ReviewContext("Local Cafe", "Cafe", "Istanbul", "Turkey")
        },
        {
            "review": "Call us now for discount! Special offer available! Visit our website!",
            "context": ReviewContext("Taste Bistro", "Restaurant", "Istanbul", "Turkey")
        },
        {
            "review": "Folk hero is a business every time amazing experience with the beautiful.",
            "context": ReviewContext("Heritage Restaurant", "Traditional Restaurant", "Istanbul", "Turkey")
        },
        {
            "review": "Food was decent but nothing special. Service was average, prices reasonable.",
            "context": ReviewContext("Downtown Diner", "Restaurant", "Istanbul", "Turkey")
        }
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} reviews:")
    
    # Test individual reviews
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}")
        classifier.analyze_review_detailed(case["review"], case["context"])
    
    # Test batch processing
    print(f"\n{'='*60}")
    print("🔄 Testing Batch Processing")
    
    reviews = [case["review"] for case in test_cases]
    contexts = [case["context"] for case in test_cases]
    
    batch_results = classifier.predict_batch(
        reviews, contexts, 
        save_results=True, 
        output_file="enhanced_test_results.json"
    )
    
    print("\n✅ Enhanced classifier testing completed!")


if __name__ == "__main__":
    test_enhanced_classifier()

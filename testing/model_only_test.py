#!/usr/bin/env python3
"""
Test the model without LLM validation (offline mode)
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from datetime import datetime

class ModelOnlyClassifier:
    """
    Classifier that works without LLM validation
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
        
        print("⚠️ LLM validation disabled (offline mode)")
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "high_confidence": 0,
            "low_confidence": 0,
        }
    
    def predict_single(self, text, business_name="Unknown Restaurant"):
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
        labels = ["INVALID", "VALID"]
        prediction = labels[int(predicted_class)]
        
        # Update statistics
        self.stats["total_processed"] += 1
        if confidence >= self.confidence_threshold:
            self.stats["high_confidence"] += 1
            final_action = "keep" if prediction == "VALID" else "flag"
            confidence_level = "High Confidence"
        else:
            self.stats["low_confidence"] += 1
            final_action = "flag"  # Flag low confidence for manual review
            confidence_level = "Low Confidence (would use LLM)"
        
        return {
            "text": text,
            "business_name": business_name,
            "model_prediction": prediction,
            "model_confidence": confidence,
            "confidence_level": confidence_level,
            "final_action": final_action,
            "probabilities": {
                "Valid": probabilities[0][1].item(),
                "Invalid": probabilities[0][0].item()
            }
        }
    
    def analyze_review(self, text, business_name="Unknown Restaurant"):
        """Analyze and display results for a single review"""
        
        result = self.predict_single(text, business_name)
        
        print(f"\n📝 Review Analysis:")
        print(f"   Text: \"{text}\"")
        print(f"   Business: {business_name}")
        print(f"\n🤖 Model Results:")
        print(f"   Prediction: {result['model_prediction']}")
        print(f"   Confidence: {result['model_confidence']:.4f}")
        print(f"   Probabilities: Valid={result['probabilities']['Valid']:.4f}, Invalid={result['probabilities']['Invalid']:.4f}")
        print(f"\n📊 Assessment: {result['confidence_level']}")
        print(f"   Final Action: {result['final_action']}")
        
        return result

def test_model_only():
    """Test the model without LLM"""
    
    print("🚀 Testing Model-Only Classifier (No LLM)")
    print("=" * 50)
    
    # Initialize classifier
    classifier = ModelOnlyClassifier(
        model_path="./models/roberta_policy_based_model", 
        confidence_threshold=0.70
    )
    
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
    
    results = []
    for i, (review, business) in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"Test Case {i}")
        result = classifier.analyze_review(review, business)
        results.append(result)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Summary")
    print(f"   Total reviews: {classifier.stats['total_processed']}")
    print(f"   High confidence: {classifier.stats['high_confidence']} ({classifier.stats['high_confidence']/classifier.stats['total_processed']*100:.1f}%)")
    print(f"   Low confidence: {classifier.stats['low_confidence']} ({classifier.stats['low_confidence']/classifier.stats['total_processed']*100:.1f}%)")
    print(f"   Reviews that would trigger LLM: {classifier.stats['low_confidence']}")
    
    return classifier, results

if __name__ == "__main__":
    test_model_only()

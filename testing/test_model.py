"""
Quick Model Tester
Interactive script to test your new RoBERTa model on individual reviews.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os


#!/usr/bin/env python3
"""
Test the trained RoBERTa model with various testing modes.
"""

import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test RoBERTa model')
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['batch', 'interactive', 'both'],
                       help='Testing mode: batch, interactive, or both')
    parser.add_argument('--model-path', type=str, default='./models/roberta_policy_based_model',
                       help='Path to the trained model')
    return parser.parse_args()

def load_model(model_path):
    """Load the trained model and tokenizer."""
    print(f"🤖 Loading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("💡 Make sure you've trained the model first!")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model size: {total_params/1e6:.1f}M parameters")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def test_single_review(text, model, tokenizer):
    """Test a single review and return prediction with confidence."""
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Extract results
        pred_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][pred_class].item()
        
        # Map to human-readable labels
        label = "✅ VALID" if pred_class == 1 else "❌ INVALID"
        
        return {
            'prediction': pred_class,
            'label': label,
            'confidence': confidence,
            'probabilities': {
                'invalid': probabilities[0][0].item(),
                'valid': probabilities[0][1].item()
            }
        }
        
    except Exception as e:
        print(f"❌ Error processing review: {e}")
        return None


def interactive_testing():
    """Interactive testing mode."""
    print("🎯 Interactive Model Testing")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model("./models/roberta_policy_based_model")
    if model is None:
        return
    
    print("\n🔍 Enter restaurant reviews to test (or 'quit' to exit):")
    print("💡 Try both valid reviews and problematic ones!")
    print()
    
    test_examples = [
        "Great restaurant with amazing food and excellent service! Highly recommend the pasta.",
        "Visit our website www.spamsite.com for amazing deals! Call 555-1234 now!",
        "This place is terrible terrible terrible terrible terrible terrible",
        "I love this place! The food is delicious and staff is friendly.",
        "Never been here but heard it's bad from my friend.",
    ]
    
    print("🎲 Example reviews to try:")
    for i, example in enumerate(test_examples, 1):
        print(f"   {i}. {example}")
    print()
    
    while True:
        try:
            user_input = input("📝 Enter review (or number 1-5 for examples): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Handle numbered examples
            if user_input.isdigit() and 1 <= int(user_input) <= 5:
                text = test_examples[int(user_input) - 1]
                print(f"📖 Testing example: {text}")
            else:
                text = user_input
            
            if not text:
                continue
                
            # Test the review
            result = test_single_review(text, model, tokenizer)
            
            if result:
                print(f"\n🔮 Prediction: {result['label']}")
                print(f"🎯 Confidence: {result['confidence']:.1%}")
                print(f"📊 Probabilities:")
                print(f"   Invalid: {result['probabilities']['invalid']:.1%}")
                print(f"   Valid: {result['probabilities']['valid']:.1%}")
                
                # Give interpretation
                if result['confidence'] > 0.9:
                    print("💪 Very confident prediction!")
                elif result['confidence'] > 0.7:
                    print("👍 Confident prediction")
                else:
                    print("🤔 Less confident - review carefully")
                
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Testing interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_testing():
    """Run batch testing on predefined test cases."""
    print("📊 Batch Testing on Sample Reviews")
    print("=" * 50)
    
    model, tokenizer = load_model("./models/roberta_policy_based_model")
    if model is None:
        return
    
    # Test cases
    test_cases = [
        # Valid reviews
        ("Amazing food and great service! The pasta was delicious.", "Expected: VALID"),
        ("Lovely atmosphere, friendly staff, reasonable prices.", "Expected: VALID"),
        ("Best restaurant in town! Highly recommend the seafood.", "Expected: VALID"),
        
        # Invalid reviews (ads)
        ("Visit our website www.bestdeals.com for 50% off!", "Expected: INVALID (Ad)"),
        ("Call 555-1234 for amazing restaurant deals today!", "Expected: INVALID (Ad)"),
        
        # Invalid reviews (spam/irrelevant)
        ("This phone app is great for downloading music", "Expected: INVALID (Irrelevant)"),
        ("aaaaaaaaaaaaaaaaaaaaaaaaa", "Expected: INVALID (Spam)"),
        
        # Invalid reviews (rants without visits)  
        ("Never been there but heard it's terrible from friends", "Expected: INVALID (Rant)"),
        ("My friend told me this place is bad", "Expected: INVALID (Rant)"),
        
        # Edge cases
        ("Good", "Expected: INVALID (Too short)"),
        ("I love it but the service was terrible and food was amazing but bad", "Expected: INVALID (Contradiction)"),
    ]
    
    correct_predictions = 0
    
    for i, (text, expected) in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {expected}")
        print(f"📝 Review: {text}")
        
        result = test_single_review(text, model, tokenizer)
        
        if result:
            print(f"🔮 Result: {result['label']} ({result['confidence']:.1%})")
            
            # Simple accuracy check (you'd need to define expected labels properly)
            if "VALID" in expected and result['prediction'] == 1:
                correct_predictions += 1
                print("✅ Correct!")
            elif "INVALID" in expected and result['prediction'] == 0:
                correct_predictions += 1
                print("✅ Correct!")
            else:
                print("❌ Incorrect")
        
        print("-" * 40)
    
    accuracy = correct_predictions / len(test_cases)
    print(f"\n📊 Batch Test Results:")
    print(f"   Correct: {correct_predictions}/{len(test_cases)}")
    print(f"   Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    args = parse_args()
    
    print("🚀 RoBERTa Model Tester")
    print("Testing your newly trained policy-based model!")
    print()
    
    # Handle command line mode selection
    if args.mode == "batch":
        batch_testing()
    elif args.mode == "interactive":
        interactive_testing()
    elif args.mode == "both":
        batch_testing()
        print("\n" + "="*60)
        interactive_testing()
    else:
        # Interactive fallback for manual execution
        print("Choose testing mode:")
        print("1. Interactive testing (test your own reviews)")
        print("2. Batch testing (predefined test cases)")
        print("3. Both")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_testing()
        elif choice == "2":
            batch_testing()
        elif choice == "3":
            batch_testing()
            print("\n" + "="*60)
            interactive_testing()
        else:
            print("Invalid choice. Running interactive mode...")
            interactive_testing()

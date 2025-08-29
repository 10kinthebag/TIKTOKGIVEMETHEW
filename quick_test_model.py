"""
Quick test script to verify your trained model works.
Run this immediately after training to confirm everything is working!
"""
from hybrid_pipeline.hybrid_pipeline import ReviewClassificationPipeline


def quick_model_test():
    print("🧪 Quick Model Test")
    print("=" * 40)
    
    # Load your trained model
    pipeline = ReviewClassificationPipeline("./final_model")
    
    # Test cases
    test_cases = [
        ("Great food and service!", True),   # Should be valid
        ("Visit www.spam.com now!", False),  # Should be invalid  
        ("Never been but heard bad things", False),  # Should be invalid
        ("Delicious pasta, friendly staff", True),   # Should be valid
    ]
    
    print("Testing your trained model:")
    print("-" * 40)
    
    correct = 0
    for text, expected in test_cases:
        result = pipeline.classify(text)
        predicted = result['is_valid']
        status = "✅" if predicted == expected else "❌"
        
        print(f"{status} '{text[:30]}...'")
        print(f"    Expected: {'Valid' if expected else 'Invalid'}")
        print(f"    Predicted: {'Valid' if predicted else 'Invalid'}")  
        print(f"    Confidence: {result['confidence']:.3f}")
        print()
        
        if predicted == expected:
            correct += 1
    
    accuracy = correct / len(test_cases)
    print(f"🎯 Quick Test Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")
    
    if accuracy >= 0.75:
        print("🎉 Your model looks good!")
    else:
        print("⚠️ Model might need more training or data adjustment")


if __name__ == "__main__":
    quick_model_test()

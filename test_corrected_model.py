"""
Test the corrected model after fixing the label mapping issue
"""
from hybrid_pipeline.hybrid_pipeline import ReviewClassificationPipeline


def test_corrected_model():
    print("🧪 Testing Corrected Model (Fixed Labels)")
    print("=" * 50)
    
    try:
        pipeline = ReviewClassificationPipeline("./final_model")
        print("✅ Loaded corrected model")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("Make sure training completed successfully!")
        return
    
    # Test cases with corrected expectations
    test_cases = [
        # Valid reviews (should predict True/1)
        ("Great food and excellent service! Highly recommend.", True),
        ("Amazing pasta and friendly staff. Will definitely come back!", True),
        ("Delicious meals, cozy atmosphere, reasonable prices.", True),
        ("Had a wonderful dining experience with my family.", True),
        ("The pizza was incredible and the service was top-notch.", True),
        ("Turkey's cheapest artisan restaurant and its food is delicious!", True),
        ("We went to Marmaris with my wife for a holiday. Great place!", True),
        
        # Invalid reviews (should predict False/0) 
        ("Visit my website www.deals.com for 50% discount!", False),
        ("Never been here but heard it's terrible from friends.", False),
        ("Click here for amazing offers! Limited time only!", False),
        ("Call 555-1234 now for restaurant discounts!", False),
        ("Check out my blog at foodreviews.net", False),
        ("asdfgh qwerty 12345", False),
        ("Follow me on Instagram @foodblogger", False),
    ]
    
    print(f"Testing {len(test_cases)} cases with corrected labels:")
    print("-" * 70)
    print("| # | Expected | Predicted | Confidence | Status | Review Text")
    print("|---|----------|-----------|------------|--------|------------")
    
    correct = 0
    for i, (text, expected) in enumerate(test_cases, 1):
        result = pipeline.classify(text)
        predicted = result['is_valid']
        confidence = result['confidence']
        status = "✅" if predicted == expected else "❌"
        
        if predicted == expected:
            correct += 1
        
        expected_str = "Valid" if expected else "Invalid"
        predicted_str = "Valid" if predicted else "Invalid"
        text_short = text[:40] + "..." if len(text) > 40 else text
        
        print(f"| {i:2d} | {expected_str:8s} | {predicted_str:9s} | {confidence:10.3f} | {status:6s} | {text_short}")
    
    accuracy = correct / len(test_cases)
    print(f"\n📊 Results:")
    print(f"Accuracy: {correct}/{len(test_cases)} ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("🎉 Excellent! Model is working much better with corrected labels!")
    elif accuracy >= 0.6:
        print("👍 Good improvement! Model is working better.")
    else:
        print("⚠️ Still needs improvement. Check if training completed properly.")
    
    # Show some detailed results
    if accuracy < 0.8:
        print(f"\n🔍 Areas for improvement:")
        for i, (text, expected) in enumerate(test_cases, 1):
            result = pipeline.classify(text)
            predicted = result['is_valid']
            if predicted != expected:
                print(f"  • Case {i}: Expected {expected}, got {predicted}")
                print(f"    Text: {text[:60]}...")


if __name__ == "__main__":
    test_corrected_model()

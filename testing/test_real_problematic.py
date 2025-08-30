#!/usr/bin/env python3
"""
Test real problematic reviews from the ground truth dataset
to see if we can trigger low confidence predictions and LLM validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.simple_llm_enhanced import SimpleEnhancedClassifier

def test_real_problematic_reviews():
    """Test with actual reviews that were labeled as invalid in ground truth"""
    
    print("🧪 Testing Real Problematic Reviews")
    print("="*50)
    
    # Initialize classifier
    classifier = SimpleEnhancedClassifier(confidence_threshold=0.80)  # Slightly higher threshold
    
    # Real problematic reviews from ground truth data (labeled as 0/invalid)
    problematic_reviews = [
        {
            "text": "During my holiday in Marmaris we ate here to fit the food. It's really good that the food is cheap and nice. Eating as much bread as you want is a big plus for those who are not satisfied without bread. It is a place that I can recommend to those who will go to Marmaris. On July 1 there was a small increase but even the price hike is cheap. I leave the photo of the latest prices and breakfast below. there was a serious queue. You proceed by taking the food you want in the form of an open buffet. Both vegetable dishes and meat dishes were plentiful. There was also dessert for those who wanted it. After you get what you want you pay at the cashier. They don't go through cards they work in cash. There was a lot of food variety. And the food prices were unbelievably cheap. We paid only 84 TL for all the meals here. It included buttermilk and bread. But unfortunately I can't say it's too clean as a place..",
            "business_name": "Marmaris Buffet Restaurant",
            "expected": "INVALID",
            "reason": "Duplicate/copy-paste content with excessive detail"
        },
        {
            "text": "Every time I go, I still experience the amazement I experienced years ago as if it were the first time. There is no need to explain. Folk hero is a business.",
            "business_name": "Folk Hero Restaurant",
            "expected": "INVALID", 
            "reason": "Vague, non-specific review without concrete details"
        },
        {
            "text": "The most f/p of all businesses I've seen.",
            "business_name": "Generic Restaurant",
            "expected": "INVALID",
            "reason": "Abbreviations and unclear meaning"
        },
        {
            "text": "Great.",
            "business_name": "One Word Restaurant",
            "expected": "INVALID",
            "reason": "Too brief, no meaningful content"
        },
        {
            "text": "Absolutely perfect come try it..",
            "business_name": "Perfect Place",
            "expected": "INVALID",
            "reason": "Generic promotional language"
        },
        {
            "text": "Slagethi all nero was excellent. Beef ribs were always tender enough. Cheek, on the other hand; has preserved its taste even though its collection is a little small. If deterioration was to come; the services were passed and the environment was much expected.",
            "business_name": "Italian Steakhouse",
            "expected": "INVALID",
            "reason": "Confusing language, possible translation issues"
        },
        {
            "text": "Good salep was good; the employees were smiling..",
            "business_name": "Salep Cafe",
            "expected": "INVALID", 
            "reason": "Redundant phrasing, minimal content"
        },
        {
            "text": "A delicious doner.",
            "business_name": "Doner Kebab Shop",
            "expected": "INVALID",
            "reason": "Too brief for meaningful review"
        },
        {
            "text": "Awesomeee",
            "business_name": "Awesome Restaurant",
            "expected": "INVALID",
            "reason": "Single word with spelling variation"
        },
        {
            "text": "Delicious!",
            "business_name": "Delicious Cafe",
            "expected": "INVALID",
            "reason": "Single word, no context"
        }
    ]
    
    print(f"🤖 Loading model...")
    print(f"⚠️ LLM validation {'available' if classifier.use_llm else 'not available (missing API key or components)'}")
    print(f"Testing {len(problematic_reviews)} reviews labeled as INVALID in ground truth:")
    print(f"Confidence threshold: {classifier.confidence_threshold}")
    print()
    
    low_confidence_count = 0
    model_errors = 0
    
    for i, review in enumerate(problematic_reviews, 1):
        print("="*40)
        print(f"Problematic Review {i}")
        print()
        
        result = classifier.predict_single(
            review["text"], 
            review["business_name"]
        )
        
        print(f"📝 Review Analysis:")
        print(f"   Text: \"{review['text'][:100]}{'...' if len(review['text']) > 100 else ''}\"")
        print(f"   Business: {review['business_name']}")
        print(f"   Expected: {review['expected']} ({review['reason']})")
        print()
        
        print(f"🤖 Model Results:")
        print(f"   Prediction: {result['model_prediction']}")
        print(f"   Confidence: {result['model_confidence']:.4f}")
        print(f"   Probabilities: Valid={result['probability_valid']:.4f}, Invalid={result['probability_invalid']:.4f}")
        print()
        
        if result['model_confidence'] <= classifier.confidence_threshold:
            low_confidence_count += 1
            print(f"🚨 Low Confidence - {'Using LLM Validation' if classifier.use_llm else 'Would Use LLM if Available'}")
            if classifier.use_llm:
                print(f"🤖 LLM Results:")
                print(f"   Prediction: {result.get('final_prediction', 'N/A')}")
                print(f"   Confidence: {result.get('llm_confidence', 'N/A')}")
                print(f"   Rationale: {result.get('llm_rationale', 'N/A')}")
                print()
        else:
            print(f"✅ High Confidence - Using Model Prediction")
            
        print(f"   Final Action: {result['final_action']}")
        
        # Check if model got it wrong compared to ground truth
        # Ground truth: 0 = invalid, 1 = valid
        # Our expectation is that these should be flagged as INVALID
        expected_prediction = "INVALID"  # All these reviews are labeled as 0 in ground truth
        if result['model_prediction'] != expected_prediction:
            model_errors += 1
            print(f"   ❌ Model disagrees with ground truth!")
        else:
            print(f"   ✅ Model matches ground truth")
            
        print()
    
    print("📊 Analysis Summary:")
    print(f"   Total problematic reviews: {len(problematic_reviews)}")
    print(f"   Low confidence (≤{classifier.confidence_threshold}): {low_confidence_count}")
    print(f"   LLM validation rate: {low_confidence_count/len(problematic_reviews)*100:.1f}%")
    print(f"   Model errors vs ground truth: {model_errors}/{len(problematic_reviews)} ({model_errors/len(problematic_reviews)*100:.1f}%)")
    
    if low_confidence_count == 0:
        print()
        print("💡 Observations:")
        print("   - Model is very confident even on problematic reviews")
        print("   - This suggests the model may be overfitting or too permissive")
        print("   - LLM validation could help catch these edge cases")
        print("   - Consider lowering confidence threshold to 0.60-0.70")
        
    print()
    print("🎯 RECOMMENDATIONS:")
    print("1. Set OPENAI_API_KEY to enable LLM validation:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("2. Consider lowering confidence threshold to 0.60-0.70")
    print("3. Use LLM to catch cases where model is overconfident")
    print("4. Monitor disagreement patterns for model improvement")

if __name__ == "__main__":
    test_real_problematic_reviews()

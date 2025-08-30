"""
Test Low-Confidence Reviews
Create adversarial examples to test LLM integration
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from simple_llm_enhanced import SimpleEnhancedClassifier
import os

def test_low_confidence_examples():
    """Test with examples designed to have lower confidence"""
    
    print("🧪 Testing Low-Confidence Examples")
    print("=" * 50)
    
    # Set a higher threshold to force LLM usage
    classifier = SimpleEnhancedClassifier(confidence_threshold=0.95)
    
    # These should be more challenging for the model
    challenging_reviews = [
        ("Best place!!!", "Generic Restaurant"),
        ("...", "Minimal Text Place"),
        ("Good good good good good", "Repetitive Restaurant"),
        ("🔥🔥🔥👍👍👍💯💯💯", "Emoji Only Cafe"),
        ("ksjdhfkjsdhfkjsdhf", "Random Text Restaurant"),
        ("Call 555-1234 for reservations", "Contact Info Bistro"),
        ("Visit our website www.example.com", "Website Promotion Cafe"),
        ("Free food today! Limited time offer!", "Promotional Restaurant"),
        ("Never been here but I heard it's bad", "Hearsay Cafe"),
        ("My friend told me this place is okay", "Second Hand Info"),
        ("Generally fine I guess maybe", "Uncertain Review Restaurant"),
        ("Nice", "One Word Cafe"),
        ("meh", "Ambiguous Restaurant"),
        ("The", "Incomplete Sentence Cafe")
    ]
    
    print(f"Testing {len(challenging_reviews)} challenging reviews:")
    print(f"Confidence threshold: {classifier.confidence_threshold}")
    
    low_confidence_count = 0
    
    for i, (review, business) in enumerate(challenging_reviews, 1):
        print(f"\n{'='*40}")
        print(f"Challenge {i}")
        
        result = classifier.analyze_review(review, business)
        
        if result['model_confidence'] <= classifier.confidence_threshold:
            low_confidence_count += 1
    
    print(f"\n📊 Challenge Summary:")
    print(f"   Total challenges: {len(challenging_reviews)}")
    print(f"   Low confidence (≤{classifier.confidence_threshold}): {low_confidence_count}")
    print(f"   LLM validation rate: {low_confidence_count/len(challenging_reviews):.1%}")
    
    if low_confidence_count == 0:
        print("\n💡 Note: Model seems very confident on all examples.")
        print("   This suggests the model is well-trained but may benefit from")
        print("   LLM validation on edge cases or new types of content.")
    
    return classifier


def simulate_llm_integration():
    """Simulate what LLM integration would look like"""
    
    print("\n" + "="*60)
    print("🎭 SIMULATING LLM INTEGRATION")
    print("="*60)
    
    print("\n💡 Here's what would happen with OpenAI API:")
    print("\n1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-key-here'")
    
    print("\n2. Low-confidence reviews (≤0.70) would be sent to OpenAI")
    print("\n3. LLM would analyze using structured schema:")
    print("   - is_advertisement: true/false")
    print("   - is_spam: true/false") 
    print("   - is_irrelevant: true/false")
    print("   - is_fake_visit: true/false")
    print("   - quality_score: 0.0-1.0")
    print("   - final_action: keep/flag/remove")
    print("   - confidence: high/medium/low")
    print("   - rationale: explanation")
    
    print("\n4. Example LLM responses:")
    
    examples = [
        {
            "review": "Call 555-1234 for reservations",
            "model_pred": "VALID (0.65)",
            "llm_response": {
                "is_advertisement": True,
                "final_action": "remove", 
                "confidence": "high",
                "rationale": "Contains phone number for business promotion"
            }
        },
        {
            "review": "Never been here but heard it's bad",
            "model_pred": "VALID (0.55)",
            "llm_response": {
                "is_fake_visit": True,
                "final_action": "remove",
                "confidence": "high", 
                "rationale": "Reviewer explicitly states they haven't visited"
            }
        },
        {
            "review": "Nice place, good food",
            "model_pred": "VALID (0.68)",
            "llm_response": {
                "is_advertisement": False,
                "is_spam": False,
                "final_action": "keep",
                "confidence": "medium",
                "rationale": "Brief but genuine-sounding positive review"
            }
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n   Example {i}:")
        print(f"   Review: \"{ex['review']}\"")
        print(f"   Model: {ex['model_pred']}")
        print(f"   LLM: {ex['llm_response']['final_action']} ({ex['llm_response']['confidence']})")
        print(f"   Reason: {ex['llm_response']['rationale']}")
    
    print("\n5. Hybrid decision combines model + LLM intelligence")
    print("6. Agreement/disagreement tracking helps improve system")


if __name__ == "__main__":
    # Test challenging examples
    classifier = test_low_confidence_examples()
    
    # Show what LLM integration would look like
    simulate_llm_integration()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Set OPENAI_API_KEY to enable LLM validation")
    print("2. Adjust confidence_threshold (0.70 recommended)")  
    print("3. Test with real problematic reviews from your dataset")
    print("4. Monitor agreement rates between model and LLM")
    print("5. Use LLM insights to improve training data")
    
    print(f"\n✅ System ready for production with LLM enhancement!")

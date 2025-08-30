#!/usr/bin/env python3
"""
Demonstrate LLM validation by lowering confidence threshold
This shows how the hybrid system would work with an API key
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testing.simple_llm_enhanced import SimpleEnhancedClassifier

def demonstrate_llm_integration():
    """Demonstrate how LLM validation would work"""
    
    print("🚀 Demonstrating LLM Integration")
    print("="*50)
    
    # Lower threshold to trigger LLM validation
    classifier = SimpleEnhancedClassifier(confidence_threshold=0.99)  # Very high to trigger on most
    
    # A few problematic examples
    test_cases = [
        {
            "text": "Great.",
            "business_name": "One Word Restaurant",
            "why_problematic": "Too brief, no meaningful content"
        },
        {
            "text": "Awesomeee",
            "business_name": "Awesome Restaurant", 
            "why_problematic": "Single word with spelling variation"
        },
        {
            "text": "The most f/p of all businesses I've seen.",
            "business_name": "Generic Restaurant",
            "why_problematic": "Abbreviations and unclear meaning"
        }
    ]
    
    print(f"🤖 Loading model with threshold: {classifier.confidence_threshold}")
    print(f"⚠️ LLM validation {'enabled' if classifier.use_llm else 'disabled (no API key)'}")
    print()
    
    for i, case in enumerate(test_cases, 1):
        print(f"📝 Test Case {i}: {case['text']}")
        print(f"   Why problematic: {case['why_problematic']}")
        
        result = classifier.predict_single(case['text'], case['business_name'])
        
        print(f"   Model: {result['model_prediction']} ({result['model_confidence']:.4f})")
        
        if result['used_llm']:
            print(f"   🧠 LLM: {result['final_prediction']} ({result.get('llm_confidence', 'N/A')})")
            print(f"   Agreement: {result.get('agreement_status', 'N/A')}")
            print(f"   Action: {result['final_action']}")
        else:
            print(f"   Would trigger LLM validation (confidence {result['model_confidence']:.4f} ≤ {classifier.confidence_threshold})")
            print(f"   Action: {result['final_action']}")
        print()
    
    # Show statistics
    stats = classifier.stats
    print("📊 Statistics:")
    print(f"   Total processed: {stats['total_processed']}")
    print(f"   LLM validated: {stats['llm_validated']}")
    print(f"   Model confident: {stats['model_confident']}")
    if stats['llm_validated'] > 0:
        print(f"   Agreements: {stats['agreements']}")
        print(f"   Disagreements: {stats['disagreements']}")
    
    print()
    print("💡 WHAT WOULD HAPPEN WITH OPENAI API:")
    print()
    print("1. Low confidence reviews sent to OpenAI")
    print("2. LLM analyzes using structured schema:")
    print("   - is_advertisement: true/false")
    print("   - is_spam: true/false") 
    print("   - is_irrelevant: true/false")
    print("   - is_fake_visit: true/false")
    print("   - quality_score: 0.0-1.0")
    print("   - final_action: keep/flag/remove")
    print("   - confidence: high/medium/low")
    print("   - rationale: explanation")
    print()
    print("3. Example LLM responses:")
    print()
    print("   Review: 'Great.'")
    print("   LLM Analysis: {")
    print("     'is_spam': false,")
    print("     'is_irrelevant': true,")
    print("     'quality_score': 0.1,")
    print("     'final_action': 'remove',")
    print("     'confidence': 'high',")
    print("     'rationale': 'Too brief to provide meaningful information'")
    print("   }")
    print()
    print("   Review: 'The most f/p of all businesses I've seen.'")
    print("   LLM Analysis: {")
    print("     'is_spam': false,")
    print("     'is_irrelevant': true,") 
    print("     'quality_score': 0.2,")
    print("     'final_action': 'remove',")
    print("     'confidence': 'high',")
    print("     'rationale': 'Contains abbreviations making it unclear'")
    print("   }")
    print()
    print("🎯 KEY INSIGHT:")
    print("Your model is actually very good at identifying legitimate reviews!")
    print("The 'problematic' examples it misses are edge cases where LLM")
    print("validation can provide the nuanced judgment needed.")
    print()
    print("🔧 TO ENABLE FULL SYSTEM:")
    print("export OPENAI_API_KEY='your-openai-api-key'")
    print("Then rerun with confidence_threshold=0.70")

if __name__ == "__main__":
    demonstrate_llm_integration()

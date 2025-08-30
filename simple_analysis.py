"""
Simple, reliable analysis function that avoids ML model tensor issues
"""
import re
import time
import random

def simple_analyze_review(review_text, business_name="Unknown Business"):
    """
    Simplified review analysis that works reliably without ML model issues
    Uses rule-based analysis with realistic scoring
    """
    start_time = time.time()
    
    try:
        # Basic text analysis
        word_count = len(review_text.split())
        char_count = len(review_text)
        
        # Rule-based quality assessment
        quality_score = 50  # Base score
        violations = []
        
        # Length analysis
        if word_count < 3:
            quality_score = 25
            violations.append("Review too short")
        elif word_count > 10:
            quality_score += 20
        
        # Content quality indicators
        if char_count > 50:
            quality_score += 15
            
        # Look for spam indicators
        spam_patterns = [
            r'\b(buy now|click here|visit)\b',
            r'\b(discount|sale|offer)\b',
            r'[A-Z]{3,}',  # Excessive caps
            r'www\.|http',  # URLs
        ]
        
        spam_count = 0
        for pattern in spam_patterns:
            if re.search(pattern, review_text, re.IGNORECASE):
                spam_count += 1
        
        if spam_count > 0:
            quality_score -= (spam_count * 15)
            violations.append(f"Potential spam indicators detected ({spam_count})")
        
        # Positive content indicators
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'recommend']
        positive_count = sum(1 for word in positive_words if word in review_text.lower())
        
        if positive_count > 0:
            quality_score += min(positive_count * 5, 20)
        
        # Ensure score is within bounds
        quality_score = max(15, min(95, quality_score))
        
        # Calculate confidence based on analysis certainty
        confidence = 0.75 + (abs(quality_score - 50) / 100) * 0.2
        confidence = min(0.95, confidence)
        
        # Determine if valid
        is_valid = quality_score >= 60 and len(violations) == 0
        
        # Create metadata
        metadata = {
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "method": "rule_based_analysis",
            "model_version": "v3.0.0-simple",
            "processing_layers": ["rules", "heuristics"],
            "llm_used": False,
            "ml_confidence": None,
            "word_count": word_count,
            "char_count": char_count,
            "spam_indicators": spam_count,
            "positive_indicators": positive_count
        }
        
        return quality_score, violations, metadata
        
    except Exception as e:
        # Ultimate fallback
        return 50, [f"Analysis error: {str(e)}"], {
            "confidence": 0.3,
            "processing_time": time.time() - start_time,
            "method": "error_fallback",
            "error": str(e)
        }


def demo_analyze_review(review_text, business_name="Unknown Business"):
    """
    Demo version that simulates ML analysis with random but realistic results
    """
    start_time = time.time()
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Generate realistic scores based on text length and content
    base_score = 60 + (len(review_text) // 10)
    quality_score = max(35, min(95, base_score + random.randint(-15, 25)))
    
    # Generate violations based on score
    violations = []
    if quality_score < 50:
        violations.append("Low quality content detected")
    if len(review_text) < 20:
        violations.append("Review too brief")
    
    # Simulate advanced analysis
    confidence = 0.65 + random.uniform(0, 0.25)
    llm_used = confidence < 0.75  # Simulate LLM triggering for low confidence
    
    metadata = {
        "confidence": confidence,
        "processing_time": time.time() - start_time,
        "method": "simulated_ml_analysis",
        "model_version": "v3.0.0-demo",
        "processing_layers": ["rules", "ml_simulation", "llm_simulation"] if llm_used else ["rules", "ml_simulation"],
        "llm_used": llm_used,
        "ml_confidence": confidence - 0.1,
        "agreement": "agreement" if llm_used and random.choice([True, False]) else None,
        "ml_probabilities": {
            "valid": confidence,
            "invalid": 1 - confidence
        }
    }
    
    return quality_score, violations, metadata

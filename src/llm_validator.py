"""
LLM-Enhanced Review Validation System
Integrates OpenAI API with structured outputs for low-confidence model predictions
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
import logging
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file (override existing ones)
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReviewContext:
    """Context information for a review"""
    business_name: str
    business_category: str = "Restaurant"
    city: str = "Unknown"
    country: str = "Unknown"
    rating: Optional[float] = None


class LLMReviewJudge:
    """OpenAI-powered review validation with structured outputs"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM judge
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        
        if not (api_key or os.getenv("OPENAI_API_KEY")):
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        # Define the structured output schema
        self.policy_schema = {
            "name": "ReviewPolicyDecision",
            "schema": {
                "type": "object",
                "properties": {
                    "is_advertisement": {
                        "type": "boolean",
                        "description": "Contains promotional content, contact info, or business solicitation"
                    },
                    "is_spam": {
                        "type": "boolean",
                        "description": "Generic template text, nonsensical content, or obvious fake review"
                    },
                    "is_irrelevant": {
                        "type": "boolean",
                        "description": "Content not about this specific location"
                    },
                    "is_fake_visit": {
                        "type": "boolean",
                        "description": "Clear signs reviewer never actually visited (contradictions, impossible details)"
                    },
                    "quality_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Overall quality/trustworthiness (0=very poor, 1=excellent)"
                    },
                    "final_action": {
                        "type": "string",
                        "enum": ["keep", "flag", "remove"],
                        "description": "Recommended action: keep (publish), flag (human review), remove (filter out)"
                    },
                    "confidence": {
                        "type": "string", 
                        "enum": ["high", "medium", "low"],
                        "description": "How confident you are in this decision"
                    },
                    "main_issue": {
                        "type": "string",
                        "enum": ["none", "advertisement", "spam", "irrelevant", "fake_visit", "low_quality"],
                        "description": "Primary problem with the review, if any"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief reason for the decision",
                        "maxLength": 150
                    }
                },
                "required": [
                    "is_advertisement", "is_spam", "is_irrelevant", "is_fake_visit", 
                    "quality_score", "final_action", "confidence", "main_issue", "rationale"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
        
        # System prompt
        self.system_prompt = """You are an expert content moderator specializing in location review authenticity. Your job is to evaluate reviews for trustworthiness and policy violations with precision and nuance.

CRITICAL: Return ONLY valid JSON matching the provided schema. No explanations, no extra text.

DECISION GUIDELINES:
KEEP (quality_score 0.7-1.0):
- Genuine personal experience with specific details
- Relevant to the business type and location
- May be negative but shows actual visit
- Note: Many reviews are from non-native English speakers - imperfect grammar, translations, or broken English does NOT indicate spam if content is meaningful and relevant

FLAG (quality_score 0.4-0.6):
- Borderline cases, unclear authenticity
- Minor policy concerns but potentially legitimate
- Needs human judgment

REMOVE (quality_score 0.0-0.3):
- Clear policy violations
- Obviously fake or promotional
- Harmful or completely irrelevant

Confidence Levels:
- High: Very obvious decision, clear evidence
- Medium: Strong indicators but some ambiguity
- Low: Difficult case, could go either way

IMPORTANT DISTINCTION - Broken English vs Nonsensical Spam:
- Broken English (KEEP): Grammar errors but meaningful content - "The food was good but service very slow waiting one hour"
- Nonsensical Spam (REMOVE): Incoherent phrases regardless of grammar - "Folk hero is a business" or "Amazing experience with the beautiful of taste"

Focus on whether the content makes logical sense, not grammar perfection."""

    def build_user_prompt(self, review_text: str, context: ReviewContext) -> str:
        """Build the user prompt for review evaluation"""
        
        return f"""POLICIES TO ENFORCE:
Advertisement: Promotional content, discount codes, phone numbers, links, business solicitation, "DM me", competitor mentions for promotion
Spam: Generic template language, nonsensical/incoherent text, meaningless phrases, copy-paste indicators, obvious bot/AI-generated content, repetitive phrases, reviews that don't make logical sense, overly perfect structure, emotionally flat language despite strong claims. Note: Do NOT flag reviews simply for imperfect grammar or non-native English - focus on meaningless or incoherent content.
Irrelevant: Content not about this specific location, talks about different business/place, completely off-topic
Fake Visit: Clear signs reviewer never visited - impossible timelines, contradictory details, "never been but heard", secondhand information presented as personal experience

LOCATION CONTEXT:
Business: {context.business_name}
Category: {context.business_category}
Location: {context.city}, {context.country}

REVIEW TO EVALUATE:
{review_text}

OUTPUT ONLY JSON - NO OTHER TEXT"""

    def judge_review(self, review_text: str, context: ReviewContext) -> Dict[str, Any]:
        """
        Judge a single review using OpenAI API with structured outputs
        
        Args:
            review_text: The review text to evaluate
            context: Business context for the review
            
        Returns:
            Dictionary containing structured judgment
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_user_prompt(review_text, context)}
            ]
            
            logger.info(f"Sending review to LLM for judgment: {review_text[:50]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": self.policy_schema
                }
            )
            
            # Extract and parse the JSON response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Add metadata
            result["source"] = "llm-judge"
            result["model"] = self.model
            result["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"LLM judgment: {result['final_action']} (confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM judgment: {e}")
            # Return a safe fallback
            return {
                "source": "llm-judge",
                "error": str(e),
                "final_action": "flag",
                "confidence": "low",
                "main_issue": "processing_error",
                "rationale": f"Error during LLM processing: {str(e)[:100]}",
                "quality_score": 0.5,
                "is_advertisement": False,
                "is_spam": False,
                "is_irrelevant": False,
                "is_fake_visit": False
            }


class HybridReviewValidator:
    """
    Hybrid validation system that uses trained model + LLM fallback
    """
    
    def __init__(self, llm_judge: LLMReviewJudge, confidence_threshold: float = 0.70):
        """
        Initialize hybrid validator
        
        Args:
            llm_judge: LLM judge instance
            confidence_threshold: Threshold below which to use LLM validation
        """
        self.llm_judge = llm_judge
        self.confidence_threshold = confidence_threshold
        self.validation_stats = {
            "total_reviews": 0,
            "model_confident": 0,
            "llm_validated": 0,
            "agreements": 0,
            "disagreements": 0
        }
    
    def validate_review(
        self, 
        review_text: str, 
        model_prediction: str, 
        model_confidence: float, 
        context: ReviewContext
    ) -> Dict[str, Any]:
        """
        Validate a review using hybrid approach
        
        Args:
            review_text: The review text
            model_prediction: Model's prediction (VALID/INVALID)
            model_confidence: Model's confidence score (0-1)
            context: Business context
            
        Returns:
            Final validation decision with metadata
        """
        self.validation_stats["total_reviews"] += 1
        
        # If model is confident, use its prediction
        if model_confidence >= self.confidence_threshold:
            self.validation_stats["model_confident"] += 1
            
            return {
                "source": "trained-model",
                "final_decision": model_prediction.lower(),
                "final_action": "keep" if model_prediction.upper() == "VALID" else "remove",
                "confidence_score": model_confidence,
                "rationale": f"High-confidence model prediction ({model_confidence:.3f})",
                "used_llm": False,
                "model_prediction": model_prediction,
                "model_confidence": model_confidence
            }
        
        # Otherwise, use LLM validation
        self.validation_stats["llm_validated"] += 1
        logger.info(f"Low confidence ({model_confidence:.3f}) - using LLM validation")
        
        llm_result = self.llm_judge.judge_review(review_text, context)
        
        # Map LLM action to model format
        llm_prediction = "VALID" if llm_result["final_action"] == "keep" else "INVALID"
        
        # Check agreement between model and LLM
        if model_prediction.upper() == llm_prediction:
            self.validation_stats["agreements"] += 1
            agreement_status = "agreement"
        else:
            self.validation_stats["disagreements"] += 1
            agreement_status = "disagreement"
        
        # Create final result
        result = {
            "source": "hybrid-validation",
            "final_decision": llm_prediction.lower(),
            "final_action": llm_result["final_action"],
            "confidence_score": model_confidence,
            "llm_confidence": llm_result["confidence"],
            "rationale": llm_result["rationale"],
            "used_llm": True,
            "model_prediction": model_prediction,
            "model_confidence": model_confidence,
            "llm_prediction": llm_prediction,
            "agreement_status": agreement_status,
            "llm_details": llm_result
        }
        
        logger.info(f"Hybrid result: {llm_prediction} (LLM confidence: {llm_result['confidence']}, Agreement: {agreement_status})")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats["total_reviews"] > 0:
            stats["model_confidence_rate"] = float(stats["model_confident"]) / stats["total_reviews"]
            stats["llm_usage_rate"] = float(stats["llm_validated"]) / stats["total_reviews"]
            
            if stats["llm_validated"] > 0:
                stats["agreement_rate"] = float(stats["agreements"]) / stats["llm_validated"]
                stats["disagreement_rate"] = float(stats["disagreements"]) / stats["llm_validated"]
        
        return stats
    
    def print_stats(self):
        """Print validation statistics"""
        stats = self.get_stats()
        print("\n📊 Hybrid Validation Statistics:")
        print("=" * 40)
        print(f"Total reviews processed: {stats['total_reviews']}")
        print(f"Model confident decisions: {stats['model_confident']} ({stats.get('model_confidence_rate', 0):.1%})")
        print(f"LLM validations needed: {stats['llm_validated']} ({stats.get('llm_usage_rate', 0):.1%})")
        
        if stats['llm_validated'] > 0:
            print(f"Model-LLM agreements: {stats['agreements']} ({stats.get('agreement_rate', 0):.1%})")
            print(f"Model-LLM disagreements: {stats['disagreements']} ({stats.get('disagreement_rate', 0):.1%})")


def test_llm_integration():
    """Test the LLM integration with sample reviews"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY environment variable not set")
        print("Please set it before running LLM validation:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🧪 Testing LLM Integration")
    print("=" * 40)
    
    # Initialize components
    llm_judge = LLMReviewJudge()
    validator = HybridReviewValidator(llm_judge, confidence_threshold=0.70)
    
    # Test reviews with various confidence levels
    test_cases = [
        {
            "review": "Great food and excellent service! The pasta was perfectly cooked.",
            "model_pred": "VALID",
            "model_conf": 0.95,  # High confidence - should use model
            "context": ReviewContext("Mario's Restaurant", "Italian Restaurant", "Istanbul", "Turkey")
        },
        {
            "review": "Call us now for discount! Special offer available!",
            "model_pred": "INVALID", 
            "model_conf": 0.60,  # Low confidence - should use LLM
            "context": ReviewContext("Taste Bistro", "Restaurant", "Istanbul", "Turkey")
        },
        {
            "review": "Ok.",
            "model_pred": "VALID",
            "model_conf": 0.65,  # Low confidence - should use LLM
            "context": ReviewContext("Local Cafe", "Cafe", "Istanbul", "Turkey")
        },
        {
            "review": "Folk hero is a business every time amazing experience.",
            "model_pred": "VALID",
            "model_conf": 0.55,  # Low confidence - should use LLM
            "context": ReviewContext("Heritage Restaurant", "Restaurant", "Istanbul", "Turkey")
        }
    ]
    
    print(f"Testing with confidence threshold: {validator.confidence_threshold}")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: \"{case['review'][:50]}{'...' if len(case['review']) > 50 else ''}\"")
        print(f"   Model: {case['model_pred']} (confidence: {case['model_conf']:.3f})")
        
        result = validator.validate_review(
            case["review"],
            case["model_pred"], 
            case["model_conf"],
            case["context"]
        )
        
        print(f"   Final: {result['final_decision'].upper()} ({result['source']})")
        if result['used_llm']:
            print(f"   LLM Confidence: {result['llm_confidence']}")
            print(f"   Agreement: {result['agreement_status']}")
            print(f"   Rationale: {result['rationale']}")
    
    # Print statistics
    validator.print_stats()


if __name__ == "__main__":
    test_llm_integration()

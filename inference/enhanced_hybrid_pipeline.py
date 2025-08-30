"""
Enhanced Review Classification Pipeline with LLM Integration
Combines rule-based filtering, ML model, and LLM validation for comprehensive review analysis.
"""

import os
import sys
import torch
from typing import Dict, Tuple, Optional, Any
import warnings

# Add project root to path
sys.path.append('.')

# Import components
try:
    from hybrid_pipeline.hybrid_pipeline import HybridReviewClassificationPipeline
    from training_scripts.model_setup import load_model
    from src.policy_module import PolicyModule
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import pipeline components: {e}")
    PIPELINE_AVAILABLE = False

# LLM integration imports
try:
    import openai
    from openai import OpenAI
    import json
    from dotenv import load_dotenv
    LLM_AVAILABLE = True
    
    # Load environment variables
    load_dotenv()
    
    # Get API key with fallback
    api_key = os.getenv('OPENAI_API_KEY')
    
    # If the key is a placeholder, try to read from .env file directly
    if not api_key or api_key == 'your-actual-api-key-here':
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip()
                        break
        except Exception:
            pass
    
    # Initialize OpenAI client
    if api_key and api_key.startswith('sk-'):
        openai_client = OpenAI(api_key=api_key)
        print(f"✅ OpenAI client initialized successfully")
    else:
        openai_client = None
        LLM_AVAILABLE = False
        print(f"⚠️ Invalid or missing OpenAI API key")
    
except ImportError as e:
    print(f"⚠️ Warning: LLM functionality not available: {e}")
    LLM_AVAILABLE = False
    openai_client = None

class ReviewContext:
    """Context class for review validation."""
    def __init__(self, business_name: str, business_category: str = "Restaurant", city: str = "Unknown"):
        self.business_name = business_name
        self.business_category = business_category
        self.city = city

class EnhancedReviewClassificationPipeline:
    """
    Enhanced 3-layer classification pipeline:
    1. Rule-based filtering (basic checks)
    2. ML model prediction (RoBERTa)
    3. LLM validation (for low-confidence cases)
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.70, use_llm: bool = True):
        """Initialize the enhanced pipeline."""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_llm = use_llm and LLM_AVAILABLE
        
        # Initialize components
        self.policy_module = None
        self.model = None
        self.tokenizer = None
        self.llm_validator = openai_client if LLM_AVAILABLE else None
        
        # Load ML model
        self._load_ml_model()
        
        print(f"✅ Enhanced pipeline initialized (LLM: {'enabled' if self.use_llm else 'disabled'})")
    
    def _load_ml_model(self):
        """Load the ML model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            # Load policy module if available
            try:
                self.policy_module = PolicyModule()
            except Exception as e:
                print(f"⚠️ Policy module not available: {e}")
            
            print(f"✅ Loaded ML model from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            self.model = None
            self.tokenizer = None
    
    def rule_based_filter(self, text: str) -> Tuple[Optional[str], float, str]:
        """First layer: Basic rule-based filtering."""
        if not text or len(text.strip()) < 5:
            return "fake", 0.95, "Too short"
        
        if len(text) > 5000:
            return "fake", 0.90, "Too long"
        
        # Check for spam patterns
        spam_keywords = ['buy now', 'click here', 'free money', 'guaranteed', 'no risk']
        text_lower = text.lower()
        spam_count = sum(1 for keyword in spam_keywords if keyword in text_lower)
        
        if spam_count >= 2:
            return "fake", 0.85, "Spam patterns detected"
        
        # Check for excessive capitalization
        if len(text) > 50:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.7:
                return "fake", 0.80, "Excessive capitalization"
        
        return None, 0.0, "Passed rule-based filter"
    
    def ml_predict(self, text: str) -> Tuple[str, float]:
        """Second layer: ML model prediction."""
        if not self.model or not self.tokenizer:
            # Fallback to simple heuristics when model is unavailable
            print("⚠️ Using fallback heuristics - ML model not available")
            if any(word in text.lower() for word in ['excellent', 'amazing', 'perfect', 'love', 'great', 'wonderful', 'fantastic']):
                return "real", 0.65
            elif any(word in text.lower() for word in ['terrible', 'awful', 'worst', 'hate', 'horrible', 'disgusting']):
                return "real", 0.60  # Negative reviews are also real reviews
            elif any(word in text.lower() for word in ['buy now', 'click here', 'free money', 'guaranteed']):
                return "fake", 0.75  # Spam patterns
            else:
                return "real", 0.45  # Default to real but low confidence
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Check if we have valid tensors
                if logits.is_meta:
                    raise RuntimeError("Model returned meta tensors - model not properly loaded")
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get predicted class and confidence with proper error handling
                predicted_class_id = torch.argmax(probabilities, dim=-1)
                confidence_tensor = torch.max(probabilities, dim=-1)[0]
                
                # Safely extract values
                predicted_class = predicted_class_id.cpu().item() if predicted_class_id.numel() > 0 else 0
                confidence = confidence_tensor.cpu().item() if confidence_tensor.numel() > 0 else 0.5
                
                # Map class ID to label (assuming 0=fake, 1=real)
                prediction = "real" if predicted_class == 1 else "fake"
                
                return prediction, confidence
                
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            # Better fallback when model fails
            if any(word in text.lower() for word in ['buy now', 'click here', 'free money', 'guaranteed']):
                return "fake", 0.70
            else:
                return "real", 0.40  # Default to real but low confidence when uncertain
    
    def llm_validate(self, text: str, ml_prediction: str, ml_confidence: float, business_name: str = "Unknown Business") -> Tuple[str, float, str]:
        """Third layer: LLM validation for low-confidence cases."""
        if not self.use_llm or not LLM_AVAILABLE:
            return ml_prediction, ml_confidence, "LLM unavailable"
        
        try:
            # Create review context
            context = ReviewContext(
                business_name=business_name,
                business_category="Restaurant",
                city="Unknown"
            )
            
            # Prepare LLM prompt
            prompt = f"""
            Analyze this restaurant review for authenticity. Focus on obvious spam signals:
            - Commercial language (buy now, click here, promotional content)
            - Completely irrelevant content (not about restaurant/food)
            - Obvious bot patterns (repetitive phrases, template-like structure)
            
            IMPORTANT: Regular customer reviews (positive, negative, or neutral) should be classified as "real" even if they are:
            - Brief or generic ("good food", "nice place")
            - Emotional ("terrible service", "amazing experience")
            - Simple opinions without detailed descriptions
            
            Only classify as "fake" if there are clear spam/promotional signals.
            
            Restaurant: {context.business_name}
            Review: "{text}"
            
            ML Model Prediction: {ml_prediction} (confidence: {ml_confidence:.2f})
            
            Respond with JSON:
            {{
                "classification": "real" or "fake",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            }}
            """
            
            # Call LLM
            response = self.llm_validator.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at detecting fake restaurant reviews. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            return result["classification"], result["confidence"], result["reasoning"]
            
        except Exception as e:
            print(f"❌ LLM validation error: {e}")
            return ml_prediction, ml_confidence, f"LLM error: {str(e)}"
    
    def classify(self, text: str, business_name: str = "Unknown Business") -> Dict[str, Any]:
        """
        Complete classification pipeline.
        Returns comprehensive analysis results.
        """
        results = {
            "text": text,
            "business_name": business_name,
            "layers": {},
            "final_prediction": None,
            "final_confidence": 0.0,
            "reasoning": ""
        }
        
        # Layer 1: Rule-based filtering
        rule_prediction, rule_confidence, rule_reason = self.rule_based_filter(text)
        results["layers"]["rule_based"] = {
            "prediction": rule_prediction,
            "confidence": rule_confidence,
            "reasoning": rule_reason
        }
        
        # If rule-based filter gives strong signal, use it
        if rule_prediction and rule_confidence > 0.80:
            results["final_prediction"] = rule_prediction
            results["final_confidence"] = rule_confidence
            results["reasoning"] = f"Rule-based: {rule_reason}"
            return results
        
        # Layer 2: ML model prediction
        ml_prediction, ml_confidence = self.ml_predict(text)
        results["layers"]["ml_model"] = {
            "prediction": ml_prediction,
            "confidence": ml_confidence,
            "reasoning": "ML model classification"
        }
        
        # If ML confidence is high, use it
        if ml_confidence >= self.confidence_threshold:
            results["final_prediction"] = ml_prediction
            results["final_confidence"] = ml_confidence
            results["reasoning"] = f"ML model (confidence: {ml_confidence:.2f})"
            return results
        
        # Layer 3: LLM validation for low-confidence cases
        llm_prediction, llm_confidence, llm_reason = self.llm_validate(
            text, ml_prediction, ml_confidence, business_name
        )
        results["layers"]["llm_validation"] = {
            "prediction": llm_prediction,
            "confidence": llm_confidence,
            "reasoning": llm_reason
        }
        
        # Use LLM result as final
        results["final_prediction"] = llm_prediction
        results["final_confidence"] = llm_confidence
        results["reasoning"] = f"LLM validation: {llm_reason}"
        
        return results
    
    def batch_classify(self, reviews: list, business_names: list = None) -> list:
        """Classify multiple reviews efficiently."""
        if business_names is None:
            business_names = ["Unknown Business"] * len(reviews)
        
        results = []
        for i, review in enumerate(reviews):
            business_name = business_names[i] if i < len(business_names) else "Unknown Business"
            result = self.classify(review, business_name)
            results.append(result)
        
        return results

# Factory function for easy instantiation
def create_enhanced_pipeline(model_path: str = "./models/roberta_policy_based_model", 
                           confidence_threshold: float = 0.70,
                           use_llm: bool = True) -> EnhancedReviewClassificationPipeline:
    """Create and return an enhanced pipeline instance."""
    return EnhancedReviewClassificationPipeline(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_llm=use_llm
    )

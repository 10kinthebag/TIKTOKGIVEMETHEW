"""
Multi-Model Testing Suite
Flexible script to test any of your trained models with easy model selection
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path


class ModelManager:
    """Manages multiple trained models and provides easy selection interface"""
    
    def __init__(self, models_dir: Optional[str] = None):
        # Auto-detect models directory based on current location
        if models_dir is None:
            current_dir = Path.cwd()
            if current_dir.name == "testing":
                models_dir = "../models"
            else:
                models_dir = "./models"
        
        self.models_dir = Path(models_dir)
        self.available_models = self._discover_models()
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
    
    def _discover_models(self) -> Dict[str, Dict[str, str]]:
        """Automatically discover all available models in the models directory"""
        models = {}
        
        if not self.models_dir.exists():
            print(f"❌ Models directory not found: {self.models_dir}")
            return models
        
        # Check each subdirectory for model files
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
                tokenizer_files = ['tokenizer_config.json', 'vocab.txt', 'tokenizer.json']
                
                # Check if it's a valid model directory
                has_model = any((model_dir / f).exists() for f in model_files)
                has_tokenizer = any((model_dir / f).exists() for f in tokenizer_files)
                
                if has_model and has_tokenizer:
                    # Get model info
                    info = self._get_model_info(model_dir)
                    models[model_dir.name] = {
                        'path': str(model_dir),
                        'name': model_dir.name,
                        'info': info
                    }
        
        return models
    
    def _get_model_info(self, model_dir: Path) -> Dict[str, str]:
        """Extract model information from config files"""
        info = {
            'architecture': 'Unknown',
            'size': 'Unknown',
            'description': 'No description available'
        }
        
        config_path = model_dir / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                info['architecture'] = config.get('model_type', 'Unknown')
                
                # Estimate size from vocab size and hidden size
                vocab_size = config.get('vocab_size', 0)
                hidden_size = config.get('hidden_size', 0)
                if vocab_size and hidden_size:
                    estimated_params = (vocab_size * hidden_size) / 1e6
                    info['size'] = f"~{estimated_params:.1f}M parameters"
            except:
                pass
        
        # Add custom descriptions based on model names
        model_descriptions = {
            'final_model': 'Latest trained model with best performance',
            'intial_model': 'First baseline model version',
            'policy_based_model': 'Model trained with policy-based filtering',
            'roberta_policy_based_model': 'RoBERTa model with policy-based training',
            'combined_training': 'Model trained on combined dataset (policy + ground truth)'
        }
        
        model_name = model_dir.name
        if model_name in model_descriptions:
            info['description'] = model_descriptions[model_name]
        
        return info
    
    def list_models(self) -> None:
        """Display all available models"""
        print("\n🤖 Available Models:")
        print("=" * 70)
        
        if not self.available_models:
            print("❌ No trained models found in the models directory")
            print(f"💡 Make sure you have trained models in: {self.models_dir}")
            return
        
        for i, (name, details) in enumerate(self.available_models.items(), 1):
            print(f"\n{i}. {name}")
            print(f"   📍 Path: {details['path']}")
            info = details['info']
            print(f"   🏗️  Architecture: {info['architecture']}")
            print(f"   📊 Size: {info['size']}")
            print(f"   📝 Description: {info['description']}")
    
    def select_model_interactive(self) -> Optional[str]:
        """Interactive model selection"""
        self.list_models()
        
        if not self.available_models:
            return None
        
        print(f"\n🎯 Select a model to test:")
        model_names = list(self.available_models.keys())
        
        while True:
            try:
                choice = input(f"\nEnter model number (1-{len(model_names)}) or model name: ").strip()
                
                # Check if it's a number
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(model_names):
                        return model_names[choice_num - 1]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(model_names)}")
                
                # Check if it's a model name
                elif choice in model_names:
                    return choice
                
                # Check for partial matches
                else:
                    matches = [name for name in model_names if choice.lower() in name.lower()]
                    if len(matches) == 1:
                        return matches[0]
                    elif len(matches) > 1:
                        print(f"❌ Multiple matches found: {', '.join(matches)}")
                        print("Please be more specific.")
                    else:
                        print(f"❌ Model '{choice}' not found")
                        print(f"Available models: {', '.join(model_names)}")
            
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        if model_name not in self.available_models:
            print(f"❌ Model '{model_name}' not found")
            return False
        
        model_path = self.available_models[model_name]['path']
        print(f"\n🔄 Loading model: {model_name}")
        print(f"📍 Path: {model_path}")
        
        try:
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.current_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.current_model_name = model_name
            
            # Count parameters
            total_params = sum(p.numel() for p in self.current_model.parameters())
            print(f"✅ Model loaded successfully!")
            print(f"📊 Total parameters: {total_params/1e6:.1f}M")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, text: str, return_details: bool = False) -> Dict:
        """Make prediction with the currently loaded model"""
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Tokenize input
        inputs = self.current_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = self.current_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Define labels (adjust based on your model)
        labels = {0: "INVALID", 1: "VALID"}
        
        result = {
            'text': text,
            'prediction': labels[int(predicted_class)],
            'confidence': confidence,
            'model_used': self.current_model_name
        }
        
        if return_details:
            result.update({
                'probability_invalid': probabilities[0][0].item(),
                'probability_valid': probabilities[0][1].item(),
                'raw_logits': logits[0].tolist()
            })
        
        return result


def run_interactive_testing(model_manager: ModelManager):
    """Run interactive testing mode"""
    print("\n🎮 Interactive Testing Mode")
    print("=" * 50)
    print("Enter restaurant reviews to test (type 'quit' to exit)")
    
    while True:
        try:
            text = input("\n📝 Enter review: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                print("❌ Please enter some text")
                continue
            
            # Make prediction
            result = model_manager.predict(text, return_details=True)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   🎯 Prediction: {result['prediction']}")
            print(f"   🔥 Confidence: {result['confidence']:.4f}")
            print(f"   🤖 Model: {result['model_used']}")
            print(f"   📈 Probabilities:")
            print(f"      Valid: {result['probability_valid']:.4f}")
            print(f"      Invalid: {result['probability_invalid']:.4f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    
    print("\n👋 Interactive testing ended")


def run_batch_testing(model_manager: ModelManager):
    """Run batch testing with sample reviews"""
    print("\n📦 Batch Testing Mode")
    print("=" * 50)
    
    # Sample test reviews
    test_reviews = [
        # Valid reviews
        "Great food and excellent service! The pasta was perfectly cooked.",
        "Amazing Italian restaurant with authentic flavors. Highly recommended!",
        "Good value for money. The portions were generous and tasty.",
        
        # Potentially invalid reviews
        "Best restaurant ever!!!!! 5 stars!!!! 👍👍👍👍👍",
        "Call us now for discount! Special offer available!",
        "Ok.",
        
        # Edge cases
        "The food was decent but nothing special. Service was average.",
        "This place is terrible. Worst experience ever.",
    ]
    
    print(f"Testing {len(test_reviews)} sample reviews...")
    
    results = []
    for i, review in enumerate(test_reviews, 1):
        result = model_manager.predict(review, return_details=True)
        results.append(result)
        
        print(f"\n{i}. {result['prediction']} ({result['confidence']:.3f})")
        print(f"   📝 \"{review[:60]}{'...' if len(review) > 60 else ''}\"")
    
    # Summary
    valid_count = sum(1 for r in results if r['prediction'] == 'VALID')
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print(f"\n📊 Batch Testing Summary:")
    print(f"   Total reviews: {len(results)}")
    print(f"   Valid predictions: {valid_count}")
    print(f"   Invalid predictions: {len(results) - valid_count}")
    print(f"   Average confidence: {avg_confidence:.4f}")
    print(f"   Model used: {model_manager.current_model_name}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Model Restaurant Review Tester')
    parser.add_argument('--model', type=str, 
                       help='Specific model name to use (optional)')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['interactive', 'batch', 'both'],
                       help='Testing mode')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    print("🚀 Multi-Model Restaurant Review Tester")
    print("=" * 60)
    
    # Initialize model manager
    model_manager = ModelManager(args.models_dir)
    
    # Model selection
    if args.model:
        # Use specified model
        if not model_manager.load_model(args.model):
            return
    else:
        # Interactive model selection
        selected_model = model_manager.select_model_interactive()
        if not selected_model:
            print("👋 No model selected. Exiting...")
            return
        
        if not model_manager.load_model(selected_model):
            return
    
    # Run testing
    if args.mode in ['batch', 'both']:
        run_batch_testing(model_manager)
    
    if args.mode in ['interactive', 'both']:
        run_interactive_testing(model_manager)
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()

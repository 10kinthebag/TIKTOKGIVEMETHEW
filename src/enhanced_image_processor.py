"""
Enhanced Image Processing Module for Multi-modal Restaurant Review Analysis
Integrates with the Kaggle dataset and provides category-aware image analysis
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Import the multimodal model
import sys
sys.path.append('/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW')
try:
    from training_scripts.multimodal_model import create_multimodal_model
except ImportError:
    # If import fails, we'll work without the multimodal model
    create_multimodal_model = None


class EnhancedImageProcessor:
    """Enhanced image processor that works with the multi-modal model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the enhanced image processor
        
        Args:
            model_path: Path to trained multi-modal model (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image preprocessing pipeline (matches training)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Load multimodal model if path provided
        self.multimodal_model = None
        if model_path and os.path.exists(model_path):
            self.load_multimodal_model(model_path)
        
        # Category mappings for image analysis
        self.category_descriptions = {
            "taste": "Food quality, presentation, and taste-related visual elements",
            "menu": "Menu boards, price lists, and food variety displays",
            "indoor_atmosphere": "Interior design, seating, lighting, and indoor ambiance",
            "outdoor_atmosphere": "Exterior views, outdoor seating, and environmental context"
        }
        
        print(f"✅ Enhanced Image Processor initialized on {self.device}")
    
    def load_multimodal_model(self, model_path: str):
        """Load a trained multi-modal model"""
        try:
            if create_multimodal_model is not None:
                self.multimodal_model = create_multimodal_model()
                self.multimodal_model.load_state_dict(torch.load(model_path, map_location=self.device))
                # Move to device - fix the syntax
                self.multimodal_model = self.multimodal_model.to(self.device)
                self.multimodal_model.eval()
                print(f"✅ Loaded multi-modal model from {model_path}")
            else:
                print("⚠️ Multi-modal model creation function not available")
                self.multimodal_model = None
        except Exception as e:
            print(f"⚠️ Failed to load multi-modal model: {e}")
            self.multimodal_model = None
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model input
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"⚠️ Error processing image {image_path}: {e}")
            # Return blank tensor
            return torch.zeros(1, 3, 224, 224)
    
    def analyze_image_content(self, image_path: str) -> Dict[str, any]:
        """
        Analyze image content using a pre-trained vision model
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image analysis results
        """
        # Load a pre-trained ResNet model for feature extraction
        resnet = models.resnet50(pretrained=True)
        resnet.eval()
        
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            # Extract features
            features = resnet(image_tensor)
            
            # Get top predictions (this is a simplified example)
            probabilities = torch.nn.functional.softmax(features, dim=1)
            top_prob, top_class = torch.topk(probabilities, 5)
            
        return {
            'image_path': image_path,
            'top_classes': top_class[0].tolist(),
            'top_probabilities': top_prob[0].tolist(),
            'feature_vector': features[0].tolist()[:10]  # First 10 features only
        }
    
    def classify_review_with_image(
        self, 
        text: str, 
        image_path: str
    ) -> Dict[str, any]:
        """
        Classify a restaurant review using both text and image
        
        Args:
            text: Review text
            image_path: Path to associated image
            
        Returns:
            Classification results with confidence scores
        """
        if self.multimodal_model is None:
            return {
                'error': 'Multi-modal model not loaded',
                'text_only_available': True
            }
        
        try:
            # Tokenize text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            image_tensor = image_tensor.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.multimodal_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=image_tensor
                )
                
                logits = outputs['logits']
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)
            
            return {
                'prediction': prediction.item(),
                'confidence': probabilities.max().item(),
                'probabilities': probabilities[0].tolist(),
                'text_features_available': True,
                'image_features_available': True
            }
            
        except Exception as e:
            return {
                'error': f'Classification failed: {e}',
                'text': text,
                'image_path': image_path
            }
    
    def analyze_kaggle_dataset_images(self, sample_size: int = 50) -> Dict[str, any]:
        """
        Analyze a sample of images from the Kaggle dataset
        
        Args:
            sample_size: Number of images to analyze
            
        Returns:
            Analysis results for the image dataset
        """
        csv_path = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/kaggle_data/reviews.csv"
        
        if not os.path.exists(csv_path):
            return {'error': 'Kaggle dataset not found'}
        
        df = pd.read_csv(csv_path)
        
        # Sample random rows
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        results = {
            'total_samples': len(sample_df),
            'categories': {},
            'image_analysis': [],
            'missing_images': []
        }
        
        for _, row in sample_df.iterrows():
            category = row['rating_category']
            image_path = row['photo']
            
            # Count categories
            if category not in results['categories']:
                results['categories'][category] = 0
            results['categories'][category] += 1
            
            # Full image path
            full_image_path = f"/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/kaggle_data/{image_path}"
            
            if os.path.exists(full_image_path):
                # Analyze image
                analysis = self.analyze_image_content(full_image_path)
                analysis['category'] = category
                analysis['text'] = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                results['image_analysis'].append(analysis)
            else:
                results['missing_images'].append(image_path)
        
        return results


def demonstrate_enhanced_image_processing():
    """Demonstrate the enhanced image processing capabilities"""
    print("🎨 Enhanced Image Processing Demonstration")
    print("=" * 50)
    
    # Initialize processor
    processor = EnhancedImageProcessor()
    
    # Analyze Kaggle dataset
    print("\n📊 Analyzing Kaggle dataset images...")
    analysis = processor.analyze_kaggle_dataset_images(sample_size=20)
    
    print(f"✅ Analysis Results:")
    print(f"   📁 Total samples analyzed: {analysis['total_samples']}")
    print(f"   🏷️ Categories found: {list(analysis['categories'].keys())}")
    print(f"   📸 Successful image analyses: {len(analysis['image_analysis'])}")
    print(f"   ❌ Missing images: {len(analysis['missing_images'])}")
    
    # Category breakdown
    print(f"\n📈 Category Distribution:")
    for category, count in analysis['categories'].items():
        print(f"   {category}: {count} samples")
    
    # Show a few example analyses
    print(f"\n🔍 Sample Image Analyses:")
    for i, img_analysis in enumerate(analysis['image_analysis'][:3]):
        print(f"\n   Example {i+1}:")
        print(f"   Category: {img_analysis['category']}")
        print(f"   Text: {img_analysis['text']}")
        print(f"   Top predicted classes: {img_analysis['top_classes'][:3]}")
    
    return analysis


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_enhanced_image_processing()
    print("\n🎉 Enhanced image processing demonstration completed!")

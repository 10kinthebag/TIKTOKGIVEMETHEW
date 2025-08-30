"""
Multi-modal dataset preparation for text + image training
Combines restaurant review text with corresponding images for enhanced model training
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional


class MultiModalRestaurantDataset(Dataset):
    """Dataset class for restaurant reviews with text and images"""
    
    def __init__(
        self, 
        csv_path: str, 
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        data_root: str = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data"
    ):
        """
        Initialize multimodal dataset
        
        Args:
            csv_path: Path to CSV file with text, photo, and labels
            tokenizer: Pre-trained tokenizer for text processing
            max_length: Maximum text sequence length
            image_size: Target image size (height, width)
            data_root: Root directory for data files
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_root = data_root
        
        # Image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])
        
        # Create labels if not present
        if 'label' not in self.df.columns:
            # For now, assume all reviews are valid (label=1)
            # You can modify this based on your labeling strategy
            self.df['label'] = 1
        
        print(f"📊 Loaded {len(self.df)} samples")
        print(f"🏷️ Label distribution: {self.df['label'].value_counts().to_dict()}")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with text and image"""
        row = self.df.iloc[idx]
        
        # Process text
        text = str(row['text'])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process image
        image_tensor = self._load_image(row['photo'])
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'image': image_tensor,
            'labels': label,
            'rating_category': row.get('rating_category', 'unknown')
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            # Handle different path formats
            if image_path.startswith('dataset/'):
                # Kaggle dataset format
                full_path = os.path.join(self.data_root, 'kaggle_data', image_path)
            else:
                # Other formats
                full_path = os.path.join(self.data_root, image_path)
            
            # Load image
            image = Image.open(full_path).convert('RGB')
            return self.image_transform(image)
            
        except Exception as e:
            print(f"⚠️ Error loading image {image_path}: {e}")
            # Return a blank image tensor if loading fails
            return torch.zeros(3, 224, 224)


def prepare_multimodal_datasets(
    kaggle_csv: str = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/kaggle_data/reviews.csv",
    existing_csv: str = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/cleanedData/reviews_cleaned.csv",
    tokenizer_name: str = "roberta-base",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[MultiModalRestaurantDataset, MultiModalRestaurantDataset, MultiModalRestaurantDataset]:
    """
    Prepare train/val/test splits for multimodal training
    
    Args:
        kaggle_csv: Path to Kaggle dataset CSV
        existing_csv: Path to existing cleaned dataset CSV  
        tokenizer_name: Name of tokenizer to use
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load and combine datasets
    print("📂 Loading datasets...")
    
    # Load Kaggle data (has images)
    kaggle_df = pd.read_csv(kaggle_csv)
    kaggle_df['source'] = 'kaggle'
    print(f"✅ Loaded {len(kaggle_df)} Kaggle samples with images")
    
    # For this implementation, we'll focus on the Kaggle data since it has images
    # You can extend this to include text-only data from existing_csv if needed
    
    # Create train/val/test splits
    n_samples = len(kaggle_df)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle the data
    kaggle_df = kaggle_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = kaggle_df[:n_train]
    val_df = kaggle_df[n_train:n_train + n_val]
    test_df = kaggle_df[n_train + n_val:]
    
    print(f"📊 Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save split CSVs
    os.makedirs("/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits", exist_ok=True)
    train_df.to_csv("/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits/train.csv", index=False)
    val_df.to_csv("/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits/val.csv", index=False)
    test_df.to_csv("/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits/test.csv", index=False)
    
    # Create dataset objects
    train_dataset = MultiModalRestaurantDataset(
        "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits/train.csv",
        tokenizer
    )
    
    val_dataset = MultiModalRestaurantDataset(
        "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits/val.csv", 
        tokenizer
    )
    
    test_dataset = MultiModalRestaurantDataset(
        "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/data/multimodal_splits/test.csv",
        tokenizer
    )
    
    print("✅ Multimodal datasets created successfully!")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset preparation
    print("🧪 Testing multimodal dataset preparation...")
    
    train_ds, val_ds, test_ds = prepare_multimodal_datasets()
    
    # Test loading a sample
    sample = train_ds[0]
    print(f"\n📋 Sample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✅ Test completed! Datasets ready for multimodal training.")

"""
Multi-modal model architecture combining RoBERTa (text) with Vision Transformer (images)
For restaurant review classification using both text and visual information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig
)
from torchvision import models
import torchvision.transforms as transforms
from typing import Dict, Optional, Tuple


class MultiModalConfig(PretrainedConfig):
    """Configuration class for multi-modal model"""
    
    model_type = "multimodal_restaurant_classifier"
    
    def __init__(
        self,
        text_model_name: str = "roberta-base",
        image_model_name: str = "resnet50",
        num_labels: int = 2,
        text_hidden_size: int = 768,
        image_hidden_size: int = 2048,
        fusion_hidden_size: int = 512,
        dropout_rate: float = 0.1,
        fusion_method: str = "concat",  # "concat", "attention", "bilinear"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.num_labels = num_labels
        self.text_hidden_size = text_hidden_size
        self.image_hidden_size = image_hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.dropout_rate = dropout_rate
        self.fusion_method = fusion_method


class MultiModalRestaurantClassifier(PreTrainedModel):
    """
    Multi-modal classifier for restaurant reviews
    Combines text features from RoBERTa with image features from ResNet
    """
    
    config_class = MultiModalConfig
    
    def __init__(self, config: MultiModalConfig):
        super().__init__(config)
        self.config = config
        
        # Text encoder (RoBERTa)
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        
        # Image encoder (ResNet)
        if config.image_model_name == "resnet50":
            self.image_encoder = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.image_encoder.fc = nn.Identity()
            image_features_dim = 2048
        elif config.image_model_name == "resnet18":
            self.image_encoder = models.resnet18(pretrained=True)
            self.image_encoder.fc = nn.Identity()
            image_features_dim = 512
        else:
            raise ValueError(f"Unsupported image model: {config.image_model_name}")
        
        # Update config with actual dimensions
        self.config.image_hidden_size = image_features_dim
        
        # Feature projection layers
        self.text_projection = nn.Linear(config.text_hidden_size, config.fusion_hidden_size)
        self.image_projection = nn.Linear(image_features_dim, config.fusion_hidden_size)
        
        # Fusion layer
        if config.fusion_method == "concat":
            fusion_input_size = config.fusion_hidden_size * 2
        elif config.fusion_method == "attention":
            fusion_input_size = config.fusion_hidden_size
            self.attention = nn.MultiheadAttention(
                embed_dim=config.fusion_hidden_size,
                num_heads=8,
                dropout=config.dropout_rate
            )
        elif config.fusion_method == "bilinear":
            fusion_input_size = config.fusion_hidden_size
            self.bilinear = nn.Bilinear(
                config.fusion_hidden_size,
                config.fusion_hidden_size,
                config.fusion_hidden_size
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.fusion_hidden_size, config.num_labels)
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multi-modal model
        
        Args:
            input_ids: Text token IDs (batch_size, seq_len)
            attention_mask: Attention mask for text (batch_size, seq_len)
            image: Image tensor (batch_size, 3, 224, 224)
            labels: Ground truth labels (batch_size,)
            return_dict: Whether to return dict format
            
        Returns:
            Dictionary with loss, logits, and other outputs
        """
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.text_projection(text_features)
        
        # Encode image
        image_features = self.image_encoder(image)
        image_features = self.image_projection(image_features)
        
        # Fusion
        if self.config.fusion_method == "concat":
            fused_features = torch.cat([text_features, image_features], dim=1)
        elif self.config.fusion_method == "attention":
            # Use cross-attention between text and image features
            text_features_unsqueezed = text_features.unsqueeze(1)
            image_features_unsqueezed = image_features.unsqueeze(1)
            
            # Cross-attention: text as query, image as key/value
            attended_features, _ = self.attention(
                text_features_unsqueezed,
                image_features_unsqueezed,
                image_features_unsqueezed
            )
            fused_features = attended_features.squeeze(1)
        elif self.config.fusion_method == "bilinear":
            fused_features = self.bilinear(text_features, image_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        
        return {
            'loss': loss,
            'logits': logits,
            'text_features': text_features,
            'image_features': image_features,
            'fused_features': fused_features
        }


def create_multimodal_model(
    text_model_name: str = "roberta-base",
    image_model_name: str = "resnet50", 
    num_labels: int = 2,
    fusion_method: str = "concat"
) -> MultiModalRestaurantClassifier:
    """
    Factory function to create a multi-modal model
    
    Args:
        text_model_name: HuggingFace model name for text encoder
        image_model_name: Model name for image encoder
        num_labels: Number of classification labels
        fusion_method: Method to fuse text and image features
        
    Returns:
        Initialized multi-modal model
    """
    
    config = MultiModalConfig(
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        num_labels=num_labels,
        fusion_method=fusion_method
    )
    
    model = MultiModalRestaurantClassifier(config)
    
    print(f"✅ Created multi-modal model:")
    print(f"   📝 Text encoder: {text_model_name}")
    print(f"   🖼️ Image encoder: {image_model_name}")
    print(f"   🔗 Fusion method: {fusion_method}")
    print(f"   🏷️ Number of labels: {num_labels}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("🧪 Testing multi-modal model...")
    
    model = create_multimodal_model()
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        image=dummy_image,
        labels=dummy_labels
    )
    
    print(f"\n📋 Model outputs:")
    print(f"   Loss: {outputs['loss'].item():.4f}")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Text features shape: {outputs['text_features'].shape}")
    print(f"   Image features shape: {outputs['image_features'].shape}")
    print(f"   Fused features shape: {outputs['fused_features'].shape}")
    
    print(f"\n✅ Model test completed successfully!")

"""
Multi-modal training script for restaurant review classification
Combines text and image data for enhanced model performance
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    EvalPrediction
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any
import json
import time

# Import our custom modules
try:
    from multimodal_dataset import prepare_multimodal_datasets, MultiModalRestaurantDataset
    from multimodal_model import create_multimodal_model, MultiModalRestaurantClassifier
except ImportError:
    from training_scripts.multimodal_dataset import prepare_multimodal_datasets, MultiModalRestaurantDataset
    from training_scripts.multimodal_model import create_multimodal_model, MultiModalRestaurantClassifier


class MultiModalDataCollator:
    """Custom data collator for multi-modal data"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        """Collate a batch of multi-modal samples"""
        batch = {}
        
        # Text features
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        
        # Image features
        batch['image'] = torch.stack([f['image'] for f in features])
        
        # Labels
        batch['labels'] = torch.stack([f['labels'] for f in features])
        
        return batch


def compute_multimodal_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for multi-modal evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class MultiModalTrainer(Trainer):
    """Custom trainer for multi-modal models"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for multi-modal training"""
        labels = inputs.pop("labels")
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image=inputs['image'],
            labels=labels
        )
        
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


def setup_training_args(
    output_dir: str = "./results/multimodal_training",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    eval_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 100
) -> TrainingArguments:
    """Setup training arguments for multi-modal training"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard for now
        remove_unused_columns=False,  # Keep all columns for multi-modal data
        dataloader_pin_memory=False,  # Disable for compatibility
    )


def train_multimodal_model(
    model_name: str = "roberta-base",
    image_model: str = "resnet50",
    fusion_method: str = "concat",
    num_epochs: int = 3,
    batch_size: int = 4,  # Smaller batch size for multi-modal training
    learning_rate: float = 1e-5,  # Lower learning rate for stable training
    output_dir: str = "./results/multimodal_training"
) -> MultiModalRestaurantClassifier:
    """
    Train a multi-modal restaurant review classifier
    
    Args:
        model_name: Text encoder model name
        image_model: Image encoder model name  
        fusion_method: Method to fuse modalities
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        output_dir: Directory to save results
        
    Returns:
        Trained multi-modal model
    """
    
    print("🚀 Starting multi-modal training...")
    print(f"📝 Text model: {model_name}")
    print(f"🖼️ Image model: {image_model}")
    print(f"🔗 Fusion: {fusion_method}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    print("📂 Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_multimodal_datasets(
        tokenizer_name=model_name
    )
    
    # Create model
    print("🏗️ Creating model...")
    model = create_multimodal_model(
        text_model_name=model_name,
        image_model_name=image_model,
        num_labels=2,
        fusion_method=fusion_method
    )
    
    # Setup training arguments
    training_args = setup_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Create data collator
    data_collator = MultiModalDataCollator(tokenizer)
    
    # Create trainer
    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_multimodal_metrics,
    )
    
    # Start training
    print("🎯 Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    print(f"✅ Training completed in {training_time:.2f} minutes")
    
    # Save model
    model_save_path = os.path.join(output_dir, "final_multimodal_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"💾 Model saved to {model_save_path}")
    
    # Evaluate on test set
    print("📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Save evaluation results
    results = {
        'training_time_minutes': training_time,
        'test_results': test_results,
        'model_config': {
            'text_model': model_name,
            'image_model': image_model,
            'fusion_method': fusion_method,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }
    
    with open(os.path.join(output_dir, "training_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📈 Test Results:")
    for key, value in test_results.items():
        if 'eval_' in key:
            metric_name = key.replace('eval_', '')
            print(f"   {metric_name}: {value:.4f}")
    
    return model


if __name__ == "__main__":
    print("🧪 Starting multi-modal training script...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Train model with different fusion methods
    fusion_methods = ["concat"]  # Start with concat, can add "attention", "bilinear"
    
    for fusion_method in fusion_methods:
        print(f"\n{'='*50}")
        print(f"Training with fusion method: {fusion_method}")
        print(f"{'='*50}")
        
        output_dir = f"./results/multimodal_{fusion_method}"
        
        try:
            model = train_multimodal_model(
                fusion_method=fusion_method,
                output_dir=output_dir,
                num_epochs=2,  # Start with fewer epochs for testing
                batch_size=2,  # Small batch size for initial testing
            )
            print(f"✅ Successfully trained model with {fusion_method} fusion")
            
        except Exception as e:
            print(f"❌ Error training with {fusion_method} fusion: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Multi-modal training script completed!")

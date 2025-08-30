"""
Simplified Multi-modal Training Script for Restaurant Reviews
Combines text and image data for enhanced sentiment classification
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
import time
import json

# Import our custom modules
from training_scripts.multimodal_dataset import prepare_multimodal_datasets
from training_scripts.multimodal_model import create_multimodal_model


def train_epoch(model, dataloader, optimizer, device, epoch_num):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    print(f"📚 Training Epoch {epoch_num}...")
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=images,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    print(f"✅ Epoch {epoch_num} completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def evaluate_model(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    print("📊 Evaluating...")
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=images,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    print(f"📈 Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def collate_multimodal_batch(batch):
    """Custom collate function for DataLoader"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'image': torch.stack([item['image'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }


def train_multimodal_restaurant_classifier():
    """Main training function"""
    print("🚀 Starting Multi-modal Restaurant Review Training!")
    print("=" * 60)
    
    # Configuration
    config = {
        'batch_size': 4,  # Small batch size for initial testing
        'learning_rate': 1e-5,
        'num_epochs': 2,
        'model_name': 'roberta-base',
        'image_model': 'resnet50',
        'fusion_method': 'concat'
    }
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Create output directory
    output_dir = "/Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/results/multimodal_simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare datasets
    print("\n📂 Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_multimodal_datasets()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_multimodal_batch,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=0
    )
    
    print(f"✅ Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Create model
    print("\n🏗️ Creating multi-modal model...")
    model = create_multimodal_model(
        text_model_name=config['model_name'],
        image_model_name=config['image_model'],
        fusion_method=config['fusion_method']
    )
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print(f"\n🎯 Starting training for {config['num_epochs']} epochs...")
    start_time = time.time()
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*40}")
        print(f"EPOCH {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*40}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    training_time = (time.time() - start_time) / 60
    print(f"\n✅ Training completed in {training_time:.2f} minutes!")
    
    # Final test evaluation
    print(f"\n🏁 Final evaluation on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    
    # Save model
    model_path = os.path.join(output_dir, "multimodal_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Save training results
    results = {
        'config': config,
        'training_time_minutes': training_time,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_test_loss': test_loss,
        'final_test_accuracy': test_acc
    }
    
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   🎯 Test Accuracy: {test_acc:.4f}")
    print(f"   📉 Test Loss: {test_loss:.4f}")
    print(f"   ⏱️ Training Time: {training_time:.2f} minutes")
    print(f"   📄 Results saved to: {results_path}")
    
    return model, results


if __name__ == "__main__":
    try:
        model, results = train_multimodal_restaurant_classifier()
        print("\n🎉 Multi-modal training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

# Model Testing Guide

## 🚀 Quick Start

You now have **5 different models** to choose from for testing:

### Available Models
1. **`combined_training_model`** 
   - Latest model trained on combined policy + ground truth data
   - Best performance: 97.73% accuracy
   - 124.6M parameters

2. **`final_model`** 
   - Previous best model with good performance
   - Stable and well-tested

3. **`roberta_policy_based_model`** ⭐ RECOMMENDED
   - RoBERTa model trained with policy-based filtering
   - Good for policy-specific tasks

4. **`policy_based_model`** ⭐ RECOMMENDED
   - Baseline model with policy-based filtering
   - Good for comparison

5. **`intial_model`**
   - First baseline model (deprecated)
   - Keep for historical comparison

## 🎯 How to Choose and Test Models

### Method 1: Interactive Menu (Easiest)
```bash
# From project root
cd /Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW
python testing/quick_test.py

# OR from testing directory
cd /Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW/testing
python quick_test.py
```
- Shows prioritized list of models
- Easy selection by number
- Choose test mode (batch/interactive/both)
- Works from any directory!

### Method 2: Direct Model Selection
```bash
# Test specific model with batch testing
python testing/quick_test.py combined_training_model

# Test specific model with interactive mode
python testing/quick_test.py combined_training_model --interactive
```

### Method 3: Full Feature Testing
```bash
# Detailed model management
python testing/multi_model_tester.py

# Test specific model with options
python testing/multi_model_tester.py --model combined_training_model --mode interactive

# Use different models directory
python testing/multi_model_tester.py --models-dir ./custom_models
```

## 📊 Testing Modes

### 1. **Batch Testing**
- Tests predefined sample reviews
- Quick performance overview
- Shows accuracy metrics

### 2. **Interactive Testing**
- Enter your own reviews to test
- Real-time predictions with confidence scores
- Type 'quit' to exit

### 3. **Both**
- Runs batch testing first
- Then switches to interactive mode

## 🎛️ Configuration

Edit `testing/model_config.json` to:
- Change model priorities
- Update descriptions
- Set default model
- Modify test samples

## 💡 Tips

1. **Start with the newest model**: `combined_training_model` has the best performance
2. **Compare models**: Test the same review on different models to see differences
3. **Use batch mode first**: Get quick overview before interactive testing
4. **Check confidence scores**: Lower confidence might indicate edge cases

## 🔧 Adding New Models

1. Save your trained model to the `models/` directory
2. Ensure it has both model files and tokenizer files
3. Update `testing/model_config.json` with model info
4. The system will auto-discover and include it

## 📈 Model Performance Comparison

| Model | Accuracy | Parameters | Best For |
|-------|----------|------------|----------|
| combined_training_model | 97.73% | 124.6M | Production use |
| final_model | ~95%+ | ~124M | Stable baseline |
| roberta_policy_based_model | ~90%+ | ~124M | Policy-specific tasks |
| policy_based_model | ~85%+ | ~110M | Quick testing |
| intial_model | ~80%+ | ~110M | Historical comparison |

## 🚨 Common Issues

- **Model not found**: Make sure you're running from the project root directory
- **Memory errors**: Use CPU mode for testing if GPU memory is limited
- **Import errors**: Ensure you're using the virtual environment

## Example Usage

```bash
# Quick test latest model
cd /Users/nicksonho/Work/TechJam/TIKTOKGIVEMETHEW
python testing/quick_test.py

# Select option 1 (combined_training_model)
# Choose mode 3 (both batch and interactive)
# Test your own reviews interactively
```

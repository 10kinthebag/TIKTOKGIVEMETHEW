# 🚀 RoBERTa Model Testing - Quick Reference

## Fast Testing Commands

### Quick Tests (< 30 seconds)
```bash
make test_roberta       # Quick batch test (11 test cases, 100% accuracy)
make test_compare       # Performance comparison (97.4% vs 27.1% baseline)
make test_all          # Run both quick tests
```

### Interactive Testing
```bash
make test_interactive   # Test your own reviews interactively
```

### Training Commands
```bash
make train_roberta     # Train RoBERTa model with policy data
```

## Direct Python Commands

### Batch Testing Only
```bash
python test_model.py --mode batch
```

### Interactive Testing Only
```bash
python test_model.py --mode interactive
```

### Both Modes
```bash
python test_model.py --mode both
# or just
python test_model.py
```

### Performance Comparison
```bash
python simple_comparison.py
```

## Model Performance Summary

- **RoBERTa Model**: 97.4% accuracy
- **Old Baseline**: 27.1% accuracy  
- **Improvement**: +70.3% (3.6x better!)
- **Model Size**: 124.6M parameters
- **Test Dataset**: 1,100 ground truth samples

## Quick Test Cases Included

✅ **Valid Reviews**:
- "Amazing food and great service! The pasta was delicious."
- "Lovely atmosphere, friendly staff, reasonable prices."
- "Best restaurant in town! Highly recommend the seafood."

❌ **Invalid Reviews** (Detected):
- Advertisements with URLs/phone numbers
- Irrelevant content (non-restaurant)
- Spam (repeated characters)
- Rants from non-visitors
- Too short reviews
- Contradictory reviews

## File Locations

- **Trained Model**: `./roberta_policy_based_model/`
- **Test Script**: `test_model.py`
- **Comparison Script**: `simple_comparison.py`
- **Makefile**: Contains all fast commands

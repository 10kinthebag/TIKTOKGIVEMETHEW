# Training Data Combination Strategies

You now have **4 different approaches** to combine your ground truth data with pseudo-labeled data for training:

## 🎯 Quick Commands

```bash
# Option 1: Simple Hybrid (RECOMMENDED)
make tokenize_hybrid && make train

# Option 2: Ground Truth Only  
make tokenize_ground_truth && make train

# Option 3: Pseudo Labels Only
make tokenize && make train

# Option 4: Progressive Training
make train_progressive

# Option 5: Weighted Training
make tokenize_weighted && make train
```

## 📊 Detailed Approaches

### 1. **Hybrid Approach (RECOMMENDED)** 
**Command:** `make tokenize_hybrid`

- **What it does:** Combines both datasets into one large training set
- **Benefits:** 
  - Maximum training data (2,200 samples vs 1,100 each)
  - Simple to implement and train
  - Good balance of quality and quantity
- **Use when:** You want the simplest approach with good results

**Data split:**
- Ground truth: 1,100 samples  
- Pseudo-labeled: 1,100 samples
- **Total: 2,200 samples**

### 2. **Progressive Training**
**Command:** `make train_progressive`

- **What it does:** 
  1. First trains on ground truth data (high learning rate)
  2. Then fine-tunes on pseudo data (low learning rate)
- **Benefits:**
  - Prioritizes high-quality labels first
  - Uses pseudo data to improve generalization
  - Prevents pseudo labels from overwhelming ground truth
- **Use when:** You want maximum control over training quality

### 3. **Weighted Training** 
**Command:** `make tokenize_weighted`

- **What it does:** Assigns higher importance to ground truth samples during training
- **Weights:** Ground truth = 2.0, Pseudo labels = 1.0
- **Benefits:** 
  - Ground truth has 2x influence on loss function
  - Single training process
  - Balanced approach
- **Use when:** You want to emphasize ground truth without separate training stages

### 4. **Ground Truth Only**
**Command:** `make tokenize_ground_truth`

- **What it does:** Trains only on human-annotated data
- **Benefits:** Highest quality training data
- **Drawbacks:** Limited to 1,100 samples
- **Use when:** You prioritize quality over quantity

### 5. **Pseudo Labels Only** 
**Command:** `make tokenize`

- **What it does:** Trains only on rule-based pseudo labels
- **Benefits:** Large dataset (1,100 samples)
- **Drawbacks:** Lower quality labels
- **Use when:** You want to compare against pseudo-only performance

## 🏆 Recommendation

**Start with the Hybrid Approach (`make tokenize_hybrid`)** because:

1. ✅ **Simple to use** - one command
2. ✅ **Maximum training data** - 2,200 samples  
3. ✅ **Good balance** - quality + quantity
4. ✅ **Fast training** - single training process
5. ✅ **Proven effective** - commonly used in ML

If you want even better results, try **Progressive Training** afterward:

```bash
# Best approach for maximum quality
make train_progressive
```

## 📈 Expected Performance

| Approach | Training Data | Expected Performance | Training Time |
|----------|---------------|---------------------|---------------|
| Ground Truth Only | 1,100 samples | High precision, may overfit | Fast |
| Pseudo Only | 1,100 samples | Good recall, less precision | Fast |
| **Hybrid** | **2,200 samples** | **Best balance** | **Medium** |
| Progressive | 2,200 samples | Highest quality | Slow |
| Weighted | 2,200 samples | High quality | Medium |

## 🚀 Quick Start

```bash
# Recommended quick start
make tokenize_hybrid
make train
make evaluate
```

Your model will now learn from both high-quality ground truth labels AND benefit from the larger dataset of pseudo labels! 🎉

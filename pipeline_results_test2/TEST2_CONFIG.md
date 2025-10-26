# Test 2 Configuration - 1000 Rows
# Updated hyperparameters based on Test 1 results

## Changes to Make Before Running Test 2

### 1. Update src/constants.py

Replace the HYPERPARAMETERS section with:

```python
HYPERPARAMETERS = {
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,        # ↑ Increased from 4
    "per_device_eval_batch_size": 8,          # ↑ Increased from 4
    "gradient_accumulation_steps": 2,         # ↓ Decreased from 4
    "learning_rate": 1.5e-4,                  # ↓ Lowered from 2e-4
    "max_length": 768,                        # Keep same
    "val_split_ratio": 0.1,                   # ↓ Decreased from 0.2
    "warmup_steps": 10,                       # ↑ Increased from 2
    "weight_decay": 0.01,                     # Keep same
}
```

Also update the dataset URI:
```python
RAW_DATASET_URI: str = f"gs://{BUCKET_NAME}/tarot_training_data_1000.csv"
```

---

## Rationale for Each Change

### Batch Size: 4 → 8
**Why**: 
- More training data (1000 vs 300 rows) provides more stable gradients
- Larger batches improve GPU utilization
- Reduces training time

**Expected Impact**:
- Faster training (fewer gradient updates needed)
- Smoother loss curves

---

### Gradient Accumulation: 4 → 2
**Why**:
- Maintain effective batch size of 16 (8 × 2 = 16, same as 4 × 4 = 16)
- Balance memory usage with training speed

**Expected Impact**:
- More frequent weight updates
- Better for monitoring training progress

---

### Learning Rate: 2e-4 → 1.5e-4
**Why**:
- More data = more stable optimization landscape
- Slightly lower LR helps prevent overshooting
- Safer convergence with larger batches

**Expected Impact**:
- Potentially smoother training
- May need 1 extra epoch if convergence is slower

**Note**: If training seems slow, can keep at 2e-4

---

### Validation Split: 0.2 → 0.1
**Why**:
- Test 1: 0.2 × 300 = 60 samples for validation (sufficient)
- Test 2: 0.1 × 1000 = 100 samples for validation (more than enough)
- More samples for training improves model

**Expected Impact**:
- 900 training samples vs 240 in Test 1 (3.75× more!)
- Better model generalization

---

### Warmup Steps: 2 → 10
**Why**:
- Test 1: 2 steps was minimal (270 samples / 16 batch = ~17 total steps)
- Test 2: 10 steps = better stability at start
- Prevents early instability with larger batches

**Expected Impact**:
- Smoother training curves from the start
- Better initial learning

---

## Expected Training Metrics

### Test 1 (300 rows) - Actual:
- Training samples: 240 (after 0.2 split)
- Validation samples: 60
- Steps per epoch: ~17
- Total steps: ~51 (3 epochs)
- Training time: ~10 minutes

### Test 2 (1000 rows) - Predicted:
- Training samples: 900 (after 0.1 split)
- Validation samples: 100
- Steps per epoch: ~57 (900 / 16 effective batch)
- Total steps: ~171 (3 epochs)
- Training time: ~20-25 minutes

---

## Performance Expectations

### Current (Test 1):
- BLEU: 0.3025
- ROUGE: 0.5529

### Target (Test 2):
- BLEU: 0.35-0.40 (+15-30%)
- ROUGE: 0.56-0.60 (+5-10%)

### Why We Expect Improvement:
1. **3.75× more training data** = better pattern learning
2. **More diverse examples** = better generalization
3. **Larger batches** = more stable gradients
4. **Better warmup** = smoother convergence

---

## Monitoring During Training

### Watch in TensorBoard:
1. **Train loss should decrease smoothly**
   - No sharp spikes
   - Steady downward trend

2. **Eval loss should track train loss**
   - Gap should be small (<10%)
   - Both should decrease together
   - If eval loss increases while train decreases → overfitting

3. **Learning rate warmup**
   - Should see gradual increase from 0 to 1.5e-4
   - Then steady or slight decay

### Red Flags:
- ❌ Eval loss diverging from train loss
- ❌ Loss not decreasing after epoch 1
- ❌ Large gradient spikes (grad_norm)
- ❌ Training taking >30 minutes

### If You See Issues:
- **Overfitting** (eval >> train): Increase dropout or weight decay
- **Slow convergence**: Increase learning rate to 2e-4
- **Instability**: Decrease batch size to 6 or learning rate to 1e-4

---

## Alternative Configurations to Try

### Option A: More Conservative (if Test 2 shows instability)
```python
"per_device_train_batch_size": 6,
"gradient_accumulation_steps": 3,  # 6×3=18 effective
"learning_rate": 1e-4,
"warmup_steps": 15,
```

### Option B: More Aggressive (if Test 2 converges too slowly)
```python
"per_device_train_batch_size": 8,
"gradient_accumulation_steps": 2,
"learning_rate": 2e-4,  # Keep original
"num_train_epochs": 4,  # Add one more epoch
```

---

## Checklist Before Running Test 2

- [ ] Prepare 1000-row dataset: `tarot_training_data_1000.csv`
- [ ] Upload to GCS: `gs://anna-pavel-bucket/tarot_training_data_1000.csv`
- [ ] Update `src/constants.py` with new hyperparameters
- [ ] Update `RAW_DATASET_URI` to point to 1000-row file
- [ ] Verify TensorBoard will log properly (already configured)
- [ ] Clear any cached pipeline artifacts (optional)
- [ ] Run: `python scripts/pipeline_runner.py`
- [ ] Monitor TensorBoard at http://localhost:6006

---

## After Test 2 Completes

### Analysis Steps:
1. Download predictions and evaluation results
2. Compare BLEU/ROUGE to Test 1
3. Review TensorBoard graphs
4. Examine sample predictions for quality
5. Document findings for Test 3 planning

### Success Criteria:
- ✅ BLEU ≥ 0.35 (vs 0.30)
- ✅ ROUGE ≥ 0.56 (vs 0.55)
- ✅ Smooth training curves
- ✅ No overfitting
- ✅ Training completes in <30 minutes


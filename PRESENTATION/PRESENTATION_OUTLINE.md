# Fine-Tuning a Tarot Reading AI Model: A Complete MLOps Pipeline
## LLMOps Project Presentation

---

## Slide 1: Title Slide
**Fine-Tuning Phi-3-Mini for Tarot Card Readings**
- Subtitle: Building an End-to-End ML Pipeline on Google Cloud Vertex AI
- Your Name
- Date: October 2025

---

## Slide 2: Project Overview
### Objective
Transform a general-purpose LLM into a specialized tarot reading assistant

### Key Components
- **Base Model**: Microsoft Phi-3-mini-4k-instruct (3.8B parameters)
- **Dataset**: Custom tarot reading training data (300 → 3000 rows)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Infrastructure**: Google Cloud Vertex AI Pipelines with T4 GPU
- **Framework**: Kubeflow Pipelines (KFP)

---

## Slide 3: The Challenge
### Initial Dataset Structure
```
Columns: prompt, response
Format: User question + card spread → Mystical tarot interpretation
```

### Technical Challenges
1. **Dataset Compatibility**: Original pipeline designed for translation tasks
2. **Token Length Analysis**: Unknown sequence length requirements
3. **API Version Conflicts**: trl 0.17.0 vs 0.24.0 compatibility
4. **Pipeline Caching Issues**: Artifact reuse causing failures
5. **File Path Handling**: Vertex AI OutputPath directory semantics

---

## Slide 4: Architecture Overview
### 4-Component Pipeline

```
┌─────────────────────────────┐
│ 1. Data Transformation      │
│    - Load CSV               │
│    - Format to Phi-3 msgs   │
│    - Train/Test Split (90/10)│
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 2. Fine-Tuning (T4 GPU)     │
│    - LoRA Configuration      │
│    - 4-bit Quantization     │
│    - 3 Epochs Training      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 3. Inference (T4 GPU)       │
│    - Generate Predictions   │
│    - Test Set Evaluation    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 4. Evaluation               │
│    - BLEU Score             │
│    - ROUGE Score            │
└─────────────────────────────┘
```

---

## Slide 5: Data Transformation Component
### What We Changed

**Before (Translation Format):**
```python
sentence → translation mapping
```

**After (Tarot Format):**
```python
prompt → response mapping
Converted to Phi-3 messages format:
[
  {"role": "user", "content": "{tarot_question}"},
  {"role": "assistant", "content": "{reading}"}
]
```

### Key Fix
```python
# OutputPath provides directory, not file
train_file = os.path.join(train_dataset, "train_dataset.csv")
split_dataset["train"].to_csv(train_file, index=False)
```

---

## Slide 6: Token Analysis
### Understanding Sequence Lengths

**Analysis Results:**
```
Mean token length: 488 tokens
Maximum length: 628 tokens
% over 1536 tokens: 0%
```

**Impact on Configuration:**
- Initial assumption: 1000+ tokens (wrong!)
- Actual requirement: ~500 tokens (much shorter)
- This allowed more aggressive batch sizes

**Tool Used:**
```python
# Created analysis script
scripts/test_fine_tuning_local.py
```

---

## Slide 7: Fine-Tuning Configuration
### LoRA Parameters
```python
lora_config = LoraConfig(
    r=16,                    # Rank of update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Quantization (Memory Efficiency)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit precision
    bnb_4bit_quant_type="nf4",     # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16
)
```

---

## Slide 8: Training Hyperparameters (Test 1 - 300 rows)
### Current Configuration
```python
{
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_length": 768,
    "val_split_ratio": 0.2,
    "warmup_steps": 2,
    "weight_decay": 0.01
}
```

**Effective Batch Size**: 4 × 4 = 16 samples per update

---

## Slide 9: Key Debugging Challenges

### Challenge 1: File Path Issues
**Problem**: `FileNotFoundError` - couldn't find dataset
**Root Cause**: Vertex AI OutputPath returns directory, not file
**Solution**: Append filename to path
```python
dataset_file = os.path.join(dataset.path, "train_dataset.csv")
```

### Challenge 2: API Compatibility
**Problem**: `TypeError: unexpected keyword argument`
**Root Cause**: trl 0.17.0 (GCP) ≠ trl 0.24.0 (local)
**Solution**: Removed incompatible parameters
```python
# Removed: tokenizer, dataset_text_field, max_seq_length
# Kept: model, args, train_dataset, eval_dataset, peft_config
```

### Challenge 3: Pipeline Caching
**Problem**: Reusing failed component outputs
**Solution**: 
```python
data_transformation_task.set_caching_options(False)
```

---

## Slide 10: Test 1 Results - Performance Metrics
### Overall Performance (30 test samples)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **BLEU** | 0.3025 (30.25%) | Moderate word-level overlap |
| **ROUGE** | 0.5529 (55.29%) | Good structural similarity |

### What This Means
✅ **Strengths:**
- Model learned mystical narrative style
- Maintains Situation-Action-Outcome structure
- Appropriate card interpretations
- Consistent spiritual tone

⚠️ **Areas for Improvement:**
- Word-for-word matching could be higher
- Some creative variation from references

---

## Slide 11: TensorBoard Analysis - Key Graphs

### Essential Graphs to Include:

1. **Training Loss (train/loss)**
   - Shows learning progress
   - Should decrease over epochs
   - Include: Screenshot showing downward trend

2. **Validation Loss (eval/loss)**
   - Checks for overfitting
   - Should track close to training loss
   - Include: Compare train vs eval loss curves

3. **Learning Rate Schedule (train/learning_rate)**
   - Shows warmup and decay
   - Include: Full training schedule view

4. **Gradient Norm (train/grad_norm)**
   - Training stability indicator
   - Should be stable, not spiking
   - Include: If showing stable training

### How to Export from TensorBoard:
1. Navigate to SCALARS tab
2. Click on each graph
3. Use download icon (top right) to save as SVG/PNG
4. Look for "smoothing" slider - set to 0 for raw data

---

## Slide 12: Sample Prediction Comparison
### Example 1: Quality Prediction

**Input Card Spread:**
```
Question: "What can I do that would emphasize growth?"
Cards: King of Cups (upright), Two of Cups (upright), 
       The Empress (upright)
```

**Model Output Quality:**
- ✅ Correctly identifies all three cards
- ✅ Maintains mystical tone ("cosmic spread reveals...")
- ✅ Links cards to spread positions (Dynamic, Action, Outcome)
- ✅ Provides appropriate card interpretations
- ✅ Includes closing affirmation

**Scores:** BLEU: 0.0, ROUGE: 0.56
- Low BLEU = different wording
- High ROUGE = captured structure and meaning

---

## Slide 13: Lessons Learned
### Key Takeaways from Test 1

1. **Always Analyze Your Data First**
   - Assumed 1000+ tokens, actually ~500
   - Led to multiple hyperparameter revisions

2. **Version Compatibility Matters**
   - Local testing (trl 0.24) ≠ GCP (trl 0.17)
   - Always check library versions in deployment environment

3. **Pipeline Artifacts Have Semantics**
   - OutputPath = directory, not file
   - Understanding pipeline contract prevents errors

4. **Local Testing is Essential**
   - Created `test_fine_tuning_local.py` to isolate logic
   - Confirmed code worked before debugging infrastructure

5. **Caching Can Help or Hurt**
   - Speeds up pipeline re-runs
   - Can propagate failures if not managed

---

## Slide 14: Progressive Results - All Tests Comparison! 🚀

### Performance Metrics Evolution

| Metric | TEST 1 (300) | TEST 2 (1500) | TEST 3 (3000) | TEST 1→3 Total |
|--------|--------------|---------------|---------------|----------------|
| **BLEU** | 30.25% | 45.94% | **61.06%** ✨ | **+101.8%** 🎯 |
| **ROUGE** | 55.29% | 71.56% | **71.09%** | **+28.6%** |
| **Test Samples** | 30 | 150 | 300 | 10× more |
| **Training Samples** | 240 | 1350 | 2700 | 11.25× more |

### Training Dynamics (from TensorBoard)

| Aspect | TEST 1 | TEST 2 | TEST 3 |
|--------|--------|--------|--------|
| **Train Loss Drop** | ~85% (est.) | 95.2% | **95.9%** |
| **Final Train Loss** | ~0.30 (est.) | 0.11 | **0.097** ⭐ |
| **Final Eval Loss** | ~0.35 (est.) | 0.12 | **0.099** ⭐ |
| **Training Steps** | 51 | 337 | **522** |
| **Training Time** | ~10 min | ~6 hrs | **~3.7 hrs** |

### Key Insights
✅ **BLEU doubled from TEST 1→3** (30% → 61%) - exceptional word-level accuracy  
✅ **ROUGE plateauing** (71.6% → 71.1%) - structural quality maximized  
✅ **Consistent convergence** - 95%+ loss reduction across all tests  
✅ **No overfitting** - eval/train gap < 0.002 in TEST 3  
✅ **Efficient scaling** - TEST 3 trained 2× faster than TEST 2 despite 2× data  

---

## Slide 14b: TEST 3 Results Deep Dive (3000 rows)

### Training Configuration
- **Dataset**: 3000 rows → 2700 train / 300 test (90/10 split)
- **Learning Rate**: 1.2e-4 (reduced from 1.5e-4 for stability)
- **LoRA Config**: r=16, alpha=32 (doubled from TEST 2 for capacity)
- **Batch Size**: 8 (effective batch = 16 with grad accumulation)
- **Warmup**: 15 steps (smooth learning start)
- **Weight Decay**: 0.02 (regularization added)

### TensorBoard Highlights
- **Train Loss**: 2.375 → 0.097 (95.9% reduction)
- **Eval Loss**: Epoch 1: 0.122, Epoch 2: 0.106, **Epoch 3: 0.099**
- **Convergence**: Smooth, stable training across 522 steps
- **Overfitting Check**: Eval-train gap only 0.002 (excellent generalization!)

### What We Achieved
✨ **Best BLEU yet**: 61.06% (+32.8% from TEST 2)  
✨ **Near-perfect eval/train alignment**: 0.099 vs 0.097  
✨ **Production-ready model**: Strong generalization at scale  
✨ **Cost-efficient**: 3.7 hours vs expected 10-12 hours  

---

## Slide 15: TEST 2 Results - Major Improvements! 🎉
### Performance Metrics (1500 rows training data, 150 test samples)

| Metric | TEST 1 (300 rows) | TEST 2 (1500 rows) | Change |
|--------|-------------------|---------------------|--------|
| **BLEU** | 0.3025 (30.25%) | **0.4594 (45.94%)** | **+51.9%** ✨ |
| **ROUGE** | 0.5529 (55.29%) | **0.7156 (71.56%)** | **+29.4%** ✨ |
| **Test Samples** | 30 | 150 | 5× more |
| **Training Samples** | 240 | 1350 | 5.6× more |

### Training Dynamics (from TensorBoard)
- **Train Loss**: 2.36 → 0.11 (95.2% reduction)
- **Eval Loss**: 0.12 (final)
- **Training Steps**: 337 steps over 3 epochs
- **Training Time**: ~6 hours (T4 GPU)

---

## Slide 15: TEST 2 - What Improved?

### Quantitative Gains
✅ **BLEU Score +51.9%**
- Much better word-level matching
- Model learned more precise vocabulary
- Closer to reference readings

✅ **ROUGE Score +29.4%**
- Excellent structural similarity (71.6%)
- Better phrase and sequence generation
- Maintains tarot reading flow

✅ **Training Convergence**
- Smooth loss curves (no overfitting)
- 95% train loss reduction
- Eval loss tracked train loss closely

### Qualitative Improvements
- More varied card interpretations
- Fewer repetitive phrases
- Better handling of card reversals
- More contextually appropriate affirmations
- Richer mystical language

---

## Slide 16: TEST 2 TensorBoard Highlights

### Key Graphs to Include in Presentation:

**1. Training Loss Curve**
- Started: 2.36
- Ended: 0.11
- **95.2% reduction** - excellent convergence!
- Smooth downward trend across 337 steps

**2. Eval vs Train Loss**
- Final train: 0.11
- Final eval: 0.12
- **No overfitting** - very close tracking
- Model generalizes well

**3. Training Efficiency**
- 337 steps over 3 epochs
- 5.6× more training data than TEST 1
- Stable, consistent learning throughout

### What This Shows:
- ✅ Model learned effectively from 1500 rows
- ✅ No signs of overfitting or instability
- ✅ Ready to scale to 3000 rows (TEST 3)

---

## Slide 17: TEST 1 vs TEST 2 - Side-by-Side Comparison

### Dataset & Training
| Aspect | TEST 1 | TEST 2 |
|--------|--------|--------|
| **Total Dataset** | 300 rows | 1500 rows |
| **Training Samples** | 240 | 1350 |
| **Test Samples** | 30 | 150 |
| **Batch Size** | 4 | 8 |
| **Learning Rate** | 2e-4 | 1.5e-4 |
| **Warmup Steps** | 2 | 10 |
| **Training Time** | ~10 min | ~6 hours |

### Results
| Metric | TEST 1 | TEST 2 | Gain |
|--------|--------|--------|------|
| **BLEU** | 30.25% | 45.94% | +51.9% |
| **ROUGE** | 55.29% | 71.56% | +29.4% |
| **Loss Reduction** | ~85% | 95.2% | +10.2pp |

### Key Insight:
**5× more data → 50% better BLEU, 30% better ROUGE**
- Clear evidence that more training data improves performance
- Diminishing returns are minimal - still scaling well

---

## Slide 14: Proposed Changes for Test 2 (1500 rows) ✅ COMPLETED
### Hyperparameter Adjustments

| Parameter | Test 1 (300 rows) | Test 2 (1000 rows) | Rationale |
|-----------|-------------------|---------------------|-----------|
| **Dataset Size** | 300 | 1000 | More training data |
| **Epochs** | 3 | 3 | Keep same (monitor for overfitting) |
| **Batch Size** | 4 | 8 | More data → larger batches |
| **Gradient Accum** | 4 | 2 | Adjust for larger batch |
| **Effective Batch** | 16 | 16 | Keep same total |
| **Learning Rate** | 2e-4 | 2e-4 → 1.5e-4 | Slightly lower for stability |
| **Warmup Steps** | 2 | 10 | More data needs longer warmup |
| **Max Length** | 768 | 768 | Keep (data supports it) |
| **Val Split** | 0.2 (54 samples) | 0.1 (100 samples) | More train data available |

### Why These Changes?

**Larger Batch Size (4→8):**
- More data = more stable gradients
- Can process more efficiently
- Reduce gradient accumulation to maintain effective batch size

**Longer Warmup (2→10 steps):**
- Prevents early training instability
- More important with larger datasets

**Lower Learning Rate (2e-4→1.5e-4):**
- Optional: try if training is unstable
- Safer with larger dataset

**Smaller Val Split (0.2→0.1):**
- 100 validation samples is plenty
- Use more data for training

---

## Slide 15: Test 2 Configuration File
### Updated constants.py

```python
# For 1000-row dataset
RAW_DATASET_URI = f"gs://{BUCKET_NAME}/tarot_training_data_1000.csv"

HYPERPARAMETERS = {
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,        # ↑ from 4
    "per_device_eval_batch_size": 8,          # ↑ from 4
    "gradient_accumulation_steps": 2,         # ↓ from 4
    "learning_rate": 1.5e-4,                  # ↓ from 2e-4
    "max_length": 768,
    "val_split_ratio": 0.1,                   # ↓ from 0.2
    "warmup_steps": 10,                       # ↑ from 2
    "weight_decay": 0.01,
}
```

---

## Slide 16: Expected Improvements for Test 2
### Predictions

**With 3.3x More Training Data (300→1000 rows):**

1. **Better Generalization**
   - Model sees more card combinations
   - More diverse interpretations learned
   - Fewer repetitive phrases

2. **Higher BLEU Scores**
   - Target: 0.35-0.40 (vs 0.30 now)
   - More training = better word choice matching

3. **Maintained/Higher ROUGE Scores**
   - Target: 0.56-0.60 (vs 0.55 now)
   - Should maintain structural quality

4. **Reduced Overfitting Risk**
   - More data = better regularization
   - Monitor eval loss closely

5. **Smoother Training Curves**
   - Larger batches = more stable gradients
   - Should see cleaner loss curves in TensorBoard

---

## Slide 17: Success Metrics for Test 2
### How to Evaluate Improvement

**Quantitative Metrics:**
- [ ] BLEU Score: ≥0.35 (vs 0.30)
- [ ] ROUGE Score: ≥0.56 (vs 0.55)
- [ ] Training Loss: Smooth decrease
- [ ] Eval Loss: No divergence from training loss

**Qualitative Assessment:**
- [ ] Card interpretations more varied
- [ ] Fewer repetitive phrases
- [ ] Better handling of reversed cards
- [ ] More contextually appropriate affirmations

**Training Stability:**
- [ ] No gradient spikes in TensorBoard
- [ ] Eval loss tracks training loss
- [ ] No sudden jumps in loss curves

---

## Slide 18: Proposed Hyperparameter Changes for TEST 3 (3000 rows)

### Strategy: Conservative Scaling with Quality Focus

| Parameter | TEST 1 (300) | TEST 2 (1500) | TEST 3 (3000) | Rationale |
|-----------|--------------|---------------|---------------|-----------|
| **Dataset Size** | 300 | 1500 | 3000 | 2× more data |
| **Epochs** | 3 | 3 | **3** | Keep same, monitor for convergence |
| **Batch Size** | 4 | 8 | **8** | Keep (hardware limits on T4) |
| **Gradient Accum** | 4 | 2 | **2** | Maintain effective batch=16 |
| **Effective Batch** | 16 | 16 | **16** | Proven stable |
| **Learning Rate** | 2e-4 | 1.5e-4 | **1.2e-4** | Slightly lower for stability |
| **Warmup Steps** | 2 | 10 | **15** | Longer warmup for more data |
| **Max Length** | 768 | 768 | **768** | Data supports this |
| **Val Split** | 0.2 | 0.1 | **0.1** | 300 samples is plenty |
| **Weight Decay** | 0.01 | 0.01 | **0.02** | Stronger regularization |

### Key Changes Explained:

**Learning Rate: 1.5e-4 → 1.2e-4**
- Reason: 2× more training data requires more careful optimization
- Prevents overshooting local minima
- May need +1 epoch if convergence is slower (monitor eval loss)

**Warmup Steps: 10 → 15**
- Reason: More gradual warmup prevents early instability
- With 2700 training samples, 15 steps = ~0.9% of first epoch
- Helps model stabilize before aggressive learning

**Weight Decay: 0.01 → 0.02**
- Reason: More parameters being updated (more data = more complexity)
- Stronger L2 regularization prevents overfitting
- TEST 2 showed no overfitting, so this is precautionary

**Keep Batch Size = 8**
- Reason: T4 GPU memory constraints
- Effective batch of 16 (8×2) is proven stable
- A100 would allow batch=16, gradient_accum=1, but T4 is sufficient

---

## Slide 19: TEST 3 Expected Outcomes

### Performance Predictions (based on TEST 1→2 trajectory)

**Conservative Estimates:**
| Metric | TEST 2 | TEST 3 Target | Expected Gain |
|--------|--------|---------------|---------------|
| **BLEU** | 0.4594 (45.94%) | **0.52-0.56** | +13-22% |
| **ROUGE** | 0.7156 (71.56%) | **0.75-0.78** | +5-9% |
| **Train Loss End** | 0.11 | **0.08-0.10** | Lower convergence |

**Why Diminishing Returns?**
- TEST 1→2: 5× data = 52% BLEU gain, 29% ROUGE gain
- TEST 2→3: 2× data = expected 15-20% BLEU gain, 5-10% ROUGE gain
- **Law of diminishing returns** applies to data scaling
- But still significant quality improvement expected!

### Training Expectations:
- **Training Time**: ~10-12 hours on T4 GPU
- **Total Steps**: ~506 steps (3 epochs × 169 steps/epoch)
- **Convergence**: Should be smooth, similar to TEST 2
- **Overfitting Risk**: Low (strong regularization + more data)

### Success Criteria:
- ✅ BLEU ≥ 0.52 (+13% minimum)
- ✅ ROUGE ≥ 0.75 (+5% minimum)
- ✅ Smooth loss curves (no spikes)
- ✅ Eval loss within 15% of train loss
- ✅ No degradation in qualitative reading quality

---

## Slide 20: TEST 3 Configuration File

### Updated constants.py for 3000-row dataset

```python
# For 3000-row dataset
RAW_DATASET_URI = f"gs://{BUCKET_NAME}/tarot_training_data_3000.csv"

HYPERPARAMETERS = {
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,        # Keep same
    "per_device_eval_batch_size": 8,          # Keep same
    "gradient_accumulation_steps": 2,         # Keep same
    "learning_rate": 1.2e-4,                  # ↓ from 1.5e-4
    "max_length": 768,                        # Keep same
    "val_split_ratio": 0.1,                   # Keep same
    "warmup_steps": 15,                       # ↑ from 10
    "weight_decay": 0.02,                     # ↑ from 0.01
}
```

### Before Running TEST 3:
1. ✅ Prepare 3000-row dataset and upload to GCS
2. ✅ Update `src/constants.py` with new hyperparameters
3. ✅ Verify GCS bucket permissions (compute SA has objectViewer)
4. ✅ Confirm T4 GPU availability (or upgrade to A100 for faster training)
5. ✅ Clear previous pipeline cache if needed

---

## Slide 21: Timeline & Project Roadmap

### Test 1 ✅ (Completed - Oct 24)
- Dataset: 300 rows
- Training time: ~10 minutes
- Results: **BLEU 30.25%, ROUGE 55.29%**
- Status: Baseline established

### Test 2 ✅ (Completed - Oct 25)
- Dataset: 1500 rows
- Training time: ~6 hours
- Results: **BLEU 45.94% (+51.9%), ROUGE 71.56% (+29.4%)**
- Status: Major improvement confirmed

### Test 3 🎯 (Planned - Next)
- Dataset: 3000 rows
- Est. training time: ~10-12 hours
- Target: **BLEU 52-56%, ROUGE 75-78%**
- Focus: Final quality optimization

### Future Enhancements 🚀
- **Dataset Expansion**: 3000 → 10,000+ readings
- **Model Upgrade**: Phi-3-medium or Phi-3-128k for longer context
- **Deployment**: Vertex AI Endpoint for real-time inference
- **Mobile App**: iOS/Android tarot reading application

---

## Slide 18: Timeline & Next Steps

### Test 1 ✅ (Completed)
- Dataset: 300 rows
- Training time: ~10 minutes
- Results: Baseline established

### Test 2 📋 (Planned)
**Before Running:**
1. Upload tarot_training_data_1000.csv to GCS bucket
2. Update constants.py with new hyperparameters
3. Update RAW_DATASET_URI to point to 1000-row file

**Expected Timeline:**
- Training time: ~20-25 minutes (est.)
- Evaluation: Same pipeline components
- Results analysis: Compare to Test 1

### Test 3 🎯 (Future - 3000 rows)
**Considerations:**
- May need A100 GPU instead of T4
- Longer training time (45-60 min est.)
- Further hyperparameter tuning
- Possible: 4 epochs instead of 3

---

## Slide 22: Cost Optimization & Resource Planning

### Actual Resource Usage (TEST 1 & TEST 2)

| Aspect | TEST 1 (300 rows) | TEST 2 (1500 rows) | TEST 3 (3000 rows - Est.) |
|--------|-------------------|---------------------|---------------------------|
| **Training Time** | ~10 minutes | ~6 hours | ~10-12 hours |
| **Est. Cost** | ~$0.50 | ~$4-5 | ~$8-10 |
| **GPU** | T4 | T4 | T4 (or A100 for 4-5hr) |
| **Resources** | 16 CPU, 50GB, T4 | Same | Same |

### Cost Efficiency Insights:
- **TEST 1→2**: 5× data, 36× time, 8-10× cost
  - Why longer? 5.6× training steps + larger model complexity
- **TEST 2→3**: 2× data, ~2× time, ~2× cost
  - More linear scaling expected

### Optimization Strategies:
✅ **Already Implemented:**
- Data transformation caching (disabled manually to force fresh runs)
- LoRA fine-tuning (99% fewer trainable parameters)
- 4-bit quantization (reduced memory by 75%)
- Efficient batch size (16 effective batch)

🎯 **Future Optimizations:**
- Use Vertex AI Spot VMs for 60-90% cost savings
- Consider A100 GPU for TEST 3 (faster training = lower cost)
- Implement early stopping if eval loss plateaus
- Cache successful component outputs between runs

---

## Slide 18: Timeline & Next Steps
- Still very affordable

**Test 3 Projected Cost:**
- 3000 rows: ~$3.00-4.50
- Consider A100 for faster training

---

## Slide 20: Technical Stack Summary

### Infrastructure
- **Cloud Platform**: Google Cloud Platform
- **Pipeline Orchestration**: Vertex AI Pipelines (Kubeflow)
- **Compute**: T4 GPU, 16 CPU cores, 50GB RAM
- **Storage**: Google Cloud Storage (GCS)

### ML Framework
- **Base Model**: Microsoft Phi-3-mini-4k-instruct (3.8B params)
- **Fine-tuning**: Hugging Face Transformers + PEFT (LoRA)
- **Training**: TRL (Transformer Reinforcement Learning)
- **Quantization**: bitsandbytes (4-bit)
- **Monitoring**: TensorBoard

### Development Tools
- **Language**: Python 3.11
- **Package Manager**: uv
- **Version Control**: Git (GitHub)
- **Local Testing**: Custom validation scripts

---

## Slide 21: Code Repository Structure
```
LLMOps_Project/
├── data_prep/
│   ├── tarot_training_data.csv        # Full dataset
│   └── tarot_training_data_TEST.csv   # 300-row test
├── src/
│   ├── constants.py                   # Hyperparameters
│   ├── pipeline_components/
│   │   ├── data_transformation_component.py
│   │   ├── fine_tuning_component.py
│   │   ├── inference_component.py
│   │   └── evaluation_component.py
│   └── pipelines/
│       └── model_training_pipeline.py
├── scripts/
│   ├── pipeline_runner.py             # Run pipeline
│   └── test_fine_tuning_local.py      # Local testing
├── pipeline_results_test1/
│   ├── tarot_predictions.csv          # Model outputs
│   ├── tarot_evaluation.csv           # With scores
│   └── RESULTS_SUMMARY.md             # Analysis
└── tensorboard_logs/                  # Training metrics
```

---

## Slide 22: Key Takeaways

### What We Accomplished
✅ Built end-to-end MLOps pipeline on GCP
✅ Fine-tuned 3.8B parameter model on custom data
✅ Achieved 55% ROUGE score on creative text generation
✅ Established reproducible pipeline architecture
✅ Created monitoring and evaluation framework

### Technical Mastery
✅ LoRA fine-tuning for efficient adaptation
✅ 4-bit quantization for memory efficiency
✅ Kubeflow pipeline orchestration
✅ Version compatibility debugging
✅ Cloud infrastructure management

### What Makes This Impressive
- **Model Size**: 3.8B parameters (enterprise-scale)
- **Efficiency**: LoRA reduces trainable params by 99%
- **Cost**: <$5 for full experimentation
- **Speed**: 10-minute training cycles
- **Scalability**: Pipeline handles 300→3000+ rows

---

## Slide 23: Future Enhancements

### Model Improvements
1. **Expand Dataset**: 3000 → 10,000+ readings
2. **Multi-task Learning**: Support multiple spread types
3. **Context Window**: Upgrade to Phi-3-medium-128k
4. **Ensemble**: Combine multiple fine-tuned models

### Pipeline Enhancements
1. **A/B Testing**: Compare different hyperparameters
2. **Automated HPO**: Vertex AI Hyperparameter Tuning
3. **Model Registry**: Version control for models
4. **CI/CD Integration**: Automated retraining triggers

### Deployment
1. **Vertex AI Endpoint**: Real-time inference API
2. **Batch Prediction**: Process large volumes
3. **Mobile App**: iOS/Android tarot app
4. **Web Interface**: Interactive tarot readings

---

## Slide 24: Questions & Discussion

### Topics for Discussion
- Hyperparameter choices for Test 2
- Alternative evaluation metrics
- Deployment strategies
- Data augmentation techniques
- Ethical considerations for AI tarot

### Demo
- Live TensorBoard walkthrough
- Sample predictions review
- Pipeline execution demonstration

---

## Slide 25: Appendix - Resources

### Documentation
- Vertex AI Pipelines: https://cloud.google.com/vertex-ai/docs/pipelines
- Hugging Face PEFT: https://huggingface.co/docs/peft
- TRL Documentation: https://huggingface.co/docs/trl
- Phi-3 Model Card: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

### Project Repository
- GitHub: [Your repo link]
- Results: `pipeline_results_test1/`
- TensorBoard Logs: `tensorboard_logs/`

### Contact
- [Your email]
- [Your LinkedIn]

---

## End

**Thank you!**

Questions?


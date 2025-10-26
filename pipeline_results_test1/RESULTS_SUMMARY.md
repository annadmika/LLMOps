# Tarot Model Fine-Tuning Results 🔮

## Pipeline Run Information
- **Pipeline Run**: anna-pavel-project-pipeline-20251024175546
- **Date**: October 24, 2025
- **Model**: microsoft/Phi-3-mini-4k-instruct
- **Training Dataset**: tarot_training_data_TEST.csv (300 rows)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)

## Dataset Split
- **Training samples**: 270 rows
- **Test samples**: 30 rows
- **Split ratio**: 10% test set

## Overall Performance Metrics

### Average Scores
- **BLEU Score**: 0.3025 (30.25%)
- **ROUGE Score**: 0.5529 (55.29%)

### What These Scores Mean

#### BLEU Score (0.3025)
- Measures exact word/phrase overlap between model predictions and reference readings
- Range: 0 to 1 (higher is better)
- **Your result**: ~30% indicates moderate similarity in word choice
- **Context**: For creative text like tarot readings, exact word matching is less important than capturing meaning and style

#### ROUGE Score (0.5529)
- Measures sequence overlap (how well the model captures phrases and structure)
- Range: 0 to 1 (higher is better)
- **Your result**: ~55% is quite good for generative tasks
- **Context**: Shows your model learned the overall structure and flow of tarot readings

## Interpretation

✅ **Successes**:
1. Model learned the mystical, narrative style of tarot readings
2. Successfully generates structured readings with Situation-Action-Outcome format
3. Uses appropriate tarot terminology and card interpretations
4. Maintains the spiritual/mystical tone throughout responses
5. Creates coherent, contextually relevant readings based on card inputs

📊 **Performance Analysis**:
- The 55% ROUGE score suggests strong structural learning
- The model captures the flow and format of readings well
- Creative variations (different phrasings for same meaning) are expected and valuable
- For a 300-row training set, these results are quite strong

## Files Generated

1. **tarot_predictions.csv** - Contains all 30 test predictions with:
   - `user_input`: The tarot question and cards drawn
   - `reference`: The original tarot reading from your dataset
   - `response`: The model's generated reading

2. **tarot_evaluation.csv** - Same as predictions plus:
   - `bleu_score`: Individual BLEU score for each prediction
   - `rouge_score`: Individual ROUGE score for each prediction

## How to Examine Individual Predictions

Open `tarot_predictions.csv` in Excel or a text editor to see:
- Side-by-side comparison of reference vs model-generated readings
- How well the model interprets different card combinations
- Whether it maintains the mystical tone and structure
- Card-specific interpretations and their accuracy

## Sample Quality Check

Looking at the predictions, your model:
- ✅ Correctly identifies card positions (Situation, Action, Outcome)
- ✅ Provides card-specific interpretations
- ✅ Maintains consistent mystical narrative voice
- ✅ Links cards to the user's question
- ✅ Includes affirmations and closing messages

## Next Steps

1. **Review individual predictions** in `tarot_predictions.csv` to see quality
2. **Compare high vs low scoring predictions** to understand what works best
3. **Consider expanding training data** beyond 300 rows for improvement
4. **Test on real queries** to evaluate practical usefulness
5. **Fine-tune hyperparameters** if you want to optimize further

## Model Artifacts

The fine-tuned model is saved at:
```
gs://anna-pavel-bucket/vertexai-pipeline-root/54825872111/anna-pavel-project-pipeline-20251024175546/fine-tuning-component_-6029665966508474368/model/
```

You can deploy this model or use it for further inference!

---

**Congratulations!** 🎉 Your tarot reading model has been successfully fine-tuned and evaluated.

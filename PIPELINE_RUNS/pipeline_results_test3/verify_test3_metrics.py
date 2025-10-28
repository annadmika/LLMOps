import pandas as pd

# Load the evaluation results
df = pd.read_csv("pipeline_results_test3/tarot_evaluation_test3.csv")

print("=== TEST 3 Verification ===\n")
print(f"Total test samples: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")

# Check if we have the metrics
if 'bleu_score' in df.columns and 'rouge_score' in df.columns:
    avg_bleu = df['bleu_score'].mean()
    avg_rouge = df['rouge_score'].mean()
    
    print(f"\nAverage BLEU: {avg_bleu:.4f} ({avg_bleu*100:.2f}%)")
    print(f"Average ROUGE: {avg_rouge:.4f} ({avg_rouge*100:.2f}%)")
    
    print(f"\nBLEU - Min: {df['bleu_score'].min():.4f}, Max: {df['bleu_score'].max():.4f}")
    print(f"ROUGE - Min: {df['rouge_score'].min():.4f}, Max: {df['rouge_score'].max():.4f}")
    
    # Show sample rows
    print("\n=== Sample Predictions (first 2) ===")
    for idx in range(min(2, len(df))):
        print(f"\nSample {idx+1}:")
        print(f"  BLEU: {df.iloc[idx]['bleu_score']:.4f}")
        print(f"  ROUGE: {df.iloc[idx]['rouge_score']:.4f}")
        if 'user_input' in df.columns:
            print(f"  Input: {df.iloc[idx]['user_input'][:100]}...")
        if 'response' in df.columns:
            print(f"  Response: {df.iloc[idx]['response'][:150]}...")
else:
    print(f"\nFirst few rows:")
    print(df.head())

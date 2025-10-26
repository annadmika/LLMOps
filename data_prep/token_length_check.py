import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
df = pd.read_csv("data_prep/tarot_training_data.csv")

lengths = []
for _, row in df.iterrows():
    text = f"{row['prompt']}\n{row['response']}"
    tokens = tokenizer.encode(text)
    lengths.append(len(tokens))

print(f"Mean length: {sum(lengths)/len(lengths):.0f} tokens")
print(f"Max length: {max(lengths)} tokens")
print(f"% over 1536: {sum(1 for l in lengths if l > 1536)/len(lengths)*100:.1f}%")
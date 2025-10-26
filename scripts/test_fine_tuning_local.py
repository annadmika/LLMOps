"""Test fine-tuning locally without quantization (macOS compatible)."""
import logging
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import ast
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fine_tuning():
    """Test fine-tuning process locally."""
    try:
        logger.info("Starting local fine-tuning test...")
        
        # Step 1: Transform the data first (simulating data_transformation_component)
        logger.info("Step 1: Transforming data...")
        raw_data_path = "/Users/annamika/Documents/AlbertSchool - MscY2/LLM_Ops/LLMOps_Project/data_prep/tarot_training_data_TEST.csv"
        
        from datasets import Dataset as HFDataset
        
        raw_df = pd.read_csv(raw_data_path).head(10)  # Only 10 rows for testing
        logger.info(f"Loaded {len(raw_df)} rows of raw data")
        logger.info(f"Raw columns: {raw_df.columns.tolist()}")
        
        # Transform to messages format
        def format_to_messages(row):
            return {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            }
        
        messages_data = [format_to_messages(row) for _, row in raw_df.iterrows()]
        df = pd.DataFrame(messages_data)
        logger.info(f"Transformed data - columns: {df.columns.tolist()}")
        logger.info(f"Sample messages: {df.iloc[0]['messages']}")
        
        # Hyperparameters (reduced for local testing)
        hyperparameters = {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "val_split_ratio": 0.2,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "learning_rate": 3e-4,
            "num_epochs": 1,  # Just 1 epoch for testing
            "batch_size": 1,  # Small batch for macOS
            "max_length": 768,
            "gradient_accumulation_steps": 2,
        }
        
        # Step 2: Parse messages (simulating what fine-tuning component does)
        logger.info("Step 2: Preparing dataset...")
        
        # No need to parse - data is already in correct format
        # Just need to split into train/eval
        full_dataset = HFDataset.from_pandas(df).train_test_split(test_size=0.2)
        train_dataset = full_dataset["train"]
        eval_dataset = full_dataset["test"]
        
        logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
        logger.info(f"Sample message structure: {train_dataset[0]['messages']}")
        
        # Step 3: Load model WITHOUT quantization (for macOS)
        logger.info("Loading model (no quantization for macOS)...")
        model = AutoModelForCausalLM.from_pretrained(
            hyperparameters["model_name"],
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",  # Force CPU on macOS
        )
        
        # Apply LoRA
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=hyperparameters["lora_r"],
            lora_alpha=hyperparameters["lora_alpha"],
            bias="none",
            lora_dropout=hyperparameters["lora_dropout"],
            task_type="CAUSAL_LM",
            target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Create tokenizer
        logger.info("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"])
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        
        # Create training config
        logger.info("Creating training configuration...")
        output_dir = repo_root / "local_test_output"
        output_dir.mkdir(exist_ok=True)
        
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=hyperparameters["batch_size"],
            max_length=hyperparameters["max_length"],  # Changed from max_seq_length
            packing=False,
            num_train_epochs=hyperparameters["num_epochs"],
            learning_rate=hyperparameters["learning_rate"],
            logging_steps=1,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            fp16=False,  # Disable mixed precision for CPU
            report_to="none",  # Disable reporting for local test
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("✅ Fine-tuning test completed successfully!")
        logger.info(f"Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Fine-tuning test failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    test_fine_tuning()

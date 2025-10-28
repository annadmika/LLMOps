"""Fine-tuning component for Vertex AI pipeline."""

from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0",
        "transformers==4.46.*",
        "peft==0.13.2",
        "accelerate==1.10.1",
        "trl==0.17.0",
        "bitsandbytes==0.47.0",
        "datasets==4.0.0",
        "huggingface-hub==0.34.4",
        "safetensors==0.6.2",
        "pandas==2.2.2",
        "numpy==2.0.2",
        "tensorboard",
        "gcsfs",
    ],
)
def fine_tuning_component(
    dataset: Input[Dataset], metrics: Output[Metrics], model: Output[Model]
):
    """Fine-tune a Phi-3 model using LoRA and integrate with Vertex AI."""
    import logging
    import time
    import ast

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig,  # pyright: ignore[reportPrivateImportUsage]
        get_peft_model,  # pyright: ignore[reportPrivateImportUsage]
        prepare_model_for_kbit_training,  # pyright: ignore[reportPrivateImportUsage]
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting fine tuning process...")

        hyperparameters = {
            "model_name": "microsoft/Phi-3-mini-4k-instruct",
            "val_split_ratio": 0.1,  # TEST 3: Reduced for larger dataset (3000 rows)
            "lora_r": 16,  # TEST 3: Increased from 8 for better capacity
            "lora_alpha": 32,  # TEST 3: Increased proportionally (2x lora_r)
            "lora_dropout": 0.1,  # TEST 3: Increased from 0.05 for regularization
            "learning_rate": 1.2e-4,  # TEST 3: Reduced from 1.5e-4 for stability
            "num_epochs": 3,
            "batch_size": 8,  # TEST 3: Increased from 4 for efficiency
            "max_length": 768,
            "gradient_accumulation_steps": 2,  # TEST 3: Reduced since batch_size doubled
            "warmup_steps": 15,  # TEST 3: Increased from 10 for smoother warmup
            "weight_decay": 0.02,  # TEST 3: Added regularization for larger scale
        }

        logger.info("Creating training configurations...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
        lora_config = LoraConfig(
            r=hyperparameters["lora_r"],
            lora_alpha=hyperparameters["lora_alpha"],
            bias="none",
            lora_dropout=hyperparameters["lora_dropout"],
            task_type="CAUSAL_LM",
            target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        )
        sft_config = SFTConfig(
            output_dir=model.path,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
            per_device_train_batch_size=hyperparameters["batch_size"],
            auto_find_batch_size=True,
            max_seq_length=hyperparameters["max_length"],
            packing=False,  # Disabled for very long sequences
            num_train_epochs=hyperparameters["num_epochs"],
            learning_rate=hyperparameters["learning_rate"],
            warmup_steps=hyperparameters["warmup_steps"],  # TEST 3: Added warmup
            weight_decay=hyperparameters["weight_decay"],  # TEST 3: Added regularization
            optim="paged_adamw_8bit",
            logging_steps=1,
            logging_dir=metrics.path,
            report_to="tensorboard",
            bf16=torch.cuda.is_bf16_supported(including_emulation=False),
            fp16=not torch.cuda.is_bf16_supported(including_emulation=False),
            do_eval=True,
            eval_strategy="epoch",
        )

        logger.info("Loading pre-trained model...")
        pre_trained_model = AutoModelForCausalLM.from_pretrained(
            hyperparameters["model_name"],
            device_map="cuda:0",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        pre_trained_model = prepare_model_for_kbit_training(pre_trained_model)
        # Don't apply PEFT here - SFTTrainer will do it with peft_config parameter

        logger.info(f"Loading dataset from {dataset.path}...")
        # Use ast.literal_eval instead of eval for safer parsing
        def parse_messages(x):
            try:
                return ast.literal_eval(x.replace("\n", ","))
            except Exception as e:
                logger.error(f"Error parsing messages: {e}, value: {x[:100]}")
                raise
        
        # OutputPath provides a directory, so we need to read the file inside
        import os
        dataset_file = os.path.join(dataset.path, "train_dataset.csv")
        logger.info(f"Reading from {dataset_file}...")
        df = pd.read_csv(dataset_file)
        logger.info(f"Loaded {len(df)} rows from dataset")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First row sample: {df.iloc[0]['messages'][:200] if 'messages' in df.columns else 'NO MESSAGES COLUMN'}")
        
        df["messages"] = df["messages"].apply(parse_messages)
        full_dataset = Dataset.from_pandas(df).train_test_split(
            test_size=hyperparameters["val_split_ratio"]
        )
        train_dataset, eval_dataset = (
            full_dataset["train"],
            full_dataset["test"],
        )
        logger.info(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

        logger.info("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"])
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id

        logger.info("Starting training...")
        # trl 0.17.0 - minimal parameters only
        trainer = SFTTrainer(
            model=pre_trained_model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
        )
        training_start_time = time.time()
        trainer.train()
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        logger.info(f"Saving model at {model.path}...")
        trainer.save_model(model.path)

        logger.info("Logging metrics...")
        metrics.log_metric("training_time", training_duration)
        for metric_name, metric_value in hyperparameters.items():
            metrics.log_metric(metric_name, metric_value)
            
    except Exception as e:
        logger.error(f"Fine-tuning failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

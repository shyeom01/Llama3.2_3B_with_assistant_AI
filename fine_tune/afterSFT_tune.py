import json
import logging
import os
import gc
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from typing import List, Dict

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # --- Updated Paths and Constants ---
    BASE_MODEL = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b"
    OUTPUT_DIR = "Tune_Results/llama3.2_SFT_Combined"
    
    # --- Load Pre-processed SFT Datasets ---
    logger.info("Loading pre-processed SFT datasets...")
    try:
        # Load OpenMath datasets
        openmath_train = load_dataset("Seono/sft-openmath-train")["train"]
        openmath_eval = load_dataset("Seono/sft-openmath-eval")["train"]
        logger.info(f"Successfully loaded OpenMath datasets: {len(openmath_train)} train, {len(openmath_eval)} eval samples")
        
        # Load SHP datasets
        shp_train = load_dataset("Seono/sft-shp-train")["train"]
        shp_eval = load_dataset("Seono/sft-shp-eval")["train"]
        logger.info(f"Successfully loaded SHP datasets: {len(shp_train)} train, {len(shp_eval)} eval samples")
        
        # Alternative: Load datasets from local disk if uploaded to Hugging Face fails
        # openmath_train = load_from_disk("./sft_datasets/openmath_train")
        # openmath_eval = load_from_disk("./sft_datasets/openmath_eval")
        # shp_train = load_from_disk("./sft_datasets/shp_train")
        # shp_eval = load_from_disk("./sft_datasets/shp_eval")
        
        # Sample datasets for faster training if needed
        # openmath_train = openmath_train.select(range(len(openmath_train)//10))
        # openmath_eval = openmath_eval.select(range(len(openmath_eval)//10))
        # shp_train = shp_train.select(range(len(shp_train)//10))
        # shp_eval = shp_eval.select(range(len(shp_eval)//10))
        
        # Combine datasets
        train_dataset = concatenate_datasets([openmath_train, shp_train])
        eval_dataset = concatenate_datasets([openmath_eval, shp_eval])
        
        # Shuffle datasets
        train_dataset = train_dataset.shuffle(seed=42)
        eval_dataset = eval_dataset.shuffle(seed=42)
        
        logger.info(f"Combined train samples: {len(train_dataset)}")
        logger.info(f"Combined eval samples: {len(eval_dataset)}")
        
        # Log a sample to verify structure
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info("Sample Data:")
            logger.info(f"  Text: {sample['text'][:150]}...")
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

    # --- Load Tokenizer and Model ---
    logger.info(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set EOS token as PAD token.")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # --- Apply LoRA ---
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["gate_proj", "up_proj", "down_proj"]  # Targeting FFN layers for Llama 3.2
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_params)
    model.config.use_cache = False
    model.print_trainable_parameters()

    # Verify trainable parameters
    trainable_params = [(name, param.shape) for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"Trainable parameters ({len(trainable_params)}):")
    for name, shape in trainable_params:
        logger.info(f"  - {name}: {shape}")
    if not trainable_params:
        logger.error("No trainable parameters found. Check LoRA config and target_modules.")
        exit(1)

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=1e-4,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False
    )

        # --- Initialize SFTTrainer ---
    # def preprocess_dataset(dataset):
    #     processed_dataset = dataset.map(
    #         lambda example: {
    #             "text": example["text"] if isinstance(example["text"], str) else str(example["text"])
    #         }
    #     )
    #     return processed_dataset

    # train_dataset = preprocess_dataset(train_dataset)
    # eval_dataset = preprocess_dataset(eval_dataset)

    logger.info(f"Checking dataset structure after loading/concatenating:")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"  Sample type of 'text' column: {type(sample.get('text'))}")
        if isinstance(sample.get('text'), str):
            logger.info(f"  Sample 'text' content start: {sample['text'][:200]}...")
        else:
            logger.info(f"  'text' column content is not a string or missing: {sample.get('text')}")
    else:
        logger.warning("Train dataset is empty!")


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,  
        peft_config=peft_params,
        max_seq_length=512,
        dataset_text_field="text"
    )

    # --- Pre-training Optimizations ---
    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Speed optimization
    model.train()

    # --- Train ---
    logger.info("Starting fine-tuning...")
    train_result = trainer.train(resume_from_checkpoint=False)

    # --- Save Final Model ---
    logger.info("Saving final model...")
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_save_path)
    logger.info(f"Final model adapters saved to: {final_save_path}")
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Tokenizer saved to: {final_save_path}")

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # --- Evaluate on Validation Set ---
    logger.info("Running evaluation on the validation set...")
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    
    # Calculate perplexity if possible
    try:
        perplexity = torch.exp(torch.tensor(eval_metrics["eval_loss"]))
        eval_metrics["perplexity"] = perplexity.item()
        logger.info(f"Validation Perplexity: {perplexity.item()}")
    except Exception as e:
        logger.warning(f"Could not calculate perplexity: {str(e)}")
    
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Fine-tuning completed successfully!")
    logger.info(f"LoRA adapters saved in: {final_save_path}")
    logger.info("To use the fine-tuned model for inference:")
    logger.info("1. Load the base model (Llama 3.2 3B).")
    logger.info(f"2. Load the LoRA adapters from '{final_save_path}'.")
    logger.info("3. Merge the adapters with the base model OR use PeftModel directly for inference.")

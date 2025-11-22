import json
import logging
import os
import gc
import random
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
    BASE_MODEL = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b"  # Adjust this to your path
    OUTPUT_DIR = "Tune_Results/llama3.2_SFT_Combined"
    
    # Downsampling factor for OpenMath dataset (to make training faster)
    OPENMATH_DOWNSAMPLE_FACTOR = 300  # Reduce dataset by this factor
    
    # --- Load Pre-processed SFT Datasets ---
    logger.info("Loading pre-processed SFT datasets...")
    try:
        # Load OpenMath datasets and downsample
        logger.info(f"Loading OpenMath datasets (with 1/{OPENMATH_DOWNSAMPLE_FACTOR} downsampling)...")
        # Try loading from HuggingFace first
        try:
            openmath_train_full = load_from_disk("./datasets/sft_datasets_dedup/openmath_train")
            openmath_eval_full = load_from_disk("./datasets/sft_datasets_dedup/openmath_eval")
            
            logger.info(f"Successfully loaded full OpenMath datasets from HF: {len(openmath_train_full)} train, {len(openmath_eval_full)} eval samples")
        except Exception as e:
            logger.warning(f"Could not load OpenMath datasets from HF, trying local disk: {str(e)}")
            openmath_train_full = load_dataset("Seono/sft-dedup-openmath-train")["train"]
            openmath_eval_full = load_dataset("Seono/sft-dedup-openmath-eval")["train"]
            logger.info(f"Successfully loaded OpenMath datasets from disk: {len(openmath_train_full)} train, {len(openmath_eval_full)} eval samples")
        
        # Downsample OpenMath datasets
        indices_train = random.sample(range(len(openmath_train_full)), max(len(openmath_train_full) // OPENMATH_DOWNSAMPLE_FACTOR, 100))
        indices_eval = random.sample(range(len(openmath_eval_full)), max(len(openmath_eval_full) // OPENMATH_DOWNSAMPLE_FACTOR, 50))
        openmath_train = openmath_train_full.select(indices_train)
        openmath_eval = openmath_eval_full.select(indices_eval)
        logger.info(f"Downsampled OpenMath datasets: {len(openmath_train)} train, {len(openmath_eval)} eval samples")
        
        # Load SHP datasets
        logger.info("Loading SHP datasets...")
        try:
            shp_train = load_from_disk("./datasets/sft_datasets_dedup/shp_train")
            shp_eval = load_from_disk("./datasets/sft_datasets_dedup/shp_eval")
            logger.info(f"Successfully loaded SHP datasets from HF: {len(shp_train)} train, {len(shp_eval)} eval samples")
        except Exception as e:
            logger.warning(f"Could not load SHP datasets from HF, trying local disk: {str(e)}")
            shp_train = load_dataset("Seono/sft-dedup-shp-train")["train"]
            shp_eval = load_dataset("Seono/sft-dedup-shp-eval")["train"]
            logger.info(f"Successfully loaded SHP datasets from disk: {len(shp_train)} train, {len(shp_eval)} eval samples")
        
        # Init flags for additional datasets
        additional_datasets_available = True
        mmlu_available = False
        
        # Load Dolly datasets (direct SFT format, not multiple choice)
        logger.info("Loading Dolly datasets...")
        try:
            dolly_train = load_from_disk("./sft_datasets/dolly_train")
            dolly_eval = load_from_disk("./sft_datasets/dolly_eval")
            logger.info(f"Successfully loaded Dolly datasets: {len(dolly_train)} train, {len(dolly_eval)} eval samples")
        except Exception as e:
            logger.warning(f"Error loading Dolly datasets, will skip: {str(e)}")
            dolly_train = None
            dolly_eval = None
        
        # Load Alpaca datasets (direct SFT format, not multiple choice)
        logger.info("Loading Alpaca datasets...")
        try:
            alpaca_train = load_from_disk("./sft_datasets/alpaca_train")
            alpaca_eval = load_from_disk("./sft_datasets/alpaca_eval")
            logger.info(f"Successfully loaded Alpaca datasets: {len(alpaca_train)} train, {len(alpaca_eval)} eval samples")
        except Exception as e:
            logger.warning(f"Error loading Alpaca datasets, will skip: {str(e)}")
            alpaca_train = None
            alpaca_eval = None
        
        # Load OpenBookQA datasets (direct SFT format, not multiple choice)
        logger.info("Loading OpenBookQA datasets...")
        try:
            obqa_train = load_from_disk("./sft_datasets/openbookqa_train")
            obqa_eval = load_from_disk("./sft_datasets/openbookqa_eval")
            logger.info(f"Successfully loaded OpenBookQA datasets: {len(obqa_train)} train, {len(obqa_eval)} eval samples")
        except Exception as e:
            logger.warning(f"Error loading OpenBookQA datasets, will skip: {str(e)}")
            obqa_train = None
            obqa_eval = None
        
        # Load MMLU datasets (if available)
        logger.info("Loading MMLU datasets...")
        try:
            mmlu_train = load_from_disk("./sft_datasets/mmlu_train")
            mmlu_eval = load_from_disk("./sft_datasets/mmlu_eval")
            logger.info(f"Successfully loaded MMLU datasets: {len(mmlu_train)} train, {len(mmlu_eval)} eval samples")
            mmlu_available = True
        except Exception as e:
            logger.warning(f"Error loading MMLU datasets, will skip: {str(e)}")
            mmlu_train = None
            mmlu_eval = None
            mmlu_available = False
        
        # Check if any additional datasets were loaded
        if dolly_train is None and alpaca_train is None and obqa_train is None and not mmlu_available:
            logger.warning("None of the additional datasets could be loaded.")
            additional_datasets_available = False
        
        # Combine datasets
        train_datasets = [openmath_train, shp_train]
        eval_datasets = [openmath_eval, shp_eval]
        
        # Add additional datasets if available
        if additional_datasets_available:
            if dolly_train is not None:
                train_datasets.append(dolly_train)
                eval_datasets.append(dolly_eval)
            
            if alpaca_train is not None:
                train_datasets.append(alpaca_train)
                eval_datasets.append(alpaca_eval)
            
            if obqa_train is not None:
                train_datasets.append(obqa_train)
                eval_datasets.append(obqa_eval)
            
            if mmlu_available:
                train_datasets.append(mmlu_train)
                eval_datasets.append(mmlu_eval)
        
        # Concatenate all datasets
        train_dataset = concatenate_datasets(train_datasets)
        eval_dataset = concatenate_datasets(eval_datasets)
        
        # Shuffle datasets
        train_dataset = train_dataset.shuffle(seed=42)
        eval_dataset = eval_dataset.shuffle(seed=42)
        
        logger.info(f"Combined train samples: {len(train_dataset)}")
        logger.info(f"Combined eval samples: {len(eval_dataset)}")
        
        # Calculate dataset composition for reporting
        composition_train = {
            "OpenMath": len(openmath_train),
            "SHP": len(shp_train)
        }
        
        composition_eval = {
            "OpenMath": len(openmath_eval),
            "SHP": len(shp_eval)
        }
        
        if additional_datasets_available:
            if dolly_train is not None:
                composition_train["Dolly"] = len(dolly_train)
                composition_eval["Dolly"] = len(dolly_eval)
            
            if alpaca_train is not None:
                composition_train["Alpaca"] = len(alpaca_train)
                composition_eval["Alpaca"] = len(alpaca_eval)
            
            if obqa_train is not None:
                composition_train["OpenBookQA"] = len(obqa_train)
                composition_eval["OpenBookQA"] = len(obqa_eval)
            
            if mmlu_available:
                composition_train["MMLU"] = len(mmlu_train)
                composition_eval["MMLU"] = len(mmlu_eval)
        
        logger.info("Training dataset composition:")
        for dataset_name, count in composition_train.items():
            percentage = (count / len(train_dataset)) * 100
            logger.info(f"  - {dataset_name}: {count} samples ({percentage:.2f}%)")
            
        logger.info("Evaluation dataset composition:")
        for dataset_name, count in composition_eval.items():
            percentage = (count / len(eval_dataset)) * 100
            logger.info(f"  - {dataset_name}: {count} samples ({percentage:.2f}%)")
        
        # Log a sample from each dataset type
        logger.info("Sample data from each dataset type:")
        
        if len(openmath_train) > 0:
            sample = openmath_train[0]
            logger.info("OpenMath Sample:")
            logger.info(f"  Text: {sample['text'][:150]}...")
        
        if len(shp_train) > 0:
            sample = shp_train[0]
            logger.info("SHP Sample:")
            logger.info(f"  Text: {sample['text'][:150]}...")
        
        if additional_datasets_available:
            if dolly_train is not None and len(dolly_train) > 0:
                sample = dolly_train[0]
                logger.info("Dolly Sample:")
                logger.info(f"  Text: {sample['text'][:150]}...")
            
            if alpaca_train is not None and len(alpaca_train) > 0:
                sample = alpaca_train[0]
                logger.info("Alpaca Sample:")
                logger.info(f"  Text: {sample['text'][:150]}...")
            
            if obqa_train is not None and len(obqa_train) > 0:
                sample = obqa_train[0]
                logger.info("OpenBookQA Sample:")
                logger.info(f"  Text: {sample['text'][:150]}...")
            
            if mmlu_available and len(mmlu_train) > 0:
                sample = mmlu_train[0]
                logger.info("MMLU Sample:")
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
        packing=True,
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
    # Uncomment one of the following lines based on whether you want to start fresh or resume
    train_result = trainer.train(resume_from_checkpoint=False)
    # train_result = trainer.train(resume_from_checkpoint="Tune_Results/llama3.2_SFT_Combined/checkpoint-XXXXX")

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
    
    max_seq_length = 512

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    logger.info("Tokenizing evaluation dataset for evaluation...")
    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names 
    )
    logger.info(f"Tokenized evaluation dataset structure: {tokenized_eval_dataset.features}")

    eval_metrics = trainer.evaluate(eval_dataset=tokenized_eval_dataset)
    
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from typing import Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_MODEL = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b"
ADAPTER_PATH = "Tune_Results/llama3.2_SFT_Combined_MC/final_checkpoint"
CACHE_DIR = "./hf_cache" 


def load_model_for_inference(base_model_path: str, adapter_path: str, cache_dir: Optional[str] = None):
    """Load the base model, tokenizer, and apply LoRA adapters."""
    logger.info(f"Loading tokenizer from adapter path: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, cache_dir=cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set PAD token to EOS token: {tokenizer.pad_token}")

    logger.info(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
        device_map="auto",          # Automatically distribute model across devices
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Successfully loaded adapters using PeftModel.")
    except Exception as e:
        logger.error(f"Error loading PeftModel from {adapter_path}: {e}")
        logger.warning("Attempting to load adapters manually if possible (this might fail).")
        # model.load_adapter(adapter_path)

    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256):
    """Format the prompt, generate response, and decode it."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    logger.info("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9, # Nucleus sampling
        # no_repeat_ngram_size=3,
        # early_stopping=True
    )

    logger.info("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

    response_ids = outputs[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    logger.info("Decoding complete.")
    return response_text


if __name__ == "__main__":
    try:
        model, tokenizer = load_model_for_inference(BASE_MODEL, ADAPTER_PATH, CACHE_DIR)

        print("\nStarting conversation with Fine-tuned model. if you quit the session, you can type 'quit' or 'exit'.")
        while True:
            try:
                user_instruction = input("\nUser Question: ")
                if user_instruction.lower() in ["exit", "quit"]:
                    break
                if not user_instruction:
                    continue

                response = generate_response(model, tokenizer, user_instruction)
                print(f"\nModel Response: {response}")

            except EOFError:
                 break
            except KeyboardInterrupt:
                 break
            except Exception as e:
                 logger.error(f"An error occurred during interaction: {e}")
                 print("An error occurred during interaction, please try it again")

        print("\nStop the conversation")

    except Exception as e:
        logger.error(f"Failed to initialize or run inference: {e}")
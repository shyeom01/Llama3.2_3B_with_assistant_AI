# Llama 3.2 3B Fine-tuning on OpenMathInstruct-1

## Overview

This project fine-tunes the Meta Llama 3.2 3B base model on the `nvidia/OpenMathInstruct-1` dataset to enhance its mathematical problem-solving capabilities. It utilizes LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning, specifically targeting only the Feed-Forward Network (FFN) layers.

This README provides a comprehensive guide covering the entire process, from downloading the base model and setting up the environment to running the fine-tuning script and optionally evaluating the resulting model. It aims to help teammates reproduce the workflow.

## Prerequisites

### 1. Hardware
* **Server:** Linux-based server (e.g., SOL supercomputer)
* **GPU:** 1 or more NVIDIA GPUs (A100 40GB/80GB recommended)
* **Memory (RAM):** 160GB or more recommended (for data loading/processing)
* **Storage:**
    * Home Directory (`~`): Sufficient space for Python environment, code (several GBs).
    * Scratch Directory (`/scratch/syeom3` or similar): Sufficient space for the base model, dataset cache, training outputs (checkpoints, adapters) (tens to hundreds of GBs).

### 2. Software
* Git and Git LFS (Requires running `git lfs install --system` once)
* `curl` (for pyenv installation)
* Python (to be installed via `pyenv`)
* `pip` (comes with Python)

### 3. Accounts
* **Hugging Face Account:**
    * Sign up and log in required ([https://huggingface.co/](https://huggingface.co/))
    * **Accept terms of use** on the Hugging Face Hub for both the Llama model and the `nvidia/OpenMathInstruct-1` dataset.
    * Generate a Hugging Face **Access Token** (Settings > Access Tokens > New token, include at least 'Read' permission).
* **(Optional) OpenAI API Key:** Required only if evaluating using MT-Bench with GPT-4/GPT-3.5 judge (incurs costs).

## Setup and Execution Steps

**Note:** All paths are examples and must be adjusted to your specific environment (`syeom3`, server structure, etc.).

### Step 1: Setup Python Environment (using pyenv)

* **Goal:** Create an isolated, stable, and up-to-date Python environment for the project.
* **Location:** Run these commands in your **home directory (`~`)**.

    ```bash
    # 0. (Optional) Check and unload any currently loaded system Python modules
    module list
    # module unload <python_module_name> # e.g., module unload shpc/python/3.9.2-slim/module

    # 1. Install pyenv (skip if already installed)
    cd ~
    # rm -rf ~/.pyenv # Warning! Deletes previous pyenv installation if you want a fresh start
    curl [https://pyenv.run](https://pyenv.run) | bash

    # 2. Configure shell environment (.bashrc or .bash_profile)
    # Add the lines indicated by the installer output to the end of your ~/.bashrc or ~/.bash_profile
    # Example:
    # export PYENV_ROOT="$HOME/.pyenv"
    # [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    # eval "$(pyenv init -)"
    # After adding, restart your terminal or run: source ~/.bashrc (or source ~/.bash_profile)

    # 3. Install Python Build Dependencies (★★★ CRITICAL ★★★)
    # This step depends heavily on the system environment (SOL).
    # Contact SOL administrators or consult documentation for the exact list of packages/modules needed to compile Python 3.12 from source.
    # Examples of required packages: gcc, make, openssl-devel, readline-devel, sqlite-devel, etc.
    # How to install/load them depends on SOL (e.g., module load gcc openssl ...)
    # The next step WILL FAIL without the correct build dependencies!

    # 4. Install desired Python version (e.g., 3.12.3 - stable version recommended)
    # (User used 3.13.3, but 3.12.x is recommended for stability)
    pyenv install 3.12.3 # This will take time (compilation)

    # 5. Set the global Python version for your user
    pyenv global 3.12.3

    # 6. Verify the setup
    python --version   # Expected: Python 3.12.3
    which python       # Expected: /home/syeom3/.pyenv/shims/python
    pip --version      # Expected: pip X.Y.Z from ... (python 3.12)
    ```

### Step 2: Create Project Directory & Install Python Libraries

* **Goal:** Set up a project directory and install necessary Python packages.
* **Location:** Recommend using scratch space due to potentially large outputs.

    ```bash
    # 1. Create project directory and navigate into it (scratch example)
    mkdir -p /scratch/syeom3/openmath_finetune
    cd /scratch/syeom3/openmath_finetune

    # 2. Create requirements.txt file
    # Save the following content as requirements.txt
    # (Versions based on pip show or recent stable releases)
    cat << EOF > requirements.txt
    torch==2.6.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    transformers==4.51.1
    accelerate>=0.30.0 # Use a verified recent version
    peft==0.15.1
    trl==0.16.1
    datasets==3.5.0
    huggingface_hub>=0.20.0
    bitsandbytes # Optional: if using 4/8bit quantization
    EOF

    # 3. Install libraries (ensure pyenv Python environment is active)
    # Use 'python -m pip' if 'pip' command points to the wrong installation
    python -m pip install --user -r requirements.txt

    # 4. (Optional but Recommended) Fix potential 'accelerate' issues
    # python -m pip install --user --upgrade --force-reinstall accelerate
    # pip show accelerate # Check version and location again
    ```

### Step 3: Hugging Face Login

* **Goal:** Authenticate your terminal session to download gated models/datasets.
* **Location:** Any directory (current terminal session).

    ```bash
    huggingface-cli login
    # Paste your Hugging Face Access Token (hf_...) when prompted
    ```

### Step 4: Download Base Model

* **Goal:** Download the Llama 3.2 3B model files.
* **Location:** Can be run from anywhere (uses absolute path for download).

    1.  **Verify/Create Model Directory:** Ensure the target directory exists.
        ```bash
        mkdir -p /scratch/syeom3/Technical_Llama3.2/llama3.2_3b
        ```

    2.  **Choose Download Method:**
        * **(Method A: Python Script - Recommended)** Create `download_model.py` with the content below and run it (`python download_model.py`).
            ```python
            # download_model.py
            from huggingface_hub import snapshot_download
            import os

            # ★★★ Verify the exact model ID on Hugging Face Hub! ★★★
            hf_model_id = "meta-llama/Llama-3.2-3B" # Example ID, confirm!
            local_model_dir = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b"
            hf_token = None # Assumes CLI login was successful

            print(f"Downloading model {hf_model_id} to {local_model_dir}...")
            try:
                snapshot_download(
                    repo_id=hf_model_id,
                    local_dir=local_model_dir,
                    token=hf_token,
                    local_dir_use_symlinks=False,
                    # Example patterns if not downloading everything:
                    # allow_patterns=["*.json", "*.safetensors"],
                    # ignore_patterns=["*.bin", "*.py", "*.md"],
                )
                print("Model download complete.")
            except Exception as e:
                print(f"Error during download: {e}")
                print("Check model ID, HF Hub access terms, login status, network, and disk space.")
            ```
        * **(Method B: Git LFS)** Requires Git LFS setup (`git lfs install --system`).
            ```bash
            # ★★★ Verify the exact model ID on Hugging Face Hub! ★★★
            git clone [https://huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) /scratch/syeom3/Technical_Llama3.2/llama3.2_3b
            ```
    3.  **Crucial:** You must **accept the model's terms of use** on its Hugging Face Hub page before downloading.

### Step 5: Prepare Fine-tuning Script

* **Goal:** Place the `instruct_tune.py` script in the project directory and ensure configurations are correct for your environment.
* **File Location:** Your project directory (e.g., `/scratch/syeom3/openmath_finetune`)

    1.  Copy or create the `instruct_tune.py` file in this directory.
    2.  **Verify and modify key configuration variables** at the top of the script:
        * `BASE_MODEL_PATH`: **Must** match the actual path where you downloaded the model (e.g., `/scratch/syeom3/Technical_Llama3.2/llama3.2_3b`). Check for typos (like `syeom3` vs `jsong132`).
        * `OUTPUT_DIR`: Relative path (e.g., `Tune_Results/...`). Outputs will be saved relative to where you run the script.
        * `DATASET_NAME`: `nvidia/OpenMathInstruct-1` (currently used).
        * `CACHE_DIR`: `./hf_cache` (relative path).
        * **Training Hyperparameters (See A100 recommendations):**
            * `PER_DEVICE_TRAIN_BATCH_SIZE = 8` (or adjust based on VRAM)
            * `GRADIENT_ACCUMULATION_STEPS = 16` (or adjust)
            * `gradient_checkpointing = True` (add to `TrainingArguments`)
            * `MAX_SEQ_LENGTH = 1024` (or experiment with 2048)
            * `LOGGING_STEPS`, `SAVE_STEPS`, `EVAL_STEPS` (e.g., 100, 1000, 1000)
            * `load_best_model_at_end = False` (Set to `False` to avoid the previous `ValueError`; set to `True` only if the underlying library issue is resolved, e.g., by updating libraries).
    3.  Verify that the **`formatting_prompts_func`** function uses the correct column names (`question`, `generated_solution`).
    4.  Verify that the column checking logic (`required_columns = ["instruction", "output"] ... raise ValueError(...)`) inside the **`load_and_prepare_dataset` function** has been **removed or commented out**.
    5.  Verify that the `tokenizer` and `max_seq_length` arguments have been **removed** from the `SFTTrainer(...)` call.

### Step 6: Run Fine-tuning

* **Goal:** Start the model training process.
* **Location:** Run from **the directory containing the script** (e.g., `/scratch/syeom3/openmath_finetune`)

    ```bash
    # 1. Navigate to the script directory
    cd /scratch/syeom3/openmath_finetune

    # 2. Ensure the correct Python environment (pyenv global) is active

    # 3. Run the script
    python instruct_tune.py
    ```
* **During Execution:** Monitor the terminal output for training logs (loss, steps, etc.). You can monitor GPU usage with `nvidia-smi` in another terminal. Training can take a long time depending on settings and data size.
* **Output:** Upon successful completion, LoRA adapter files will be saved in the specified `OUTPUT_DIR` (e.g., `Tune_Results/llama3.2_OpenMath_FFN_Instruction/final_checkpoint`).

### Step 7: (Optional) Evaluation using `lm-evaluation-harness` (Free Method)

* **Goal:** Evaluate the performance of the fine-tuned model (with LoRA adapters applied) on standard benchmarks like MMLU, GSM8K without incurring API costs.
* **Preparation:**
    1.  **Merge LoRA Adapters:** You need to merge the trained LoRA adapters with the base model first, as `lm-evaluation-harness` typically expects a single model path. Use a script (e.g., `merge_lora.py`, refer to previous conversation) to load the base model and adapters, merge them (`model.merge_and_unload()`), and save the merged model to a new directory (e.g., `/scratch/syeom3/merged_models/llama-3.2-3b-openmath-lora`).
    2.  **Install `lm-evaluation-harness`** (Refer to previous conversation: `cd ~`, `git clone ...`, `cd lm-evaluation-harness`, `python -m pip install --user -e .`)

* **Location:** Run from inside the `lm-evaluation-harness` directory (`/home/syeom3/lm-evaluation-harness`).

    ```bash
    # --- Define settings ---
    # ★★★ Path to the MERGED model ★★★
    export MERGED_MODEL_PATH="/scratch/syeom3/merged_models/llama-3.2-3b-openmath-lora"
    # Path to save results
    export RESULTS_PATH="/scratch/syeom3/eval_results/llama-3.2-3b-openmath-lora-mmlu-gsm8k.json"
    # Create results directory
    mkdir -p /scratch/syeom3/eval_results

    # --- Run evaluation (MMLU, GSM8K example) ---
    lm_eval --model hf \
        --model_args pretrained=$MERGED_MODEL_PATH,trust_remote_code=True,dtype=bfloat16 \
        --tasks mmlu,gsm8k \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --output_path $RESULTS_PATH
    ```
* **Output:** Scores will be printed to the terminal, and detailed results saved to the specified JSON file.

### Step 8: Using the Fine-tuned Model (Inference)

* **Goal:** Load the base model and apply the trained LoRA adapters for inference.
* **Method:** Use the `peft` library in a Python script.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    base_model_path = "/scratch/syeom3/Technical_Llama3.2/llama3.2_3b"
    # ★★★ Path to your saved adapters ★★★
    adapter_path = "/scratch/syeom3/openmath_finetune/Tune_Results/llama3.2_OpenMath_FFN_Instruction/final_checkpoint"

    # 1. Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, # Use the same dtype as training
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 2. Load LoRA adapters onto the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 3. (Optional) Merge for faster inference (cannot un-merge after this)
    # model = model.merge_and_unload()

    # 4. Run inference
    prompt = "### Instruction:\nSolve the following math problem: What is the square root of 144?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id) # Added pad_token_id
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```

---

Hopefully, this README provides a clear guide for your teammates. Remember to verify all paths and settings according to your specific environment on the SOL system.

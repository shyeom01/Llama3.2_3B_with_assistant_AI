# Deploying Your Fine-Tuned Model to Hugging Face Spaces

This guide will help you deploy your Streamlit application to Hugging Face Spaces for free.

## Prerequisites

1.  **Hugging Face Account**: You need an account on [huggingface.co](https://huggingface.co/).
2.  **Llama 3.2 License**: Ensure you have accepted the license for `meta-llama/Llama-3.2-3B` on its model card page.
3.  **Access Token**: Get a User Access Token (Write) from your [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Option 1: Automated Deployment (Recommended)
We have included a script to automate the entire process.

1.  Run the script:
    ```bash
    python deploy.py
    ```
2.  Paste your Hugging Face Token (Write permission) when prompted.
3.  Enter a name for your Space.
4.  The script will create the Space, upload the files, and configure the secrets for you.

## Option 2: Manual Deployment

### 1. Create a New Space
1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Space Name**: Give it a name (e.g., `my-finetuned-llama`).
3.  **License**: Choose a license (e.g., MIT).
4.  **SDK**: Select **Streamlit**.
5.  **Hardware**: Select **CPU Basic (Free)**.
    *   *Note*: Llama 3.2 3B might be slow or run out of memory on the free CPU tier. If possible, select a small GPU (paid) or ensure your model is quantized (the code attempts to load in bfloat16/float32, but bitsandbytes quantization might be needed for free tier).
6.  **Visibility**: Public.
7.  Click **Create Space**.

### 2. Upload Files
You can upload files directly via the browser or use git.
**Files to upload:**
- `app.py`
- `requirements.txt`

**Via Browser:**
1.  In your Space, go to the **Files** tab.
2.  Click **Add file** -> **Upload files**.
3.  Drag and drop `app.py` and `requirements.txt`.
4.  Commit changes.

### 3. Configure Secrets (Important!)
Since `meta-llama/Llama-3.2-3B` is a gated model, you need to provide your token.
1.  In your Space, go to **Settings**.
2.  Scroll down to **Variables and secrets**.
3.  Click **New secret**.
4.  **Name**: `HF_TOKEN`
5.  **Value**: Paste your Hugging Face Access Token.
6.  Click **Save**.

### 4. Run
The Space will automatically build and run. You can see the logs in the **App** tab.
Once built, your app will be live!

## Troubleshooting
- **Memory Error**: If the app crashes with an OOM (Out of Memory) error, the model is too big for the free tier. You might need to:
    - Use a smaller base model.
    - Use 4-bit or 8-bit quantization (requires `bitsandbytes` in `requirements.txt` and `load_in_4bit=True` in `app.py`).
    - Upgrade to a GPU Space.

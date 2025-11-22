import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import os

# Page Config
st.set_page_config(
    page_title="Instruction Fine-Tuned Model",
    page_icon="🤖",
    layout="wide"
)

# Title and Description
st.title("🤖 Instruction Fine-Tuned Llama 3.2")
st.markdown("""
This is a simple interface to interact with the fine-tuned Llama 3.2 model.
The model is hosted on Hugging Face: [Seono/Instruction_Fine_Tune](https://huggingface.co/Seono/Instruction_Fine_Tune)
""")

# Sidebar for Configuration
st.sidebar.header("Model Configuration")
base_model_id = st.sidebar.text_input("Base Model ID", value="meta-llama/Llama-3.2-3B")
adapter_model_id = st.sidebar.text_input("Adapter Model ID", value="Seono/Instruction_Fine_Tune")
hf_token = st.sidebar.text_input("Hugging Face Token (Read)", type="password", help="Required for gated models like Llama 3.2")

st.sidebar.divider()
st.sidebar.header("Generation Parameters")
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
max_new_tokens = st.sidebar.slider("Max New Tokens", 64, 1024, 256, 64)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9, 0.1)

# Caching the model loading
@st.cache_resource
def load_model(base_model, adapter_model, token=None):
    try:
        if not token:
             # Try to get from env var if not provided in UI
             token = os.getenv("HF_TOKEN")
        
        login_kwargs = {"token": token} if token else {}

        st.info(f"Loading Tokenizer for {adapter_model}...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_model, trust_remote_code=True, **login_kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        st.info(f"Loading Base Model {base_model} (4-bit Quantization)...")
        
        # Quantization Config for Free Tier (Memory Optimization)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            offload_folder="offload", # Safety net for OOM
            offload_buffers=True,
            **login_kwargs
        )

        st.info(f"Loading Adapter {adapter_model}...")
        model = PeftModel.from_pretrained(
            model, 
            adapter_model, 
            offload_folder="offload",
            **login_kwargs
        )
        model.eval()
        
        st.success("Model Loaded Successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What is your instruction?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if token is provided (if needed)
    if not hf_token and "meta-llama" in base_model_id and not os.getenv("HF_TOKEN"):
        st.warning("Please provide a Hugging Face Token in the sidebar to access the gated Llama model.")
    else:
        # Load model
        model, tokenizer = load_model(base_model_id, adapter_model_id, hf_token)

        if model and tokenizer:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
                        
                        inputs = tokenizer(formatted_prompt, return_tensors="pt", return_attention_mask=True)
                        
                        device = next(model.parameters()).device
                        input_ids = inputs["input_ids"].to(device)
                        attention_mask = inputs["attention_mask"].to(device)

                        generation_config = GenerationConfig(
                            max_new_tokens=max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                        )

                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                generation_config=generation_config
                            )
                        
                        response_ids = outputs[0][input_ids.shape[1]:]
                        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                        
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    except Exception as e:
                        st.error(f"An error occurred during generation: {e}")


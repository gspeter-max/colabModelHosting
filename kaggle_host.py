#!/usr/bin/env python3
"""
Kaggle Model Host — Simple script to run in Kaggle notebooks
Just paste and run in a single cell!

Usage:
    # In a Kaggle notebook cell:
    from kaggle_host import main
    main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
"""

import subprocess
import sys
import os
import time

# ============================================================
# Install dependencies
# ============================================================

def install_deps():
    """Install required packages."""
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "pyngrok",
        "sentencepiece",
        "bitsandbytes",
    ]

    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg]
            )

install_deps()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import model_info
from fastapi import FastAPI, HTTPException
import uvicorn
import uuid
import nest_asyncio
import asyncio

# Apply nest_asyncio for Kaggle/Colab compatibility
nest_asyncio.apply()

# ============================================================
# Detect hardware
# ============================================================

def detect_hardware():
    """Detect available hardware."""
    info = {
        "device": "cpu",
        "dtype": torch.float32,
        "gpu_name": None,
        "gpu_memory_gb": 0,
        "quantize": None,
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )

        if torch.cuda.is_bf16_supported():
            info["dtype"] = torch.bfloat16
        else:
            info["dtype"] = torch.float16

        # Auto-quantize for small GPUs
        if info["gpu_memory_gb"] < 8:
            info["quantize"] = "4bit"
        elif info["gpu_memory_gb"] < 16:
            info["quantize"] = "8bit"

    return info

# ============================================================
# Load model
# ============================================================

def load_model(model_name: str, hw: dict):
    """Download and load model."""

    print(f"\n{'='*60}")
    print(f"MODEL:    {model_name}")
    print(f"DEVICE:   {hw['device']}")
    print(f"DTYPE:    {hw['dtype']}")
    print(f"GPU:      {hw['gpu_name'] or 'None'}")
    print(f"GPU RAM:  {hw['gpu_memory_gb']}GB")
    print(f"QUANTIZE: {hw['quantize'] or 'None'}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
        "torch_dtype": hw["dtype"],
    }

    # Handle quantization
    if hw["quantize"] == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=hw["dtype"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
        print("Using 4-bit quantization")

    elif hw["quantize"] == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["device_map"] = "auto"
        print("Using 8-bit quantization")

    elif hw["device"] == "cuda":
        model_kwargs["device_map"] = "auto"

    # Load model
    print("Loading model (this may take a few minutes)...")
    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    elapsed = round(time.time() - start, 1)
    print(f"Model loaded in {elapsed}s")

    if hw["device"] == "cuda":
        allocated = round(torch.cuda.memory_allocated() / 1e9, 2)
        print(f"GPU memory — allocated: {allocated}GB")

    return model, tokenizer

# ============================================================
# Generate response
# ============================================================

def generate_response(model, tokenizer, messages: list, max_tokens: int = 1024,
                      temperature: float = 0.7, top_p: float = 0.9, device: str = "cuda"):
    """Generate response from chat messages."""

    # Try chat template first
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: manual formatting
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text += f"System: {content}\n"
            elif role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n"
        text += "Assistant: "

    inputs = tokenizer(text, return_tensors="pt")

    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip(), len(new_tokens)

# ============================================================
# Create API
# ============================================================

def create_api(model, tokenizer, hw: dict, model_name: str):
    """Create FastAPI server."""

    app = FastAPI(title="Model Host", version="1.0.0")

    @app.get("/")
    def root():
        return {"status": "running", "model": model_name, "device": hw["device"]}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "self-hosted"}]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 1024)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)

        if not messages:
            raise HTTPException(400, "messages is required")

        try:
            response_text, tokens_used = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=hw["device"],
            )
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {str(e)}")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens_used,
                "total_tokens": tokens_used
            },
        }

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": model_name, "device": hw["device"]}

    return app

# ============================================================
# Setup tunnel
# ============================================================

def setup_tunnel(port: int) -> str:
    """Create public URL using ngrok."""
    try:
        from pyngrok import ngrok
        ngrok.kill()
        tunnel = ngrok.connect(port)
        public_url = tunnel.public_url

        print(f"\n{'='*60}")
        print(f"PUBLIC URL: {public_url}")
        print(f"{'='*60}")
        print(f"\nTest it:")
        print(f"  curl {public_url}/v1/models")
        print(f"\nUse with OpenAI client:")
        print(f'  client = OpenAI(base_url="{public_url}/v1", api_key="none")')

        return public_url
    except Exception as e:
        print(f"Tunnel failed: {e}")
        return ""

# ============================================================
# MAIN
# ============================================================

def main(model_name: str, port: int = 8000):
    """Main entry point for Kaggle."""

    print(f"\n🚀 Kaggle Model Host")
    print(f"{'='*60}")

    # Detect hardware
    hw = detect_hardware()

    # Load model
    model, tokenizer = load_model(model_name, hw)

    # Create API
    app = create_api(model, tokenizer, hw, model_name)

    # Setup tunnel
    setup_tunnel(port)

    print(f"\n{'='*60}")
    print("Server starting...")
    print(f"{'='*60}\n")

    # Run server
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())

    print("\n✅ Server is running! Keep this cell running.\n")

    # Keep the cell alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")

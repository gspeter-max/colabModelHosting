#!/usr/bin/env python3
"""
Universal Model Host — paste into Colab/Kaggle/any GPU machine.
Give it a HuggingFace repo name, it serves an OpenAI-compatible API.

Usage:
    python host.py --model "microsoft/Phi-3-mini-4k-instruct"
    python host.py --model "google/gemma-2b-it"
    python host.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

In Colab, just run the cell and it handles everything.
"""

import subprocess
import sys
import os
import argparse
import json
import time
import threading
from pathlib import Path


# ============================================================
# STEP 1: Auto-install dependencies
# ============================================================

def install_deps():
    """Install all required packages. Works in Colab/Kaggle/bare metal."""
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "fastapi",
        "uvicorn",
        "pyngrok",
        "sentencepiece",
        "protobuf",
    ]

    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg]
            )

    # Try installing bitsandbytes for quantization (optional, GPU only)
    try:
        __import__("bitsandbytes")
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"]
            )
        except Exception:
            print("bitsandbytes not available — quantization disabled")


install_deps()


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, model_info
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import uuid
from datetime import datetime


# ============================================================
# STEP 2: Detect hardware
# ============================================================

def detect_hardware():
    """Detect available hardware and return config."""
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

        # Pick dtype based on GPU
        if torch.cuda.is_bf16_supported():
            info["dtype"] = torch.bfloat16
        else:
            info["dtype"] = torch.float16

        # Auto-quantize for small GPUs
        if info["gpu_memory_gb"] < 8:
            info["quantize"] = "4bit"
        elif info["gpu_memory_gb"] < 16:
            info["quantize"] = "8bit"

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["dtype"] = torch.float16

    return info


# ============================================================
# STEP 3: Download and load model
# ============================================================

def get_model_size_estimate(model_name: str) -> float:
    """Estimate model size in GB from HuggingFace."""
    try:
        info = model_info(model_name)
        siblings = info.siblings or []
        total = sum(
            s.size for s in siblings
            if s.size and s.rfilename.endswith((".safetensors", ".bin"))
        )
        return round(total / 1e9, 1)
    except Exception:
        return 0.0


def load_model(model_name: str, hw: dict):
    """Download and load model with automatic optimization."""

    print(f"\n{'='*60}")
    print(f"MODEL:    {model_name}")
    print(f"DEVICE:   {hw['device']}")
    print(f"DTYPE:    {hw['dtype']}")
    print(f"GPU:      {hw['gpu_name'] or 'None'}")
    print(f"GPU RAM:  {hw['gpu_memory_gb']}GB")
    print(f"QUANTIZE: {hw['quantize'] or 'None'}")
    print(f"{'='*60}\n")

    # Estimate size
    est_size = get_model_size_estimate(model_name)
    if est_size > 0:
        print(f"Estimated model size: {est_size}GB")

        if hw["device"] == "cuda" and est_size > hw["gpu_memory_gb"] * 2:
            print(f"WARNING: Model may be too large for your GPU.")
            print(f"Will attempt 4-bit quantization.")
            hw["quantize"] = "4bit"

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
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=hw["dtype"],
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
            print("Using 4-bit quantization")
        except ImportError:
            print("bitsandbytes not available, loading without quantization")
            model_kwargs["device_map"] = "auto"

    elif hw["quantize"] == "8bit":
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["device_map"] = "auto"
            print("Using 8-bit quantization")
        except ImportError:
            print("bitsandbytes not available, loading without quantization")
            model_kwargs["device_map"] = "auto"

    elif hw["device"] == "cuda":
        model_kwargs["device_map"] = "auto"

    # Load model
    print("Loading model (this may take a few minutes)...")
    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Move to device if not using device_map
    if "device_map" not in model_kwargs and hw["device"] != "cpu":
        model = model.to(hw["device"])

    model.eval()

    elapsed = round(time.time() - start, 1)
    print(f"Model loaded in {elapsed}s")

    # Print memory usage
    if hw["device"] == "cuda":
        allocated = round(torch.cuda.memory_allocated() / 1e9, 2)
        reserved = round(torch.cuda.memory_reserved() / 1e9, 2)
        print(f"GPU memory — allocated: {allocated}GB, reserved: {reserved}GB")

    return model, tokenizer


# ============================================================
# STEP 4: Generation logic
# ============================================================

def generate_response(
    model,
    tokenizer,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
):
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

    # Move inputs to correct device
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    # Generate
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

    # Decode only new tokens
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip(), len(new_tokens)


# ============================================================
# STEP 5: OpenAI-compatible API server
# ============================================================

def create_api(model, tokenizer, hw: dict, model_name: str):
    """Create FastAPI server with OpenAI-compatible endpoints."""

    app = FastAPI(title="Model Host", version="1.0.0")

    @app.get("/")
    def root():
        return {
            "status": "running",
            "model": model_name,
            "device": hw["device"],
            "gpu": hw["gpu_name"],
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "self-hosted",
                }
            ],
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(
                503,
                "GPU out of memory. Try smaller max_tokens or a smaller model.",
            )
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {str(e)}")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens_used,
                "total_tokens": tokens_used,
            },
        }

    @app.get("/health")
    def health():
        gpu_info = {}
        if hw["device"] == "cuda":
            gpu_info = {
                "allocated_gb": round(
                    torch.cuda.memory_allocated() / 1e9, 2
                ),
                "reserved_gb": round(
                    torch.cuda.memory_reserved() / 1e9, 2
                ),
            }
        return {
            "status": "healthy",
            "model": model_name,
            "device": hw["device"],
            "gpu": gpu_info,
        }

    return app


# ============================================================
# STEP 6: Tunnel for Colab/Kaggle (public URL)
# ============================================================

def setup_tunnel(port: int) -> str:
    """Create public URL using ngrok. Returns URL or empty string."""
    try:
        from pyngrok import ngrok

        # Kill existing tunnels
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
        print("Server still running on localhost")
        return ""


# ============================================================
# STEP 7: Colab-specific cell runner
# ============================================================

def is_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


# ============================================================
# MAIN
# ============================================================

def main(model_name: str, port: int = 8000, tunnel: bool = True):
    """Main entry point."""

    print(f"\n🚀 Universal Model Host")
    print(f"{'='*60}")

    # Detect hardware
    hw = detect_hardware()

    # Load model
    model, tokenizer = load_model(model_name, hw)

    # Create API
    app = create_api(model, tokenizer, hw, model_name)

    # Setup tunnel if in Colab/Kaggle
    if tunnel and (is_colab() or is_kaggle()):
        setup_tunnel(port)
    elif tunnel:
        print(f"\nServer will run on: http://localhost:{port}")
        print(f"Test: curl http://localhost:{port}/v1/models")
        print(f"\nUse with OpenAI client:")
        print(
            f'  client = OpenAI(base_url="http://localhost:{port}/v1",'
            f' api_key="none")'
        )

    print(f"\n{'='*60}")
    print("Server starting...")
    print(f"{'='*60}\n")

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=port)


# ============================================================
# ENTRY POINTS
# ============================================================

# --- For Colab: paste this in a cell ---
# !pip install -q torch transformers accelerate huggingface_hub fastapi uvicorn pyngrok sentencepiece
#
# # Then in next cell, paste this entire script, then run:
# main("microsoft/Phi-3-mini-4k-instruct")
#
# # Or for smaller model:
# main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# ----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host any HuggingFace model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. microsoft/Phi-3-mini-4k-instruct)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Disable ngrok tunnel",
    )

    args = parser.parse_args()
    main(args.model, args.port, tunnel=not args.no_tunnel)

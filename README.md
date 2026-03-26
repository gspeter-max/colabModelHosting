# Universal Model Host

Run any HuggingFace model as an OpenAI-compatible API. Works on Colab, Kaggle, or any machine with GPU/CPU.

## Quick Start

### Google Colab (Free GPU)

```python
# Cell 1: Install dependencies
!pip install -q torch transformers accelerate huggingface_hub fastapi uvicorn pyngrok sentencepiece bitsandbytes

# Cell 2: Run the host
# Paste the entire host.py script here, then:

from host import main

# Small model (works on free tier):
main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Better model (needs T4 GPU):
main("microsoft/Phi-3-mini-4k-instruct")
```

### Kaggle

```python
# Same as Colab - just paste and run
from host import main
main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Local Machine

```bash
python host.py --model "microsoft/Phi-3-mini-4k-instruct"
```

## Use with OpenAI Client

Once running, you get a public URL (in Colab/Kaggle) or `http://localhost:8000`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-url.ngrok.io/v1",  # or http://localhost:8000/v1
    api_key="none"  # Not used, but required by SDK
)

response = client.chat.completions.create(
    model="any-name-here",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## API Endpoints

- `GET /` - Server status
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)
- `GET /health` - Health check with GPU memory info

## Features

| Feature | Description |
|---------|-------------|
| **Auto hardware detection** | Detects GPU, CPU, or Apple Silicon |
| **Auto quantization** | 4-bit for small GPUs, 8-bit for medium |
| **Auto dtype** | Uses bfloat16/float16 on supported GPUs |
| **Public URL** | ngrok tunnel in Colab/Kaggle |
| **OpenAI compatible** | Drop-in replacement for OpenAI API |

## Recommended Models

| GPU | Model |
|-----|-------|
| No GPU / CPU | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| T4 Free (16GB) | `microsoft/Phi-3-mini-4k-instruct` |
| T4 Free (16GB) | `google/gemma-2b-it` |
| A100 / V100 | `meta-llama/Llama-2-7b-chat-hf` |

## Tips

- **Free Colab**: The T4 GPU has ~16GB VRAM - use quantized models
- **CPU only**: Will work but will be slow
- **ngrok free tier**: Has rate limits, consider upgrading for heavy use
- **Model size**: Script auto-detects and quantizes if too large

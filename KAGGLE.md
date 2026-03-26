# Kaggle Quick Start

## Option 1: Clone and Run (Recommended)

```python
# Cell 1: Clone and install
!git clone https://github.com/gspeter-max/colabModelHosting.git
%cd colabModelHosting
!pip install -q torch transformers accelerate huggingface_hub fastapi uvicorn pyngrok sentencepiece bitsandbytes

# Cell 2: Run the server
from kaggle_host import main

# Small model (works on free GPU):
main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Or better model (needs T4/P100 GPU):
# main("microsoft/Phi-3-mini-4k-instruct")
```

## Option 2: Copy-Paste Script

1. Copy the entire `kaggle_host.py` file
2. Paste it into a Kaggle cell
3. Run it, then run:

```python
from kaggle_host import main
main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

## Test It (in new notebook)

```python
from openai import OpenAI

# Use the ngrok URL from the server output
client = OpenAI(
    base_url="https://xxxx-xxxx.ngrok.io/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Write a haiku about AI"}]
)

print(response.choices[0].message.content)
```

## Recommended Models for Kaggle

| GPU Type | Model |
|----------|-------|
| T4 (free tier) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| T4 (free tier) | `microsoft/Phi-3-mini-4k-instruct` |
| P100 | `google/gemma-2b-it` |
| V100/A100 | `meta-llama/Llama-2-7b-chat-hf` |

## Notes

- ✅ Works with any Kaggle GPU accelerator
- ✅ Auto-detects hardware and quantizes if needed
- ✅ Creates public URL via ngrok
- ✅ OpenAI-compatible API
- ✅ Keep the cell running to keep server alive

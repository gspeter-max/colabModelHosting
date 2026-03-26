# Colab Quick Copy-Paste

## Cell 1: Install Dependencies

```python
!pip install -q torch transformers accelerate huggingface_hub fastapi uvicorn pyngrok sentencepiece bitsandbytes
```

## Cell 2: Paste host.py Script

Copy the entire `host.py` file and paste it here.

## Cell 3: Run the Server

### Option A: Small Model (Works on Free Tier)

```python
from host import main
main("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Option B: Better Model (Requires T4 GPU - Colab Free)

```python
from host import main
main("microsoft/Phi-3-mini-4k-instruct")
```

### Option C: Gemma 2B (Good balance)

```python
from host import main
main("google/gemma-2b-it")
```

## Cell 4: Test It (in new notebook)

```python
from openai import OpenAI

# Use the ngrok URL from the server output
client = OpenAI(
    base_url="https://xxxx-xx-xx-xx-xx.ngrok.io/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": "Write a haiku about coding"}]
)

print(response.choices[0].message.content)
```

---

**That's it!** The script handles everything else automatically:
- ✅ Detects your GPU
- ✅ Downloads the model
- ✅ Creates a public URL
- ✅ Serves OpenAI-compatible API

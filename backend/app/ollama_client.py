# backend/app/ollama_client.py
import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def ollama_generate(prompt: str, model="llama3"):
    payload = {"model": model, "prompt": prompt}

    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=30)
    r.raise_for_status()

    data = r.json()
    return data.get("response") or data.get("text") or ""

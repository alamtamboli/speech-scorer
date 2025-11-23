# backend/app/groq_client.py
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def groq_complete(prompt: str, model="llama3-8b", max_tokens=256, temperature=0.3):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    r = requests.post(url, json=body, headers=headers, timeout=20)
    r.raise_for_status()

    data = r.json()
    return data["choices"][0]["message"]["content"]

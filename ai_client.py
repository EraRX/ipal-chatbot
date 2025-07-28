# ai_client.py

import os
import openai
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

# Probeer RateLimitError uit de submodule te importeren, anders uit de top-level
try:
    from openai.error import RateLimitError
except ImportError:
    try:
        RateLimitError = openai.RateLimitError  # fallback als package structuur anders is
    except AttributeError:
        # laatste fallback: vang alle exceptions
        RateLimitError = Exception

# Model vanuit ENV, default gpt-4o-mini
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def openai_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 800) -> str:
    """Algemene wrapper om OpenAI ChatCompletion aan te roepen met retry-logica."""
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def rewrite_answer(text: str) -> str:
    """Herschrijf antwoord eenvoudig en vriendelijk."""
    system = "Herschrijf dit antwoord eenvoudig en vriendelijk."
    return openai_chat(
        [{"role": "system", "content": system},
         {"role": "user",   "content": text}],
        temperature=0.2,
        max_tokens=800,
    )

def get_ai_answer(prompt: str) -> str:
    """
    Vraag een antwoord aan de AI.
    - prompt: een volledige user prompt (inclusief module-prefix indien gew

# ai_client.py

import os
import openai
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

# Probeer RateLimitError uit de submodule, anders uit top-level, anders fallback op Exception
try:
    from openai.error import RateLimitError
except ImportError:
    try:
        RateLimitError = openai.RateLimitError
    except AttributeError:
        RateLimitError = Exception

# Model vanuit ENV, default gpt-4o-mini
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def openai_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 800) -> str:
    """
    Algemene helper om een OpenAI ChatCompletion request te doen met retry-logica.
    - messages: lijst van dicts met {'role': ..., 'content': ...}
    - temperature: creativiteit
    - max_tokens: maximum tokens in antwoord
    """
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def rewrite_answer(text: str) -> str:
    """
    Herschrijf een bestaand antwoord eenvoudig en vriendelijk.
    - text: de originele tekst vanuit FAQ of AI
    """
    system_prompt = "Herschrijf dit antwoord eenvoudig en vriendelijk."
    return openai_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text}
        ],
        temperature=0.2,
        max_tokens=800,
    )


def get_ai_answer(prompt: str) -> str:
    """
    Vraag een volledig AI-antwoord op basis van de IPAL Chatbox context.
    - prompt: volledige user prompt, inclusief module-prefix indien gewenst
    """
    system_prompt = "Je bent de IPAL Chatbox, een behulpzame Nederlandse helpdeskassistent."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt}
    ]
    return openai_chat(messages)

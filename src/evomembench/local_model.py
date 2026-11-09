"""Simple OpenAI-compatible worker for a locally hosted open-weight Qwen model."""
from __future__ import annotations

import re
from .hidden_rules import GATES

class LocalWorker:
    def __init__(self, model: str, base_url: str, api_key: str = "local", temperature: float = 0.0):
        if not model or not base_url:
            raise ValueError("Provide model and OpenAI-compatible base URL.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the inference extra with: pip install -e '.[eval,test]'") from exc
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0, max_retries=1)

    def generate(self, system: str, prompt: str, *, max_tokens: int = 128) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content or ""
        usage = response.usage
        return answer, {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

def normalize_answer(text: str) -> str | None:
    # Identical parser for every condition. No condition-specific correction.
    found = re.findall(r"\b(?:" + "|".join(GATES) + r")\b", text.upper())
    return found[-1] if found else None

"""Minimal Kimi OpenAI-compatible client using only the Python standard library."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KimiConfig:
    api_key: str
    model: str
    base_url: str = "https://api.moonshot.cn/v1"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 1024
    timeout: int = 120


class KimiApiError(RuntimeError):
    """Raised when Kimi returns an unsuccessful response."""


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return max(1.0, float(2**attempt))


def load_config() -> KimiConfig:
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY is not set; refusing to run without an explicit environment variable.")
    return KimiConfig(
        api_key=api_key,
        model=os.environ.get("MOONSHOT_MODEL", "kimi-k3.1-pro"),
    )


def complete(config: KimiConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    request = urllib.request.Request(
        f"{config.base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 429 and attempt < 4:
                time.sleep(_retry_delay(error, attempt))
                continue
            raise KimiApiError(f"Kimi HTTP {error.code}: {detail[:500]}") from error
        except urllib.error.URLError as error:
            raise KimiApiError(f"Kimi connection failed: {error.reason}") from error
    body["_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return body


def list_models(config: KimiConfig) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{config.base_url}/models",
        headers={"Authorization": f"Bearer {config.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise KimiApiError(f"Kimi model-list HTTP {error.code}: {detail[:500]}") from error
    except urllib.error.URLError as error:
        raise KimiApiError(f"Kimi model-list connection failed: {error.reason}") from error
    models = body.get("data")
    if not isinstance(models, list):
        raise KimiApiError("Kimi model-list response did not contain a data list.")
    return [model for model in models if isinstance(model, dict)]


def extract_text(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as error:
        raise KimiApiError("Kimi response did not contain choices[0].message.content.") from error

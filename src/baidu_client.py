"""Baidu Qianfan v2 client for DeepSeek-compatible models."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaiduConfig:
    api_key: str
    model: str
    base_url: str = "https://qianfan.baidubce.com/v2"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 8192
    timeout: int = 120


class BaiduApiError(RuntimeError):
    """Raised when Baidu Qianfan returns an unsuccessful response."""


def load_config() -> BaiduConfig:
    api_key = os.environ.get("BAIDU_API_KEY")
    if not api_key:
        raise RuntimeError("BAIDU_API_KEY is not set; refusing to run without an explicit environment variable.")
    return BaiduConfig(
        api_key=api_key,
        model=os.environ.get("BAIDU_MODEL", "deepseek-v4.1-flash"),
    )


def complete(config: BaiduConfig, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
                retry_after = error.headers.get("Retry-After")
                try:
                    delay = max(1.0, float(retry_after)) if retry_after else float(2**attempt)
                except ValueError:
                    delay = float(2**attempt)
                time.sleep(delay)
                continue
            raise BaiduApiError(f"Baidu HTTP {error.code}: {detail[:500]}") from error
        except urllib.error.URLError as error:
            raise BaiduApiError(f"Baidu connection failed: {error.reason}") from error
    body["_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return body


def extract_text(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as error:
        raise BaiduApiError("Baidu response did not contain choices[0].message.content.") from error

"""Model-assisted seven-dimension scoring for completed translation results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from baidu_client import BaiduApiError, BaiduConfig, complete, extract_text, load_config

ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "results" / "baidu_scores_raw"

EVALUATOR_SYSTEM = """You are a strict evaluator of classical Chinese poetry translations.
Evaluate the supplied source and English translation independently. Return only one JSON object
with exactly D1,D2,D3,D4,D5,D6,D7,total. D1-D7 are weighted points with maxima
30,10,15,15,10,10,10. total must equal their sum and be at most 100.
Use these dimensions: D1 key meaning and cultural relations; D2 completeness and unsupported
additions; D3 imagery and rhetoric; D4 emotion, tone and progression; D5 English grammar;
D6 naturalness and readability; D7 style, rhythm and poetic effect. Check negation, time,
agents, direction, causality and image relations before rewarding fluency. Accept reasonable
alternative interpretations. Do not reward extra explanation or author metadata as translation
quality. Do not output markdown or reasons."""


def parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Evaluator did not return a JSON object.")
    value = json.loads(match.group(0))
    fields = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "total")
    if set(value) != set(fields):
        raise ValueError("Evaluator JSON fields are not exactly D1-D7 and total.")
    values = {field: float(value[field]) for field in fields}
    if abs(sum(values[f"D{i}"] for i in range(1, 8)) - values["total"]) > 0.11:
        raise ValueError("Evaluator total does not equal the dimension sum.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config()
    config = BaiduConfig(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=16384,
        timeout=config.timeout,
    )
    source_dir = ROOT / "results" / "baidu_raw"
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.json")):
        output = SCORE_DIR / path.name
        if args.resume and output.exists():
            try:
                if json.loads(output.read_text(encoding="utf-8")).get("status") == "ok":
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        record = json.loads(path.read_text(encoding="utf-8"))
        poem = record["poem"]
        translation = record["response"]["choices"][0]["message"]["content"]
        user_prompt = (
            f"Source title: {poem['title']}\nAuthor: {poem['author']}\n"
            f"Original:\n{poem['text']}\n\nEnglish translation:\n{translation}"
        )
        response: dict[str, Any] | None = None
        try:
            response = complete(config, EVALUATOR_SYSTEM, user_prompt)
            scores = parse_json(extract_text(response))
            result = {
                "status": "ok",
                "poem_id": poem["id"],
                "condition_id": record["condition_id"],
                "repeat": record["repeat"],
                "scores": scores,
                "usage": response.get("usage"),
                "response": response,
            }
        except (BaiduApiError, ValueError, json.JSONDecodeError) as error:
            result = {
                "status": "error",
                "poem_id": poem["id"],
                "condition_id": record["condition_id"],
                "repeat": record["repeat"],
                "error": str(error),
                "response": response if "response" in locals() else None,
            }
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"ERROR {output.name}: {error}", file=sys.stderr)
            raise SystemExit(1)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"scored {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

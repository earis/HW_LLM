"""Run the hw-4 prompt ablation experiment and persist raw results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def build_user_prompt(poem: dict[str, Any]) -> str:
    return (
        f"Translate this poem into literary English.\n"
        f"Title: {poem['title']}\nAuthor: {poem['author']}\n"
        f"Genre: {poem['genre']}\nSource:\n{poem['text']}"
    )


def output_path(raw_dir: Path, poem_id: str, condition: str, repeat: int) -> Path:
    return raw_dir / f"{poem_id}__{condition}__r{repeat}.json"


def run(args: argparse.Namespace) -> int:
    if args.provider == "baidu":
        from baidu_client import BaiduApiError, complete, extract_text, load_config
        api_error = BaiduApiError
    else:
        from kimi_client import KimiApiError, complete, extract_text, list_models, load_config
        api_error = KimiApiError
    config = load_config()
    if args.list_models:
        if args.provider != "kimi":
            raise ValueError("--list-models is currently supported only for --provider kimi.")
        for model in list_models(config):
            print(model.get("id", ""))
        return 0
    poems = read_json(DATA / "poems.json")
    prompt_config = read_json(DATA / "prompts.json")
    conditions = prompt_config["conditions"]
    selected_poems = [poem for poem in poems if poem["split"] == args.split]
    if args.smoke:
        selected_poems = selected_poems[:2]
        selected_conditions = ["C0", "C1", "C2"]
        repeats = 1
    else:
        selected_conditions = list(conditions)
        repeats = args.repeats
    if not selected_poems:
        raise ValueError(f"No poems found for split {args.split!r}.")
    raw_dir = RESULTS / f"{args.provider}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for poem in selected_poems:
        for condition_id in selected_conditions:
            condition = conditions[condition_id]
            system_prompt = prompt_config["base_system"]
            if condition["system_addition"]:
                system_prompt += "\n\n" + condition["system_addition"]
            for repeat in range(1, repeats + 1):
                path = output_path(raw_dir, poem["id"], condition_id, repeat)
                if args.resume and path.exists():
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        existing = {}
                    response = existing.get("response") or {}
                    choices = response.get("choices") or []
                    choice = choices[0] if choices else {}
                    message = choice.get("message") or {}
                    translation = (message.get("content") or "").strip()
                    if existing.get("status") == "ok" and translation and choice.get("finish_reason") == "stop":
                        continue
                user_prompt = build_user_prompt(poem)
                record: dict[str, Any] = {
                    "poem": poem,
                    "condition_id": condition_id,
                    "condition_name": condition["name"],
                    "repeat": repeat,
                    "model": config.model,
                    "system_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
                    "user_prompt": user_prompt,
                    "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                try:
                    response = complete(config, system_prompt, user_prompt)
                    record["translation"] = extract_text(response)
                    record["usage"] = response.get("usage")
                    record["elapsed_seconds"] = response.get("_elapsed_seconds")
                    record["response"] = response
                    record["status"] = "ok"
                except api_error as error:
                    record["status"] = "error"
                    record["error"] = str(error)
                    print(f"ERROR {path.name}: {error}", file=sys.stderr)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                if record["status"] == "error":
                    return 1
                print(f"saved {path.relative_to(ROOT)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run two development poems with C0-C2 once.")
    parser.add_argument("--split", choices=("dev", "validation"), default="dev")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--provider", choices=("kimi", "baidu"), default="kimi")
    parser.add_argument("--list-models", action="store_true", help="List models available to the configured API key.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

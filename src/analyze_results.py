"""Summarize successful raw translations and API usage."""

from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("kimi", "baidu"), default="kimi")
    args = parser.parse_args()
    raw = ROOT / "results" / f"{args.provider}_raw"
    rows = []
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0})
    for path in sorted(raw.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            continue
        usage = record.get("usage") or {}
        row = {
            "poem_id": record["poem"]["id"],
            "split": record["poem"]["split"],
            "condition_id": record["condition_id"],
            "condition_name": record["condition_name"],
            "repeat": record["repeat"],
            "model": record["model"],
            "input_tokens": usage.get("prompt_tokens", ""),
            "output_tokens": usage.get("completion_tokens", ""),
            "elapsed_seconds": record.get("elapsed_seconds", ""),
            "translation": record["translation"],
        }
        rows.append(row)
        total = totals[record["condition_id"]]
        total["count"] += 1
        total["input_tokens"] += usage.get("prompt_tokens", 0) or 0
        total["output_tokens"] += usage.get("completion_tokens", 0) or 0
        total["seconds"] += record.get("elapsed_seconds", 0) or 0
    if not rows:
        raise SystemExit("No successful raw results found.")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{args.provider}_translations.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / f"{args.provider}_usage_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("condition_id", "count", "input_tokens", "output_tokens", "seconds"))
        writer.writeheader()
        for condition_id, values in sorted(totals.items()):
            writer.writerow({"condition_id": condition_id, **values})
    print(f"Summarized {len(rows)} translations.")


if __name__ == "__main__":
    main()

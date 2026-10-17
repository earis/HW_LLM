from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.predict import LABELS, make_windows, predict_label_ids


def load_text(path: Path, limit: int, repeats: int) -> str:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return "".join(str(row["text"]) for row in rows) * repeats


def serial_predict(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    window_size: int,
    commit_size: int,
) -> tuple[str, int]:
    position = 0
    output: list[str] = []
    windows = 0
    while position < len(text):
        window_text = text[position : position + window_size]
        labels = predict_label_ids(
            window_text, model, tokenizer, device, 1, window_size, window_size
        )
        windows += 1
        if len(window_text) < window_size:
            submit = len(window_text)
        else:
            search_end = min(commit_size, len(labels))
            sentence_positions = [
                index for index in range(search_end) if LABELS[labels[index]] == "。"
            ]
            submit = (sentence_positions[-1] + 1) if sentence_positions else commit_size
        for char, label_id in zip(window_text[:submit], labels[:submit]):
            output.append(char)
            label = LABELS[label_id]
            if label != "O":
                output.append(label)
        position += submit
    return "".join(output), windows


def batched_predict(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    window_size: int,
    stride: int,
    batch_size: int,
) -> tuple[str, int]:
    labels = predict_label_ids(
        text, model, tokenizer, device, batch_size, window_size, stride
    )
    output: list[str] = []
    for char, label_id in zip(text, labels):
        output.append(char)
        label = LABELS[label_id]
        if label != "O":
            output.append(label)
    return "".join(output), len(make_windows(text, window_size, stride))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(
    name: str,
    runner,
    device: torch.device,
    repeats: int,
) -> dict[str, object]:
    runner()
    synchronize(device)
    durations = []
    window_count = 0
    output_length = 0
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        output, window_count = runner()
        synchronize(device)
        durations.append(time.perf_counter() - started)
        output_length = len(output)
    return {
        "name": name,
        "durations": durations,
        "median_seconds": statistics.median(durations),
        "windows": window_count,
        "output_characters": output_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair serial vs Batch inference benchmark.")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/full"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--text-limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--commit-size", type=int, default=300)
    parser.add_argument("--strides", default="300")
    parser.add_argument("--batch-sizes", default="1,4,8,16")
    parser.add_argument("--data-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    text = load_text(args.test_data, args.text_limit, args.data_repeats)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]

    strides = [int(value) for value in args.strides.split(",")]
    runners = {
        "serial_sentence_boundary": lambda: serial_predict(
            text, model, tokenizer, device, args.window_size, args.commit_size
        )
    }
    for stride in strides:
        for batch_size in batch_sizes:
            runners[f"fixed_window_stride_{stride}_batch_{batch_size}"] = (
                lambda stride=stride, batch_size=batch_size: batched_predict(
                    text,
                    model,
                    tokenizer,
                    device,
                    args.window_size,
                    stride,
                    batch_size,
                )
            )

    order = list(runners)
    measured: dict[str, dict[str, object]] = {}
    for round_index in range(args.repeats):
        if round_index % 3 == 1:
            current_order = list(reversed(order))
        elif round_index % 3 == 2:
            current_order = order[2:] + order[:2]
        else:
            current_order = order
        for name in current_order:
            result = measure(name, runners[name], device, 1)
            measured.setdefault(name, {"durations": [], "windows": 0, "output_characters": 0})
            measured[name]["durations"].extend(result["durations"])
            measured[name]["windows"] = result["windows"]
            measured[name]["output_characters"] = result["output_characters"]

    serial_seconds = statistics.median(measured["serial_sentence_boundary"]["durations"])
    results = []
    for name, value in measured.items():
        median_seconds = statistics.median(value["durations"])
        results.append(
            {
                "name": name,
                "durations": value["durations"],
                "median_seconds": median_seconds,
                "windows": value["windows"],
                "output_characters": value["output_characters"],
                "characters": len(text),
                "characters_per_second": len(text) / median_seconds,
                "speedup_vs_serial": serial_seconds / median_seconds,
            }
        )
    result = {
        "device": str(device),
        "characters": len(text),
        "window_size": args.window_size,
        "commit_size": args.commit_size,
        "strides": strides,
        "warmup": True,
        "repeats": args.repeats,
        "order_policy": "forward, reverse, rotated",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

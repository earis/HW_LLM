from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABELS = ("O", "，", "。", "？", "！", "；", "：")
IGNORE_INDEX = -100


def load_rows(path: Path, limit: int | None) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows if limit is None else rows[:limit]


def make_windows(text: str, window_size: int, stride: int) -> list[tuple[int, str]]:
    if not text:
        return []
    if stride <= 0 or stride > window_size:
        raise ValueError("stride must be in the range (0, window_size]")
    windows = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        windows.append((start, text[start:end]))
        if end == len(text):
            break
        start += stride
    return windows


class WindowDataset(Dataset[dict[str, object]]):
    def __init__(self, windows: list[tuple[int, str]], tokenizer, max_length: int):
        self.windows = windows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        start, text = self.windows[index]
        encoded = self.tokenizer(
            list(text),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        encoded["window_start"] = start
        encoded["window_text_length"] = len(text)
        encoded["word_ids"] = encoded.word_ids()
        return encoded


class WindowCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        starts = [feature.pop("window_start") for feature in features]
        lengths = [feature.pop("window_text_length") for feature in features]
        word_ids = [feature.pop("word_ids") for feature in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        return {"batch": batch, "starts": starts, "lengths": lengths, "word_ids": word_ids}


def predict_label_ids(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    window_size: int,
    stride: int,
) -> str:
    windows = make_windows(text, window_size, stride)
    dataset = WindowDataset(windows, tokenizer, max_length=window_size + 2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=WindowCollator(tokenizer),
    )
    score_sum = torch.zeros((len(text), len(LABELS)), device=device)
    score_count = torch.zeros(len(text), device=device)

    model.eval()
    with torch.no_grad():
        for item in loader:
            batch = {key: value.to(device) for key, value in item["batch"].items()}
            logits = model(**batch).logits
            for batch_index, (start, length) in enumerate(
                zip(item["starts"], item["lengths"])
            ):
                word_ids = item["word_ids"][batch_index]
                seen: set[int] = set()
                for token_index, word_id in enumerate(word_ids):
                    if word_id is None or word_id in seen or word_id >= length:
                        continue
                    seen.add(word_id)
                    position = start + word_id
                    score_sum[position] += logits[batch_index, token_index]
                    score_count[position] += 1

    return (score_sum / score_count.clamp_min(1).unsqueeze(-1)).argmax(dim=-1).tolist()


def predict_text(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    window_size: int,
    stride: int,
) -> str:
    labels = predict_label_ids(
        text, model, tokenizer, device, batch_size, window_size, stride
    )
    output: list[str] = []
    for char, label_id in zip(text, labels):
        output.append(char)
        label = LABELS[label_id]
        if label != "O":
            output.append(label)
    return "".join(output)


def evaluate(
    rows: list[dict[str, object]],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    window_size: int,
    stride: int,
) -> dict[str, float]:
    predictions: list[int] = []
    targets: list[int] = []
    for row in rows:
        text = str(row["text"])
        windows = make_windows(text, window_size, stride)
        # Evaluation rows are short enough for one window in this dataset.
        if len(windows) != 1:
            continue
        predicted_labels = predict_label_ids(
            text, model, tokenizer, device, batch_size, window_size, stride
        )
        predictions.extend(predicted_labels)
        targets.extend(LABELS.index(label) for label in row["labels"])
    non_o = list(range(1, len(LABELS)))
    return {
        "samples": float(len(rows)),
        "tokens": float(len(targets)),
        "accuracy": sum(a == b for a, b in zip(targets, predictions)) / max(len(targets), 1),
        "macro_f1_non_o": f1_score(
            targets, predictions, labels=non_o, average="macro", zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERT punctuation restoration inference.")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/full"))
    parser.add_argument("--text", default="")
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--batch-sizes", default="1,4,8,16")
    parser.add_argument("--window-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results/inference.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    sample_texts = [
        args.text or "孔子曰学而时习之不亦说乎",
        "今天我们学习自然语言处理这是一个很有趣的方向",
    ]
    samples = [
        {
            "input": text,
            "output": predict_text(
                text, model, tokenizer, device, max(batch_sizes), args.window_size, args.stride
            ),
        }
        for text in sample_texts
    ]

    rows = load_rows(args.test_data, args.test_limit)
    metrics = evaluate(
        rows, model, tokenizer, device, max(batch_sizes), args.window_size, args.stride
    )
    benchmark_text = "".join(str(row["text"]) for row in rows) * 5
    benchmark = []
    for batch_size in batch_sizes:
        predict_text(
            benchmark_text[: min(len(benchmark_text), args.window_size)],
            model,
            tokenizer,
            device,
            batch_size,
            args.window_size,
            args.stride,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        predict_text(
            benchmark_text,
            model,
            tokenizer,
            device,
            batch_size,
            args.window_size,
            args.stride,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        benchmark.append(
            {
                "batch_size": batch_size,
                "seconds": elapsed,
                "characters": len(benchmark_text),
                "characters_per_second": len(benchmark_text) / max(elapsed, 1e-9),
            }
        )

    result = {
        "device": str(device),
        "model_dir": str(args.model_dir),
        "window_size": args.window_size,
        "stride": args.stride,
        "samples": samples,
        "test_metrics": metrics,
        "benchmark": benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

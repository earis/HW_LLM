from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


PUNCTUATION_LABELS = ("O", "，", "。", "？", "！", "；", "：")
PUNCTUATION = set(PUNCTUATION_LABELS[1:])
READING_MARK_RE = re.compile(r"\{[^{}\s]{1,16}\}")
SPACE_RE = re.compile(r"\s+")


def normalize_line(line: str, stats: Counter[str] | None = None) -> str:
    """Remove corpus metadata, token spaces, and pronunciation annotations."""
    parts = line.strip().split(maxsplit=1)
    if len(parts) < 2:
        return ""
    text = parts[1]
    if stats is not None:
        matches = list(READING_MARK_RE.finditer(text))
        stats["reading_annotations_removed"] += len(matches)
        stats["reading_characters_removed"] += sum(len(match.group()) for match in matches)
    text = READING_MARK_RE.sub("", text)
    return SPACE_RE.sub("", text)


def sentence_to_labels(
    text: str, stats: Counter[str] | None = None
) -> tuple[str, list[str]]:
    """Convert a punctuated sentence into characters and labels after each character."""
    chars: list[str] = []
    labels: list[str] = []
    pending = "O"

    for char in text:
        if char in PUNCTUATION:
            if chars:
                # If multiple punctuation marks occur together, retain the last
                # supported mark as the boundary label.
                pending = char
            continue
        if unicodedata.category(char).startswith(("P", "S")):
            if stats is not None:
                stats["other_punctuation_removed"] += 1
                stats[f"removed_symbol_{char}"] += 1
            continue
        if char.isspace():
            continue
        if pending != "O" and labels:
            labels[-1] = pending
        chars.append(char)
        labels.append("O")
        pending = "O"

    if pending != "O" and labels:
        labels[-1] = pending
    return "".join(chars), labels


def iter_records(
    raw_path: Path, stats: Counter[str] | None = None
) -> Iterable[dict[str, object]]:
    with raw_path.open("r", encoding="gb18030", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            normalized = normalize_line(line, stats)
            if not normalized:
                continue
            text, labels = sentence_to_labels(normalized, stats)
            if len(text) < 8:
                continue
            yield {
                "id": f"{raw_path.stem}-{line_number}",
                "text": text,
                "labels": labels,
            }


def split_records(
    records: list[dict[str, object]], seed: int = 42
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    val_end = train_end + int(len(shuffled) * 0.1)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(raw_paths: list[Path], output_dir: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    cleaning_counts: Counter[str] = Counter()
    source_counts: dict[str, int] = {}
    for raw_path in raw_paths:
        source_records = list(iter_records(raw_path, cleaning_counts))
        source_counts[raw_path.name] = len(source_records)
        records.extend(source_records)

    train, val, test = split_records(records)
    split_counts = {
        "train": write_jsonl(output_dir / "train.jsonl", train),
        "val": write_jsonl(output_dir / "val.jsonl", val),
        "test": write_jsonl(output_dir / "test.jsonl", test),
    }
    label_counts = Counter(
        label for record in records for label in record["labels"]  # type: ignore[index]
    )
    statistics = {
        "raw_files": [str(path) for path in raw_paths],
        "source_counts": source_counts,
        "total_records": len(records),
        "split_counts": split_counts,
        "label_counts": dict(label_counts),
        "label_names": list(PUNCTUATION_LABELS),
        "cleaning_counts": {
            "reading_annotations_removed": cleaning_counts[
                "reading_annotations_removed"
            ],
            "reading_characters_removed": cleaning_counts[
                "reading_characters_removed"
            ],
            "other_punctuation_removed": cleaning_counts[
                "other_punctuation_removed"
            ],
            "target_punctuation_retained": sum(
                count for label, count in label_counts.items() if label != "O"
            ),
        },
        "removed_symbol_counts": {
            key.removeprefix("removed_symbol_"): value
            for key, value in cleaning_counts.items()
            if key.startswith("removed_symbol_")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare hw3 punctuation labels.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_paths = sorted(args.raw_dir.glob("*.txt"))
    if not raw_paths:
        raise FileNotFoundError(f"No .txt files found in {args.raw_dir}")
    statistics = build_dataset(raw_paths, args.output_dir)
    print(json.dumps(statistics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

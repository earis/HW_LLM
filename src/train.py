from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABELS = ("O", "，", "。", "？", "！", "；", "：")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
IGNORE_INDEX = -100


def load_rows(path: Path, limit: int | None, seed: int) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


class PunctuationDataset(Dataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        encoded = self.tokenizer(
            list(row["text"]),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        labels = [LABEL_TO_ID[label] for label in row["labels"]]
        word_ids = encoded.word_ids()
        seen_word_ids: set[int] = set()
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None or word_id in seen_word_ids:
                aligned_labels.append(IGNORE_INDEX)
            else:
                aligned_labels.append(labels[word_id])
                seen_word_ids.add(word_id)
        encoded["labels"] = aligned_labels
        return encoded


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [label + [IGNORE_INDEX] * (max_length - len(label)) for label in labels],
            dtype=torch.long,
        )
        return batch


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            total_loss += float(output.loss.detach().cpu())
            pred = output.logits.argmax(dim=-1)
            mask = batch["labels"] != IGNORE_INDEX
            predictions.extend(pred[mask].cpu().tolist())
            targets.extend(batch["labels"][mask].cpu().tolist())
    non_o = [label for label in range(1, len(LABELS))]
    return {
        "loss": total_loss / max(len(loader), 1),
        "macro_f1_non_o": f1_score(
            targets, predictions, labels=non_o, average="macro", zero_division=0
        ),
        "accuracy": sum(x == y for x, y in zip(targets, predictions)) / max(len(targets), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for punctuation restoration.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/smoke"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-chinese", cache_dir=args.cache_dir
    )
    train_rows = load_rows(args.data_dir / "train.jsonl", args.train_limit, args.seed)
    val_rows = load_rows(args.data_dir / "val.jsonl", args.val_limit, args.seed)
    collator = Collator(tokenizer)
    train_loader = DataLoader(
        PunctuationDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        PunctuationDataset(val_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        "google-bert/bert-base-chinese",
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
        cache_dir=args.cache_dir,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    history = []

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{output.loss.item():.4f}")
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "history.json").write_text(
        json.dumps({"device": str(device), "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

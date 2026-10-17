from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABELS = ("O", "，", "。", "？", "！", "；", "：")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def load_rows(path: Path, limit: int) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny BERT training smoke test.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-chinese", cache_dir=args.cache_dir
    )
    rows = load_rows(args.data, args.limit)
    encoded = tokenizer(
        [list(row["text"]) for row in rows],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    labels = []
    for index, row in enumerate(rows):
        row_labels = [LABEL_TO_ID[label] for label in row["labels"]]
        word_ids = encoded.word_ids(batch_index=index)
        seen_word_ids: set[int] = set()
        aligned = []
        for word_id in word_ids:
            if word_id is None or word_id in seen_word_ids:
                aligned.append(-100)
            else:
                aligned.append(row_labels[word_id])
                seen_word_ids.add(word_id)
        labels.append(aligned)
    max_label_length = encoded["input_ids"].shape[1]
    labels_tensor = torch.tensor(
        [row_labels[:max_label_length] + [-100] * (max_label_length - len(row_labels))
         for row_labels in labels],
        dtype=torch.long,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        "google-bert/bert-base-chinese",
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
        cache_dir=args.cache_dir,
    ).to(device)
    batch = {key: value.to(device) for key, value in encoded.items()}
    labels_tensor = labels_tensor.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad()
    output = model(**batch, labels=labels_tensor)
    output.loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(**batch).logits.argmax(dim=-1)
    print(
        json.dumps(
            {
                "device": str(device),
                "rows": len(rows),
                "batch_shape": list(encoded["input_ids"].shape),
                "loss_after_forward": float(output.loss.detach().cpu()),
                "prediction_shape": list(prediction.shape),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

"""Train the three ModernBERT-inspired mini-BERT ablations.

The script uses the People's Daily 1998 text files by default, but the data
directory can be overridden with --data-dir. It intentionally keeps the
implementation small and dependency-light: PyTorch is the only runtime
dependency.

Examples:
    python train_ablation.py --prepare-only
    python train_ablation.py --variant all --steps 20000 --seeds 17 29 43
    python train_ablation.py --variant rope_abs --steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "hw3"
    / "1998-上半年-带音(1)"
    / "1998-上半年-带音"
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_length: int = 512
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_size: int = 1024
    window_size: int = 128
    global_every: int = 2
    rope: bool = True


@dataclass
class Sentence:
    tokens: list[str]
    section: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--variant",
        choices=("baseline", "window256", "no_global", "rope_abs", "all"),
        default="all",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def clean_token(token: str) -> str:
    token = re.sub(r"/[A-Za-z]+$", "", token)
    return re.sub(r"\{[^{}]+\}$", "", token)


def read_sentences(data_dir: Path, months: Iterable[int]) -> list[Sentence]:
    sentences: list[Sentence] = []
    for month in months:
        files = sorted(data_dir.glob(f"1998-{month:02d}-*.txt"))
        if not files:
            raise FileNotFoundError(f"No text file found for month {month:02d} in {data_dir}")
        for path in files:
            with path.open("r", encoding="gb18030", errors="replace") as handle:
                for raw_line in handle:
                    fields = raw_line.strip().split()
                    if len(fields) < 2:
                        continue
                    identifier = fields[0]
                    tokens = [clean_token(item) for item in fields[1:]]
                    tokens = [item for item in tokens if item]
                    if tokens:
                        # The second identifier field is a compact page/section label.
                        section = identifier.split("-")[1]
                        sentences.append(Sentence(tokens=tokens, section=section))
    if not sentences:
        raise ValueError(f"No usable sentences found in {data_dir}")
    return sentences


def build_vocab(sentences: list[Sentence], limit: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        for token in sentence.tokens:
            counts[token] = counts.get(token, 0) + 1
    special = ["<pad>", "<unk>", "<mask>", "<cls>", "<sep>"]
    words = sorted(counts, key=lambda item: (-counts[item], item))
    words = words[: max(0, limit - len(special))]
    return {token: index for index, token in enumerate(special + words)}


def encode_sentence(sentence: Sentence, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(token, unk) for token in sentence.tokens]


def make_blocks(
    sentences: list[Sentence],
    vocab: dict[str, int],
    max_length: int,
) -> tuple[Tensor, Tensor]:
    cls_id, sep_id, pad_id = vocab["<cls>"], vocab["<sep>"], vocab["<pad>"]
    encoded: list[list[int]] = []
    labels: list[int] = []
    sections = {value: index for index, value in enumerate(sorted({s.section for s in sentences}))}
    for sentence in sentences:
        ids = encode_sentence(sentence, vocab)[: max_length - 2]
        encoded.append([cls_id, *ids, sep_id])
        labels.append(sections[sentence.section])
    ids_tensor = torch.full((len(encoded), max_length), pad_id, dtype=torch.long)
    for row, ids in enumerate(encoded):
        ids_tensor[row, : len(ids)] = torch.tensor(ids)
    return ids_tensor, torch.tensor(labels, dtype=torch.long)


def make_packed_blocks(
    sentences: list[Sentence],
    vocab: dict[str, int],
    max_length: int,
) -> Tensor:
    cls_id, sep_id, pad_id = vocab["<cls>"], vocab["<sep>"], vocab["<pad>"]
    blocks: list[list[int]] = []
    current = [cls_id]
    for sentence in sentences:
        sentence_ids = encode_sentence(sentence, vocab)
        for token_id in sentence_ids:
            if len(current) >= max_length - 1:
                current.append(sep_id)
                blocks.append(current)
                current = [cls_id]
            current.append(token_id)
        if len(current) >= max_length - 1:
            current.append(sep_id)
            blocks.append(current)
            current = [cls_id]
        else:
            current.append(sep_id)
    if len(current) > 1:
        current.extend([pad_id] * (max_length - len(current)))
        blocks.append(current)
    return torch.tensor(blocks, dtype=torch.long)


def rope_frequencies(length: int, head_dim: int, device: torch.device) -> tuple[Tensor, Tensor]:
    positions = torch.arange(length, device=device, dtype=torch.float32)
    inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(positions, inverse)
    return angles.cos(), angles.sin()


def apply_rope(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    length, head_dim = query.shape[-2], query.shape[-1]
    cos, sin = rope_frequencies(length, head_dim, query.device)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]

    def rotate(value: Tensor) -> Tensor:
        even, odd = value[..., 0::2], value[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return value * cos.repeat_interleave(2, dim=-1) + rotated * sin.repeat_interleave(2, dim=-1)

    return rotate(query), rotate(key)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, use_global: bool):
        super().__init__()
        self.config = config
        self.use_global = use_global
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        head_dim = width // self.config.num_heads
        qkv = self.qkv(hidden).view(batch, length, 3, self.config.num_heads, head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.config.rope:
            query, key = apply_rope(query, key)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if not self.use_global:
            positions = torch.arange(length, device=hidden.device)
            distance = (positions[:, None] - positions[None, :]).abs()
            scores = scores.masked_fill(distance > self.config.window_size, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, width)
        return self.output(context)


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, use_global: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = Attention(config, use_global)
        self.ffn_in = nn.Linear(config.hidden_size, config.ffn_size * 2)
        self.ffn_out = nn.Linear(config.ffn_size, config.hidden_size)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.norm1(hidden))
        gated, value = self.ffn_in(self.norm2(hidden)).chunk(2, dim=-1)
        return hidden + self.ffn_out(F.gelu(gated) * value)


class MiniBertForMLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.absolute_position = nn.Embedding(config.max_length, config.hidden_size)
        self.layers = nn.ModuleList(
            EncoderLayer(
                config,
                use_global=config.global_every > 0 and (index + 1) % config.global_every == 0,
            )
            for index in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.mlm = nn.Linear(config.hidden_size, config.vocab_size)

    def encode(self, ids: Tensor) -> Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.embedding(ids)
        if not self.config.rope:
            hidden = hidden + self.absolute_position(positions)[None, :, :]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.final_norm(hidden)

    def forward(self, ids: Tensor) -> Tensor:
        return self.mlm(self.encode(ids))


def mask_tokens(
    ids: Tensor,
    mask_id: int,
    pad_id: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    probability = torch.full(ids.shape, 0.15, device=ids.device)
    probability[ids == pad_id] = 0
    probability[:, 0] = 0
    probability[:, -1] = 0
    masked = torch.bernoulli(probability, generator=generator).bool()
    labels = ids.masked_fill(~masked, -100)
    inputs = ids.clone()
    random_values = torch.randint(vocab_size, ids.shape, generator=generator, device=ids.device)
    replace_random = masked & (torch.rand(ids.shape, generator=generator, device=ids.device) < 0.1)
    replace_mask = masked & ~replace_random
    inputs[replace_mask] = mask_id
    inputs[replace_random] = random_values[replace_random]
    return inputs, labels


def train_one(
    model: MiniBertForMLM,
    ids: Tensor,
    device: torch.device,
    steps: int,
    batch_size: int,
    accumulation: int,
    seed: int,
    vocab: dict[str, int],
) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(accumulation):
            indices = torch.randint(ids.shape[0], (batch_size,), generator=generator, device=device)
            batch = ids[indices]
            inputs, labels = mask_tokens(
                batch, vocab["<mask>"], vocab["<pad>"], len(vocab), generator
            )
            loss = F.cross_entropy(model(inputs).reshape(-1, len(vocab)), labels.reshape(-1))
            (loss / accumulation).backward()
            loss_total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step={step + 1}/{steps} loss={loss_total / accumulation:.4f}", flush=True)
    return time.perf_counter() - start


@torch.no_grad()
def evaluate_mlm(model: MiniBertForMLM, ids: Tensor, device: torch.device, vocab: dict[str, int]) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(1001)
    batch = ids[: min(256, len(ids))].to(device)
    inputs, labels = mask_tokens(
        batch, vocab["<mask>"], vocab["<pad>"], len(vocab), generator
    )
    return F.cross_entropy(model(inputs).reshape(-1, len(vocab)), labels.reshape(-1)).item()


@torch.no_grad()
def sentence_features(
    model: MiniBertForMLM,
    ids: Tensor,
    device: torch.device,
    pad_id: int,
    batch_size: int = 64,
) -> Tensor:
    model.eval()
    features = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size].to(device)
        hidden = model.encode(batch)
        valid = (batch != pad_id).unsqueeze(-1)
        features.append((hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1))
    return torch.cat(features)


def evaluate_probe(
    model: MiniBertForMLM,
    train_ids: Tensor,
    train_labels: Tensor,
    test_ids: Tensor,
    test_labels: Tensor,
    device: torch.device,
    pad_id: int,
    epochs: int,
) -> float:
    train_features = sentence_features(model, train_ids, device, pad_id).detach()
    test_features = sentence_features(model, test_ids, device, pad_id).detach()
    probe = nn.Linear(train_features.shape[-1], int(train_labels.max().item() + 1)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        logits = probe(train_features)
        loss = F.cross_entropy(logits, train_labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    prediction = probe(test_features).argmax(dim=-1)
    return (prediction == test_labels.to(device)).float().mean().item()


@torch.no_grad()
def benchmark_inference(
    model: MiniBertForMLM,
    vocab_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    measurements: dict[str, float] = {}
    for length in (128, 256, max_length):
        if length > max_length:
            continue
        inputs = torch.randint(vocab_size, (1, length), device=device)
        for _ in range(2):
            model.encode(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        repeats = 3
        for _ in range(repeats):
            model.encode(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / repeats
        measurements[f"latency_{length}_ms"] = round(elapsed_ms, 3)
        measurements[f"tokens_per_second_{length}"] = round(
            length / (elapsed_ms / 1000), 2
        )
    return measurements


def run_variant(
    name: str,
    base: ModelConfig,
    train_mlm_ids: Tensor,
    validation_mlm_ids: Tensor,
    test_mlm_ids: Tensor,
    train_sentence_ids: Tensor,
    train_labels: Tensor,
    test_sentence_ids: Tensor,
    test_labels: Tensor,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    vocab: dict[str, int],
) -> dict[str, object]:
    config = base
    if name == "window256":
        config = ModelConfig(**{**asdict(base), "window_size": 256})
    elif name == "no_global":
        config = ModelConfig(**{**asdict(base), "global_every": 0})
    elif name == "rope_abs":
        config = ModelConfig(**{**asdict(base), "rope": False})
    torch.manual_seed(seed)
    model = MiniBertForMLM(config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_seconds = train_one(
        model, train_mlm_ids.to(device), device, args.steps, args.batch_size,
        args.grad_accumulation, seed, vocab,
    )
    result = {
        "variant": name,
        "seed": seed,
        "config": asdict(config),
        "train_steps": args.steps,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "tokens_seen": args.steps * args.batch_size * args.grad_accumulation * args.max_length,
        "train_seconds": round(train_seconds, 2),
        "validation_mlm_loss": round(
            evaluate_mlm(model, validation_mlm_ids, device, vocab), 5
        ),
        "heldout_mlm_loss": round(
            evaluate_mlm(model, test_mlm_ids, device, vocab), 5
        ),
        "section_probe_accuracy": round(
            evaluate_probe(
                model, train_sentence_ids, train_labels, test_sentence_ids,
                test_labels, device, vocab["<pad>"], args.probe_epochs,
            ),
            5,
        ),
    }
    result.update(benchmark_inference(model, len(vocab), args.max_length, device))
    result["peak_memory_mb"] = round(
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda" else 0.0,
        2,
    )
    if args.save_checkpoints:
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{name}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "variant": name,
                "seed": seed,
            },
            checkpoint_path,
        )
        result["checkpoint"] = str(checkpoint_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    train_sentences = read_sentences(args.data_dir, range(1, 5))
    validation_sentences = read_sentences(args.data_dir, [5])
    test_sentences = read_sentences(args.data_dir, [6])
    vocab = build_vocab(train_sentences, args.vocab_size)
    train_sentence_ids, train_labels = make_blocks(
        train_sentences, vocab, args.max_length
    )
    train_ids = make_packed_blocks(train_sentences, vocab, args.max_length)
    validation_ids, validation_labels = make_blocks(
        validation_sentences, vocab, args.max_length
    )
    validation_mlm_ids = make_packed_blocks(
        validation_sentences, vocab, args.max_length
    )
    test_ids, test_labels = make_blocks(test_sentences, vocab, args.max_length)
    test_mlm_ids = make_packed_blocks(test_sentences, vocab, args.max_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "train_sentences": len(train_sentences),
        "validation_sentences": len(validation_sentences),
        "test_sentences": len(test_sentences),
        "train_sentence_blocks": len(train_sentence_ids),
        "train_mlm_blocks": len(train_ids),
        "validation_blocks": len(validation_ids),
        "test_blocks": len(test_ids),
        "train_sections": int(train_labels.max().item() + 1),
        "validation_sections": int(validation_labels.max().item() + 1),
        "test_sections": int(test_labels.max().item() + 1),
        "vocab_size": len(vocab),
        "max_length": args.max_length,
        "data_dir": str(args.data_dir),
    }
    (args.output_dir / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    base = ModelConfig(vocab_size=len(vocab), max_length=args.max_length)
    variants = ["baseline", "window256", "no_global", "rope_abs"] if args.variant == "all" else [args.variant]
    results: list[dict[str, object]] = []
    print(f"device={device} variants={variants}", flush=True)
    for variant in variants:
        for seed in args.seeds:
            print(f"=== variant={variant} seed={seed} ===", flush=True)
            results.append(
                run_variant(
                    variant, base, train_ids, validation_mlm_ids, test_mlm_ids,
                    train_sentence_ids, train_labels, test_ids, test_labels,
                    device, args, seed, vocab,
                )
            )
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            grouped: dict[str, list[dict[str, object]]] = {}
            for item in results:
                grouped.setdefault(str(item["variant"]), []).append(item)
            summary: dict[str, object] = {}
            metric_names = (
                "validation_mlm_loss",
                "heldout_mlm_loss",
                "section_probe_accuracy",
                "train_seconds",
                "peak_memory_mb",
                "latency_128_ms",
                "latency_256_ms",
                "latency_512_ms",
            )
            for variant_name, items in grouped.items():
                summary[variant_name] = {}
                for metric in metric_names:
                    values = [float(item[metric]) for item in items if metric in item]
                    if not values:
                        continue
                    mean = sum(values) / len(values)
                    std = (
                        sum((value - mean) ** 2 for value in values) / len(values)
                    ) ** 0.5
                    summary[variant_name][f"{metric}_mean"] = round(mean, 5)
                    summary[variant_name][f"{metric}_std"] = round(std, 5)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()

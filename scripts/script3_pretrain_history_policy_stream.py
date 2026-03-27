#!/usr/bin/env python3
"""Streamed pretraining entrypoint for large multi-player JSONL on limited RAM."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from history_policy_lib import (
    PolicySample,
    FactorizedPolicyModel,
    batch_loss,
    collate_batch,
    evaluate,
    format_eta,
    save_checkpoint,
    set_seed,
)
from pipeline_config import HISTORY_PLIES, PRETRAIN_DATASET_TAG, TARGET_USERNAME, models_dir, processed_dir


def normalize_player_id(name: str) -> str:
    """Normalize player name into player_id format used in encoded samples."""
    return " ".join(name.strip().split()).lower()


CONFIG: Dict[str, Any] = {
    "dataset_tag": PRETRAIN_DATASET_TAG,
    "history_plies": HISTORY_PLIES,
    "input_jsonl": processed_dir(PRETRAIN_DATASET_TAG) / "policy_samples.jsonl",
    "model_output_path": models_dir(PRETRAIN_DATASET_TAG) / "history_policy_pretrain_stream.pt",
    "metrics_output_path": models_dir(PRETRAIN_DATASET_TAG) / "history_policy_pretrain_stream_metrics.json",
    "valid_size": 0.10,
    "random_seed": 42,
    "epochs": 8,
    "epochs_cpu": 4,
    "batch_size": 256,
    "batch_size_cpu": 48,
    "print_every": 20,
    "eval_batch_size": 256,
    "eval_batch_size_cpu": 128,
    "grad_accum_steps": 1,
    "grad_accum_steps_cpu": 2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "feature_vocab_size": 81924,
    "feature_embed_dim": 64,
    "dense_state_dim": 6,
    "context_dim": 2,
    "history_hidden_dim": 96,
    "shared_hidden_dim": 128,
    "dropout": 0.10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Set strict_target_isolation=True to remove target player from pretraining data.
    "strict_target_isolation": False,
    "excluded_player_ids": [normalize_player_id(TARGET_USERNAME)],
    # Streaming-specific controls.
    "shuffle_buffer_size": 20000,
    "scan_progress_every": 50000,
    "stream_progress_every": 100000,
    "train_eval_max_samples": 50000,
    "valid_eval_max_samples": 50000,
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_to_sample(row: Dict[str, Any]) -> PolicySample:
    """Convert one JSON row into PolicySample."""
    return PolicySample(
        game_id=str(row.get("game_id", "")),
        player_id=str(row.get("player_id", "unknown")),
        active_feature_indices=[int(x) for x in row["active_feature_indices"]],
        dense_state=[float(x) for x in row["dense_state"]],
        history_event=[[float(v) for v in x] for x in row["history_event"]],
        history_delta=[[float(v) for v in x] for x in row["history_delta"]],
        context=[float(x) for x in row["context"]],
        legal_from_mask=[int(x) for x in row["legal_from_mask"]],
        legal_to_by_from={str(k): [int(v) for v in vals] for k, vals in row["legal_to_by_from"].items()},
        target_from_sq=int(row["target_from_sq"]),
        target_to_sq=int(row["target_to_sq"]),
        target_promotion=int(row.get("target_promotion", 0)),
    )


def keep_sample(player_id: str, config: Dict[str, Any]) -> bool:
    """Apply optional target-isolation filter."""
    if not bool(config.get("strict_target_isolation", False)):
        return True
    excluded = {str(x).strip().lower() for x in config.get("excluded_player_ids", []) if str(x).strip()}
    return str(player_id).strip().lower() not in excluded


def split_hash_is_valid(game_id: str, valid_size: float, seed: int) -> bool:
    """Deterministically assign a game_id to valid/train without storing all IDs."""
    key = f"{seed}:{game_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha1(key).digest()[:8], byteorder="big")
    ratio = bucket / float(2**64)
    return ratio < valid_size


def reservoir_push(reservoir: List[PolicySample], sample: PolicySample, cap: int, seen: int, rng: random.Random) -> None:
    """Maintain a reservoir sample of fixed size from a stream."""
    if cap <= 0:
        return
    if len(reservoir) < cap:
        reservoir.append(sample)
        return
    j = rng.randrange(seen)
    if j < cap:
        reservoir[j] = sample


def scan_stream_metadata(config: Dict[str, Any]) -> Tuple[Dict[str, int], List[PolicySample], List[PolicySample]]:
    """One streaming pass to count split sizes and build eval reservoirs."""
    in_path = Path(config["input_jsonl"])
    valid_size = float(config["valid_size"])
    seed = int(config["random_seed"])
    progress_every = int(config.get("scan_progress_every", 0))

    train_eval_cap = int(config.get("train_eval_max_samples", 0) or 0)
    valid_eval_cap = int(config.get("valid_eval_max_samples", 0) or 0)

    train_eval: List[PolicySample] = []
    valid_eval: List[PolicySample] = []

    train_eval_seen = 0
    valid_eval_seen = 0

    counts = {
        "rows_total": 0,
        "rows_kept": 0,
        "rows_filtered_out": 0,
        "train_rows": 0,
        "valid_rows": 0,
    }

    rng = random.Random(seed + 17)
    start = time.time()

    for row in iter_jsonl(in_path):
        counts["rows_total"] += 1

        player_id = str(row.get("player_id", "unknown"))
        if not keep_sample(player_id, config):
            counts["rows_filtered_out"] += 1
            continue

        counts["rows_kept"] += 1
        game_id = str(row.get("game_id", ""))
        is_valid = split_hash_is_valid(game_id, valid_size=valid_size, seed=seed)

        sample = row_to_sample(row)

        if is_valid:
            counts["valid_rows"] += 1
            valid_eval_seen += 1
            reservoir_push(valid_eval, sample, valid_eval_cap, valid_eval_seen, rng)
        else:
            counts["train_rows"] += 1
            train_eval_seen += 1
            reservoir_push(train_eval, sample, train_eval_cap, train_eval_seen, rng)

        if progress_every > 0 and counts["rows_total"] % progress_every == 0:
            elapsed = time.time() - start
            rate = counts["rows_total"] / max(elapsed, 1e-6)
            print(
                "[scan] "
                f"rows={counts['rows_total']:,} kept={counts['rows_kept']:,} "
                f"train={counts['train_rows']:,} valid={counts['valid_rows']:,} "
                f"speed={rate:,.0f} rows/s"
            )

    elapsed = time.time() - start
    rate = counts["rows_total"] / max(elapsed, 1e-6) if counts["rows_total"] > 0 else 0.0
    print(
        "[scan] done | "
        f"rows={counts['rows_total']:,} kept={counts['rows_kept']:,} "
        f"train={counts['train_rows']:,} valid={counts['valid_rows']:,} "
        f"filtered={counts['rows_filtered_out']:,} | "
        f"elapsed={elapsed:.1f}s speed={rate:,.0f} rows/s"
    )

    if counts["train_rows"] == 0 or counts["valid_rows"] == 0:
        raise ValueError("Deterministic split produced empty train or valid set")

    return counts, train_eval, valid_eval


def iter_train_batches_stream(
    config: Dict[str, Any],
    epoch_seed: int,
    batch_size: int,
) -> Iterable[List[PolicySample]]:
    """Yield shuffled train mini-batches from stream with bounded memory."""
    in_path = Path(config["input_jsonl"])
    valid_size = float(config["valid_size"])
    seed = int(config["random_seed"])
    buffer_size = int(config.get("shuffle_buffer_size", 20000))

    if buffer_size < batch_size:
        buffer_size = batch_size

    rng = random.Random(epoch_seed)

    buffer: List[PolicySample] = []
    carry: List[PolicySample] = []

    for row in iter_jsonl(in_path):
        player_id = str(row.get("player_id", "unknown"))
        if not keep_sample(player_id, config):
            continue

        game_id = str(row.get("game_id", ""))
        if split_hash_is_valid(game_id, valid_size=valid_size, seed=seed):
            continue

        buffer.append(row_to_sample(row))

        if len(buffer) >= buffer_size:
            rng.shuffle(buffer)
            mixed = carry + buffer
            full = (len(mixed) // batch_size) * batch_size
            for i in range(0, full, batch_size):
                yield mixed[i:i + batch_size]
            carry = mixed[full:]
            buffer = []

    if buffer:
        rng.shuffle(buffer)
        mixed = carry + buffer
    else:
        mixed = carry

    for i in range(0, len(mixed), batch_size):
        yield mixed[i:i + batch_size]


def train_one_epoch_stream(
    model: FactorizedPolicyModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict[str, Any],
    epoch_index: int,
    total_epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    n_train_rows: int,
) -> float:
    """Stream one epoch from disk with bounded RAM and periodic progress logs."""
    model.train()

    progress_every = int(config.get("print_every", 20))
    stream_progress_every = int(config.get("stream_progress_every", 0))

    n_batches = max(1, math.ceil(n_train_rows / batch_size))
    total_loss = 0.0
    step = 0
    rows_seen = 0
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for batch_samples in iter_train_batches_stream(config, epoch_seed=int(config["random_seed"]) + epoch_index, batch_size=batch_size):
        if not batch_samples:
            continue

        step += 1
        rows_seen += len(batch_samples)

        batch = collate_batch(batch_samples, device=device)
        loss = batch_loss(model, batch)
        loss_value = float(loss.item())

        if not math.isfinite(loss_value):
            print(
                f"[warn] Non-finite loss at epoch {epoch_index}, step {step}. "
                "Skipping this batch update."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if loss_value > 1e4:
            print(f"[warn] Loss spike detected at epoch {epoch_index}, step {step}: loss={loss_value:.2f}")

        (loss / grad_accum_steps).backward()

        if step % grad_accum_steps == 0 or step == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss_value

        if step % progress_every == 0 or step == n_batches:
            elapsed = time.time() - start
            rate = step / max(elapsed, 1e-6)
            eta = (n_batches - step) / max(rate, 1e-6)
            avg_loss = total_loss / step
            print(
                f"Epoch {epoch_index:02d}/{total_epochs:02d} | "
                f"step {step:4d}/{n_batches:4d} | "
                f"rows={rows_seen:,}/{n_train_rows:,} | "
                f"loss={avg_loss:.4f} | "
                f"eta={format_eta(eta)}"
            )

        if stream_progress_every > 0 and rows_seen % stream_progress_every == 0:
            elapsed = time.time() - start
            speed = rows_seen / max(elapsed, 1e-6)
            print(f"[stream] epoch={epoch_index:02d} rows_seen={rows_seen:,} speed={speed:,.0f} rows/s")

    if step == 0:
        raise ValueError("No training batches were produced in stream epoch")

    return total_loss / step


def main() -> None:
    """Run streamed multi-player pretraining and save best checkpoint by valid move accuracy."""
    set_seed(int(CONFIG["random_seed"]))

    in_path = Path(CONFIG["input_jsonl"])
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    device = torch.device(str(CONFIG["device"]))
    is_cpu = device.type == "cpu"

    effective_epochs = int(CONFIG["epochs_cpu"] if is_cpu else CONFIG["epochs"])
    batch_size = int(CONFIG["batch_size_cpu"] if is_cpu else CONFIG["batch_size"])
    eval_batch_size = int(CONFIG["eval_batch_size_cpu"] if is_cpu else CONFIG["eval_batch_size"])
    grad_accum_steps = int(CONFIG["grad_accum_steps_cpu"] if is_cpu else CONFIG["grad_accum_steps"])

    counts, train_eval_samples, valid_eval_samples = scan_stream_metadata(CONFIG)
    print(
        "Eval reservoirs | "
        f"train_eval={len(train_eval_samples):,} "
        f"valid_eval={len(valid_eval_samples):,}"
    )

    model = FactorizedPolicyModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )

    best_move_acc = -1.0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, effective_epochs + 1):
        train_loss = train_one_epoch_stream(
            model=model,
            optimizer=optimizer,
            device=device,
            config=CONFIG,
            epoch_index=epoch,
            total_epochs=effective_epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            n_train_rows=int(counts["train_rows"]),
        )

        train_metrics = evaluate(model, train_eval_samples, device=device, batch_size=eval_batch_size)
        valid_metrics = evaluate(model, valid_eval_samples, device=device, batch_size=eval_batch_size)

        print(
            f"Epoch {epoch:02d} done | "
            f"loss={train_loss:.4f} | "
            f"train_move_acc={train_metrics['move_acc']:.4f} | "
            f"valid_move_acc={valid_metrics['move_acc']:.4f}"
        )

        if valid_metrics["move_acc"] > best_move_acc:
            best_move_acc = valid_metrics["move_acc"]
            best_metrics = {
                "best_epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
                "history_plies": int(CONFIG["history_plies"]),
                "mode": "pretrain_stream",
                "strict_target_isolation": bool(CONFIG.get("strict_target_isolation", False)),
                "stream_counts": counts,
                "train_eval_reservoir_size": len(train_eval_samples),
                "valid_eval_reservoir_size": len(valid_eval_samples),
            }
            save_checkpoint(model, dict(CONFIG), best_metrics, Path(CONFIG["model_output_path"]))

    metrics_path = Path(CONFIG["metrics_output_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)

    print(f"Best valid move_acc: {best_move_acc:.4f}")
    print(f"Model saved: {CONFIG['model_output_path']}")
    print(f"Metrics saved: {CONFIG['metrics_output_path']}")


if __name__ == "__main__":
    main()

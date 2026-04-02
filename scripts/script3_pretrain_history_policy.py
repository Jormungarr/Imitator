#!/usr/bin/env python3
"""Pretrain representation encoders on multi-player policy samples."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

from history_policy_lib import (
    FactorizedPolicyModel,
    evaluate,
    load_samples,
    save_checkpoint,
    set_seed,
    split_by_game,
    train_one_epoch,
)
from pipeline_config import HISTORY_PLIES, PRETRAIN_DATASET_TAG, TARGET_USERNAME, models_dir, processed_dir


def normalize_player_id(name: str) -> str:
    """Normalize player name into player_id format used in encoded samples."""
    return " ".join(name.strip().split()).lower()


CONFIG: Dict[str, Any] = {
    "dataset_tag": PRETRAIN_DATASET_TAG,
    "history_plies": HISTORY_PLIES,
    "input_jsonl": processed_dir(PRETRAIN_DATASET_TAG) / "policy_samples.jsonl",
    "model_output_path": models_dir(PRETRAIN_DATASET_TAG) / "history_policy_pretrain.pt",
    "metrics_output_path": models_dir(PRETRAIN_DATASET_TAG) / "history_policy_pretrain_metrics.json",
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
    "dense_state_dim": 29,
    "context_dim": 1,
    "history_event_dim": 16,
    "history_delta_dim": 11,
    "history_hidden_dim": 96,
    "shared_hidden_dim": 128,
    "dropout": 0.10,
    "enable_threat_head": True,
    "threat_loss_weight": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Set strict_target_isolation=True to remove target player from pretraining data.
    "strict_target_isolation": False,
    "max_samples": None,
    "max_samples_cpu": 250000,
    "reservoir_sample_for_cap": True,
    "load_progress_every": 50000,
    "train_eval_max_samples": 50000,
    "valid_eval_max_samples": 50000,
    "excluded_player_ids": [normalize_player_id(TARGET_USERNAME)],
}


def maybe_filter_samples(samples: List[Any], config: Dict[str, Any]) -> List[Any]:
    """Optionally remove excluded players from pretraining samples."""
    if not bool(config.get("strict_target_isolation", False)):
        return samples

    excluded = {str(x).strip().lower() for x in config.get("excluded_player_ids", []) if str(x).strip()}
    if not excluded:
        return samples

    kept = [s for s in samples if str(getattr(s, "player_id", "")).strip().lower() not in excluded]
    removed = len(samples) - len(kept)
    print(f"Strict isolation enabled: removed {removed} samples from excluded players: {sorted(excluded)}")
    if not kept:
        raise ValueError("All samples removed by strict_target_isolation filter")
    return kept


def maybe_subsample(samples: List[Any], max_n: int | None, seed: int) -> List[Any]:
    """Randomly subsample list for faster/lower-memory evaluation."""
    if max_n is None or max_n <= 0 or len(samples) <= max_n:
        return samples
    rng = random.Random(seed)
    idx = rng.sample(range(len(samples)), int(max_n))
    return [samples[i] for i in idx]


def main() -> None:
    """Run multi-player pretraining and save best checkpoint by valid move accuracy."""
    set_seed(int(CONFIG["random_seed"]))

    in_path = Path(CONFIG["input_jsonl"])
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    device = torch.device(str(CONFIG["device"]))
    is_cpu = device.type == "cpu"

    max_samples_raw = CONFIG.get("max_samples_cpu") if is_cpu else CONFIG.get("max_samples")
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None

    samples = load_samples(
        in_path,
        max_samples=max_samples,
        reservoir=bool(CONFIG.get("reservoir_sample_for_cap", False)),
        seed=int(CONFIG["random_seed"]),
        progress_every=int(CONFIG.get("load_progress_every", 0)),
        progress_label=f"{CONFIG['dataset_tag']}:load",
    )
    if max_samples is not None:
        print(f"Loaded capped sample set for training: {len(samples)} rows (max_samples={max_samples})")
    else:
        print(f"Loaded full sample set for training: {len(samples)} rows")

    samples = maybe_filter_samples(samples, CONFIG)
    train_samples, valid_samples = split_by_game(samples, float(CONFIG["valid_size"]), int(CONFIG["random_seed"]))

    effective_epochs = int(CONFIG["epochs_cpu"] if is_cpu else CONFIG["epochs"])
    batch_size = int(CONFIG["batch_size_cpu"] if is_cpu else CONFIG["batch_size"])
    eval_batch_size = int(CONFIG["eval_batch_size_cpu"] if is_cpu else CONFIG["eval_batch_size"])
    grad_accum_steps = int(CONFIG["grad_accum_steps_cpu"] if is_cpu else CONFIG["grad_accum_steps"])

    model = FactorizedPolicyModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["learning_rate"]),
        weight_decay=float(CONFIG["weight_decay"]),
    )

    best_move_acc = -1.0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, effective_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            samples=train_samples,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            print_every=int(CONFIG["print_every"]),
            epoch_index=epoch,
            total_epochs=effective_epochs,
            grad_accum_steps=grad_accum_steps,
        )

        eval_train_samples = maybe_subsample(
            train_samples,
            int(CONFIG["train_eval_max_samples"]) if CONFIG.get("train_eval_max_samples") is not None else None,
            seed=int(CONFIG["random_seed"]) + epoch,
        )
        eval_valid_samples = maybe_subsample(
            valid_samples,
            int(CONFIG["valid_eval_max_samples"]) if CONFIG.get("valid_eval_max_samples") is not None else None,
            seed=int(CONFIG["random_seed"]) + 1000 + epoch,
        )

        train_metrics = evaluate(model, eval_train_samples, device=device, batch_size=eval_batch_size)
        valid_metrics = evaluate(model, eval_valid_samples, device=device, batch_size=eval_batch_size)

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
                "mode": "pretrain",
                "strict_target_isolation": bool(CONFIG.get("strict_target_isolation", False)),
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





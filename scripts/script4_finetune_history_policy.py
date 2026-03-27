#!/usr/bin/env python3
"""Fine-tune target-player policy from optional representation pretraining."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from history_policy_lib import (
    FactorizedPolicyModel,
    evaluate,
    load_checkpoint,
    load_pretrained_encoders,
    load_samples,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
from pipeline_config import DATASET_TAG, HISTORY_PLIES, models_dir, processed_dir


CONFIG: Dict[str, Any] = {
    "dataset_tag": DATASET_TAG,
    "history_plies": HISTORY_PLIES,
    "input_jsonl": processed_dir(DATASET_TAG) / "policy_samples.jsonl",
    "model_output_path": models_dir(DATASET_TAG) / "history_policy.pt", # history_policy_scratch
    "metrics_output_path": models_dir(DATASET_TAG) / "history_policy_metrics.json", # history_policy_scratmetrics
    "honest_split_output_path": models_dir(DATASET_TAG) / "honest_split_game_ids.json", # honest_split_game_ids_scrat
    "pretrained_path": models_dir("pretrain_multi") / "history_policy_pretrain_stream.pt",
    "use_pretrained_encoders": True, # False no pretrain
    "valid_size": 0.10,
    "test_size": 0.10,
    "random_seed": 42,
    "epochs": 12,
    "epochs_cpu": 8,
    "batch_size": 192,
    "batch_size_cpu": 64,
    "print_every": 20,
    "eval_batch_size": 192,
    "eval_batch_size_cpu": 128,
    "grad_accum_steps": 1,
    "grad_accum_steps_cpu": 2,
    "max_samples": None,
    "max_samples_cpu": 200000,
    "reservoir_sample_for_cap": True,
    "load_progress_every": 50000,
    "encoder_learning_rate": 3e-4,
    "head_learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "feature_vocab_size": 81924,
    "feature_embed_dim": 64,
    "dense_state_dim": 6,
    "context_dim": 2,
    "history_hidden_dim": 96,
    "shared_hidden_dim": 128,
    "dropout": 0.10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def split_by_game_train_valid_test(
    samples: List[Any], valid_size: float, test_size: float, seed: int
) -> Tuple[List[Any], List[Any], List[Any], Dict[str, List[str]]]:
    """Split samples by game id into train/valid/test with disjoint games."""
    game_ids = sorted({s.game_id for s in samples})
    if len(game_ids) < 3:
        raise ValueError("Need at least 3 unique games for train/valid/test split")

    rng = random.Random(seed)
    rng.shuffle(game_ids)

    n_total = len(game_ids)
    n_test = max(1, int(n_total * test_size))
    n_valid = max(1, int(n_total * valid_size))

    if n_test + n_valid >= n_total:
        n_test = 1
        n_valid = 1

    test_games = set(game_ids[:n_test])
    valid_games = set(game_ids[n_test:n_test + n_valid])
    train_games = set(game_ids[n_test + n_valid:])

    train = [s for s in samples if s.game_id in train_games]
    valid = [s for s in samples if s.game_id in valid_games]
    test = [s for s in samples if s.game_id in test_games]

    if not train or not valid or not test:
        raise ValueError("Split produced empty train/valid/test set")

    split_info = {
        "train_game_ids": sorted(train_games),
        "valid_game_ids": sorted(valid_games),
        "test_game_ids": sorted(test_games),
    }
    return train, valid, test, split_info


def build_optimizer(model: FactorizedPolicyModel, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer with separate lrs for encoders vs policy heads."""
    encoder_prefixes = ("state_encoder.", "history_encoder.")

    encoder_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(encoder_prefixes):
            encoder_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(config["encoder_learning_rate"])},
            {"params": head_params, "lr": float(config["head_learning_rate"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )


def main() -> None:
    """Run player-specific adaptation and save best checkpoint by valid move accuracy."""
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
        print(f"Loaded capped fine-tune sample set: {len(samples)} rows (max_samples={max_samples})")
    else:
        print(f"Loaded full fine-tune sample set: {len(samples)} rows")

    train_samples, valid_samples, test_samples, split_info = split_by_game_train_valid_test(
        samples=samples,
        valid_size=float(CONFIG["valid_size"]),
        test_size=float(CONFIG["test_size"]),
        seed=int(CONFIG["random_seed"]),
    )

    split_out = Path(CONFIG["honest_split_output_path"])
    split_out.parent.mkdir(parents=True, exist_ok=True)
    with split_out.open("w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2)

    print(
        "Split summary | "
        f"train_games={len(split_info['train_game_ids'])}, "
        f"valid_games={len(split_info['valid_game_ids'])}, "
        f"test_games={len(split_info['test_game_ids'])}"
    )
    print(f"Honest split saved to: {split_out}")

    effective_epochs = int(CONFIG["epochs_cpu"] if is_cpu else CONFIG["epochs"])
    batch_size = int(CONFIG["batch_size_cpu"] if is_cpu else CONFIG["batch_size"])
    eval_batch_size = int(CONFIG["eval_batch_size_cpu"] if is_cpu else CONFIG["eval_batch_size"])
    grad_accum_steps = int(CONFIG["grad_accum_steps_cpu"] if is_cpu else CONFIG["grad_accum_steps"])

    model = FactorizedPolicyModel(CONFIG).to(device)

    loaded_keys = 0
    if bool(CONFIG.get("use_pretrained_encoders", False)):
        pre_path = Path(CONFIG["pretrained_path"])
        if pre_path.exists():
            loaded_keys = load_pretrained_encoders(model, pre_path, device=device)
            print(f"Loaded pretrained encoder tensors: {loaded_keys}")
        else:
            print(f"Pretrained checkpoint not found, training from scratch: {pre_path}")

    optimizer = build_optimizer(model, CONFIG)

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

        train_metrics = evaluate(model, train_samples, device=device, batch_size=eval_batch_size)
        valid_metrics = evaluate(model, valid_samples, device=device, batch_size=eval_batch_size)

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
                "mode": "finetune",
                "loaded_pretrained_encoder_tensors": loaded_keys,
                "honest_split_path": str(split_out),
            }
            save_checkpoint(model, dict(CONFIG), best_metrics, Path(CONFIG["model_output_path"]))

    # Honest test evaluation only once on the best checkpoint.
    ckpt = load_checkpoint(Path(CONFIG["model_output_path"]), device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_metrics = evaluate(model, test_samples, device=device, batch_size=eval_batch_size)

    best_metrics["test"] = test_metrics
    best_metrics["honest_test_note"] = "Test games are disjoint from train and valid by game_id."

    metrics_path = Path(CONFIG["metrics_output_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)

    print(f"Best valid move_acc: {best_move_acc:.4f}")
    print(f"Honest test move_acc: {test_metrics['move_acc']:.4f}")
    print(f"Model saved: {CONFIG['model_output_path']}")
    print(f"Metrics saved: {CONFIG['metrics_output_path']}")


if __name__ == "__main__":
    main()

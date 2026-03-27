#!/usr/bin/env python3
"""Shared dataset/model/training utilities for history-conditioned policy learning."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicySample:
    """In-memory sample for one move decision."""

    game_id: str
    player_id: str
    active_feature_indices: List[int]
    dense_state: List[float]
    history_event: List[List[float]]
    history_delta: List[List[float]]
    context: List[float]
    legal_from_mask: List[int]
    legal_to_by_from: Dict[str, List[int]]
    target_from_sq: int
    target_to_sq: int
    target_promotion: int


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _row_to_sample(row: Dict[str, Any]) -> PolicySample:
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


def load_samples(
    path: Path,
    max_samples: int | None = None,
    reservoir: bool = False,
    seed: int = 42,
    progress_every: int = 0,
    progress_label: str = "load_samples",
) -> List[PolicySample]:
    """Load policy samples from encoded JSONL with optional memory-safe capping."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    samples: List[PolicySample] = []
    rng = random.Random(seed)
    seen = 0
    start = time.time()

    for row in iter_jsonl(path):
        sample = _row_to_sample(row)
        seen += 1

        if progress_every > 0 and seen % progress_every == 0:
            elapsed = time.time() - start
            speed = seen / max(elapsed, 1e-6)
            print(
                f"[{progress_label}] read={seen:,} kept={len(samples):,} "
                f"speed={speed:,.0f} rows/s"
            )

        if max_samples is None:
            samples.append(sample)
            continue

        if len(samples) < max_samples:
            samples.append(sample)
            continue

        if reservoir:
            j = rng.randrange(seen)
            if j < max_samples:
                samples[j] = sample
        else:
            break

    elapsed = time.time() - start
    speed = seen / max(elapsed, 1e-6) if seen > 0 else 0.0
    print(
        f"[{progress_label}] done: read={seen:,} kept={len(samples):,} "
        f"elapsed={elapsed:.1f}s speed={speed:,.0f} rows/s"
    )

    return samples

def split_by_game(samples: List[PolicySample], valid_size: float, seed: int) -> Tuple[List[PolicySample], List[PolicySample]]:
    """Split samples by game id to reduce leakage."""
    game_ids = sorted({s.game_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(game_ids)

    n_valid_games = max(1, int(len(game_ids) * valid_size))
    valid_set = set(game_ids[:n_valid_games])

    train = [s for s in samples if s.game_id not in valid_set]
    valid = [s for s in samples if s.game_id in valid_set]

    if not train or not valid:
        raise ValueError("Split produced empty train or valid set")

    return train, valid


def batch_iter(samples: List[PolicySample], batch_size: int, shuffle: bool) -> Iterable[List[PolicySample]]:
    """Yield mini-batches from sample list."""
    idx = list(range(len(samples)))
    if shuffle:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield [samples[i] for i in batch_idx]


def _legal_to_mask(sample: PolicySample, from_sq: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(64, dtype=torch.float32, device=device)
    targets = sample.legal_to_by_from.get(str(int(from_sq)), [])
    if not targets:
        mask.fill_(1.0)
        return mask
    for to_sq in targets:
        mask[int(to_sq)] = 1.0
    return mask


def collate_batch(batch: List[PolicySample], device: torch.device) -> Dict[str, Any]:
    """Collate variable-length sparse indices and fixed tensors for a batch."""
    bsz = len(batch)

    dense_state = torch.tensor([s.dense_state for s in batch], dtype=torch.float32, device=device)
    history_event = torch.tensor([s.history_event for s in batch], dtype=torch.float32, device=device)
    history_delta = torch.tensor([s.history_delta for s in batch], dtype=torch.float32, device=device)
    context = torch.tensor([s.context for s in batch], dtype=torch.float32, device=device)

    # Harden training against occasional data inconsistencies: always keep target squares legal.
    legal_from_rows: List[List[int]] = []
    for s in batch:
        row = [int(v) for v in s.legal_from_mask]
        if len(row) < 64:
            row = row + [0] * (64 - len(row))
        else:
            row = row[:64]
        if 0 <= int(s.target_from_sq) < 64:
            row[int(s.target_from_sq)] = 1
        legal_from_rows.append(row)
    legal_from_mask = torch.tensor(legal_from_rows, dtype=torch.float32, device=device)

    target_from = torch.tensor([s.target_from_sq for s in batch], dtype=torch.long, device=device)
    target_to = torch.tensor([s.target_to_sq for s in batch], dtype=torch.long, device=device)
    target_promo = torch.tensor([s.target_promotion for s in batch], dtype=torch.long, device=device)

    flat_indices: List[int] = []
    offsets: List[int] = []
    current = 0
    for s in batch:
        offsets.append(current)
        feats = s.active_feature_indices if s.active_feature_indices else [0]
        flat_indices.extend(feats)
        current += len(feats)

    sparse_indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    sparse_offsets = torch.tensor(offsets, dtype=torch.long, device=device)

    legal_to_mask_rows: List[torch.Tensor] = []
    for s in batch:
        m = _legal_to_mask(s, s.target_from_sq, device=device)
        if 0 <= int(s.target_to_sq) < 64:
            m[int(s.target_to_sq)] = 1.0
        legal_to_mask_rows.append(m)
    legal_to_mask_target = torch.stack(legal_to_mask_rows, dim=0)

    return {
        "bsz": bsz,
        "dense_state": dense_state,
        "history_event": history_event,
        "history_delta": history_delta,
        "context": context,
        "legal_from_mask": legal_from_mask,
        "target_from": target_from,
        "target_to": target_to,
        "target_promo": target_promo,
        "sparse_indices": sparse_indices,
        "sparse_offsets": sparse_offsets,
        "legal_to_mask_target": legal_to_mask_target,
        "batch_samples": batch,
    }


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal logits to a large negative value."""
    return torch.where(mask > 0, logits, torch.full_like(logits, -1e9))


class StateEncoder(nn.Module):
    """Encode sparse HalfKP features and dense state summaries."""

    def __init__(self, feature_vocab_size: int, feature_embed_dim: int, dense_state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.sparse_bag = nn.EmbeddingBag(feature_vocab_size, feature_embed_dim, mode="mean")
        self.dense_proj = nn.Sequential(nn.Linear(dense_state_dim, hidden_dim // 2), nn.ReLU())
        self.out_dim = feature_embed_dim + hidden_dim // 2

    def forward(self, sparse_indices: torch.Tensor, sparse_offsets: torch.Tensor, dense_state: torch.Tensor) -> torch.Tensor:
        sparse_repr = self.sparse_bag(sparse_indices, sparse_offsets)
        dense_repr = self.dense_proj(dense_state)
        return torch.cat([sparse_repr, dense_repr], dim=1)


class HistoryEncoder(nn.Module):
    """Encode move-event and state-delta sequences using GRU."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        emb = 16
        self.mover_emb = nn.Embedding(2, emb)
        self.piece_emb = nn.Embedding(7, emb)
        self.square_emb = nn.Embedding(64, emb)
        self.flag_proj = nn.Linear(4, 16)
        self.delta_proj = nn.Linear(3, 16)
        self.gru = nn.GRU(input_size=emb * 4 + 16 + 16, hidden_size=hidden_dim, batch_first=True)

    def forward(self, history_event: torch.Tensor, history_delta: torch.Tensor) -> torch.Tensor:
        mover = history_event[:, :, 0].long().clamp(min=0, max=1)
        piece = history_event[:, :, 1].long().clamp(min=0, max=6)
        from_sq = history_event[:, :, 2].long().clamp(min=0, max=63)
        to_sq = history_event[:, :, 3].long().clamp(min=0, max=63)
        flags = history_event[:, :, 4:8]

        seq = torch.cat(
            [
                self.mover_emb(mover),
                self.piece_emb(piece),
                self.square_emb(from_sq),
                self.square_emb(to_sq),
                self.flag_proj(flags),
                self.delta_proj(history_delta),
            ],
            dim=-1,
        )
        output, _ = self.gru(seq)
        return output[:, -1, :]


class FactorizedPolicyModel(nn.Module):
    """History-conditioned factorized policy network with legal masking heads."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        shared_hidden = int(config["shared_hidden_dim"])

        self.state_encoder = StateEncoder(
            feature_vocab_size=int(config["feature_vocab_size"]),
            feature_embed_dim=int(config["feature_embed_dim"]),
            dense_state_dim=int(config["dense_state_dim"]),
            hidden_dim=shared_hidden,
        )
        self.history_encoder = HistoryEncoder(hidden_dim=int(config["history_hidden_dim"]))
        self.context_proj = nn.Linear(int(config["context_dim"]), 16)

        fused_dim = self.state_encoder.out_dim + int(config["history_hidden_dim"]) + 16
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, shared_hidden),
            nn.ReLU(),
            nn.Dropout(float(config["dropout"])),
        )

        self.square_emb = nn.Embedding(64, 16)
        self.from_head = nn.Linear(shared_hidden, 64)
        self.to_head = nn.Sequential(
            nn.Linear(shared_hidden + 16, shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, 64),
        )
        self.promo_head = nn.Sequential(
            nn.Linear(shared_hidden + 32, shared_hidden),
            nn.ReLU(),
            nn.Linear(shared_hidden, 5),
        )

    def fused_repr(self, batch: Dict[str, Any]) -> torch.Tensor:
        state = self.state_encoder(batch["sparse_indices"], batch["sparse_offsets"], batch["dense_state"])
        hist = self.history_encoder(batch["history_event"], batch["history_delta"])
        ctx = self.context_proj(batch["context"])
        return self.fusion(torch.cat([state, hist, ctx], dim=1))

    def from_logits(self, fused: torch.Tensor) -> torch.Tensor:
        return self.from_head(fused)

    def to_logits(self, fused: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        from_repr = self.square_emb(from_sq.long())
        return self.to_head(torch.cat([fused, from_repr], dim=1))

    def promo_logits(self, fused: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor) -> torch.Tensor:
        from_repr = self.square_emb(from_sq.long())
        to_repr = self.square_emb(to_sq.long())
        return self.promo_head(torch.cat([fused, from_repr, to_repr], dim=1))


def batch_loss(model: FactorizedPolicyModel, batch: Dict[str, Any]) -> torch.Tensor:
    """Compute factorized move loss for one mini-batch."""
    fused = model.fused_repr(batch)

    from_logits = masked_logits(model.from_logits(fused), batch["legal_from_mask"])
    from_loss = F.cross_entropy(from_logits, batch["target_from"])

    to_logits = masked_logits(model.to_logits(fused, batch["target_from"]), batch["legal_to_mask_target"])
    to_loss = F.cross_entropy(to_logits, batch["target_to"])

    promo_logits = model.promo_logits(fused, batch["target_from"], batch["target_to"])
    promo_loss = F.cross_entropy(promo_logits, batch["target_promo"])

    return from_loss + to_loss + 0.25 * promo_loss


@torch.no_grad()
def evaluate(model: FactorizedPolicyModel, samples: List[PolicySample], device: torch.device, batch_size: int = 256) -> Dict[str, float]:
    """Compute from/to/promo and exact move accuracy."""
    model.eval()

    total = 0
    from_ok = 0
    to_ok = 0
    promo_ok = 0
    move_ok = 0

    for batch_samples in batch_iter(samples, batch_size=batch_size, shuffle=False):
        batch = collate_batch(batch_samples, device=device)
        fused = model.fused_repr(batch)

        from_pred = torch.argmax(masked_logits(model.from_logits(fused), batch["legal_from_mask"]), dim=1)

        to_masks = torch.stack([
            _legal_to_mask(s, int(from_pred[i].item()), device=device) for i, s in enumerate(batch_samples)
        ], dim=0)
        to_pred = torch.argmax(masked_logits(model.to_logits(fused, from_pred), to_masks), dim=1)

        promo_pred = torch.argmax(model.promo_logits(fused, from_pred, to_pred), dim=1)

        tf = batch["target_from"]
        tt = batch["target_to"]
        tp = batch["target_promo"]

        bsz = tf.shape[0]
        total += bsz
        from_ok += int((from_pred == tf).sum().item())
        to_ok += int((to_pred == tt).sum().item())
        promo_ok += int((promo_pred == tp).sum().item())
        move_ok += int(((from_pred == tf) & (to_pred == tt) & (promo_pred == tp)).sum().item())

    return {
        "from_acc": from_ok / max(total, 1),
        "to_acc": to_ok / max(total, 1),
        "promo_acc": promo_ok / max(total, 1),
        "move_acc": move_ok / max(total, 1),
        "num_samples": float(total),
    }


def format_eta(seconds: float) -> str:
    """Format seconds as mm:ss."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def train_one_epoch(
    model: FactorizedPolicyModel,
    samples: List[PolicySample],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    print_every: int,
    epoch_index: int,
    total_epochs: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    loss_spike_threshold: float = 1e4,
) -> float:
    """Run one epoch with progress display and mini-batch updates."""
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    model.train()
    n_batches = max(1, math.ceil(len(samples) / batch_size))

    total_loss = 0.0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch_samples in enumerate(batch_iter(samples, batch_size=batch_size, shuffle=True), start=1):
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

        if loss_value > loss_spike_threshold:
            print(
                f"[warn] Loss spike detected at epoch {epoch_index}, step {step}: "
                f"loss={loss_value:.2f}"
            )

        (loss / grad_accum_steps).backward()

        if step % grad_accum_steps == 0 or step == n_batches:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss_value

        if step % print_every == 0 or step == n_batches:
            elapsed = time.time() - start
            rate = step / max(elapsed, 1e-6)
            eta = (n_batches - step) / max(rate, 1e-6)
            avg_loss = total_loss / step
            print(
                f"Epoch {epoch_index:02d}/{total_epochs:02d} | "
                f"step {step:4d}/{n_batches:4d} | "
                f"loss={avg_loss:.4f} | "
                f"eta={format_eta(eta)}"
            )

    return total_loss / n_batches


def save_checkpoint(model: FactorizedPolicyModel, config: Dict[str, Any], metrics: Dict[str, Any], path: Path) -> None:
    """Save model state, config, and metrics payload."""
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load torch checkpoint payload."""
    return torch.load(path, map_location=device, weights_only=False)


def load_pretrained_encoders(model: FactorizedPolicyModel, ckpt_path: Path, device: torch.device) -> int:
    """Load only state/history encoder weights from a pretrained checkpoint."""
    ckpt = load_checkpoint(ckpt_path, device)
    state = ckpt["model_state_dict"]
    own = model.state_dict()

    copied = 0
    for key, tensor in state.items():
        if key.startswith("state_encoder.") or key.startswith("history_encoder."):
            if key in own and own[key].shape == tensor.shape:
                own[key] = tensor
                copied += 1

    model.load_state_dict(own)
    return copied

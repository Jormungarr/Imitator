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
    history_mask: List[int]
    context: List[float]
    piece_slot_to_square: List[int]
    legal_piece_slot_mask: List[int]
    legal_from_mask: List[int]
    legal_to_by_from: Dict[str, List[int]]
    target_piece_slot: int
    target_from_sq: int
    target_to_sq: int
    target_promotion: int
    target_under_threat: int


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
        history_mask=[int(x) for x in row.get("history_mask", [1] * len(row["history_event"]))],
        context=[float(x) for x in row["context"]],
        piece_slot_to_square=[int(x) for x in row.get("piece_slot_to_square", [-1] * 16)],
        legal_piece_slot_mask=[int(x) for x in row.get("legal_piece_slot_mask", [0] * 16)],
        legal_from_mask=[int(x) for x in row["legal_from_mask"]],
        legal_to_by_from={str(k): [int(v) for v in vals] for k, vals in row["legal_to_by_from"].items()},
        target_piece_slot=int(row.get("target_piece_slot", 0)),
        target_from_sq=int(row["target_from_sq"]),
        target_to_sq=int(row["target_to_sq"]),
        target_promotion=int(row.get("target_promotion", 0)),
        target_under_threat=int(row.get("target_under_threat", 0)),
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


PIECE_SLOT_DIM = 16


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
    history_mask = torch.tensor([s.history_mask for s in batch], dtype=torch.float32, device=device)
    context = torch.tensor([s.context for s in batch], dtype=torch.float32, device=device)
    piece_slot_to_square = torch.tensor([s.piece_slot_to_square for s in batch], dtype=torch.long, device=device)


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

    target_piece = torch.tensor([s.target_piece_slot for s in batch], dtype=torch.long, device=device)
    target_from = torch.tensor([s.target_from_sq for s in batch], dtype=torch.long, device=device)
    target_to = torch.tensor([s.target_to_sq for s in batch], dtype=torch.long, device=device)
    target_promo = torch.tensor([s.target_promotion for s in batch], dtype=torch.long, device=device)
    target_under_threat = torch.tensor([s.target_under_threat for s in batch], dtype=torch.float32, device=device)

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

    legal_piece_rows: List[List[int]] = []
    for s in batch:
        row = [int(v) for v in s.legal_piece_slot_mask]
        if len(row) < PIECE_SLOT_DIM:
            row = row + [0] * (PIECE_SLOT_DIM - len(row))
        else:
            row = row[:PIECE_SLOT_DIM]
        if 0 <= int(s.target_piece_slot) < PIECE_SLOT_DIM:
            row[int(s.target_piece_slot)] = 1
        legal_piece_rows.append(row)
    legal_piece_slot_mask = torch.tensor(legal_piece_rows, dtype=torch.float32, device=device)

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
        "history_mask": history_mask,
        "context": context,
        "piece_slot_to_square": piece_slot_to_square,
        "legal_piece_slot_mask": legal_piece_slot_mask,
        "legal_from_mask": legal_from_mask,
        "target_piece": target_piece,
        "target_from": target_from,
        "target_to": target_to,
        "target_promo": target_promo,
        "target_under_threat": target_under_threat,
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

    def __init__(self, hidden_dim: int, event_flag_dim: int, delta_dim: int) -> None:
        super().__init__()
        emb = 16
        self.mover_emb = nn.Embedding(2, emb)
        self.piece_emb = nn.Embedding(7, emb)
        self.square_emb = nn.Embedding(64, emb)
        self.flag_proj = nn.Linear(event_flag_dim, 24)
        self.delta_proj = nn.Linear(delta_dim, 24)
        self.gru = nn.GRU(input_size=emb * 4 + 24 + 24, hidden_size=hidden_dim, batch_first=True)

    def forward(self, history_event: torch.Tensor, history_delta: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        mover = history_event[:, :, 0].long().clamp(min=0, max=1)
        piece = history_event[:, :, 1].long().clamp(min=0, max=6)
        from_sq = history_event[:, :, 2].long().clamp(min=0, max=63)
        to_sq = history_event[:, :, 3].long().clamp(min=0, max=63)
        flags = history_event[:, :, 4:]

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
        mask = history_mask.unsqueeze(-1).to(output.dtype)
        masked_sum = (output * mask).sum(dim=1)
        valid_steps = mask.sum(dim=1).clamp_min(1.0)
        return masked_sum / valid_steps


class FactorizedPolicyModel(nn.Module):
    """History-conditioned factorized policy network with legal masking heads."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.enable_threat_head = bool(config.get("enable_threat_head", False))
        self.threat_loss_weight = float(config.get("threat_loss_weight", 0.2))
        shared_hidden = int(config["shared_hidden_dim"])

        self.state_encoder = StateEncoder(
            feature_vocab_size=int(config["feature_vocab_size"]),
            feature_embed_dim=int(config["feature_embed_dim"]),
            dense_state_dim=int(config["dense_state_dim"]),
            hidden_dim=shared_hidden,
        )
        self.history_encoder = HistoryEncoder(
            hidden_dim=int(config["history_hidden_dim"]),
            event_flag_dim=int(config["history_event_dim"]) - 4,
            delta_dim=int(config["history_delta_dim"]),
        )
        self.context_proj = nn.Linear(int(config["context_dim"]), 16)

        fused_dim = self.state_encoder.out_dim + int(config["history_hidden_dim"]) + 16
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, shared_hidden),
            nn.ReLU(),
            nn.Dropout(float(config["dropout"])),
        )

        self.square_emb = nn.Embedding(64, 16)
        self.piece_head = nn.Linear(shared_hidden, PIECE_SLOT_DIM)
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
        if self.enable_threat_head:
            self.threat_head = nn.Linear(shared_hidden, 1)

    def fused_repr(self, batch: Dict[str, Any]) -> torch.Tensor:
        state = self.state_encoder(batch["sparse_indices"], batch["sparse_offsets"], batch["dense_state"])
        hist = self.history_encoder(batch["history_event"], batch["history_delta"], batch["history_mask"])
        ctx = self.context_proj(batch["context"])
        return self.fusion(torch.cat([state, hist, ctx], dim=1))

    def piece_logits(self, fused: torch.Tensor) -> torch.Tensor:
        return self.piece_head(fused)

    def to_logits(self, fused: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        from_repr = self.square_emb(from_sq.long())
        return self.to_head(torch.cat([fused, from_repr], dim=1))

    def promo_logits(self, fused: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor) -> torch.Tensor:
        from_repr = self.square_emb(from_sq.long())
        to_repr = self.square_emb(to_sq.long())
        return self.promo_head(torch.cat([fused, from_repr, to_repr], dim=1))

    def threat_logits(self, fused: torch.Tensor) -> torch.Tensor:
        if not self.enable_threat_head:
            raise RuntimeError("Threat head disabled for this checkpoint/config")
        return self.threat_head(fused).squeeze(1)


def resolve_from_square(piece_slot_to_square: torch.Tensor, piece_slot: torch.Tensor) -> torch.Tensor:
    """Gather current from-square from predicted or target piece slot."""
    idx = piece_slot.long().unsqueeze(1)
    resolved = piece_slot_to_square.gather(1, idx).squeeze(1)
    return resolved.clamp(min=0)


def batch_loss(model: FactorizedPolicyModel, batch: Dict[str, Any]) -> torch.Tensor:
    """Compute factorized move loss for one mini-batch."""
    fused = model.fused_repr(batch)

    piece_logits = masked_logits(model.piece_logits(fused), batch["legal_piece_slot_mask"])
    piece_loss = F.cross_entropy(piece_logits, batch["target_piece"])

    target_from = resolve_from_square(batch["piece_slot_to_square"], batch["target_piece"])
    to_logits = masked_logits(model.to_logits(fused, target_from), batch["legal_to_mask_target"])
    to_loss = F.cross_entropy(to_logits, batch["target_to"])

    promo_logits = model.promo_logits(fused, target_from, batch["target_to"])
    promo_loss = F.cross_entropy(promo_logits, batch["target_promo"])

    total_loss = piece_loss + to_loss + 0.25 * promo_loss
    if model.enable_threat_head:
        threat_logits = model.threat_logits(fused)
        threat_loss = F.binary_cross_entropy_with_logits(threat_logits, batch["target_under_threat"])
        total_loss = total_loss + model.threat_loss_weight * threat_loss

    return total_loss


@torch.no_grad()
def evaluate(model: FactorizedPolicyModel, samples: List[PolicySample], device: torch.device, batch_size: int = 256) -> Dict[str, float]:
    """Compute piece/to/promo and exact move accuracy."""
    model.eval()

    total = 0
    piece_ok = 0
    from_ok = 0
    to_ok = 0
    promo_ok = 0
    move_ok = 0
    threat_ok = 0
    threat_tp = 0
    threat_fp = 0
    threat_fn = 0
    threat_pos = 0

    for batch_samples in batch_iter(samples, batch_size=batch_size, shuffle=False):
        batch = collate_batch(batch_samples, device=device)
        fused = model.fused_repr(batch)

        piece_pred = torch.argmax(masked_logits(model.piece_logits(fused), batch["legal_piece_slot_mask"]), dim=1)
        from_pred = resolve_from_square(batch["piece_slot_to_square"], piece_pred)

        to_masks = torch.stack([
            _legal_to_mask(s, int(from_pred[i].item()), device=device) for i, s in enumerate(batch_samples)
        ], dim=0)
        to_pred = torch.argmax(masked_logits(model.to_logits(fused, from_pred), to_masks), dim=1)

        promo_pred = torch.argmax(model.promo_logits(fused, from_pred, to_pred), dim=1)
        if model.enable_threat_head:
            threat_pred = (torch.sigmoid(model.threat_logits(fused)) >= 0.5).long()

        tpiece = batch["target_piece"]
        tf = batch["target_from"]
        tt = batch["target_to"]
        tp = batch["target_promo"]
        tut = batch["target_under_threat"].long()

        bsz = tf.shape[0]
        total += bsz
        piece_ok += int((piece_pred == tpiece).sum().item())
        from_ok += int((from_pred == tf).sum().item())
        to_ok += int((to_pred == tt).sum().item())
        promo_ok += int((promo_pred == tp).sum().item())
        move_ok += int(((piece_pred == tpiece) & (from_pred == tf) & (to_pred == tt) & (promo_pred == tp)).sum().item())
        if model.enable_threat_head:
            threat_ok += int((threat_pred == tut).sum().item())
            threat_tp += int(((threat_pred == 1) & (tut == 1)).sum().item())
            threat_fp += int(((threat_pred == 1) & (tut == 0)).sum().item())
            threat_fn += int(((threat_pred == 0) & (tut == 1)).sum().item())
            threat_pos += int((tut == 1).sum().item())

    metrics = {
        "piece_acc": piece_ok / max(total, 1),
        "from_acc": from_ok / max(total, 1),
        "to_acc": to_ok / max(total, 1),
        "promo_acc": promo_ok / max(total, 1),
        "move_acc": move_ok / max(total, 1),
        "num_samples": float(total),
    }
    if model.enable_threat_head:
        precision = threat_tp / max(threat_tp + threat_fp, 1)
        recall = threat_tp / max(threat_tp + threat_fn, 1)
        f1 = 0.0 if precision + recall <= 0 else (2.0 * precision * recall) / (precision + recall)
        metrics["threat_acc"] = threat_ok / max(total, 1)
        metrics["threat_positive_rate"] = threat_pos / max(total, 1)
        metrics["threat_precision"] = precision
        metrics["threat_recall"] = recall
        metrics["threat_f1"] = f1
    return metrics


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



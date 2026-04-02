#!/usr/bin/env python3
"""Inspect one encoded sample: feature proxies plus piece/to/promo head outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from chess_feature_utils import ORIGINAL_PIECE_SLOTS
from history_policy_lib import FactorizedPolicyModel, PolicySample, collate_batch, load_checkpoint, masked_logits, resolve_from_square
from pipeline_config import DATASET_TAG, models_dir, processed_dir


CONFIG: Dict[str, Any] = {
    "dataset_tag": DATASET_TAG,
    "encoded_jsonl": processed_dir(DATASET_TAG) / "policy_samples.jsonl",
    "model_path": models_dir(DATASET_TAG) / "history_policy.pt",
    "sample_index": 0,
    "top_k": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

PROMO_NAMES = ["none", "knight", "bishop", "rook", "queen"]


def load_encoded_row(path: Path, sample_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index {sample_index} out of range for {path}")


def row_to_sample(row: Dict[str, Any]) -> PolicySample:
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
    )


def topk_pairs(values: torch.Tensor, labels: List[str], k: int) -> List[str]:
    k = max(1, min(int(k), int(values.shape[0])))
    topv, topi = torch.topk(values, k=k)
    out = []
    for score, idx in zip(topv.tolist(), topi.tolist()):
        label = labels[int(idx)] if int(idx) < len(labels) else str(int(idx))
        out.append(f"{label}: {float(score):.4f}")
    return out


def dense_feature_proxy(sample: PolicySample, model: FactorizedPolicyModel) -> List[str]:
    linear = model.state_encoder.dense_proj[0]
    weight_norm = torch.norm(linear.weight.detach(), dim=0)
    pairs = []
    for i, value in enumerate(sample.dense_state):
        proxy = abs(float(value)) * float(weight_norm[i].item())
        pairs.append((proxy, i, float(value)))
    pairs.sort(reverse=True)
    return [f"dense[{i}] value={value:.4f} proxy={proxy:.4f}" for proxy, i, value in pairs]


def sparse_feature_proxy(sample: PolicySample, model: FactorizedPolicyModel) -> List[str]:
    weight = model.state_encoder.sparse_bag.weight.detach()
    denom = max(1, len(sample.active_feature_indices))
    pairs = []
    for idx in sample.active_feature_indices:
        norm = float(torch.norm(weight[int(idx)]).item()) / denom
        pairs.append((norm, int(idx)))
    pairs.sort(reverse=True)
    return [f"feature[{idx}] proxy={score:.4f}" for score, idx in pairs]


def main() -> None:
    row = load_encoded_row(Path(CONFIG["encoded_jsonl"]), int(CONFIG["sample_index"]))
    sample = row_to_sample(row)
    device = torch.device(str(CONFIG["device"]))

    ckpt = load_checkpoint(Path(CONFIG["model_path"]), device=device)
    model = FactorizedPolicyModel(dict(ckpt.get("config", {}))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = collate_batch([sample], device=device)
    with torch.no_grad():
        fused = model.fused_repr(batch)
        piece_logits = masked_logits(model.piece_logits(fused), batch["legal_piece_slot_mask"])[0]
        piece_probs = torch.softmax(piece_logits, dim=0)
        pred_piece = int(torch.argmax(piece_logits).item())
        pred_from = int(resolve_from_square(batch["piece_slot_to_square"], torch.tensor([pred_piece], device=device))[0].item())
        actual_piece = int(sample.target_piece_slot)
        actual_from = int(sample.target_from_sq)

        pred_to_mask = torch.zeros(64, dtype=torch.float32, device=device)
        for to_sq in sample.legal_to_by_from.get(str(pred_from), []):
            pred_to_mask[int(to_sq)] = 1.0
        if float(pred_to_mask.sum().item()) <= 0:
            pred_to_mask.fill_(1.0)
        pred_to_logits = masked_logits(model.to_logits(fused, torch.tensor([pred_from], device=device))[0], pred_to_mask)
        pred_to_probs = torch.softmax(pred_to_logits, dim=0)
        pred_to = int(torch.argmax(pred_to_logits).item())

        actual_to_mask = torch.zeros(64, dtype=torch.float32, device=device)
        for to_sq in sample.legal_to_by_from.get(str(actual_from), []):
            actual_to_mask[int(to_sq)] = 1.0
        if float(actual_to_mask.sum().item()) <= 0:
            actual_to_mask.fill_(1.0)
        actual_to_logits = masked_logits(model.to_logits(fused, torch.tensor([actual_from], device=device))[0], actual_to_mask)
        actual_to_probs = torch.softmax(actual_to_logits, dim=0)

        promo_logits = model.promo_logits(fused, torch.tensor([pred_from], device=device), torch.tensor([pred_to], device=device))[0]
        promo_probs = torch.softmax(promo_logits, dim=0)

    print(f"dataset={CONFIG['dataset_tag']} sample_index={CONFIG['sample_index']}")
    print(f"game_id={sample.game_id} player_id={sample.player_id}")
    print(f"target_piece_slot={ORIGINAL_PIECE_SLOTS[sample.target_piece_slot]} target_from={sample.target_from_sq} target_to={sample.target_to_sq} target_promo={PROMO_NAMES[sample.target_promotion]}")
    print("\nCurrent piece-slot -> square map:")
    for idx, sq in enumerate(sample.piece_slot_to_square):
        print(f"  {ORIGINAL_PIECE_SLOTS[idx]} -> {sq}")

    print("\nPiece head top-k:")
    for line in topk_pairs(piece_probs, ORIGINAL_PIECE_SLOTS, int(CONFIG["top_k"])):
        print(f"  {line}")
    print(f"  predicted_piece={ORIGINAL_PIECE_SLOTS[pred_piece]} resolved_from={pred_from}")
    print(f"  actual_piece={ORIGINAL_PIECE_SLOTS[actual_piece]} actual_from={actual_from}")

    square_labels = [str(i) for i in range(64)]
    print("\nTo head top-k given predicted source:")
    for line in topk_pairs(pred_to_probs, square_labels, int(CONFIG["top_k"])):
        print(f"  {line}")
    print("\nTo head top-k given actual source:")
    for line in topk_pairs(actual_to_probs, square_labels, int(CONFIG["top_k"])):
        print(f"  {line}")

    print("\nPromotion head:")
    for line in topk_pairs(promo_probs, PROMO_NAMES, len(PROMO_NAMES)):
        print(f"  {line}")

    print("\nTop sparse feature proxies (approx):")
    for line in sparse_feature_proxy(sample, model)[: int(CONFIG["top_k"])]:
        print(f"  {line}")

    print("\nTop dense feature proxies (approx):")
    for line in dense_feature_proxy(sample, model)[: int(CONFIG["top_k"])]:
        print(f"  {line}")


if __name__ == "__main__":
    main()

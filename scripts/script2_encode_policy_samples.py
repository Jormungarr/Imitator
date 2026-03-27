#!/usr/bin/env python3
"""Encode history-aware position rows into model-ready JSONL samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import chess

from pipeline_config import DATASET_TAG, HISTORY_PLIES, processed_dir


CONFIG = {
    "dataset_tag": DATASET_TAG,
    "history_plies": HISTORY_PLIES,
    "input_jsonl": processed_dir(DATASET_TAG) / "positions_history.jsonl",
    "output_jsonl": processed_dir(DATASET_TAG) / "policy_samples.jsonl",
    "progress_every": 5000,
    "verbose": True,
}

NUM_SQUARES = 64
NUM_NON_KING_PLANES = 10
HALFKP_BLOCK_DIM = NUM_SQUARES * NUM_NON_KING_PLANES * NUM_SQUARES
HALFKP_OWN_OFFSET = 0
HALFKP_OPP_OFFSET = HALFKP_BLOCK_DIM
CASTLE_OWN_K_IDX = 2 * HALFKP_BLOCK_DIM
CASTLE_OWN_Q_IDX = 2 * HALFKP_BLOCK_DIM + 1
CASTLE_OPP_K_IDX = 2 * HALFKP_BLOCK_DIM + 2
CASTLE_OPP_Q_IDX = 2 * HALFKP_BLOCK_DIM + 3
TOTAL_FEATURE_DIM = 2 * HALFKP_BLOCK_DIM + 4


def normalize_square(square: int, mover_is_white: bool) -> int:
    """Return square index in mover-relative coordinates."""
    return square if mover_is_white else chess.square_mirror(square)


def halfkp_non_king_plane_id_relative(piece: chess.Piece, mover_color: bool) -> int | None:
    """Return HalfKP non-king piece plane id in mover-relative perspective."""
    if piece.piece_type == chess.KING:
        return None
    base = piece.piece_type - 1
    return base if piece.color == mover_color else 5 + base


def halfkp_index(king_rel_sq: int, piece_plane: int, piece_rel_sq: int, block_offset: int) -> int:
    """Return flattened HalfKP feature index."""
    return block_offset + ((king_rel_sq * NUM_NON_KING_PLANES + piece_plane) * NUM_SQUARES + piece_rel_sq)


def encode_board_sparse_indices(fen: str) -> List[int]:
    """Encode board as active HalfKP indices plus castling bits."""
    board = chess.Board(fen)
    active: List[int] = []

    mover_color = board.turn
    opp_color = not mover_color

    own_king_sq = board.king(mover_color)
    opp_king_sq = board.king(opp_color)
    if own_king_sq is None or opp_king_sq is None:
        raise ValueError("Invalid board without kings")

    own_king_rel_sq = normalize_square(own_king_sq, mover_color)
    opp_king_rel_sq = normalize_square(opp_king_sq, mover_color)

    for square, piece in board.piece_map().items():
        plane = halfkp_non_king_plane_id_relative(piece, mover_color)
        if plane is None:
            continue
        piece_rel_sq = normalize_square(square, mover_color)
        active.append(halfkp_index(own_king_rel_sq, plane, piece_rel_sq, HALFKP_OWN_OFFSET))
        active.append(halfkp_index(opp_king_rel_sq, plane, piece_rel_sq, HALFKP_OPP_OFFSET))

    if mover_color == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            active.append(CASTLE_OWN_K_IDX)
        if board.has_queenside_castling_rights(chess.WHITE):
            active.append(CASTLE_OWN_Q_IDX)
        if board.has_kingside_castling_rights(chess.BLACK):
            active.append(CASTLE_OPP_K_IDX)
        if board.has_queenside_castling_rights(chess.BLACK):
            active.append(CASTLE_OPP_Q_IDX)
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            active.append(CASTLE_OWN_K_IDX)
        if board.has_queenside_castling_rights(chess.BLACK):
            active.append(CASTLE_OWN_Q_IDX)
        if board.has_kingside_castling_rights(chess.WHITE):
            active.append(CASTLE_OPP_K_IDX)
        if board.has_queenside_castling_rights(chess.WHITE):
            active.append(CASTLE_OPP_Q_IDX)

    active.sort()
    return active


def material_balance_from_mover(board: chess.Board) -> float:
    """Return mover-centric material balance."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    own = 0
    opp = 0
    for _, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        value = values.get(piece.piece_type, 0)
        if piece.color == board.turn:
            own += value
        else:
            opp += value
    return float(own - opp)


def legal_to_by_from(board: chess.Board) -> Dict[str, List[int]]:
    """Return legal target squares keyed by normalized from-square."""
    mapping: Dict[str, List[int]] = {}
    mover_is_white = bool(board.turn == chess.WHITE)
    for mv in board.legal_moves:
        rel_from = normalize_square(mv.from_square, mover_is_white)
        rel_to = normalize_square(mv.to_square, mover_is_white)
        key = str(rel_from)
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(rel_to)

    for key in mapping:
        mapping[key] = sorted(set(mapping[key]))
    return mapping


def legal_from_mask(mapping: Dict[str, List[int]]) -> List[int]:
    """Return 64-dim binary mask for legal from-squares."""
    mask = [0] * 64
    for key in mapping.keys():
        mask[int(key)] = 1
    return mask


def history_to_arrays(history: Iterable[Dict[str, Any]], history_plies: int) -> Dict[str, List[List[float]] | List[int]]:
    """Pad/truncate history and convert to event/delta arrays."""
    hist_list = list(history)[-history_plies:]

    event_rows: List[List[float]] = []
    delta_rows: List[List[float]] = []
    mask: List[int] = []

    for item in hist_list:
        event = item.get("event", {})
        delta = item.get("delta", {})
        event_rows.append([
            float(event.get("mover_is_target", 0)),
            float(event.get("piece_type", 0)),
            float(event.get("from_sq", 0)),
            float(event.get("to_sq", 0)),
            float(event.get("is_capture", 0)),
            float(event.get("is_check", 0)),
            float(event.get("is_castling", 0)),
            float(event.get("is_promotion", 0)),
        ])
        delta_rows.append([
            float(delta.get("material_delta", 0.0)),
            float(delta.get("king_safety_delta", 0.0)),
            float(delta.get("pawn_structure_delta", 0.0)),
        ])
        mask.append(1)

    pad = history_plies - len(event_rows)
    if pad > 0:
        event_rows = [[0.0] * 8 for _ in range(pad)] + event_rows
        delta_rows = [[0.0] * 3 for _ in range(pad)] + delta_rows
        mask = [0] * pad + mask

    return {
        "history_event": event_rows,
        "history_delta": delta_rows,
        "history_mask": mask,
    }


def build_dense_state(board: chess.Board, row: Dict[str, Any]) -> List[float]:
    """Build compact dense state summary features for the current position."""
    return [
        material_balance_from_mover(board),
        float(row.get("phase_code", 0)),
        float(board.legal_moves.count()),
        float(board.has_kingside_castling_rights(board.turn)),
        float(board.has_queenside_castling_rights(board.turn)),
        float(board.fullmove_number),
    ]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSONL rows as dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def encode_row(row: Dict[str, Any], history_plies: int) -> Dict[str, Any]:
    """Encode one parsed position row into a policy training sample."""
    fen = str(row["fen_before"])
    board = chess.Board(fen)

    legal_map = legal_to_by_from(board)
    history_arrays = history_to_arrays(row.get("history", []), history_plies)

    sample = {
        "game_id": row.get("game_id"),
        "player_id": str(row.get("player_id") or row.get("target_username") or "unknown").strip().lower(),
        "ply_index": int(row.get("ply_index", 0)),
        "fen_before": fen,
        "active_feature_indices": encode_board_sparse_indices(fen),
        "dense_state": build_dense_state(board, row),
        "history_event": history_arrays["history_event"],
        "history_delta": history_arrays["history_delta"],
        "history_mask": history_arrays["history_mask"],
        "context": [
            float(row.get("target_is_white", 0)),
            float(row.get("clock_after_move_sec") or 0.0),
        ],
        "legal_from_mask": legal_from_mask(legal_map),
        "legal_to_by_from": legal_map,
        "target_from_sq": int(row["target_from_sq"]),
        "target_to_sq": int(row["target_to_sq"]),
        "target_promotion": int(row.get("target_promotion", 0)),
        "target_uci": row.get("played_uci"),
    }
    return sample


def main() -> None:
    """Run stage 2 and write encoded policy samples as JSONL."""
    in_path = Path(CONFIG["input_jsonl"])
    out_path = Path(CONFIG["output_jsonl"])
    history_plies = int(CONFIG["history_plies"])
    progress_every = int(CONFIG.get("progress_every", 0))

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(in_path):
            sample = encode_row(row, history_plies)
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if bool(CONFIG.get("verbose", True)) and progress_every > 0 and count % progress_every == 0:
                print(f"Encoded {count} samples...")

    if bool(CONFIG.get("verbose", True)):
        print(f"Encoded {count} samples -> {out_path}")
        print(f"Feature vocab size (HalfKP): {TOTAL_FEATURE_DIM}")


if __name__ == "__main__":
    main()

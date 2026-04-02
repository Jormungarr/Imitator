#!/usr/bin/env python3
"""Parse PGN and build target-move samples with structured history."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import chess
import chess.pgn

from chess_feature_utils import (
    apply_piece_identity_move,
    build_history_entry,
    canonical_piece_slot,
    current_original_piece_slot_square_map,
    current_piece_identity,
    initialize_piece_identity_tracker,
    normalize_square,
    phase_code_from_fullmove,
)
from pipeline_config import (
    DATASET_TAG,
    HISTORY_PLIES,
    TARGET_NAME_ALIASES,
    TARGET_USERNAME,
    processed_dir,
    raw_pgn,
)


CONFIG = {
    "dataset_tag": DATASET_TAG,
    "pgn_path": raw_pgn(DATASET_TAG),
    "target_username": TARGET_USERNAME,
    "target_name_aliases": TARGET_NAME_ALIASES,
    "history_plies": HISTORY_PLIES,
    "output_jsonl": processed_dir(DATASET_TAG) / "positions_history.jsonl",
    "verbose": True,
}

CLK_PATTERN = re.compile(r"\[%clk\s+([0-9]+):([0-9]{2}):([0-9]{2})\]")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
PARENS_RE = re.compile(r"\([^)]*\)")


def parse_clock_to_seconds(comment: str) -> Optional[int]:
    """Return clock seconds from a Lichess-style [%clk h:mm:ss] comment."""
    if not comment:
        return None
    match = CLK_PATTERN.search(comment)
    if not match:
        return None
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def safe_int(x: Optional[str]) -> Optional[int]:
    """Convert header text to int when possible."""
    if x is None:
        return None
    x = x.strip()
    if not x or x == "?":
        return None
    try:
        return int(x)
    except ValueError:
        return None


def clean_name(name: str) -> str:
    """Normalize player names for robust fuzzy matching."""
    text = PARENS_RE.sub(" ", (name or "").lower())
    text = NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def name_tokens(name: str) -> List[str]:
    """Return cleaned non-empty tokens from player name."""
    return [t for t in clean_name(name).split() if t]


def build_target_name_profiles(target_username: str, aliases: List[str]) -> List[Dict[str, Any]]:
    """Build normalized target name profiles from canonical name and aliases."""
    names = [target_username] + [a for a in aliases if str(a).strip()]
    profiles: List[Dict[str, Any]] = []

    seen = set()
    for raw in names:
        cleaned = clean_name(raw)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        tokens = name_tokens(raw)
        if not tokens:
            continue
        profiles.append(
            {
                "raw": raw,
                "cleaned": cleaned,
                "tokens": set(tokens),
                "surname": tokens[-1],
                "first_initial": tokens[0][0],
            }
        )

    if not profiles:
        raise ValueError("No valid target name profile could be built")
    return profiles


def match_score(candidate_name: str, profile: Dict[str, Any]) -> int:
    """Return fuzzy match score between candidate header name and one target profile."""
    candidate_clean = clean_name(candidate_name)
    if not candidate_clean:
        return 0

    if candidate_clean == profile["cleaned"]:
        return 100

    if candidate_clean in profile["cleaned"] or profile["cleaned"] in candidate_clean:
        if len(candidate_clean) >= 4:
            return 90

    cand_tokens = set(name_tokens(candidate_name))
    overlap = len(cand_tokens & profile["tokens"])

    if overlap >= 2:
        return 80

    if profile["surname"] in cand_tokens:
        cand_first_initial = ""
        cand_list = name_tokens(candidate_name)
        if cand_list:
            cand_first_initial = cand_list[0][0]
        if overlap >= 1 and cand_first_initial == profile["first_initial"]:
            return 75
        if overlap >= 1:
            return 70

    return 0


def match_target_color(white_name: str, black_name: str, profiles: List[Dict[str, Any]]) -> Optional[chess.Color]:
    """Infer whether target is white/black using robust name matching."""
    white_best = max(match_score(white_name, p) for p in profiles)
    black_best = max(match_score(black_name, p) for p in profiles)

    threshold = 70
    white_hit = white_best >= threshold
    black_hit = black_best >= threshold

    if white_hit and not black_hit:
        return chess.WHITE
    if black_hit and not white_hit:
        return chess.BLACK
    if white_hit and black_hit:
        return chess.WHITE if white_best >= black_best else chess.BLACK
    return None


def promotion_index(move: chess.Move) -> int:
    """Map promotion piece type to compact id (0 none, 1 N, 2 B, 3 R, 4 Q)."""
    if move.promotion is None:
        return 0
    mapping = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    return int(mapping.get(move.promotion, 0))


def iter_target_player_positions(pgn_path: Path, target_username: str, aliases: List[str], history_plies: int) -> Iterator[Dict[str, Any]]:
    """Yield one row per target move with trailing history from recent plies."""
    target_profiles = build_target_name_profiles(target_username, aliases)

    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        game_index = 0

        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            game_index += 1
            headers = game.headers

            white = headers.get("White", "")
            black = headers.get("Black", "")

            target_color = match_target_color(white, black, target_profiles)
            if target_color is None:
                continue

            target_is_white = bool(target_color == chess.WHITE)
            history: Deque[Dict[str, Any]] = deque(maxlen=history_plies)

            board = game.board()
            piece_id_by_square, promotion_counters = initialize_piece_identity_tracker(board)
            game_id = headers.get("GameId") or f"game_{game_index}"

            node = game
            ply_index = 0
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move

                if move == chess.Move.null():
                    # Some source PGNs encode missing/unknown plies as "--".
                    # Preserve downstream move alignment by advancing the board turn,
                    # but do not emit a sample or inject a fake history event.
                    board.push(move)
                    node = next_node
                    ply_index += 1
                    continue

                moved_piece_id = current_piece_identity(piece_id_by_square, move.from_square)
                moved_piece_slot = canonical_piece_slot(moved_piece_id)
                piece_slot_to_square = current_original_piece_slot_square_map(piece_id_by_square, target_color, bool(target_is_white))

                if board.turn == target_color:
                    row = {
                        "dataset_tag": DATASET_TAG,
                        "game_id": game_id,
                        "game_index": game_index,
                        "ply_index": ply_index,
                        "fullmove_number": board.fullmove_number,
                        "fen_before": board.fen(),
                        "played_uci": move.uci(),
                        "moved_piece_id": moved_piece_id,
                        "moved_piece_slot": moved_piece_slot or "",
                        "piece_slot_to_square": piece_slot_to_square,
                        "target_from_sq": normalize_square(move.from_square, target_is_white),
                        "target_to_sq": normalize_square(move.to_square, target_is_white),
                        "target_promotion": promotion_index(move),
                        "target_username": target_username,
                        "target_is_white": int(target_is_white),
                        "white": white,
                        "black": black,
                        "white_elo": safe_int(headers.get("WhiteElo")),
                        "black_elo": safe_int(headers.get("BlackElo")),
                        "eco": headers.get("ECO"),
                        "opening": headers.get("Opening"),
                        "time_control": headers.get("TimeControl"),
                        "result": headers.get("Result"),
                        "phase_code": phase_code_from_fullmove(board.fullmove_number),
                        "history": list(history),
                    }
                    yield row

                history.append(build_history_entry(board, move, target_color, target_is_white, moved_piece_id=moved_piece_id))
                apply_piece_identity_move(board, piece_id_by_square, move, promotion_counters)
                board.push(move)
                node = next_node
                ply_index += 1


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required inputs and constraints before parsing."""
    pgn_path = Path(config["pgn_path"])
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")
    if int(config["history_plies"]) <= 0:
        raise ValueError("history_plies must be positive")
    if not str(config["target_username"]).strip():
        raise ValueError("target_username cannot be empty")


def main() -> None:
    """Run stage 1 and save history-aware position samples as JSONL."""
    validate_config(CONFIG)

    out_path = Path(CONFIG["output_jsonl"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in iter_target_player_positions(
            pgn_path=Path(CONFIG["pgn_path"]),
            target_username=str(CONFIG["target_username"]),
            aliases=list(CONFIG.get("target_name_aliases", [])),
            history_plies=int(CONFIG["history_plies"]),
        ):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1

    if bool(CONFIG.get("verbose", True)):
        print(f"Saved {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()






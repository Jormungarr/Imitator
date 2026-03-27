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
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


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


def normalize_square(square: int, target_is_white: bool) -> int:
    """Map board square to target-relative square coordinates."""
    return square if target_is_white else chess.square_mirror(square)


def phase_code_from_fullmove(fullmove_number: int) -> int:
    """Return a coarse phase id (0 opening, 1 middlegame, 2 endgame)."""
    if fullmove_number < 10:
        return 0
    if fullmove_number < 30:
        return 1
    return 2


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

    # Common variant: different first name but same family name.
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


def material_balance(board: chess.Board, target_color: bool) -> float:
    """Return target-centric material balance."""
    own = 0
    opp = 0
    for _, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == target_color:
            own += value
        else:
            opp += value
    return float(own - opp)


def king_safety_proxy(board: chess.Board, target_color: bool) -> float:
    """Return a simple king-pressure differential proxy."""
    own_king_sq = board.king(target_color)
    opp_king_sq = board.king(not target_color)
    if own_king_sq is None or opp_king_sq is None:
        return 0.0

    own_pressure = len(board.attackers(not target_color, own_king_sq))
    opp_pressure = len(board.attackers(target_color, opp_king_sq))
    return float(opp_pressure - own_pressure)


def pawn_structure_proxy(board: chess.Board, target_color: bool) -> float:
    """Return target-centric pawn island differential."""

    def pawn_islands(color: bool) -> int:
        files_with_pawns = set()
        for sq, piece in board.piece_map().items():
            if piece.color == color and piece.piece_type == chess.PAWN:
                files_with_pawns.add(chess.square_file(sq))
        if not files_with_pawns:
            return 0
        islands = 0
        prev = None
        for file_idx in sorted(files_with_pawns):
            if prev is None or file_idx != prev + 1:
                islands += 1
            prev = file_idx
        return islands

    own = pawn_islands(target_color)
    opp = pawn_islands(not target_color)
    return float(opp - own)


def promotion_index(move: chess.Move) -> int:
    """Map promotion piece type to compact id (0 none, 1 N, 2 B, 3 R, 4 Q)."""
    if move.promotion is None:
        return 0
    mapping = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    return int(mapping.get(move.promotion, 0))


def build_history_entry(board_before: chess.Board, move: chess.Move, target_color: bool, target_is_white: bool) -> Dict[str, Any]:
    """Build one history step with move-event and state-delta features."""
    piece = board_before.piece_at(move.from_square)
    moving_piece_type = 0 if piece is None else int(piece.piece_type)

    mat_before = material_balance(board_before, target_color)
    king_before = king_safety_proxy(board_before, target_color)
    pawn_before = pawn_structure_proxy(board_before, target_color)

    is_capture = int(board_before.is_capture(move))
    is_check = int(board_before.gives_check(move))
    is_castling = int(board_before.is_castling(move))
    is_promotion = int(move.promotion is not None)

    board_after = board_before.copy(stack=False)
    board_after.push(move)

    mat_after = material_balance(board_after, target_color)
    king_after = king_safety_proxy(board_after, target_color)
    pawn_after = pawn_structure_proxy(board_after, target_color)

    return {
        "event": {
            "mover_is_target": int(board_before.turn == target_color),
            "piece_type": moving_piece_type,
            "from_sq": normalize_square(move.from_square, target_is_white),
            "to_sq": normalize_square(move.to_square, target_is_white),
            "is_capture": is_capture,
            "is_check": is_check,
            "is_castling": is_castling,
            "is_promotion": is_promotion,
        },
        "delta": {
            "material_delta": mat_after - mat_before,
            "king_safety_delta": king_after - king_before,
            "pawn_structure_delta": pawn_after - pawn_before,
        },
    }


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
            game_id = headers.get("GameId") or f"game_{game_index}"

            node = game
            ply_index = 0
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move

                if board.turn == target_color:
                    row = {
                        "dataset_tag": DATASET_TAG,
                        "game_id": game_id,
                        "game_index": game_index,
                        "ply_index": ply_index,
                        "fullmove_number": board.fullmove_number,
                        "fen_before": board.fen(),
                        "played_uci": move.uci(),
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
                        "clock_after_move_sec": parse_clock_to_seconds(next_node.comment),
                        "phase_code": phase_code_from_fullmove(board.fullmove_number),
                        "history": list(history),
                    }
                    yield row

                history.append(build_history_entry(board, move, target_color, target_is_white))
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
    seen_games = set()

    with out_path.open("w", encoding="utf-8") as out:
        for row in iter_target_player_positions(
            pgn_path=Path(CONFIG["pgn_path"]),
            target_username=str(CONFIG["target_username"]),
            aliases=list(CONFIG.get("target_name_aliases", [])),
            history_plies=int(CONFIG["history_plies"]),
        ):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
            seen_games.add(row["game_id"])

    if bool(CONFIG.get("verbose", True)):
        print(f"Saved {n_rows} target-move rows from {len(seen_games)} games -> {out_path}")


if __name__ == "__main__":
    main()

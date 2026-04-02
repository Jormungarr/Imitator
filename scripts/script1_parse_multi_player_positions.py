#!/usr/bin/env python3
"""Parse merged pretraining PGN and emit all-player move samples with history."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, Optional

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
from pipeline_config import HISTORY_PLIES, PRETRAIN_DATASET_TAG, pretrain_merged_pgn, processed_dir


CONFIG = {
    "dataset_tag": PRETRAIN_DATASET_TAG,
    "input_pgn": pretrain_merged_pgn(),
    "history_plies": HISTORY_PLIES,
    "output_jsonl": processed_dir(PRETRAIN_DATASET_TAG) / "positions_history.jsonl",
    "progress_every_games": 100,
    "verbose": True,
}

CLK_PATTERN = re.compile(r"\[%clk\s+([0-9]+):([0-9]{2}):([0-9]{2})\]")


def parse_clock_to_seconds(comment: str) -> Optional[int]:
    """Return clock seconds from [%clk h:mm:ss] when present."""
    if not comment:
        return None
    m = CLK_PATTERN.search(comment)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def safe_int(x: Optional[str]) -> Optional[int]:
    """Convert optional string to int."""
    if x is None:
        return None
    x = x.strip()
    if not x or x == "?":
        return None
    try:
        return int(x)
    except ValueError:
        return None


def promotion_index(move: chess.Move) -> int:
    """Map promotion piece type to compact promotion id."""
    if move.promotion is None:
        return 0
    return {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}.get(move.promotion, 0)


def player_id(name: str) -> str:
    """Convert player display name into stable lowercase id."""
    cleaned = " ".join(name.strip().split())
    return cleaned.lower() if cleaned else "unknown"


def iter_all_player_positions(pgn_path: Path, history_plies: int) -> Iterator[Dict[str, Any]]:
    """Yield one row per move, treating side-to-move as the target player."""
    with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
        game_index = 0

        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            game_index += 1
            headers = game.headers
            if headers.get("SetUp") == "1":
                continue

            white_name = headers.get("White", "White")
            black_name = headers.get("Black", "Black")
            white_id = player_id(white_name)
            black_id = player_id(black_name)

            history_white: Deque[Dict[str, Any]] = deque(maxlen=history_plies)
            history_black: Deque[Dict[str, Any]] = deque(maxlen=history_plies)

            board = game.board()
            piece_id_by_square, promotion_counters = initialize_piece_identity_tracker(board)
            game_id_base = headers.get("GameId") or f"game_{game_index}"
            game_id = f"{pgn_path.stem}:{game_id_base}"

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

                mover_is_white = bool(board.turn == chess.WHITE)
                moved_piece_id = current_piece_identity(piece_id_by_square, move.from_square)
                moved_piece_slot = canonical_piece_slot(moved_piece_id)
                piece_slot_to_square = current_original_piece_slot_square_map(piece_id_by_square, board.turn, bool(mover_is_white))
                target_name = white_name if mover_is_white else black_name
                target_pid = white_id if mover_is_white else black_id
                target_hist = history_white if mover_is_white else history_black

                row = {
                    "dataset_tag": str(CONFIG["dataset_tag"]),
                    "game_id": game_id,
                    "game_index": game_index,
                    "ply_index": ply_index,
                    "fullmove_number": board.fullmove_number,
                    "fen_before": board.fen(),
                    "played_uci": move.uci(),
                    "moved_piece_id": moved_piece_id,
                    "moved_piece_slot": moved_piece_slot or "",
                    "piece_slot_to_square": piece_slot_to_square,
                    "target_from_sq": normalize_square(move.from_square, mover_is_white),
                    "target_to_sq": normalize_square(move.to_square, mover_is_white),
                    "target_promotion": promotion_index(move),
                    "target_username": target_name,
                    "player_id": target_pid,
                    "target_is_white": int(mover_is_white),
                    "white": white_name,
                    "black": black_name,
                    "white_elo": safe_int(headers.get("WhiteElo")),
                    "black_elo": safe_int(headers.get("BlackElo")),
                    "eco": headers.get("ECO"),
                    "opening": headers.get("Opening"),
                    "time_control": headers.get("TimeControl"),
                    "result": headers.get("Result"),
                    "phase_code": phase_code_from_fullmove(board.fullmove_number),
                    "history": list(target_hist),
                }
                yield row

                history_white.append(build_history_entry(board, move, chess.WHITE, True, moved_piece_id=moved_piece_id))
                history_black.append(build_history_entry(board, move, chess.BLACK, False, moved_piece_id=moved_piece_id))

                apply_piece_identity_move(board, piece_id_by_square, move, promotion_counters)
                board.push(move)
                node = next_node
                ply_index += 1


def main() -> None:
    """Run multi-player parsing from merged pretraining PGN."""
    input_pgn = Path(CONFIG["input_pgn"])
    if not input_pgn.exists():
        raise FileNotFoundError(f"Merged pretrain PGN not found: {input_pgn}")

    out_path = Path(CONFIG["output_jsonl"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    games = 0
    progress_every = int(CONFIG["progress_every_games"])

    with out_path.open("w", encoding="utf-8") as out:
        for row in iter_all_player_positions(input_pgn, int(CONFIG["history_plies"])):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            if int(row["ply_index"]) == 0:
                games += 1
                if bool(CONFIG.get("verbose", True)) and progress_every > 0 and games % progress_every == 0:
                    print(f"Processed games={games}, rows={rows}")

    if bool(CONFIG.get("verbose", True)):
        print(f"Saved rows={rows} from games={games} -> {out_path}")


if __name__ == "__main__":
    main()






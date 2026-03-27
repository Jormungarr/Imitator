#!/usr/bin/env python3
"""Parse merged pretraining PGN and emit all-player move samples with history."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional

import chess
import chess.pgn

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
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


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


def normalize_square(square: int, target_is_white: bool) -> int:
    """Normalize square to player-relative perspective."""
    return square if target_is_white else chess.square_mirror(square)


def phase_code_from_fullmove(fullmove_number: int) -> int:
    """Return coarse phase id."""
    if fullmove_number < 10:
        return 0
    if fullmove_number < 30:
        return 1
    return 2


def material_balance(board: chess.Board, target_color: bool) -> float:
    """Return target-centric material balance."""
    own = 0
    opp = 0
    for _, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == target_color:
            own += val
        else:
            opp += val
    return float(own - opp)


def king_safety_proxy(board: chess.Board, target_color: bool) -> float:
    """Return simple king-pressure differential proxy."""
    own_king = board.king(target_color)
    opp_king = board.king(not target_color)
    if own_king is None or opp_king is None:
        return 0.0
    own_pressure = len(board.attackers(not target_color, own_king))
    opp_pressure = len(board.attackers(target_color, opp_king))
    return float(opp_pressure - own_pressure)


def pawn_structure_proxy(board: chess.Board, target_color: bool) -> float:
    """Return pawn-island differential proxy."""

    def pawn_islands(color: bool) -> int:
        files = set()
        for sq, piece in board.piece_map().items():
            if piece.color == color and piece.piece_type == chess.PAWN:
                files.add(chess.square_file(sq))
        if not files:
            return 0
        islands = 0
        prev = None
        for file_idx in sorted(files):
            if prev is None or file_idx != prev + 1:
                islands += 1
            prev = file_idx
        return islands

    own = pawn_islands(target_color)
    opp = pawn_islands(not target_color)
    return float(opp - own)


def promotion_index(move: chess.Move) -> int:
    """Map promotion piece type to compact promotion id."""
    if move.promotion is None:
        return 0
    return {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}.get(move.promotion, 0)


def build_history_entry(board_before: chess.Board, move: chess.Move, target_color: bool, target_is_white: bool) -> Dict[str, Any]:
    """Build one history event/delta entry for a specific target perspective."""
    piece = board_before.piece_at(move.from_square)
    moving_piece_type = 0 if piece is None else int(piece.piece_type)

    mat_before = material_balance(board_before, target_color)
    king_before = king_safety_proxy(board_before, target_color)
    pawn_before = pawn_structure_proxy(board_before, target_color)

    board_after = board_before.copy(stack=False)
    board_after.push(move)

    return {
        "event": {
            "mover_is_target": int(board_before.turn == target_color),
            "piece_type": moving_piece_type,
            "from_sq": normalize_square(move.from_square, target_is_white),
            "to_sq": normalize_square(move.to_square, target_is_white),
            "is_capture": int(board_before.is_capture(move)),
            "is_check": int(board_before.gives_check(move)),
            "is_castling": int(board_before.is_castling(move)),
            "is_promotion": int(move.promotion is not None),
        },
        "delta": {
            "material_delta": material_balance(board_after, target_color) - mat_before,
            "king_safety_delta": king_safety_proxy(board_after, target_color) - king_before,
            "pawn_structure_delta": pawn_structure_proxy(board_after, target_color) - pawn_before,
        },
    }


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

            white_name = headers.get("White", "White")
            black_name = headers.get("Black", "Black")
            white_id = player_id(white_name)
            black_id = player_id(black_name)

            history_white: Deque[Dict[str, Any]] = deque(maxlen=history_plies)
            history_black: Deque[Dict[str, Any]] = deque(maxlen=history_plies)

            board = game.board()
            game_id_base = headers.get("GameId") or f"game_{game_index}"
            game_id = f"{pgn_path.stem}:{game_id_base}"

            node = game
            ply_index = 0
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move

                mover_is_white = bool(board.turn == chess.WHITE)
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
                    "clock_after_move_sec": parse_clock_to_seconds(next_node.comment),
                    "phase_code": phase_code_from_fullmove(board.fullmove_number),
                    "history": list(target_hist),
                }
                yield row

                history_white.append(build_history_entry(board, move, chess.WHITE, True))
                history_black.append(build_history_entry(board, move, chess.BLACK, False))

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

#!/usr/bin/env python3
"""Shared handcrafted chess feature helpers for policy parsing and encoding."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import chess


PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

FILE_NAMES = "abcdefgh"
PROMOTION_PIECE_TAGS: Dict[int, str] = {
    chess.KNIGHT: "Nprom",
    chess.BISHOP: "Bprom",
    chess.ROOK: "Rprom",
    chess.QUEEN: "Qprom",
}

ORIGINAL_PIECE_SLOTS: List[str] = [
    "K",
    "Q",
    "Rq",
    "Rk",
    "Bq",
    "Bk",
    "Nq",
    "Nk",
    "Pa",
    "Pb",
    "Pc",
    "Pd",
    "Pe",
    "Pf",
    "Pg",
    "Ph",
]
ORIGINAL_SLOT_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ORIGINAL_PIECE_SLOTS)}


def canonical_piece_slot(piece_id: str) -> Optional[str]:
    """Return the original 16-slot identity carried by a tracked piece id."""
    text = str(piece_id or "").strip()
    parts = text.split("_")
    if len(parts) < 2:
        return None
    slot = parts[1]
    return slot if slot in ORIGINAL_SLOT_TO_INDEX else None


def current_original_piece_slot_square_map(
    piece_id_by_square: Dict[int, str],
    target_color: bool,
    target_is_white: bool,
) -> Dict[str, int]:
    """Return current normalized square for each surviving original piece slot."""
    prefix = f"{_color_tag(target_color)}_"
    mapping: Dict[str, int] = {}
    for square, piece_id in piece_id_by_square.items():
        text = str(piece_id)
        if not text.startswith(prefix):
            continue
        slot = canonical_piece_slot(text)
        if slot is None:
            continue
        mapping[slot] = normalize_square(int(square), target_is_white)
    return mapping


def _color_tag(color: bool) -> str:
    return "w" if color == chess.WHITE else "b"


def _origin_piece_ids(color: bool) -> Dict[int, str]:
    rank = 0 if color == chess.WHITE else 7
    color_tag = _color_tag(color)
    ids = {
        chess.square(chess.FILE_NAMES.index("e"), rank): f"{color_tag}_K",
        chess.square(chess.FILE_NAMES.index("d"), rank): f"{color_tag}_Q",
        chess.square(chess.FILE_NAMES.index("a"), rank): f"{color_tag}_Rq",
        chess.square(chess.FILE_NAMES.index("h"), rank): f"{color_tag}_Rk",
        chess.square(chess.FILE_NAMES.index("c"), rank): f"{color_tag}_Bq",
        chess.square(chess.FILE_NAMES.index("f"), rank): f"{color_tag}_Bk",
        chess.square(chess.FILE_NAMES.index("b"), rank): f"{color_tag}_Nq",
        chess.square(chess.FILE_NAMES.index("g"), rank): f"{color_tag}_Nk",
    }
    pawn_rank = 1 if color == chess.WHITE else 6
    for file_idx, file_name in enumerate(FILE_NAMES):
        ids[chess.square(file_idx, pawn_rank)] = f"{color_tag}_P{file_name}"
    return ids


def _slot_home_square(color: bool, slot: str) -> int:
    """Return the classical home square for one original slot."""
    rank = 0 if color == chess.WHITE else 7
    pawn_rank = 1 if color == chess.WHITE else 6
    if slot == "K":
        return chess.square(chess.FILE_NAMES.index("e"), rank)
    if slot == "Q":
        return chess.square(chess.FILE_NAMES.index("d"), rank)
    if slot == "Rq":
        return chess.square(chess.FILE_NAMES.index("a"), rank)
    if slot == "Rk":
        return chess.square(chess.FILE_NAMES.index("h"), rank)
    if slot == "Bq":
        return chess.square(chess.FILE_NAMES.index("c"), rank)
    if slot == "Bk":
        return chess.square(chess.FILE_NAMES.index("f"), rank)
    if slot == "Nq":
        return chess.square(chess.FILE_NAMES.index("b"), rank)
    if slot == "Nk":
        return chess.square(chess.FILE_NAMES.index("g"), rank)
    if slot.startswith("P") and len(slot) == 2 and slot[1] in FILE_NAMES:
        return chess.square(FILE_NAMES.index(slot[1]), pawn_rank)
    raise KeyError(f"Unknown original slot: {slot}")


def _piece_distance(square_a: int, square_b: int) -> int:
    """Cheap square distance for deterministic fallback assignment."""
    return abs(chess.square_file(square_a) - chess.square_file(square_b)) + abs(chess.square_rank(square_a) - chess.square_rank(square_b))


def _assign_slots_by_home_distance(
    piece_squares: Iterable[int],
    slot_names: Iterable[str],
    color: bool,
) -> Dict[int, str]:
    """Greedily assign current pieces to canonical slots of the same type."""
    remaining_squares = sorted(int(sq) for sq in piece_squares)
    remaining_slots = list(slot_names)
    assigned: Dict[int, str] = {}

    while remaining_squares and remaining_slots:
        best: Optional[Tuple[int, int, int, str]] = None
        for square in remaining_squares:
            for slot in remaining_slots:
                candidate = (_piece_distance(square, _slot_home_square(color, slot)), square, ORIGINAL_SLOT_TO_INDEX[slot], slot)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, square, _, slot = best
        assigned[square] = slot
        remaining_squares.remove(square)
        remaining_slots.remove(slot)
    return assigned


def _initial_back_rank_slots(board: chess.Board, color: bool) -> Optional[Dict[int, str]]:
    """Infer original-piece slots directly from an initial back rank, including Chess960."""
    rank = 0 if color == chess.WHITE else 7
    back_rank_squares = [chess.square(file_idx, rank) for file_idx in range(8)]
    pieces = [(sq, board.piece_at(sq)) for sq in back_rank_squares]
    if any(piece is None or piece.color != color or piece.piece_type == chess.PAWN for _, piece in pieces):
        return None

    by_type: Dict[int, List[int]] = {}
    for square, piece in pieces:
        by_type.setdefault(int(piece.piece_type), []).append(square)

    expected = {
        chess.KING: 1,
        chess.QUEEN: 1,
        chess.ROOK: 2,
        chess.BISHOP: 2,
        chess.KNIGHT: 2,
    }
    if any(len(by_type.get(piece_type, [])) != count for piece_type, count in expected.items()):
        return None

    king_square = by_type[chess.KING][0]
    king_file = chess.square_file(king_square)
    rook_squares = sorted(by_type[chess.ROOK], key=chess.square_file)
    bishop_squares = sorted(by_type[chess.BISHOP], key=chess.square_file)
    knight_squares = sorted(by_type[chess.KNIGHT], key=chess.square_file)

    q_rooks = [sq for sq in rook_squares if chess.square_file(sq) < king_file]
    k_rooks = [sq for sq in rook_squares if chess.square_file(sq) > king_file]
    if len(q_rooks) != 1 or len(k_rooks) != 1:
        return None

    return {
        king_square: "K",
        by_type[chess.QUEEN][0]: "Q",
        q_rooks[0]: "Rq",
        k_rooks[0]: "Rk",
        bishop_squares[0]: "Bq",
        bishop_squares[1]: "Bk",
        knight_squares[0]: "Nq",
        knight_squares[1]: "Nk",
    }


def _board_piece_ids(board: chess.Board, color: bool) -> Tuple[Dict[int, str], Dict[Tuple[bool, int], int]]:
    """Build tracked ids from an arbitrary board position."""
    color_tag = _color_tag(color)
    tracker: Dict[int, str] = {}
    promotion_counters: Dict[Tuple[bool, int], int] = {}

    back_rank_slots = _initial_back_rank_slots(board, color)
    if back_rank_slots is not None:
        for square, slot in back_rank_slots.items():
            tracker[square] = f"{color_tag}_{slot}"
    else:
        piece_type_to_slots = {
            chess.KING: ["K"],
            chess.QUEEN: ["Q"],
            chess.ROOK: ["Rq", "Rk"],
            chess.BISHOP: ["Bq", "Bk"],
            chess.KNIGHT: ["Nq", "Nk"],
        }
        for piece_type, slot_names in piece_type_to_slots.items():
            squares = board.pieces(piece_type, color)
            assigned = _assign_slots_by_home_distance(squares, slot_names, color)
            for square, slot in assigned.items():
                tracker[square] = f"{color_tag}_{slot}"

            extra_squares = sorted(int(sq) for sq in squares if int(sq) not in assigned)
            if extra_squares:
                promo_tag = PROMOTION_PIECE_TAGS.get(piece_type, "Prom")
                for idx, square in enumerate(extra_squares, start=1):
                    tracker[square] = f"{color_tag}_{promo_tag}{idx}"
                promotion_counters[(color, piece_type)] = len(extra_squares)

    for file_idx, file_name in enumerate(FILE_NAMES):
        square = chess.square(file_idx, 1 if color == chess.WHITE else 6)
        piece = board.piece_at(square)
        if piece is not None and piece.color == color and piece.piece_type == chess.PAWN:
            tracker[square] = f"{color_tag}_P{file_name}"

    pawn_slots = [f"P{file_name}" for file_name in FILE_NAMES]
    remaining_pawn_squares = sorted(
        int(sq)
        for sq in board.pieces(chess.PAWN, color)
        if int(sq) not in tracker
    )
    assigned_pawns = _assign_slots_by_home_distance(remaining_pawn_squares, pawn_slots, color)
    for square, slot in assigned_pawns.items():
        tracker[square] = f"{color_tag}_{slot}"

    extra_pawns = sorted(int(sq) for sq in remaining_pawn_squares if int(sq) not in assigned_pawns)
    if extra_pawns:
        for idx, square in enumerate(extra_pawns, start=1):
            tracker[square] = f"{color_tag}_Pprom{idx}"
        promotion_counters[(color, chess.PAWN)] = len(extra_pawns)

    return tracker, promotion_counters


def initialize_piece_identity_tracker(board: Optional[chess.Board] = None) -> Tuple[Dict[int, str], Dict[Tuple[bool, int], int]]:
    """Return current square->piece-id map plus promotion counters."""
    if board is None:
        tracker: Dict[int, str] = {}
        tracker.update(_origin_piece_ids(chess.WHITE))
        tracker.update(_origin_piece_ids(chess.BLACK))
        return tracker, {}

    tracker: Dict[int, str] = {}
    promotion_counters: Dict[Tuple[bool, int], int] = {}
    for color in (chess.WHITE, chess.BLACK):
        color_tracker, color_promos = _board_piece_ids(board, color)
        tracker.update(color_tracker)
        promotion_counters.update(color_promos)
    return tracker, promotion_counters


def current_piece_identity(piece_id_by_square: Dict[int, str], from_square: int) -> str:
    """Return the current tracked identity for the moving piece on a square."""
    moved_piece_id = piece_id_by_square.get(int(from_square))
    if not moved_piece_id:
        raise KeyError(f"Missing tracked piece id for square {from_square}")
    return moved_piece_id


def apply_piece_identity_move(
    board_before: chess.Board,
    piece_id_by_square: Dict[int, str],
    move: chess.Move,
    promotion_counters: Dict[Tuple[bool, int], int],
) -> str:
    """Update tracked piece identities after one move and return the post-move id."""
    moved_piece_id = current_piece_identity(piece_id_by_square, move.from_square)
    mover_color = bool(board_before.turn)
    moving_piece = board_before.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"No moving piece on square {move.from_square}")

    piece_id_by_square.pop(move.from_square, None)

    if board_before.is_en_passant(move):
        captured_square = move.to_square - 8 if mover_color == chess.WHITE else move.to_square + 8
        piece_id_by_square.pop(captured_square, None)
    elif board_before.is_capture(move):
        piece_id_by_square.pop(move.to_square, None)

    if board_before.is_castling(move):
        rank = 0 if mover_color == chess.WHITE else 7
        if move.to_square > move.from_square:
            rook_from = chess.square(chess.FILE_NAMES.index("h"), rank)
            rook_to = chess.square(chess.FILE_NAMES.index("f"), rank)
        else:
            rook_from = chess.square(chess.FILE_NAMES.index("a"), rank)
            rook_to = chess.square(chess.FILE_NAMES.index("d"), rank)
        rook_id = piece_id_by_square.pop(rook_from, None)
        if rook_id is not None:
            piece_id_by_square[rook_to] = rook_id

    post_move_id = moved_piece_id
    if moving_piece.piece_type == chess.PAWN and move.promotion is not None:
        promo_piece = int(move.promotion)
        promo_key = (mover_color, promo_piece)
        promo_index = promotion_counters.get(promo_key, 0) + 1
        promotion_counters[promo_key] = promo_index
        promo_tag = PROMOTION_PIECE_TAGS.get(promo_piece, "Prom")
        origin_slot = canonical_piece_slot(moved_piece_id) or "P?"
        post_move_id = f"{_color_tag(mover_color)}_{origin_slot}_{promo_tag}{promo_index}"

    piece_id_by_square[move.to_square] = post_move_id
    return post_move_id


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


def count_non_king_pieces(board: chess.Board, color: bool) -> int:
    """Count all non-king pieces for one side."""
    total = 0
    for piece_type in PIECE_VALUES:
        total += len(board.pieces(piece_type, color))
    return total


def count_piece_type(board: chess.Board, color: bool, piece_type: int) -> int:
    """Count pieces of one type for one side."""
    return len(board.pieces(piece_type, color))


def _pawn_file_counts(board: chess.Board, color: bool) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for sq in board.pieces(chess.PAWN, color):
        file_idx = chess.square_file(sq)
        counts[file_idx] = counts.get(file_idx, 0) + 1
    return counts


def pawn_islands(board: chess.Board, color: bool) -> int:
    """Count pawn islands for one side."""
    files = sorted(_pawn_file_counts(board, color).keys())
    if not files:
        return 0
    islands = 0
    prev = None
    for file_idx in files:
        if prev is None or file_idx != prev + 1:
            islands += 1
        prev = file_idx
    return islands


def doubled_pawns(board: chess.Board, color: bool) -> int:
    """Count doubled pawns as extra pawns beyond the first on each file."""
    return sum(max(0, count - 1) for count in _pawn_file_counts(board, color).values())


def isolated_pawns(board: chess.Board, color: bool) -> int:
    """Count isolated pawns for one side."""
    file_counts = _pawn_file_counts(board, color)
    total = 0
    for sq in board.pieces(chess.PAWN, color):
        file_idx = chess.square_file(sq)
        if file_counts.get(file_idx - 1, 0) == 0 and file_counts.get(file_idx + 1, 0) == 0:
            total += 1
    return total


def passed_pawns(board: chess.Board, color: bool) -> int:
    """Count passed pawns for one side."""
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    total = 0
    direction = 1 if color == chess.WHITE else -1
    for sq in board.pieces(chess.PAWN, color):
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        blocked = False
        for enemy_sq in enemy_pawns:
            enemy_file = chess.square_file(enemy_sq)
            enemy_rank = chess.square_rank(enemy_sq)
            if abs(enemy_file - file_idx) > 1:
                continue
            if direction == 1 and enemy_rank > rank_idx:
                blocked = True
                break
            if direction == -1 and enemy_rank < rank_idx:
                blocked = True
                break
        if not blocked:
            total += 1
    return total


def bishop_pair(board: chess.Board, color: bool) -> int:
    """Return 1 if side has bishop pair else 0."""
    return int(count_piece_type(board, color, chess.BISHOP) >= 2)


def open_files(board: chess.Board, color: bool) -> int:
    """Count open files beneficial to one side."""
    own = _pawn_file_counts(board, color)
    opp = _pawn_file_counts(board, not color)
    total = 0
    for file_idx in range(8):
        if own.get(file_idx, 0) == 0 and opp.get(file_idx, 0) == 0:
            total += 1
    return total


def semi_open_files(board: chess.Board, color: bool) -> int:
    """Count semi-open files for one side."""
    own = _pawn_file_counts(board, color)
    opp = _pawn_file_counts(board, not color)
    total = 0
    for file_idx in range(8):
        if own.get(file_idx, 0) == 0 and opp.get(file_idx, 0) > 0:
            total += 1
    return total


def _mobility(board: chess.Board, color: bool) -> int:
    tmp = board.copy(stack=False)
    tmp.turn = color
    return tmp.legal_moves.count()


def mobility_diff(board: chess.Board, target_color: bool) -> float:
    """Return target-centric legal-move mobility differential."""
    return float(_mobility(board, target_color) - _mobility(board, not target_color))


def center_control(board: chess.Board, color: bool) -> int:
    """Count attacks on the four center squares."""
    centers = [chess.D4, chess.E4, chess.D5, chess.E5]
    return sum(len(board.attackers(color, sq)) for sq in centers)


def center_control_diff(board: chess.Board, target_color: bool) -> float:
    """Return target-centric center-control differential."""
    return float(center_control(board, target_color) - center_control(board, not target_color))


def king_ring_pressure(board: chess.Board, attacker_color: bool, defender_color: bool) -> int:
    """Count attacker pressure on the defender king and adjacent squares."""
    king_sq = board.king(defender_color)
    if king_sq is None:
        return 0
    ring = [king_sq]
    k_file = chess.square_file(king_sq)
    k_rank = chess.square_rank(king_sq)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf = k_file + df
            nr = k_rank + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                ring.append(chess.square(nf, nr))
    return sum(len(board.attackers(attacker_color, sq)) for sq in ring)


def king_safety_proxy(board: chess.Board, target_color: bool) -> float:
    """Return target-centric king pressure differential."""
    own_pressure = king_ring_pressure(board, not target_color, target_color)
    opp_pressure = king_ring_pressure(board, target_color, not target_color)
    return float(opp_pressure - own_pressure)


def king_activity(board: chess.Board, color: bool) -> float:
    """Crude king-activity score, useful mostly in endgames."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0
    file_idx = chess.square_file(king_sq)
    rank_idx = chess.square_rank(king_sq)
    center_dist = abs(file_idx - 3.5) + abs(rank_idx - 3.5)
    return float(7.0 - center_dist)


def king_activity_diff(board: chess.Board, target_color: bool) -> float:
    """Return target-centric king activity differential."""
    return king_activity(board, target_color) - king_activity(board, not target_color)


def threatened_non_pawn_material(board: chess.Board, color: bool) -> int:
    """Count attacked non-pawn pieces for one side."""
    total = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(piece_type, color):
            if board.is_attacked_by(not color, sq):
                total += 1
    return total


def _least_attacker_value(board: chess.Board, victim_color: bool, square: int) -> Optional[int]:
    """Return the cheapest enemy attacker value on a square."""
    values: List[int] = []
    for attacker_sq in board.attackers(not victim_color, square):
        attacker = board.piece_at(attacker_sq)
        if attacker is None:
            continue
        values.append(PIECE_VALUES.get(attacker.piece_type, 100))
    return min(values) if values else None


def _least_defender_value(board: chess.Board, color: bool, square: int) -> Optional[int]:
    """Return the cheapest defender value on a square."""
    values: List[int] = []
    for defender_sq in board.attackers(color, square):
        defender = board.piece_at(defender_sq)
        if defender is None:
            continue
        values.append(PIECE_VALUES.get(defender.piece_type, 100))
    return min(values) if values else None


def piece_is_under_tactical_pressure(board: chess.Board, color: bool, square: int) -> bool:
    """Approximate whether keeping the piece on its square is tactically uncomfortable."""
    piece = board.piece_at(square)
    if piece is None or piece.color != color or piece.piece_type == chess.KING:
        return False
    if not board.is_attacked_by(not color, square):
        return False

    piece_value = PIECE_VALUES.get(piece.piece_type, 0)
    least_attacker = _least_attacker_value(board, color, square)
    least_defender = _least_defender_value(board, color, square)
    if least_attacker is None:
        return False
    if least_defender is None:
        return True
    if least_attacker <= piece_value:
        return True
    return least_attacker < least_defender


def hanging_non_king_piece_count(board: chess.Board, color: bool) -> int:
    """Count own non-king pieces under tactical pressure."""
    total = 0
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if piece_is_under_tactical_pressure(board, color, sq):
            total += 1
    return total


def hanging_non_king_piece_value(board: chess.Board, color: bool) -> float:
    """Return total value of own pressured non-king pieces."""
    total = 0.0
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        if piece_is_under_tactical_pressure(board, color, sq):
            total += float(PIECE_VALUES.get(piece.piece_type, 0))
    return total


def is_under_immediate_threat(board: chess.Board, target_color: bool) -> int:
    """Flag positions where the target side faces concrete tactical pressure."""
    king_sq = board.king(target_color)
    in_check = int(king_sq is not None and board.is_attacked_by(not target_color, king_sq))
    pressured_count = hanging_non_king_piece_count(board, target_color)
    pressured_value = hanging_non_king_piece_value(board, target_color)

    high_value_piece_under_pressure = 0
    for sq, piece in board.piece_map().items():
        if piece.color != target_color or piece.piece_type not in (chess.QUEEN, chess.ROOK):
            continue
        if piece_is_under_tactical_pressure(board, target_color, sq):
            high_value_piece_under_pressure = 1
            break

    return int(
        in_check
        or high_value_piece_under_pressure
        or pressured_value >= 3.0
        or pressured_count >= 2
    )


def threatened_piece_diff(board: chess.Board, target_color: bool) -> float:
    """Return target-centric threatened-piece differential."""
    own = threatened_non_pawn_material(board, target_color)
    opp = threatened_non_pawn_material(board, not target_color)
    return float(opp - own)


def count_piece_attacks_near_king(board: chess.Board, target_color: bool) -> float:
    """Return target-centric pressure near the opponent king."""
    return float(king_ring_pressure(board, target_color, not target_color))


def simplification_score(board: chess.Board, target_color: bool) -> float:
    """Return negative total non-king piece count; higher means simpler board."""
    total = count_non_king_pieces(board, chess.WHITE) + count_non_king_pieces(board, chess.BLACK)
    return float(-total)


def state_summary(board: chess.Board, target_color: bool) -> Dict[str, float]:
    """Return handcrafted current-state summary features."""
    return {
        "material_balance": material_balance(board, target_color),
        "legal_move_count": float(_mobility(board, target_color)),
        "king_safety": king_safety_proxy(board, target_color),
        "pawn_island_diff": float(pawn_islands(board, not target_color) - pawn_islands(board, target_color)),
        "doubled_pawn_diff": float(doubled_pawns(board, not target_color) - doubled_pawns(board, target_color)),
        "isolated_pawn_diff": float(isolated_pawns(board, not target_color) - isolated_pawns(board, target_color)),
        "passed_pawn_diff": float(passed_pawns(board, target_color) - passed_pawns(board, not target_color)),
        "mobility_diff": mobility_diff(board, target_color),
        "center_control_diff": center_control_diff(board, target_color),
        "open_file_diff": float(open_files(board, target_color) - open_files(board, not target_color)),
        "semi_open_file_diff": float(semi_open_files(board, target_color) - semi_open_files(board, not target_color)),
        "bishop_pair_diff": float(bishop_pair(board, target_color) - bishop_pair(board, not target_color)),
        "king_activity_diff": king_activity_diff(board, target_color),
        "threatened_piece_diff": threatened_piece_diff(board, target_color),
        "opp_king_pressure": count_piece_attacks_near_king(board, target_color),
        "simplification": simplification_score(board, target_color),
        "queen_diff": float(count_piece_type(board, target_color, chess.QUEEN) - count_piece_type(board, not target_color, chess.QUEEN)),
        "rook_diff": float(count_piece_type(board, target_color, chess.ROOK) - count_piece_type(board, not target_color, chess.ROOK)),
        "own_hanging_count": float(hanging_non_king_piece_count(board, target_color)),
        "opp_hanging_count": float(hanging_non_king_piece_count(board, not target_color)),
        "own_hanging_value": hanging_non_king_piece_value(board, target_color),
        "opp_hanging_value": hanging_non_king_piece_value(board, not target_color),
        "under_immediate_threat": float(is_under_immediate_threat(board, target_color)),
    }


def dense_state_vector(board: chess.Board, target_color: bool, phase_code: int, fullmove_number: int) -> List[float]:
    """Return the dense handcrafted feature vector for the current position."""
    summary = state_summary(board, target_color)
    return [
        summary["material_balance"],
        float(phase_code),
        summary["legal_move_count"],
        float(board.has_kingside_castling_rights(target_color)),
        float(board.has_queenside_castling_rights(target_color)),
        float(board.has_kingside_castling_rights(not target_color)),
        float(board.has_queenside_castling_rights(not target_color)),
        summary["king_safety"],
        summary["pawn_island_diff"],
        summary["doubled_pawn_diff"],
        summary["isolated_pawn_diff"],
        summary["passed_pawn_diff"],
        summary["mobility_diff"],
        summary["center_control_diff"],
        summary["open_file_diff"],
        summary["semi_open_file_diff"],
        summary["bishop_pair_diff"],
        summary["king_activity_diff"],
        summary["threatened_piece_diff"],
        summary["opp_king_pressure"],
        summary["simplification"],
        summary["queen_diff"],
        summary["rook_diff"],
        summary["own_hanging_count"],
        summary["opp_hanging_count"],
        summary["own_hanging_value"],
        summary["opp_hanging_value"],
        summary["under_immediate_threat"],
        float(fullmove_number),
    ]


def _captured_piece_type(board: chess.Board, move: chess.Move) -> Optional[int]:
    if board.is_en_passant(move):
        return chess.PAWN
    captured = board.piece_at(move.to_square)
    if captured is None:
        return None
    return captured.piece_type


def _is_exchange(board: chess.Board, move: chess.Move, moving_piece_type: int) -> int:
    captured_type = _captured_piece_type(board, move)
    if captured_type is None:
        return 0
    return int(PIECE_VALUES.get(captured_type, 0) >= PIECE_VALUES.get(moving_piece_type, 0))


def _is_pawn_break(board: chess.Board, move: chess.Move, moving_piece_type: int, mover_color: bool) -> int:
    if moving_piece_type != chess.PAWN:
        return 0
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    for df in (-1, 1):
        nf = to_file + df
        if not (0 <= nf < 8):
            continue
        for dr in (-1, 0, 1):
            nr = to_rank + dr
            if 0 <= nr < 8:
                sq = chess.square(nf, nr)
                piece = board.piece_at(sq)
                if piece is not None and piece.color != mover_color and piece.piece_type == chess.PAWN:
                    return 1
    return 0


def move_attacks_higher_value_piece(board_before: chess.Board, move: chess.Move, mover_color: bool) -> int:
    """Flag moves that newly attack an opponent piece worth at least a minor piece."""
    board_after = board_before.copy(stack=False)
    board_after.push(move)
    moved_piece = board_after.piece_at(move.to_square)
    if moved_piece is None:
        return 0
    for target_sq in board_after.attacks(move.to_square):
        target_piece = board_after.piece_at(target_sq)
        if target_piece is None or target_piece.color == mover_color or target_piece.piece_type == chess.KING:
            continue
        if PIECE_VALUES.get(target_piece.piece_type, 0) >= 3:
            return 1
    return 0


def build_history_entry(
    board_before: chess.Board,
    move: chess.Move,
    target_color: bool,
    target_is_white: bool,
    moved_piece_id: Optional[str] = None,
) -> Dict[str, object]:
    """Build one history step with richer move-event and state-delta features."""
    piece = board_before.piece_at(move.from_square)
    moving_piece_type = 0 if piece is None else int(piece.piece_type)
    mover_color = bool(board_before.turn)

    summary_before = state_summary(board_before, target_color)
    board_after = board_before.copy(stack=False)
    board_after.push(move)
    summary_after = state_summary(board_after, target_color)

    pressure_before = king_ring_pressure(board_before, mover_color, not mover_color)
    pressure_after = king_ring_pressure(board_after, mover_color, not mover_color)
    non_king_before = count_non_king_pieces(board_before, chess.WHITE) + count_non_king_pieces(board_before, chess.BLACK)
    non_king_after = count_non_king_pieces(board_after, chess.WHITE) + count_non_king_pieces(board_after, chess.BLACK)
    own_hanging_before = hanging_non_king_piece_value(board_before, target_color)
    own_hanging_after = hanging_non_king_piece_value(board_after, target_color)
    opp_hanging_before = hanging_non_king_piece_value(board_before, not target_color)
    opp_hanging_after = hanging_non_king_piece_value(board_after, not target_color)

    return {
        "event": {
            "mover_is_target": int(mover_color == target_color),
            "piece_type": moving_piece_type,
            "piece_id": moved_piece_id or "",
            "from_sq": normalize_square(move.from_square, target_is_white),
            "to_sq": normalize_square(move.to_square, target_is_white),
            "is_capture": int(board_before.is_capture(move)),
            "is_check": int(board_before.gives_check(move)),
            "is_castling": int(board_before.is_castling(move)),
            "is_promotion": int(move.promotion is not None),
            "is_exchange": _is_exchange(board_before, move, moving_piece_type),
            "is_pawn_break": _is_pawn_break(board_before, move, moving_piece_type, mover_color),
            "king_pressure_gain": int(pressure_after > pressure_before),
            "pawn_structure_change": int(
                summary_after["pawn_island_diff"] != summary_before["pawn_island_diff"]
                or summary_after["doubled_pawn_diff"] != summary_before["doubled_pawn_diff"]
                or summary_after["isolated_pawn_diff"] != summary_before["isolated_pawn_diff"]
                or summary_after["passed_pawn_diff"] != summary_before["passed_pawn_diff"]
            ),
            "is_simplification": int(non_king_after < non_king_before),
            "creates_opp_hanging": int(opp_hanging_after > opp_hanging_before),
            "creates_own_hanging": int(own_hanging_after > own_hanging_before),
            "attacks_high_value_piece": move_attacks_higher_value_piece(board_before, move, mover_color),
        },
        "delta": {
            "material_delta": summary_after["material_balance"] - summary_before["material_balance"],
            "king_safety_delta": summary_after["king_safety"] - summary_before["king_safety"],
            "pawn_structure_delta": (
                (summary_after["pawn_island_diff"] - summary_before["pawn_island_diff"])
                + (summary_after["doubled_pawn_diff"] - summary_before["doubled_pawn_diff"])
                + (summary_after["isolated_pawn_diff"] - summary_before["isolated_pawn_diff"])
                + (summary_after["passed_pawn_diff"] - summary_before["passed_pawn_diff"])
            ),
            "mobility_delta": summary_after["mobility_diff"] - summary_before["mobility_diff"],
            "center_control_delta": summary_after["center_control_diff"] - summary_before["center_control_diff"],
            "king_activity_delta": summary_after["king_activity_diff"] - summary_before["king_activity_diff"],
            "opp_king_pressure_delta": summary_after["opp_king_pressure"] - summary_before["opp_king_pressure"],
            "threat_delta": summary_after["threatened_piece_diff"] - summary_before["threatened_piece_diff"],
            "simplification_delta": summary_after["simplification"] - summary_before["simplification"],
            "own_hanging_value_delta": own_hanging_after - own_hanging_before,
            "opp_hanging_value_delta": opp_hanging_after - opp_hanging_before,
        },
    }

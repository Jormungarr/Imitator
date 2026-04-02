#!/usr/bin/env python3
"""Local desktop chess app to play against selectable player-style checkpoints."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import chess
import torch
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from history_policy_lib import FactorizedPolicyModel, PolicySample, collate_batch, load_checkpoint, masked_logits, resolve_from_square
from pipeline_config import PROJECT_ROOT
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
from script1_parse_pgn_to_positions import promotion_index
from script2_encode_policy_samples import encode_row


PIECE_TO_GLYPH = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}

LIGHT_SQ = "#efe1ca"
DARK_SQ = "#6f4e37"
SELECT_SQ = "#f2b84b"
HINT_SQ = "#8ab17d"


@dataclass
class LoadedModel:
    slug: str
    name: str
    path: Path
    model: FactorizedPolicyModel
    config: Dict[str, Any]
    history_plies: int


class ModelRegistry:
    def __init__(self, models_root: Path, device: torch.device) -> None:
        self.models_root = models_root
        self.device = device
        self.entries: Dict[str, Tuple[str, Path]] = {}
        self.loaded: Dict[str, LoadedModel] = {}

    @staticmethod
    def _pretty_name(slug: str) -> str:
        return slug.replace("_", " ").replace("-", " ").strip()

    def discover(self) -> None:
        self.entries.clear()
        if not self.models_root.exists():
            return
        for child in sorted(self.models_root.iterdir()):
            if not child.is_dir():
                continue
            ckpt = child / "history_policy.pt"
            if ckpt.exists():
                self.entries[child.name] = (self._pretty_name(child.name), ckpt)

    def register(self, slug: str, ckpt_path: Path) -> None:
        if ckpt_path.exists():
            self.entries[slug] = (self._pretty_name(slug), ckpt_path)

    def list_players(self) -> List[Dict[str, str]]:
        out = [{"id": slug, "name": name} for slug, (name, _) in self.entries.items()]
        out.sort(key=lambda x: x["name"].lower())
        return out

    def get(self, slug: str) -> LoadedModel:
        if slug in self.loaded:
            return self.loaded[slug]
        if slug not in self.entries:
            raise KeyError(f"Unknown player: {slug}")

        name, ckpt_path = self.entries[slug]
        ckpt = load_checkpoint(ckpt_path, device=self.device)
        config = dict(ckpt.get("config", {}))
        if not config:
            raise ValueError(f"Checkpoint missing config: {ckpt_path}")

        model = FactorizedPolicyModel(config).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        loaded = LoadedModel(
            slug=slug,
            name=name,
            path=ckpt_path,
            model=model,
            config=config,
            history_plies=int(config.get("history_plies", 8)),
        )
        self.loaded[slug] = loaded
        return loaded


class ChessStyleSession:
    def __init__(self, registry: ModelRegistry, device: torch.device) -> None:
        self.registry = registry
        self.device = device
        self.board = chess.Board()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=256)
        self.player_slug: Optional[str] = None
        self.player_model: Optional[LoadedModel] = None
        self.human_color = chess.WHITE
        self.model_color = chess.BLACK
        self.last_model_move = ""
        self.piece_id_by_square, self.promotion_counters = initialize_piece_identity_tracker(self.board)

    @staticmethod
    def _sample_from_encoded(enc: Dict[str, Any]) -> PolicySample:
        return PolicySample(
            game_id=str(enc.get("game_id", "")),
            player_id=str(enc.get("player_id", "unknown")),
            active_feature_indices=[int(x) for x in enc["active_feature_indices"]],
            dense_state=[float(x) for x in enc["dense_state"]],
            history_event=[[float(v) for v in x] for x in enc["history_event"]],
            history_delta=[[float(v) for v in x] for x in enc["history_delta"]],
            history_mask=[int(x) for x in enc.get("history_mask", [1] * len(enc["history_event"]))],
            context=[float(x) for x in enc["context"]],
            piece_slot_to_square=[int(x) for x in enc.get("piece_slot_to_square", [-1] * 16)],
            legal_piece_slot_mask=[int(x) for x in enc.get("legal_piece_slot_mask", [0] * 16)],
            legal_from_mask=[int(x) for x in enc["legal_from_mask"]],
            legal_to_by_from={str(k): [int(v) for v in vals] for k, vals in enc["legal_to_by_from"].items()},
            target_piece_slot=int(enc.get("target_piece_slot", 0)),
            target_from_sq=int(enc["target_from_sq"]),
            target_to_sq=int(enc["target_to_sq"]),
            target_promotion=int(enc.get("target_promotion", 0)),
        )

    @staticmethod
    def _to_mask_for_from(sample: PolicySample, from_sq: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(64, dtype=torch.float32, device=device)
        for to_sq in sample.legal_to_by_from.get(str(int(from_sq)), []):
            mask[int(to_sq)] = 1.0
        if float(mask.sum().item()) <= 0:
            mask.fill_(1.0)
        return mask

    def _record_move(self, move: chess.Move) -> None:
        target_is_white = bool(self.model_color == chess.WHITE)
        moved_piece_id = current_piece_identity(self.piece_id_by_square, move.from_square)
        entry = build_history_entry(
            board_before=self.board,
            move=move,
            target_color=self.model_color,
            target_is_white=target_is_white,
            moved_piece_id=moved_piece_id,
        )
        self.history.append(entry)
        apply_piece_identity_move(self.board, self.piece_id_by_square, move, self.promotion_counters)
        self.board.push(move)

    def _encode_current(self) -> PolicySample:
        assert self.player_model is not None
        target_is_white = bool(self.model_color == chess.WHITE)

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        dummy = None
        moved_piece_id = ""
        moved_piece_slot = ""
        for mv in legal_moves:
            pid = current_piece_identity(self.piece_id_by_square, mv.from_square)
            slot = canonical_piece_slot(pid)
            if slot is not None:
                dummy = mv
                moved_piece_id = pid
                moved_piece_slot = slot
                break
        if dummy is None:
            raise ValueError("No legal move from canonical original piece slots")

        row = {
            "game_id": "live_game",
            "player_id": self.player_model.slug.lower(),
            "fen_before": self.board.fen(),
            "phase_code": phase_code_from_fullmove(self.board.fullmove_number),
            "target_is_white": int(target_is_white),
            "history": list(self.history)[-self.player_model.history_plies :],
            "moved_piece_id": moved_piece_id,
            "moved_piece_slot": moved_piece_slot,
            "piece_slot_to_square": current_original_piece_slot_square_map(self.piece_id_by_square, self.model_color, target_is_white),
            "target_from_sq": normalize_square(dummy.from_square, target_is_white),
            "target_to_sq": normalize_square(dummy.to_square, target_is_white),
            "target_promotion": promotion_index(dummy),
        }
        enc = encode_row(row, history_plies=self.player_model.history_plies)
        return self._sample_from_encoded(enc)

    def _choose_model_move(self) -> chess.Move:
        assert self.player_model is not None
        sample = self._encode_current()
        batch = collate_batch([sample], device=self.device)

        legal_moves = list(self.board.legal_moves)
        target_is_white = bool(self.model_color == chess.WHITE)

        with torch.no_grad():
            fused = self.player_model.model.fused_repr(batch)
            piece_logits = masked_logits(self.player_model.model.piece_logits(fused), batch["legal_piece_slot_mask"])[0]
            piece_probs = torch.softmax(piece_logits, dim=0)

            to_cache: Dict[int, torch.Tensor] = {}
            promo_cache: Dict[Tuple[int, int], torch.Tensor] = {}

            best_move = legal_moves[0]
            best_score = -1.0

            for mv in legal_moves:
                rel_from = int(normalize_square(mv.from_square, target_is_white))
                rel_to = int(normalize_square(mv.to_square, target_is_white))
                rel_promo = int(promotion_index(mv))
                piece_slot = int(batch["target_piece"][0].item()) if rel_from == int(batch["target_from"][0].item()) else None
                for idx, sq in enumerate(sample.piece_slot_to_square):
                    if sq == rel_from:
                        piece_slot = idx
                        break
                if piece_slot is None:
                    continue

                if rel_from not in to_cache:
                    to_mask = self._to_mask_for_from(sample, rel_from, self.device)
                    to_logits = masked_logits(
                        self.player_model.model.to_logits(fused, torch.tensor([rel_from], device=self.device))[0],
                        to_mask,
                    )
                    to_cache[rel_from] = torch.softmax(to_logits, dim=0)

                key = (rel_from, rel_to)
                if key not in promo_cache:
                    promo_logits = self.player_model.model.promo_logits(
                        fused,
                        torch.tensor([rel_from], device=self.device),
                        torch.tensor([rel_to], device=self.device),
                    )[0]
                    promo_cache[key] = torch.softmax(promo_logits, dim=0)

                p = float(piece_probs[piece_slot].item())
                p *= float(to_cache[rel_from][rel_to].item())
                p *= float(promo_cache[key][rel_promo].item())
                if p > best_score:
                    best_score = p
                    best_move = mv

        return best_move

    def new_game(self, player_slug: str, human_color: str = "white") -> None:
        human_color = human_color.strip().lower()
        if human_color not in {"white", "black"}:
            raise ValueError("human_color must be 'white' or 'black'")

        self.player_model = self.registry.get(player_slug)
        self.player_slug = player_slug
        self.board = chess.Board()
        self.history.clear()
        self.last_model_move = ""
        self.piece_id_by_square, self.promotion_counters = initialize_piece_identity_tracker(self.board)

        self.human_color = chess.WHITE if human_color == "white" else chess.BLACK
        self.model_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE

        if self.board.turn == self.model_color and not self.board.is_game_over():
            mv = self._choose_model_move()
            self._record_move(mv)
            self.last_model_move = mv.uci()

    def play_human_move(self, uci: str) -> None:
        if self.player_model is None:
            raise ValueError("Start a game first.")
        if self.board.is_game_over():
            raise ValueError("Game already finished.")
        if self.board.turn != self.human_color:
            raise ValueError("It is not your turn.")

        try:
            mv = chess.Move.from_uci(uci.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move: {uci}") from exc

        if mv not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {uci}")

        self._record_move(mv)
        self.last_model_move = ""
        self.piece_id_by_square, self.promotion_counters = initialize_piece_identity_tracker(self.board)

        if not self.board.is_game_over() and self.board.turn == self.model_color:
            model_mv = self._choose_model_move()
            self._record_move(model_mv)
            self.last_model_move = model_mv.uci()

    def result_text(self) -> str:
        if not self.board.is_game_over():
            return ""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return self.board.result(claim_draw=True)
        if outcome.winner is None:
            winner = "draw"
        else:
            winner = "white" if outcome.winner == chess.WHITE else "black"
        return f"{self.board.result(claim_draw=True)} ({winner}, {outcome.termination.name.lower()})"


class ChessStyleApp:
    def __init__(self, root: tk.Tk, session: ChessStyleSession, players: List[Dict[str, str]]) -> None:
        self.root = root
        self.session = session
        self.players = players
        self.selected_from: Optional[chess.Square] = None

        self.player_name_to_slug = {p["name"]: p["id"] for p in players}
        self.orientation = chess.WHITE

        self.root.title("Chess Style Imitator - Local")
        self.root.geometry("980x760")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="Player model:").pack(side=tk.LEFT)
        self.player_var = tk.StringVar(value=players[0]["name"])
        self.player_combo = ttk.Combobox(toolbar, state="readonly", textvariable=self.player_var, width=28)
        self.player_combo["values"] = [p["name"] for p in players]
        self.player_combo.pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(toolbar, text="Play as:").pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="white")
        ttk.Radiobutton(toolbar, text="White", variable=self.color_var, value="white").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Radiobutton(toolbar, text="Black", variable=self.color_var, value="black").pack(side=tk.LEFT, padx=(2, 12))

        ttk.Button(toolbar, text="New Game", command=self.new_game).pack(side=tk.LEFT)

        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))

        board_frame = tk.Frame(left, bg="#8a6f54", bd=2)
        board_frame.pack(anchor=tk.NW)

        self.board_buttons: List[tk.Button] = []
        for rank in range(8):
            for file_idx in range(8):
                is_light = (rank + file_idx) % 2 == 0
                base = LIGHT_SQ if is_light else DARK_SQ
                fg = "#1e1b16" if is_light else "#ffffff"
                btn = tk.Button(
                    board_frame,
                    text="",
                    width=4,
                    height=2,
                    font=("Segoe UI Symbol", 26),
                    bg=base,
                    fg=fg,
                    activebackground=base,
                    relief=tk.FLAT,
                    command=lambda idx=len(self.board_buttons): self.on_grid_click(idx),
                )
                btn.grid(row=rank, column=file_idx, sticky="nsew")
                board_frame.grid_columnconfigure(file_idx, weight=1)
                board_frame.grid_rowconfigure(rank, weight=1)
                self.board_buttons.append(btn)

        ttk.Label(right, text="Status").pack(anchor=tk.W)
        self.status_text = tk.Text(right, width=36, height=10, wrap=tk.WORD)
        self.status_text.pack(anchor=tk.W, pady=(4, 10))

        ttk.Label(right, text="Moves (UCI)").pack(anchor=tk.W)
        self.moves_text = tk.Text(right, width=36, height=20, wrap=tk.WORD)
        self.moves_text.pack(anchor=tk.W, pady=(4, 0))

        self.new_game()

    def board_coords_in_orientation(self) -> List[chess.Square]:
        squares: List[chess.Square] = []
        rank_range = range(7, -1, -1) if self.orientation == chess.WHITE else range(0, 8)
        file_range = range(0, 8) if self.orientation == chess.WHITE else range(7, -1, -1)
        for rank in rank_range:
            for file_idx in file_range:
                squares.append(chess.square(file_idx, rank))
        return squares

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        widget.configure(state=tk.DISABLED)

    def render(self) -> None:
        board = self.session.board
        view_squares = self.board_coords_in_orientation()

        for idx, sq in enumerate(view_squares):
            piece = board.piece_at(sq)
            glyph = PIECE_TO_GLYPH.get(piece.symbol(), "") if piece else ""

            rank = idx // 8
            file_idx = idx % 8
            is_light = (rank + file_idx) % 2 == 0
            base = LIGHT_SQ if is_light else DARK_SQ
            fg = "#1e1b16" if is_light else "#ffffff"

            btn = self.board_buttons[idx]
            btn.configure(text=glyph, bg=base, fg=fg, activebackground=base)

        if self.selected_from is not None:
            for idx, sq in enumerate(view_squares):
                if sq == self.selected_from:
                    self.board_buttons[idx].configure(bg=SELECT_SQ, activebackground=SELECT_SQ)
            for mv in self.session.board.legal_moves:
                if mv.from_square == self.selected_from:
                    for idx, sq in enumerate(view_squares):
                        if sq == mv.to_square:
                            self.board_buttons[idx].configure(bg=HINT_SQ, activebackground=HINT_SQ)

        turn = "white" if board.turn == chess.WHITE else "black"
        human = "white" if self.session.human_color == chess.WHITE else "black"
        model = "white" if self.session.model_color == chess.WHITE else "black"
        lines = [
            f"Model: {self.session.player_slug or ''}",
            f"You: {human} | Model: {model}",
            f"Turn: {turn}",
            f"Game over: {'yes' if board.is_game_over() else 'no'}",
        ]
        if board.is_game_over():
            lines.append(f"Result: {self.session.result_text()}")
        if self.session.last_model_move:
            lines.append(f"Model played: {self.session.last_model_move}")

        self._set_text(self.status_text, "\n".join(lines))
        self._set_text(self.moves_text, " ".join(mv.uci() for mv in board.move_stack))

    def _is_human_turn(self) -> bool:
        return (
            not self.session.board.is_game_over()
            and self.session.board.turn == self.session.human_color
        )

    def _try_move(self, from_sq: chess.Square, to_sq: chess.Square) -> bool:
        moves = [mv for mv in self.session.board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
        if not moves:
            return False

        if len(moves) == 1:
            uci = moves[0].uci()
        else:
            promo = simpledialog.askstring("Promotion", "Promotion piece? q/r/b/n", parent=self.root)
            promo = (promo or "q").strip().lower()
            if promo not in {"q", "r", "b", "n"}:
                promo = "q"
            wanted = chess.Move.from_uci(chess.square_name(from_sq) + chess.square_name(to_sq) + promo)
            uci = wanted.uci() if wanted in moves else moves[0].uci()

        self.session.play_human_move(uci)
        return True

    def on_grid_click(self, idx: int) -> None:
        squares = self.board_coords_in_orientation()
        if idx < 0 or idx >= len(squares):
            return
        self.on_square_click(squares[idx])

    def on_square_click(self, sq: chess.Square) -> None:
        try:
            if not self._is_human_turn():
                return

            piece = self.session.board.piece_at(sq)
            if self.selected_from is None:
                if piece is None:
                    return
                if piece.color != self.session.human_color:
                    return
                has_moves = any(mv.from_square == sq for mv in self.session.board.legal_moves)
                if not has_moves:
                    return
                self.selected_from = sq
                self.render()
                return

            if sq == self.selected_from:
                self.selected_from = None
                self.render()
                return

            moved = self._try_move(self.selected_from, sq)
            self.selected_from = None

            if not moved:
                if piece is not None and piece.color == self.session.human_color:
                    has_moves = any(mv.from_square == sq for mv in self.session.board.legal_moves)
                    self.selected_from = sq if has_moves else None
                self.render()
                return

            self.render()
            if self.session.board.is_game_over():
                messagebox.showinfo("Game over", self.session.result_text(), parent=self.root)
        except Exception as exc:
            messagebox.showerror("Move error", str(exc), parent=self.root)
            self.render()

    def new_game(self) -> None:
        try:
            name = self.player_var.get()
            if name not in self.player_name_to_slug:
                raise ValueError("Select a player model")
            slug = self.player_name_to_slug[name]
            color = self.color_var.get().strip().lower()
            self.session.new_game(slug, color)
            self.orientation = self.session.human_color
            self.selected_from = None
            self.render()
        except Exception as exc:
            messagebox.showerror("New game error", str(exc), parent=self.root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local chess app for style checkpoints")
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--register",
        action="append",
        default=[],
        help="Extra mapping as slug=checkpoint_path (repeatable)",
    )
    return parser.parse_args()


def parse_register_item(item: str) -> Tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Invalid --register '{item}', expected slug=path")
    slug, path_str = item.split("=", 1)
    slug = slug.strip()
    if not slug:
        raise ValueError(f"Invalid --register '{item}', empty slug")
    return slug, Path(path_str).expanduser().resolve()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    registry = ModelRegistry(args.models_root.resolve(), device)
    registry.discover()

    for item in args.register:
        slug, ckpt = parse_register_item(item)
        registry.register(slug, ckpt)

    players = registry.list_players()
    if not players:
        raise FileNotFoundError(
            f"No checkpoints found. Expected models like: {args.models_root}/<player>/history_policy.pt"
        )

    session = ChessStyleSession(registry, device)

    root = tk.Tk()
    ChessStyleApp(root, session, players)
    root.mainloop()


if __name__ == "__main__":
    main()




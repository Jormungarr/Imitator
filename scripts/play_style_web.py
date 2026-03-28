#!/usr/bin/env python3
"""Serve a local HTML chess UI that plays with selected player-style checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import chess
import torch

from history_policy_lib import FactorizedPolicyModel, PolicySample, collate_batch, load_checkpoint, masked_logits
from pipeline_config import PROJECT_ROOT
from script1_parse_pgn_to_positions import (
    build_history_entry,
    normalize_square,
    phase_code_from_fullmove,
    promotion_index,
)
from script2_encode_policy_samples import encode_row


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chess Style Imitator</title>
  <style>
    :root {
      --bg: #f4efe7;
      --paper: #fffaf2;
      --ink: #1e1b16;
      --accent: #9f3a2f;
      --dark: #6f4e37;
      --light: #efe1ca;
      --sel: #f2b84b;
      --move: #8ab17d;
    }
    body {
      margin: 0;
      font-family: "Georgia", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at 15% 15%, #fff8ea 0%, transparent 35%),
        radial-gradient(circle at 85% 30%, #f7e8d2 0%, transparent 45%),
        var(--bg);
      color: var(--ink);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
      box-sizing: border-box;
    }
    .app {
      width: min(980px, 100%);
      background: var(--paper);
      border: 2px solid #d8c8ac;
      border-radius: 14px;
      box-shadow: 0 14px 36px rgba(0, 0, 0, 0.12);
      padding: 18px;
      box-sizing: border-box;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 14px;
    }
    select, button {
      font: inherit;
      border-radius: 8px;
      border: 1px solid #b9a88a;
      padding: 7px 10px;
      background: #fff;
      color: var(--ink);
    }
    button {
      background: var(--accent);
      color: #fff;
      border-color: #7c271d;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.05); }
    .layout {
      display: grid;
      grid-template-columns: 1fr 300px;
      gap: 16px;
    }
    .board-wrap {
      display: grid;
      place-items: center;
    }
    .board {
      width: min(82vw, 620px);
      aspect-ratio: 1;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      border: 2px solid #8a6f54;
    }
    .sq {
      border: none;
      font-size: clamp(22px, 3.8vw, 42px);
      line-height: 1;
      cursor: pointer;
      display: grid;
      place-items: center;
      user-select: none;
      transition: transform 100ms ease;
    }
    .sq:active { transform: scale(0.97); }
    .light { background: var(--light); }
    .dark { background: var(--dark); color: #fff; }
    .sel { outline: 3px solid var(--sel); z-index: 2; }
    .hint { box-shadow: inset 0 0 0 4px var(--move); }
    .side {
      border: 1px solid #dbcbb0;
      border-radius: 10px;
      background: #fff;
      padding: 10px;
      box-sizing: border-box;
      height: fit-content;
    }
    .label { font-size: 13px; opacity: 0.8; margin-top: 10px; margin-bottom: 4px; }
    .mono {
      font-family: "Courier New", monospace;
      font-size: 13px;
      background: #fbf7f0;
      border: 1px solid #e4d8c5;
      border-radius: 6px;
      padding: 7px;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      .board { width: min(92vw, 520px); }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <label>Player model:</label>
      <select id="playerSelect"></select>
      <label>Play as:</label>
      <select id="colorSelect">
        <option value="white">White</option>
        <option value="black">Black</option>
      </select>
      <button id="newGameBtn">New Game</button>
    </div>
    <div class="layout">
      <div class="board-wrap">
        <div id="board" class="board"></div>
      </div>
      <div class="side">
        <div class="label">Status</div>
        <div id="status" class="mono"></div>
        <div class="label">Moves (UCI)</div>
        <div id="moves" class="mono"></div>
      </div>
    </div>
  </div>

  <script>
    const pieceToGlyph = {
      'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
      'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    };
    const files = ['a','b','c','d','e','f','g','h'];

    let state = null;
    let selectedFrom = null;
    let orientation = 'white';

    function squareName(fileIdx, rankIdx) {
      return files[fileIdx] + String(8 - rankIdx);
    }

    function parseFenBoard(fen) {
      const part = fen.split(' ')[0];
      const rows = part.split('/');
      const board = [];
      for (const row of rows) {
        const line = [];
        for (const ch of row) {
          if (/[1-8]/.test(ch)) {
            const n = parseInt(ch, 10);
            for (let i = 0; i < n; i++) line.push('.');
          } else {
            line.push(ch);
          }
        }
        board.push(line);
      }
      return board;
    }

    function boardCoords() {
      const coords = [];
      const ranks = orientation === 'white' ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
      const filesIdx = orientation === 'white' ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
      for (const r of ranks) {
        for (const f of filesIdx) {
          coords.push([r, f]);
        }
      }
      return coords;
    }

    function legalMovesFrom(fromSq) {
      if (!state || !state.legal_moves) return [];
      return state.legal_moves.filter(m => m.startsWith(fromSq));
    }

    async function api(path, method='GET', body=null) {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body !== null) opts.body = JSON.stringify(body);
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      return data;
    }

    function clearSelection() {
      selectedFrom = null;
    }

    function boardPieceAt(square) {
      if (!state) return '.';
      const b = parseFenBoard(state.fen);
      const file = square.charCodeAt(0) - 97;
      const rank = 8 - parseInt(square[1], 10);
      return b[rank][file];
    }

    function renderBoard() {
      const boardEl = document.getElementById('board');
      boardEl.innerHTML = '';
      if (!state) return;

      const b = parseFenBoard(state.fen);
      const hintTargets = selectedFrom ? legalMovesFrom(selectedFrom).map(m => m.slice(2, 4)) : [];

      for (const [rankIdx, fileIdx] of boardCoords()) {
        const sq = squareName(fileIdx, rankIdx);
        const piece = b[rankIdx][fileIdx];
        const isLight = (rankIdx + fileIdx) % 2 === 0;
        const btn = document.createElement('button');
        btn.className = `sq ${isLight ? 'light' : 'dark'}`;
        if (selectedFrom === sq) btn.classList.add('sel');
        if (hintTargets.includes(sq)) btn.classList.add('hint');
        btn.textContent = pieceToGlyph[piece] || '';
        btn.onclick = () => onSquareClick(sq);
        boardEl.appendChild(btn);
      }
    }

    function renderStatus() {
      if (!state) return;
      const lines = [];
      lines.push(`Model: ${state.selected_player}`);
      lines.push(`You: ${state.human_color} | Model: ${state.model_color}`);
      lines.push(`Turn: ${state.turn}`);
      lines.push(`Game over: ${state.game_over ? 'yes' : 'no'}`);
      if (state.result) lines.push(`Result: ${state.result}`);
      if (state.last_model_move) lines.push(`Model played: ${state.last_model_move}`);
      if (state.error) lines.push(`Error: ${state.error}`);
      if (state.message) lines.push(state.message);
      document.getElementById('status').textContent = lines.join('\n');
      document.getElementById('moves').textContent = (state.moves_uci || []).join(' ');
    }

    async function refreshState() {
      state = await api('/api/state');
      renderBoard();
      renderStatus();
    }

    function normalizePromotion(baseMove, candidates) {
      if (candidates.length === 1) return candidates[0];
      const hasPromo = candidates.some(m => m.length === 5);
      if (!hasPromo) return baseMove;
      const pick = (prompt('Promotion piece? q/r/b/n', 'q') || 'q').toLowerCase().trim();
      const wanted = ['q', 'r', 'b', 'n'].includes(pick) ? pick : 'q';
      const candidate = candidates.find(m => m === baseMove + wanted);
      return candidate || (baseMove + 'q');
    }

    async function onSquareClick(square) {
      if (!state || state.game_over || !state.is_human_turn) return;

      if (!selectedFrom) {
        const piece = boardPieceAt(square);
        if (piece === '.') return;
        const isWhitePiece = piece === piece.toUpperCase();
        const humanIsWhite = state.human_color === 'white';
        if (isWhitePiece !== humanIsWhite) return;
        const moves = legalMovesFrom(square);
        if (!moves.length) return;
        selectedFrom = square;
        renderBoard();
        return;
      }

      if (selectedFrom === square) {
        clearSelection();
        renderBoard();
        return;
      }

      const candidates = legalMovesFrom(selectedFrom).filter(m => m.slice(2,4) === square);
      if (!candidates.length) {
        selectedFrom = square;
        const moves = legalMovesFrom(square);
        if (!moves.length) selectedFrom = null;
        renderBoard();
        return;
      }

      const baseMove = selectedFrom + square;
      const uci = normalizePromotion(baseMove, candidates);
      clearSelection();

      try {
        state = await api('/api/human_move', 'POST', { uci });
      } catch (err) {
        state = { ...(state || {}), error: String(err) };
      }
      renderBoard();
      renderStatus();
    }

    async function loadPlayers() {
      const data = await api('/api/players');
      const sel = document.getElementById('playerSelect');
      sel.innerHTML = '';
      for (const p of data.players) {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name;
        sel.appendChild(opt);
      }
      if (data.players.length === 0) {
        throw new Error('No checkpoints found under models/*/history_policy.pt');
      }
    }

    async function newGame() {
      const player = document.getElementById('playerSelect').value;
      const humanColor = document.getElementById('colorSelect').value;
      orientation = humanColor;
      state = await api('/api/new_game', 'POST', {
        player,
        human_color: humanColor
      });
      clearSelection();
      renderBoard();
      renderStatus();
    }

    document.getElementById('newGameBtn').onclick = () => {
      newGame().catch(err => {
        state = { error: String(err), fen: '8/8/8/8/8/8/8/8 w - - 0 1', legal_moves: [] };
        renderBoard();
        renderStatus();
      });
    };

    (async () => {
      try {
        await loadPlayers();
        await newGame();
      } catch (err) {
        state = { error: String(err), fen: '8/8/8/8/8/8/8/8 w - - 0 1', legal_moves: [] };
        renderBoard();
        renderStatus();
      }
    })();
  </script>
</body>
</html>
"""


@dataclass
class LoadedModel:
    """Runtime model bundle for one player checkpoint."""

    slug: str
    name: str
    path: Path
    model: FactorizedPolicyModel
    config: Dict[str, Any]
    history_plies: int


class ModelRegistry:
    """Discover checkpoints and lazily load style models by player slug."""

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
                slug = child.name
                self.entries[slug] = (self._pretty_name(slug), ckpt)

    def register(self, slug: str, ckpt_path: Path) -> None:
        if ckpt_path.exists():
            self.entries[slug] = (self._pretty_name(slug), ckpt_path)

    def list_players(self) -> List[Dict[str, str]]:
        out = [{"id": s, "name": n} for s, (n, _) in self.entries.items()]
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
        bundle = LoadedModel(
            slug=slug,
            name=name,
            path=ckpt_path,
            model=model,
            config=config,
            history_plies=int(config.get("history_plies", 8)),
        )
        self.loaded[slug] = bundle
        return bundle


class ChessStyleSession:
    """Single in-memory game session against one selected style model."""

    def __init__(self, registry: ModelRegistry, device: torch.device) -> None:
        self.registry = registry
        self.device = device
        self.board = chess.Board()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=256)
        self.player_slug: Optional[str] = None
        self.player_model: Optional[LoadedModel] = None
        self.human_color = chess.WHITE
        self.model_color = chess.BLACK
        self.last_model_move: str = ""

    @staticmethod
    def _sample_from_encoded(enc: Dict[str, Any]) -> PolicySample:
        return PolicySample(
            game_id=str(enc.get("game_id", "")),
            player_id=str(enc.get("player_id", "unknown")),
            active_feature_indices=[int(x) for x in enc["active_feature_indices"]],
            dense_state=[float(x) for x in enc["dense_state"]],
            history_event=[[float(v) for v in x] for x in enc["history_event"]],
            history_delta=[[float(v) for v in x] for x in enc["history_delta"]],
            context=[float(x) for x in enc["context"]],
            legal_from_mask=[int(x) for x in enc["legal_from_mask"]],
            legal_to_by_from={str(k): [int(v) for v in vals] for k, vals in enc["legal_to_by_from"].items()},
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
        entry = build_history_entry(
            board_before=self.board,
            move=move,
            target_color=self.model_color,
            target_is_white=target_is_white,
        )
        self.history.append(entry)
        self.board.push(move)

    def _encode_current(self) -> PolicySample:
        assert self.player_model is not None
        target_is_white = bool(self.model_color == chess.WHITE)

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        dummy = legal_moves[0]
        row = {
            "game_id": "live_game",
            "player_id": self.player_model.slug.lower(),
            "fen_before": self.board.fen(),
            "phase_code": phase_code_from_fullmove(self.board.fullmove_number),
            "target_is_white": int(target_is_white),
            "clock_after_move_sec": 0.0,
            "history": list(self.history)[-self.player_model.history_plies :],
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
            from_logits = masked_logits(self.player_model.model.from_logits(fused), batch["legal_from_mask"])[0]
            from_probs = torch.softmax(from_logits, dim=0)

            to_cache: Dict[int, torch.Tensor] = {}
            promo_cache: Dict[Tuple[int, int], torch.Tensor] = {}

            best_move = legal_moves[0]
            best_score = -1.0

            for mv in legal_moves:
                rel_from = int(normalize_square(mv.from_square, target_is_white))
                rel_to = int(normalize_square(mv.to_square, target_is_white))
                rel_promo = int(promotion_index(mv))

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

                p = float(from_probs[rel_from].item())
                p *= float(to_cache[rel_from][rel_to].item())
                p *= float(promo_cache[key][rel_promo].item())
                if p > best_score:
                    best_score = p
                    best_move = mv

        return best_move

    def new_game(self, player_slug: str, human_color: str = "white") -> None:
        human_color_norm = human_color.strip().lower()
        if human_color_norm not in {"white", "black"}:
            raise ValueError("human_color must be 'white' or 'black'")

        self.player_model = self.registry.get(player_slug)
        self.player_slug = player_slug
        self.board = chess.Board()
        self.history.clear()
        self.last_model_move = ""

        self.human_color = chess.WHITE if human_color_norm == "white" else chess.BLACK
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
            raise ValueError("It is not the human turn.")

        try:
            mv = chess.Move.from_uci(uci.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move: {uci}") from exc

        if mv not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {uci}")

        self._record_move(mv)
        self.last_model_move = ""

        if not self.board.is_game_over() and self.board.turn == self.model_color:
            model_mv = self._choose_model_move()
            self._record_move(model_mv)
            self.last_model_move = model_mv.uci()

    def _result_text(self) -> str:
        if not self.board.is_game_over():
            return ""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return self.board.result(claim_draw=True)
        winner = outcome.winner
        if winner is None:
            who = "draw"
        elif winner == chess.WHITE:
            who = "white"
        else:
            who = "black"
        return f"{self.board.result(claim_draw=True)} ({who}, {outcome.termination.name.lower()})"

    def state_payload(self) -> Dict[str, Any]:
        turn = "white" if self.board.turn == chess.WHITE else "black"
        is_human_turn = bool(self.board.turn == self.human_color and not self.board.is_game_over())
        legal_moves = [mv.uci() for mv in self.board.legal_moves] if is_human_turn else []

        return {
            "fen": self.board.fen(),
            "turn": turn,
            "selected_player": self.player_slug or "",
            "human_color": "white" if self.human_color == chess.WHITE else "black",
            "model_color": "white" if self.model_color == chess.WHITE else "black",
            "is_human_turn": is_human_turn,
            "game_over": bool(self.board.is_game_over()),
            "result": self._result_text(),
            "last_model_move": self.last_model_move,
            "legal_moves": legal_moves,
            "moves_uci": [mv.uci() for mv in self.board.move_stack],
        }


class Handler(BaseHTTPRequestHandler):
    """HTTP API + HTML handler."""

    session: ChessStyleSession
    registry: ModelRegistry

    def _json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, payload: str) -> None:
        body = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data) if data.strip() else {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._html(HTML_PAGE)
            return
        if self.path == "/api/players":
            self._json({"players": self.registry.list_players()})
            return
        if self.path == "/api/state":
            self._json(self.session.state_payload())
            return
        self._json({"error": f"Not found: {self.path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._read_json()
        except Exception:
            self._json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if self.path == "/api/new_game":
                player = str(body.get("player", "")).strip()
                human_color = str(body.get("human_color", "white")).strip().lower()
                if not player:
                    raise ValueError("Missing 'player'.")
                self.session.new_game(player_slug=player, human_color=human_color)
                self._json(self.session.state_payload())
                return

            if self.path == "/api/human_move":
                uci = str(body.get("uci", "")).strip()
                if not uci:
                    raise ValueError("Missing 'uci'.")
                self.session.play_human_move(uci=uci)
                self._json(self.session.state_payload())
                return

            self._json({"error": f"Not found: {self.path}"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": str(exc), **self.session.state_payload()}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep default stdout cleaner for this local UI server.
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local chess UI to play against player-style checkpoints."
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Root containing per-player checkpoint directories",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--register",
        action="append",
        default=[],
        help="Extra player mapping as slug=checkpoint_path (repeatable)",
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
    registry = ModelRegistry(models_root=args.models_root.resolve(), device=device)
    registry.discover()
    for item in args.register:
        slug, path = parse_register_item(item)
        registry.register(slug, path)

    players = registry.list_players()
    if not players:
        raise FileNotFoundError(
            f"No checkpoints found. Expected files like: {args.models_root}/<player>/history_policy.pt"
        )

    session = ChessStyleSession(registry=registry, device=device)
    first_player = players[0]["id"]
    session.new_game(player_slug=first_player, human_color="white")

    Handler.session = session
    Handler.registry = registry
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    url = f"http://{args.host}:{args.port}"

    print(f"Serving chess style UI at {url}")
    print("Available players:")
    for p in players:
        print(f"  - {p['name']} [{p['id']}]")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

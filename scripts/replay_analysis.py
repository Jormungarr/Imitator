#!/usr/bin/env python3
"""Replay one target game and analyze move-by-move policy predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.svg
import torch

from history_policy_lib import FactorizedPolicyModel, PolicySample, collate_batch, load_checkpoint, masked_logits, resolve_from_square
from pipeline_config import DATASET_TAG, HISTORY_PLIES, TARGET_NAME_ALIASES, TARGET_USERNAME, models_dir, outputs_dir, raw_pgn
from script1_parse_pgn_to_positions import iter_target_player_positions
from script2_encode_policy_samples import encode_row


CONFIG: Dict[str, Any] = {
    "dataset_tag": DATASET_TAG,
    "model_path": models_dir(DATASET_TAG) / "history_policy.pt",
    "pgn_path": raw_pgn(DATASET_TAG),
    "target_username": TARGET_USERNAME,
    "target_name_aliases": TARGET_NAME_ALIASES,
    # Pick one selection mode: game id (preferred) or 1-based index among matched games.
    "target_game_id": None,
    "target_game_index": 50,
    "max_moves": 0,
    "top_k": 5,
    # Confidence gate for replay diagnostics.
    # If predicted factorized prob < threshold, mark as low-confidence.
    "confidence_threshold": 0.02,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir_prefix": "replay",
    "verbose": True,
}

PROMO_IDX_TO_CHAR = {
    0: "",
    1: "n",
    2: "b",
    3: "r",
    4: "q",
}


def denormalize_square(square: int, target_is_white: bool) -> int:
    """Map target-relative square index back to absolute board index."""
    return square if target_is_white else chess.square_mirror(square)


def phase_label(fullmove_number: int) -> str:
    """Map fullmove number to coarse phase label."""
    if fullmove_number < 10:
        return "opening"
    if fullmove_number < 30:
        return "middlegame"
    return "endgame"


def sample_from_encoded_row(row: Dict[str, Any]) -> PolicySample:
    """Convert encoded row to PolicySample."""
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


def choose_legal_move(
    board: chess.Board,
    pred_from_rel: int,
    pred_to_rel: int,
    pred_promo_idx: int,
    target_is_white: bool,
) -> chess.Move:
    """Create best legal move from factorized outputs."""
    from_abs = denormalize_square(int(pred_from_rel), target_is_white)
    to_abs = denormalize_square(int(pred_to_rel), target_is_white)

    promo_char = PROMO_IDX_TO_CHAR.get(int(pred_promo_idx), "")
    promo_piece = None
    if promo_char:
        promo_piece = {
            "n": chess.KNIGHT,
            "b": chess.BISHOP,
            "r": chess.ROOK,
            "q": chess.QUEEN,
        }[promo_char]

    candidates = [m for m in board.legal_moves if m.from_square == from_abs and m.to_square == to_abs]
    if not candidates:
        return chess.Move.null()

    if promo_piece is not None:
        for mv in candidates:
            if mv.promotion == promo_piece:
                return mv

    for mv in candidates:
        if mv.promotion is None:
            return mv
    return candidates[0]


def factorized_prob(
    piece_probs: torch.Tensor,
    to_probs: torch.Tensor,
    promo_probs: torch.Tensor,
    from_sq: int,
    to_sq: int,
    promo: int,
) -> float:
    """Compute factorized probability for one move tuple."""
    p = float(piece_probs[int(from_sq)].item())
    p *= float(to_probs[int(to_sq)].item())
    p *= float(promo_probs[int(promo)].item())
    return p


def save_move_svg(
    board: chess.Board,
    actual_uci: str,
    pred_uci: str,
    out_path: Path,
    board_size: int = 640,
) -> None:
    """Save one SVG board with actual/predicted arrows."""
    actual_move = chess.Move.from_uci(actual_uci)
    pred_move = chess.Move.from_uci(pred_uci) if pred_uci != "0000" else None

    arrows = [chess.svg.Arrow(actual_move.from_square, actual_move.to_square, color="#1565c0")]
    if pred_move is not None:
        arrows.append(chess.svg.Arrow(pred_move.from_square, pred_move.to_square, color="#b71c1c"))

    fill = {
        actual_move.from_square: "#bbdefb",
        actual_move.to_square: "#90caf9",
    }
    if pred_move is not None:
        fill[pred_move.from_square] = "#ffcdd2"
        fill[pred_move.to_square] = "#ef9a9a"

    svg = chess.svg.board(
        board=board,
        size=board_size,
        coordinates=True,
        arrows=arrows,
        fill=fill,
        lastmove=actual_move,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")


def to_mask_for_from(sample: PolicySample, from_sq: int, device: torch.device) -> torch.Tensor:
    """Build legal to-square mask for a chosen from-square."""
    m = torch.zeros(64, dtype=torch.float32, device=device)
    for to_sq in sample.legal_to_by_from.get(str(int(from_sq)), []):
        m[int(to_sq)] = 1.0
    if float(m.sum().item()) <= 0:
        m.fill_(1.0)
    return m


def topk_for_from(
    model: FactorizedPolicyModel,
    fused: torch.Tensor,
    from_sq: int,
    sample: PolicySample,
    device: torch.device,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Return top-k move candidates for a fixed from-square."""
    to_mask = to_mask_for_from(sample, from_sq, device)
    to_logits = masked_logits(model.to_logits(fused, torch.tensor([from_sq], device=device))[0], to_mask)
    to_probs = torch.softmax(to_logits, dim=0)

    promo_by_to: Dict[int, torch.Tensor] = {}
    for to_sq in sample.legal_to_by_from.get(str(int(from_sq)), []):
        promo_logits = model.promo_logits(
            fused,
            torch.tensor([from_sq], device=device),
            torch.tensor([int(to_sq)], device=device),
        )[0]
        promo_by_to[int(to_sq)] = torch.softmax(promo_logits, dim=0)

    cand: List[Dict[str, Any]] = []
    for to_sq in sample.legal_to_by_from.get(str(int(from_sq)), []):
        to_sq_i = int(to_sq)
        promo_probs = promo_by_to[to_sq_i]
        best_promo = int(torch.argmax(promo_probs).item())
        p = float(to_probs[to_sq_i].item()) * float(promo_probs[best_promo].item())
        cand.append(
            {
                "from_sq": int(from_sq),
                "to_sq": to_sq_i,
                "promo": best_promo,
                "local_prob": p,
            }
        )

    cand.sort(key=lambda x: x["local_prob"], reverse=True)
    return cand[: max(1, int(top_k))]


def analyze_row(
    model: FactorizedPolicyModel,
    row: Dict[str, Any],
    history_plies: int,
    device: torch.device,
    top_k: int,
    confidence_threshold: float,
    board_svg_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze one move row and return prediction details."""
    enc = encode_row(row, history_plies=history_plies)
    sample = sample_from_encoded_row(enc)
    batch = collate_batch([sample], device=device)
    board = chess.Board(str(row["fen_before"]))
    target_is_white = bool(int(row.get("target_is_white", 1)))

    with torch.no_grad():
        fused = model.fused_repr(batch)

        piece_logits = masked_logits(model.piece_logits(fused), batch["legal_piece_slot_mask"])[0]
        piece_probs = torch.softmax(piece_logits, dim=0)
        pred_piece = int(torch.argmax(piece_logits).item())
        pred_from = int(resolve_from_square(batch["piece_slot_to_square"], torch.tensor([pred_piece], device=device))[0].item())

        pred_to_mask = to_mask_for_from(sample, pred_from, device)
        to_logits_pred = masked_logits(model.to_logits(fused, torch.tensor([pred_from], device=device))[0], pred_to_mask)
        to_probs_pred = torch.softmax(to_logits_pred, dim=0)
        pred_to = int(torch.argmax(to_logits_pred).item())

        promo_logits_pred = model.promo_logits(
            fused,
            torch.tensor([pred_from], device=device),
            torch.tensor([pred_to], device=device),
        )[0]
        promo_probs_pred = torch.softmax(promo_logits_pred, dim=0)
        pred_promo = int(torch.argmax(promo_logits_pred).item())

        actual_piece = int(enc.get("target_piece_slot", 0))
        actual_from = int(row["target_from_sq"])
        actual_to = int(row["target_to_sq"])
        actual_promo = int(row.get("target_promotion", 0))

        actual_to_mask = to_mask_for_from(sample, actual_from, device)
        to_logits_actual = masked_logits(
            model.to_logits(fused, torch.tensor([actual_from], device=device))[0],
            actual_to_mask,
        )
        to_probs_actual = torch.softmax(to_logits_actual, dim=0)

        promo_logits_actual = model.promo_logits(
            fused,
            torch.tensor([actual_from], device=device),
            torch.tensor([actual_to], device=device),
        )[0]
        promo_probs_actual = torch.softmax(promo_logits_actual, dim=0)

        topk_moves = topk_for_from(
            model=model,
            fused=fused,
            from_sq=actual_from,
            sample=sample,
            device=device,
            top_k=top_k,
        )

    pred_move = choose_legal_move(board, pred_from, pred_to, pred_promo, target_is_white)
    pred_uci = pred_move.uci() if pred_move != chess.Move.null() else "0000"

    actual_uci = str(row.get("played_uci", ""))
    if board_svg_path is not None:
        save_move_svg(board=board, actual_uci=actual_uci, pred_uci=pred_uci, out_path=board_svg_path)

    topk_uci = []
    for item in topk_moves:
        from_abs = denormalize_square(int(item["from_sq"]), target_is_white)
        to_abs = denormalize_square(int(item["to_sq"]), target_is_white)
        mv = chess.Move(from_abs, to_abs)
        promo_char = PROMO_IDX_TO_CHAR.get(int(item["promo"]), "")
        if promo_char:
            mv = chess.Move.from_uci(f"{chess.square_name(from_abs)}{chess.square_name(to_abs)}{promo_char}")
        topk_uci.append(mv.uci())

    actual_rank: Optional[int] = None
    for i, u in enumerate(topk_uci, start=1):
        if u == actual_uci:
            actual_rank = i
            break

    pred_prob = factorized_prob(piece_probs, to_probs_pred, promo_probs_pred, pred_piece, pred_to, pred_promo)
    actual_prob = factorized_prob(
        piece_probs,
        to_probs_actual,
        promo_probs_actual,
        actual_piece,
        actual_to,
        actual_promo,
    )
    is_low_conf = int(pred_prob < float(confidence_threshold))
    is_match = int(actual_uci == pred_uci)
    if is_match:
        decision = "match"
    elif is_low_conf:
        decision = "uncertain_miss"
    else:
        decision = "confident_miss"

    fullmove = int(row.get("fullmove_number", 0))

    return {
        "game_id": row.get("game_id"),
        "ply_index": int(row.get("ply_index", 0)),
        "fullmove_number": fullmove,
        "phase": phase_label(fullmove),
        "actual_uci": actual_uci,
        "pred_uci": pred_uci,
        "match": is_match,
        "is_low_confidence": is_low_conf,
        "decision": decision,
        "actual_rank_in_topk_given_from": actual_rank,
        "top1_hit": int(actual_rank == 1 if actual_rank is not None else False),
        "top3_hit": int(actual_rank is not None and actual_rank <= 3),
        "top5_hit": int(actual_rank is not None and actual_rank <= 5),
        "pred_factorized_prob": pred_prob,
        "actual_factorized_prob": actual_prob,
        "topk_given_from": [
            {
                "rank": i + 1,
                "uci": topk_uci[i],
                "local_prob": float(topk_moves[i]["local_prob"]),
            }
            for i in range(len(topk_moves))
        ],
        "board_svg_path": str(board_svg_path) if board_svg_path is not None else "",
    }


def select_target_game_rows(
    pgn_path: Path,
    target_username: str,
    aliases: List[str],
    history_plies: int,
    target_game_id: Optional[str],
    target_game_index: int,
) -> List[Dict[str, Any]]:
    """Select rows for one target game by GameId or matched-game index."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []

    for row in iter_target_player_positions(
        pgn_path=pgn_path,
        target_username=target_username,
        aliases=aliases,
        history_plies=history_plies,
    ):
        gid = str(row.get("game_id", ""))
        if gid not in grouped:
            grouped[gid] = []
            order.append(gid)
        grouped[gid].append(row)

    if not order:
        return []

    if target_game_id is not None:
        return grouped.get(str(target_game_id), [])

    idx = int(target_game_index)
    if idx <= 0 or idx > len(order):
        return []
    return grouped[order[idx - 1]]


def phase_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute per-phase metrics."""
    out: Dict[str, Dict[str, float]] = {}
    phases = ["opening", "middlegame", "endgame"]
    for ph in phases:
        sub = [r for r in rows if str(r.get("phase")) == ph]
        n = len(sub)
        if n == 0:
            out[ph] = {
                "num_moves": 0,
                "exact_move_acc": 0.0,
                "uncertain_rate": 0.0,
                "confident_miss_rate": 0.0,
            }
            continue

        exact = sum(int(r.get("match", 0)) for r in sub) / n
        uncertain = sum(int(r.get("decision") == "uncertain_miss") for r in sub) / n
        confident = sum(int(r.get("decision") == "confident_miss") for r in sub) / n
        out[ph] = {
            "num_moves": n,
            "exact_move_acc": exact,
            "uncertain_rate": uncertain,
            "confident_miss_rate": confident,
        }
    return out


def write_index_html(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """Write a simple HTML viewer for browsing move boards sequentially."""
    index_path = out_dir / "index.html"
    payload = [
        {
            "i": i + 1,
            "ply": int(r.get("ply_index", 0)),
            "phase": str(r.get("phase", "")),
            "actual": str(r.get("actual_uci", "")),
            "pred": str(r.get("pred_uci", "")),
            "decision": str(r.get("decision", "")),
            "prob": float(r.get("pred_factorized_prob", 0.0)),
            "svg": str(r.get("board_svg_path", "")),
        }
        for i, r in enumerate(rows)
    ]
    data_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Replay Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    .bar {{ display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }}
    button {{ padding: 6px 10px; }}
    #meta {{ font-weight: 600; margin-bottom: 8px; }}
    #imgwrap {{ border: 1px solid #ddd; padding: 8px; width: fit-content; background: #fafafa; }}
    img {{ max-width: min(90vw, 740px); height: auto; }}
    .match {{ color: #0b6b0b; }}
    .uncertain {{ color: #6a1b9a; }}
    .confident {{ color: #b71c1c; }}
  </style>
</head>
<body>
  <h2>Replay Viewer</h2>
  <div class=\"bar\">
    <button id=\"prev\">Prev</button>
    <button id=\"next\">Next</button>
    <label>Jump:</label>
    <input id=\"jump\" type=\"number\" min=\"1\" style=\"width:70px\" />
    <button id=\"go\">Go</button>
  </div>
  <div id=\"meta\"></div>
  <div id=\"imgwrap\"><img id=\"board\" alt=\"board\" /></div>

  <script>
    const rows = {data_json};
    let idx = 0;

    function clamp(v) {{
      if (rows.length === 0) return 0;
      return Math.max(0, Math.min(rows.length - 1, v));
    }}

    function classForDecision(d) {{
      if (d === 'match') return 'match';
      if (d === 'uncertain_miss') return 'uncertain';
      return 'confident';
    }}

    function render() {{
      if (rows.length === 0) return;
      idx = clamp(idx);
      const r = rows[idx];
      document.getElementById('board').src = r.svg;
      const cls = classForDecision(r.decision);
      document.getElementById('meta').innerHTML =
        `Move ${{r.i}}/${{rows.length}} | ply=${{r.ply}} | phase=${{r.phase}} | actual=${{r.actual}} | pred=${{r.pred}} | ` +
        `p(pred)=${{r.prob.toFixed(6)}} | <span class="${{cls}}">${{r.decision}}</span>`;
      document.getElementById('jump').value = r.i;
    }}

    document.getElementById('prev').onclick = () => {{ idx -= 1; render(); }};
    document.getElementById('next').onclick = () => {{ idx += 1; render(); }};
    document.getElementById('go').onclick = () => {{
      const v = parseInt(document.getElementById('jump').value || '1', 10);
      idx = clamp(v - 1);
      render();
    }};

    render();
  </script>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")
    return index_path


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, confidence_threshold: float) -> None:
    """Write replay analysis outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    details_jsonl = out_dir / "replay_details.jsonl"
    with details_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = out_dir / "replay_summary.csv"
    fields = [
        "game_id",
        "ply_index",
        "fullmove_number",
        "phase",
        "actual_uci",
        "pred_uci",
        "match",
        "is_low_confidence",
        "decision",
        "actual_rank_in_topk_given_from",
        "top1_hit",
        "top3_hit",
        "top5_hit",
        "pred_factorized_prob",
        "actual_factorized_prob",
        "board_svg_path",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    index_html = write_index_html(rows, out_dir)

    n = len(rows)
    top1 = sum(int(r["match"]) for r in rows) / max(n, 1)
    top3 = sum(int(r["top3_hit"]) for r in rows) / max(n, 1)
    top5 = sum(int(r["top5_hit"]) for r in rows) / max(n, 1)
    avg_actual_prob = sum(float(r["actual_factorized_prob"]) for r in rows) / max(n, 1)
    uncertain_rate = sum(int(r["decision"] == "uncertain_miss") for r in rows) / max(n, 1)
    confident_miss_rate = sum(int(r["decision"] == "confident_miss") for r in rows) / max(n, 1)

    overall = {
        "num_target_moves": n,
        "exact_move_acc": top1,
        "top3_given_actual_from": top3,
        "top5_given_actual_from": top5,
        "avg_actual_factorized_prob": avg_actual_prob,
        "confidence_threshold": float(confidence_threshold),
        "uncertain_miss_rate": uncertain_rate,
        "confident_miss_rate": confident_miss_rate,
        "phase_metrics": phase_metrics(rows),
        "replay_details_jsonl": str(details_jsonl),
        "replay_summary_csv": str(summary_csv),
        "index_html": str(index_html),
    }

    with (out_dir / "overall_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)


def main() -> None:
    model_path = Path(CONFIG["model_path"])
    pgn_path = Path(CONFIG["pgn_path"])
    target_username = str(CONFIG["target_username"])
    aliases = [str(x).strip() for x in CONFIG.get("target_name_aliases", []) if str(x).strip()]
    target_game_id = CONFIG.get("target_game_id")
    target_game_index = int(CONFIG.get("target_game_index", 1))
    top_k = int(CONFIG.get("top_k", 5))
    max_moves = int(CONFIG.get("max_moves", 0))
    confidence_threshold = float(CONFIG.get("confidence_threshold", 0.02))
    verbose = bool(CONFIG.get("verbose", True))

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")
    if not target_username.strip():
        raise ValueError("target_username cannot be empty")

    device = torch.device(str(CONFIG.get("device", "cpu")))

    ckpt = load_checkpoint(model_path, device=device)
    train_config = dict(ckpt.get("config", {}))
    history_plies = int(train_config.get("history_plies", HISTORY_PLIES))

    model = FactorizedPolicyModel(train_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = select_target_game_rows(
        pgn_path=pgn_path,
        target_username=target_username,
        aliases=aliases,
        history_plies=history_plies,
        target_game_id=target_game_id,
        target_game_index=target_game_index,
    )
    if not rows:
        raise ValueError("No target game rows found. Check target username/game selection.")

    if max_moves > 0:
        rows = rows[:max_moves]

    game_id = str(rows[0].get("game_id", f"game_{target_game_index}"))
    out_dir = outputs_dir(str(CONFIG["dataset_tag"])) / f"{CONFIG['output_dir_prefix']}_{game_id}"
    boards_dir = out_dir / "boards"

    analysis_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        board_svg_path = boards_dir / f"move_{i:03d}_ply_{int(row.get('ply_index', 0)):03d}.svg"
        result = analyze_row(
            model=model,
            row=row,
            history_plies=history_plies,
            device=device,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            board_svg_path=board_svg_path,
        )
        result["board_svg_path"] = str(board_svg_path.relative_to(out_dir))
        analysis_rows.append(result)

        if verbose:
            print(
                f"Move {i:02d} | ply={int(result['ply_index']):03d} | "
                f"phase={result['phase']} | decision={result['decision']} | "
                f"actual={result['actual_uci']} | pred={result['pred_uci']}"
            )

    write_outputs(analysis_rows, out_dir, confidence_threshold=confidence_threshold)

    n = len(analysis_rows)
    exact = sum(int(r["match"]) for r in analysis_rows) / max(n, 1)
    print("\nDone.")
    print(f"Target moves analyzed: {n}")
    print(f"Exact move acc: {exact:.4f}")
    print(f"Confidence threshold: {confidence_threshold:.6f}")
    print(f"Output dir: {out_dir}")
    print(f"Boards dir: {boards_dir}")
    print(f"Viewer: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

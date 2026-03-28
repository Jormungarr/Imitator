# Imitator (History-Conditioned Policy)

This project follows `AGENTS.md`: model player move choice directly with current state + recent trajectory, not candidate ranking against engine lines.

## Workflows

### A) Multi-player representation pretraining

1. Collect many player PGNs under `data/raw/pretrain_multi/`.
2. Run:

```bash
python scripts/run_pretrain_pipeline.py
```

This now includes merge-first flow:
- Stage 0: merge many pretrain PGNs into one file `data/raw/pretrain_multi/pretrain_multi_merged.pgn`
- Stage 1: parse merged pretrain PGN
- Stage 2: encode samples
- Stage 3: pretrain model

Low-RAM full-data option (streamed Stage 3 only):

```bash
python scripts/script3_pretrain_history_policy_stream.py
```

Outputs:
- `data/processed/pretrain_multi/positions_history.jsonl`
- `data/processed/pretrain_multi/policy_samples.jsonl`
- `models/pretrain_multi/history_policy_pretrain.pt` (standard Stage 3)
- `models/pretrain_multi/history_policy_pretrain_stream.pt` (stream Stage 3)

Strict isolation option:
- In `script3_pretrain_history_policy.py`, set `strict_target_isolation = True` to remove target player IDs from pretraining samples.

### B) Target-player adaptation (fine-tuning)

You can collect target games ahead of pretraining and store them separately.

1. Prepare target PGN at `data/raw/finetune_players/<DATASET_TAG>.pgn`.
2. Set `DATASET_TAG` and `TARGET_USERNAME` in `scripts/pipeline_config.py`.
3. Run:

```bash
python scripts/run_pipeline.py
```

Outputs:
- `data/processed/<dataset>/positions_history.jsonl`
- `data/processed/<dataset>/policy_samples.jsonl`
- `models/<dataset>/history_policy.pt`
- `models/<dataset>/history_policy_metrics.json`
- `models/<dataset>/honest_split_game_ids.json`

### C) Replay analysis (game-level visual diagnostics)

Run:

```bash
python scripts/replay_analysis.py
```

Configure target/model/game in `scripts/replay_analysis.py` `CONFIG`.

Outputs under:
- `outputs/<dataset>/replay_<game_id>/`

Includes:
- `boards/*.svg` (actual vs predicted arrows)
- `replay_details.jsonl`
- `replay_summary.csv`
- `overall_metrics.json` (with confidence gate + phase metrics)
- `index.html` (interactive move-by-move board viewer)

## Scraper Browser Settings

In `script_chesscom_db_bulk_download.py`:

- `headless`: `False` shows browser window; `True` runs hidden.
- `slow_mo_ms`: delay per Playwright action (ms). Higher is slower but easier to debug.
- `timeout_ms`: max wait per action before timeout.
- `wait_for_manual_login`: pause for manual login/captcha, resume on Enter.
- `human_like`: enable random pauses between actions.
- `human_pause_min_sec` / `human_pause_max_sec`: pause range used when `human_like=True`.

Recommended stable setup:
- `headless=False`
- `wait_for_manual_login=True`
- `human_like=True`
- `slow_mo_ms=50~120`
- `timeout_ms=20000~30000`

## Honest test split

Fine-tuning uses game-level `train/valid/test` split.
- Model selection uses `valid` only.
- Final `test` is evaluated once from best-valid checkpoint.
- Test games are disjoint from train/valid by `game_id`.

## Notes

- Fine-tuning loads only pretrained `state_encoder` + `history_encoder` by default.
- Policy heads stay highly adaptable for style fidelity.
- Training uses mini-batches (`EmbeddingBag`) with per-epoch progress and ETA.

## Current policy factorization

The current model represents a move as the tuple `m = (f,t,p)` and predicts it with factorized heads:

$$ \hat P(m \mid s,h,c) = \hat P(f,t,p \mid s,h,c) = \hat P(f \mid s,h,c)\hat P(t \mid f,s,h,c)\hat P(p \mid f,t,s,h,c) $$

where:
- `s` = current board state
- `h` = recent move/state history
- `c` = optional context
- `f` = from-square
- `t` = to-square
- `p` = promotion choice

Equivalently:

$$ \hat P(f,t,p \mid s,h,c) = \hat P_{\mathrm{from}}(f \mid s,h,c)\hat P_{\mathrm{to}}(t \mid f,s,h,c)\hat P_{\mathrm{promo}}(p \mid f,t,s,h,c) $$

At inference time this is combined with legal-move constraints, so illegal `from` and `to` choices are masked out by the current position.

## Derivation

Let:
- $x_s$ = sparse current-state features
- $x_d$ = dense current-state features
- $x_h^{evt}$ = move-event history sequence
- $x_h^{delta}$ = state-delta history sequence
- $x_c$ = context features

The encoder stack is:

$$ z_s = f_{\mathrm{state}}(x_s, x_d) $$

$$ z_h = f_{\mathrm{hist}}(x_h^{evt}, x_h^{delta}) $$

$$ z_c = f_{\mathrm{ctx}}(x_c) $$

$$ z = f_{\mathrm{fuse}}([z_s, z_h, z_c]) $$

The move is parameterized as `m = (f,t,p)`, where:
- $f$ = from-square
- $t$ = to-square
- $p$ = promotion class

The model factorization is:

$$ \hat P(m \mid s,h,c) = \hat P(f,t,p \mid s,h,c) = \hat P(f \mid z)\hat P(t \mid z,f)\hat P(p \mid z,f,t) $$

The three heads are:

$$ \ell_f = g_{\mathrm{from}}(z) $$

$$ \hat P(f \mid z) = \mathrm{Softmax}(\mathrm{MaskFrom}(\ell_f)) $$

$$ e_f = E_{\square}(f) $$

$$ \ell_t = g_{\mathrm{to}}([z, e_f]) $$

$$ \hat P(t \mid z,f) = \mathrm{Softmax}(\mathrm{MaskTo}(\ell_t; f)) $$

$$ e_t = E_{\square}(t) $$

$$ \ell_p = g_{\mathrm{promo}}([z, e_f, e_t]) $$

$$ \hat P(p \mid z,f,t) = \mathrm{Softmax}(\ell_p) $$

So the full move probability is:

$$ \hat P(f,t,p \mid s,h,c) = \mathrm{Softmax}(\mathrm{MaskFrom}(g_{\mathrm{from}}(z)))[f]\cdot \mathrm{Softmax}(\mathrm{MaskTo}(g_{\mathrm{to}}([z, E_{\square}(f)]); f))[t]\cdot \mathrm{Softmax}(g_{\mathrm{promo}}([z, E_{\square}(f), E_{\square}(t)]))[p] $$

with

$$ z = f_{\mathrm{fuse}}([f_{\mathrm{state}}(x_s, x_d), f_{\mathrm{hist}}(x_h^{evt}, x_h^{delta}), f_{\mathrm{ctx}}(x_c)]) $$

The training objective is the weighted negative log-likelihood of the observed move components:

$$ \mathcal L = -\log \hat P(f^\ast \mid z) - \log \hat P(t^\ast \mid z,f^\ast) - 0.25 \log \hat P(p^\ast \mid z,f^\ast,t^\ast) $$

## Data Sources

- Pretraining data reference: [docs/data_sources.md](docs/data_sources.md)
- TWIC (The Week in Chess): https://theweekinchess.com/twic
- Chess.com Games (target-player fine-tune source): https://www.chess.com/games

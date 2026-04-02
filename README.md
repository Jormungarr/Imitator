# Imitator (History-Conditioned Policy)

This project follows `AGENTS.md`: model player move choice directly with current state + recent trajectory, not candidate ranking against engine lines.

Current implementation status:
- State encoder: HalfKP-style sparse board features plus dense state summaries.
- History encoder: GRU over structured move-event and state-delta sequences.
- Policy head: factorized `from -> to -> promotion` prediction with legal-move masking.
- Tooling: replay diagnostics, local Tkinter app, local web UI, and Kaggle packaging helpers.

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
2. If scraping from Chess.com, `script_chesscom_db_bulk_download.py` can store page PGNs under `data/raw/finetune_players/<DATASET_TAG>/db_pages/` and merge them into the canonical fine-tune PGN.
3. Set `DATASET_TAG` and `TARGET_USERNAME` in `scripts/pipeline_config.py`.
4. Run:

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

### D) Interactive play interfaces

Local Tkinter app:

```bash
python scripts/play_style_local_app.py
```

Local web UI:

```bash
python scripts/play_style_web.py
```

Both interfaces load checkpoints from `models/<dataset>/history_policy.pt` and let you play against a selected player-style model from the current position.

### E) Kaggle notebook / upload helpers

Kaggle-related helpers live under `kaggle/`.

- `kaggle/imitator_kaggle.ipynb`: notebook entry point for GPU training on Kaggle.
- `kaggle/build_imitator_kaggle_notebook.py`: rebuilds the notebook payload from the repo scripts.
- `kaggle/upload_project_dataset.py`: uploads a code-focused project snapshot.
- `kaggle/upload_model.py`: uploads trained checkpoints separately.

Use these helpers when you want reproducible remote training without bundling local model artifacts into the repository.

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
- Split IDs are saved alongside metrics for reproducible evaluation.

## Notes

- Fine-tuning loads only pretrained `state_encoder` + `history_encoder` by default.
- Policy heads stay highly adaptable for style fidelity.
- Training uses mini-batches (`EmbeddingBag`) with per-epoch progress and ETA.
- Replay and play tools use the same factorized inference path as training-time move decoding.
- Raw source PGNs may be kept locally for scraping and Kaggle packaging, while processed tensors, checkpoints, and reports stay outside versioned source outputs.

## Current policy factorization

The current implementation predicts a move through the tuple `m = (u,f,t,p)`, where `u` is the canonical moving piece slot and `f` is the resolved current square of that slot.

$$ \hat P(m \mid s,h,c) = \hat P(u,t,p \mid s,h,c) = \hat P(u \mid s,h,c)\hat P(t \mid u,s,h,c)\hat P(p \mid u,t,s,h,c) $$

where:
- `s` = current board state
- `h` = recent move/state history
- `c` = optional context
- `u` = canonical piece slot
- `f` = resolved from-square from the current `piece_slot_to_square` map
- `t` = to-square
- `p` = promotion choice

Equivalently:

$$ \hat P(u,t,p \mid s,h,c) = \hat P_{\mathrm{piece}}(u \mid s,h,c)\hat P_{\mathrm{to}}(t \mid f(u),s,h,c)\hat P_{\mathrm{promo}}(p \mid f(u),t,s,h,c) $$

At inference time the piece-slot head is masked by legal movable piece slots, the resolved `from-square` is obtained by gathering from `piece_slot_to_square`, and the `to-square` head is masked by legal destinations for that resolved square.

## Derivation

Let:
- $x_s$ = sparse current-state features
- $x_d$ = dense current-state features
- $x_h^{evt}$ = move-event history sequence
- $x_h^{delta}$ = state-delta history sequence
- $x_c$ = context features

The encoder stack is:

$$ z_s = f_{\mathrm{state}}(x_s, x_d) $$

$$ \tilde h_{1:K} = \mathrm{GRU}\left(\left[\mathrm{Emb}_{mover}, \mathrm{Emb}_{piece}, \mathrm{Emb}_{from}, \mathrm{Emb}_{to}, W_{flag}x_{flag}, W_{delta}x_h^{delta}\right]_{1:K}\right) $$

$$ z_h = \frac{\sum_{k=1}^{K} m_k \tilde h_k}{\max\left(\sum_{k=1}^{K} m_k, 1\right)} $$

$$ z_c = f_{\mathrm{ctx}}(x_c) $$

$$ z = f_{\mathrm{fuse}}([z_s, z_h, z_c]) $$

The move is parameterized as `m = (u,f,t,p)`, where:
- $u$ = piece slot
- $f = \mathrm{Resolve}(u)$ from the current piece-slot map
- $t$ = to-square
- $p$ = promotion class

The model factorization is:

$$ \hat P(m \mid s,h,c) = \hat P(u,t,p \mid s,h,c) = \hat P(u \mid z)\hat P(t \mid z,f(u))\hat P(p \mid z,f(u),t) $$

The three heads are:

$$ \ell_u = g_{\mathrm{piece}}(z) $$

$$ \hat P(u \mid z) = \mathrm{Softmax}(\mathrm{MaskPiece}(\ell_u)) $$

$$ f = \mathrm{Gather}(M_{\mathrm{slot}\to\mathrm{sq}}, u) $$

$$ e_f = E_{\square}(f) $$

$$ \ell_t = g_{\mathrm{to}}([z, e_f]) $$

$$ \hat P(t \mid z,f) = \mathrm{Softmax}(\mathrm{MaskTo}(\ell_t; f)) $$

$$ e_t = E_{\square}(t) $$

$$ \ell_p = g_{\mathrm{promo}}([z, e_f, e_t]) $$

$$ \hat P(p \mid z,f,t) = \mathrm{Softmax}(\ell_p) $$

So the full move probability is:

$$ \hat P(u,t,p \mid s,h,c) = \mathrm{Softmax}(\mathrm{MaskPiece}(g_{\mathrm{piece}}(z)))[u]\cdot \mathrm{Softmax}(\mathrm{MaskTo}(g_{\mathrm{to}}([z, E_{\square}(f(u))]); f(u)))[t]\cdot \mathrm{Softmax}(g_{\mathrm{promo}}([z, E_{\square}(f(u)), E_{\square}(t)]))[p] $$

with

$$ z = f_{\mathrm{fuse}}([f_{\mathrm{state}}(x_s, x_d), f_{\mathrm{hist}}(x_h^{evt}, x_h^{delta}), f_{\mathrm{ctx}}(x_c)]) $$

The training objective is the weighted negative log-likelihood of the observed move components, with an optional auxiliary threat head:

$$ \mathcal L_{\mathrm{move}} = -\log \hat P(u^\ast \mid z) - \log \hat P(t^\ast \mid z,f^\ast) - 0.25 \log \hat P(p^\ast \mid z,f^\ast,t^\ast) $$

$$ \mathcal L = \mathcal L_{\mathrm{move}} + \lambda_{\mathrm{threat}} \, \mathcal L_{\mathrm{threat}} \qquad \text{if the auxiliary threat head is enabled} $$

## Data Sources

- Pretraining data reference: [docs/data_sources.md](docs/data_sources.md)
- TWIC (The Week in Chess): https://theweekinchess.com/twic
- Chess.com Games (target-player fine-tune source): https://www.chess.com/games
- Kaggle helper notes: [kaggle/README.md](kaggle/README.md)

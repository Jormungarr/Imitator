# Chess Style Imitation Model Design Notes

## Purpose

This document summarizes the current design direction for a **player-style imitation model** in chess, based on discussion around Stockfish / NNUE ideas, but **adapted to the real objective**:

- We are **not** trying to build a strongest-move engine.
- We are **not** trying to reproduce Stockfish evaluation.
- We **are** trying to model **how a specific player tends to choose moves**, including imperfect play and player-specific tendencies.

This summary is intended for Codex to understand the design direction and implementation priorities.

---

## Problem Setting

### Goal
Model the move choice distribution of a target player:

$$
P(m_t \mid s_t, h_t, c_t)
$$

where:

- `s_t`: current board state
- `h_t`: recent move / state history
- `c_t`: optional context (opening family, color, time control, etc.)
- `m_t`: actual move chosen by the target player

### Important assumptions

- Dataset size for a single target player is usually only **1000-3000 games**.
- **Blunders are part of style**, assuming the games reflect the player's real level.
- We do **not** want to filter training data using Stockfish candidate generation at this stage.
- Decision quality depends not only on the current position, but also on:
  - recent position changes
  - recent plans / move trajectory
  - opponent interaction
  - momentum / local context

---

## Core Modeling Conclusion

The task should be treated as:

> **History-conditioned human move policy modeling**

not as:

- engine evaluation learning
- best-move prediction
- candidate ranking over Stockfish-generated moves

So the model should directly learn the player's move distribution, conditioned on:

1. current board representation
2. recent trajectory / evolution
3. optional game context

---

## How NNUE / Stockfish Should Influence the Design

### What to borrow

We should borrow **representation ideas**, not the original task formulation.

Most valuable ideas from NNUE / Stockfish:

1. **King-relative sparse board encoding**
   - piece features anchored by king square
   - captures positional relationships around king safety / attack structure
   - strong, mature board representation idea

2. **Efficiently updatable / delta mindset**
   - moves change only a small portion of state
   - recent history can be represented by state deltas instead of full board recomputation

3. **Strong input design + relatively modest network**
   - important under small-data constraints

### What not to copy directly

Do **not** directly copy:

- Stockfish's original objective: position evaluation for alpha-beta search
- full NNUE training regime
- large sparse feature spaces designed for billions of positions
- search-dependent engine pipeline
- candidate generation driven by engine best lines

Reason:
our task is **player policy imitation**, not engine evaluation.

---

## Why Static Current-Position Modeling Is Not Enough

A player's move often depends on **how the current position was reached**.

Two identical board states may produce different likely moves depending on:

- whether the last few moves were kingside expansion or central simplification
- whether the player has been attacking or defending
- whether a recent exchange changed the plan
- whether the player has just missed tactics or entered a quieter phase
- whether the recent trajectory suggests continued commitment to a plan

Therefore, **history must be a first-class input**.

---

## Main Design Direction

## Recommended Model Family

A **history-conditioned policy network** with three components:

1. **Current state encoder**
2. **History encoder**
3. **Policy head with legality constraints**

---

## Component A: Current State Encoder

### Objective
Encode the current board in a way that captures structure, king relationships, material, and strategic shape.

### Recommended inputs

#### A1. NNUE-inspired sparse board features
For each non-king piece, activate features relative to:

- own king square
- opponent king square

Plus:

- side to move
- castling rights

This is a **HalfKP-like / king-relative representation**, adapted for our use.

#### A2. Dense strategic state summaries
Add compact handcrafted state features such as:

- material vector
- phase (opening / middlegame / endgame)
- pawn structure summary
  - isolated pawns
  - doubled pawns
  - passed pawns
  - backward pawns
- king safety summary
- mobility summary
- open / semi-open files
- bishop pair
- center / space measures

### Reason
King-relative sparse structure gives a strong board encoding base.
Dense summaries stabilize learning and reduce dependence on ultra-sparse features under limited data.

---

## Component B: History Encoder

This is the most important addition relative to earlier static approaches.

History should not be represented only as raw UCI move strings.
Instead, use **structured recent trajectory encoding**.

### B1. Move-event sequence

For the last `K` plies (recommended initially: `8-16` plies), encode each move as an event vector containing information such as:

- mover side
- moved piece type
- from square
- to square
- capture flag
- check flag
- castling flag
- promotion flag
- exchange flag
- pawn break flag
- whether it increased pressure near king
- whether it changed pawn structure
- whether it simplified or complicated the position

This captures **action semantics**.

---

### B2. State-delta sequence

In addition to move-event tokens, encode recent **state changes** between successive positions.

For each recent step, record deltas such as:

- material delta
- pawn structure delta
- king safety delta
- mobility delta
- center / space delta
- file openness delta
- pressure near king delta

This captures **result semantics**:
not only what move was played, but what effect it had on the position.

---

### B3. Optional plan-trace summary

History is not only short tactical memory; it can also reflect local plans.

Useful summary features over the last several plies may include:

- kingside activity ratio
- queenside activity ratio
- central activity ratio
- simplification ratio
- forcing-move ratio
- repeated maneuver indicators
- pressure on same file / same wing
- continuation of prior strategic theme

This may be implemented as:
- explicit summary features, or
- implicit sequence encoding output

---

## Recommended History Encoder Architecture

Under small data constraints, prefer **lightweight sequence modules**.

### Preferred first option
- GRU over move-event sequence
- GRU or temporal MLP / 1D CNN over state-delta sequence
- fuse both outputs

### Why not a large Transformer first
Dataset is too small for a large fully generic sequence model.
A lightweight recurrent or temporal convolutional encoder is more stable.

---

## Component C: Context Encoder (Optional but Useful)

Optional context features:

- player color
- move number
- ECO family / opening family
- time control
- clock bucket (if available)
- opponent rating bucket
- tournament / casual flag (if available)

These should not dominate the model, but can help explain style variation by context.

---

## Output Layer / Policy Head

Since we are **not** using engine-generated candidate sets, the output must define a move distribution directly.

### Important design choice
Avoid naive huge one-shot move classification if possible.

Instead prefer a **factorized policy head**.

### Recommended factorization
Predict move in stages, for example:

1. which piece / from-square is likely to move
2. where it is likely to go
3. special case outputs (promotion, special move type)

Then reconstruct legal move probabilities under a **legality mask**.

### Why factorized output is better here

- better parameter sharing
- more data-efficient
- better for small datasets
- style often appears as:
  - which pieces a player prefers to move
  - whether the player pushes pawns vs maneuvers pieces
  - whether the player tends toward certain regions / plans

---

## Alternative Output Options Considered

### Option 1: Full fixed move vocabulary classification
Possible, but less preferred initially.

Pros:
- simple
- direct

Cons:
- too many rare moves
- many illegal outputs
- harder under small data

### Option 2: Factorized move policy
**Preferred**

Pros:
- more data-efficient
- better inductive bias
- easier to learn stylistic tendencies

### Option 3: Fully autoregressive move generation
Not recommended initially due to data scale.

---

## Blunders and Imperfect Play

We explicitly assume:

> **Blunders are part of the player's style distribution**

Therefore:

- do **not** filter them out just because they are suboptimal
- do **not** force the model to imitate only engine-approved moves

However, the model should not treat all bad moves as random noise either.

Instead, it should learn:

- in what contexts the player tends to make mistakes
- what kinds of mistakes occur
- how mistakes correlate with recent trajectory / pressure / structure

This means training labels remain the **actual human move**.

Useful contextual features for this include:

- legal move count
- tactical volatility proxies
- king exposure
- recent forcing sequence count
- material imbalance
- clock information (if available)

---

## Final Recommended Architecture


## Pretraining and Style Preservation

### Important risk

A naive pretraining strategy can **wash out player-specific style**.

This happens if we first train a strong generic human policy model and then only lightly fine-tune it on one target player.
In that case, the final model often becomes:

- generic human move prior
- plus a very small player-specific correction

That is **not** the desired behavior.
The goal is not to imitate an average human.
The goal is to imitate **this specific player**.

---

### What pretraining is allowed to do

Pretraining should mainly help the model learn:

- how to encode board structure
- how to encode king-relative piece relationships
- how to encode recent move / state history
- how to compress trajectory information into stable embeddings

In other words:

> **Pretraining should mainly learn representation, not a rigid generic policy.**

---

### What pretraining should NOT do

Do **not** make the generic pretrained model too strong at deciding moves in a way that dominates later adaptation.

Avoid the following pattern:

1. train a strong generic human policy model
2. freeze most layers
3. apply very weak fine-tuning on the target player

This is likely to suppress the player's individual tendencies.

Also avoid these failure modes:

- too much freezing
- too little learning rate during adaptation
- too much regularization during player-specific training
- early stopping before player-specific deviations are learned
- using only "strong / correct / standard" human data for pretraining

---

### Recommended implementation rule

If pretraining is used, use it mainly for:

- `state_encoder`
- `history_encoder`

Do **not** rely on a heavily pretrained final `policy_head` as the main decision maker for the target player.

Preferred default:

- pretrained `state_encoder`
- pretrained `history_encoder`
- lightly pretrained or randomly initialized `policy_head`
- then player-specific training on top

This keeps the board understanding transferable, while leaving the final move choice distribution flexible.

---

### Better formulation: player style as deviation from a base policy

A better conceptual target is:

\[
P_{player}(m \mid s,h,c) = P_{base}(m \mid s,h,c) + \Delta_{player}(m \mid s,h,c)
\]

where:

- `P_base` = broad human prior
- `Delta_player` = player-specific deviation

Implementation does not need to literally use additive probabilities, but the architecture should reflect the idea that:

- the model should know general human chess structure
- the player-specific part must be large enough to matter

The target player's style is not just noise around a generic model.
It is a meaningful structured deviation.

---

### Strong recommendation for Codex

When implementing adaptation to a target player:

1. treat generic pretraining as **representation pretraining**
2. keep the final policy layer highly adaptable
3. allow at least part of the history module to update during player-specific training
4. do not over-freeze the model
5. evaluate whether the adapted model remains distinguishable from the generic human baseline

A practical diagnostic is:

- compare the target-player model against a generic-human baseline
- verify that the target-player model changes move preferences in a noticeable and consistent way
- verify that player-specific tendencies are not collapsed into the mean policy

---

### Bottom line

Use pretraining only if it improves representation quality without collapsing player individuality.

If there is a conflict between:

- stronger generic policy performance
- stronger player-specific style fidelity

prefer **player-specific style fidelity**.


## High-level pipeline

### Input
- current board state
- recent move-event history
- recent state-delta history
- optional context metadata

### Encoders
- sparse + dense state encoder
- lightweight history encoder
- optional context encoder

### Fusion
Concatenate all embeddings into a shared representation.

### Policy head
Factorized move prediction with legal-move masking.

---

## Suggested Minimal Viable Version (MVP)

If implementing the first version now, keep it small and stable.

### State inputs
- king-relative piece features
- castling rights
- side to move
- material
- phase

### History inputs
Last `8` plies:
- moved piece
- from square
- to square
- capture
- check
- castle
- promotion

Plus simple post-move deltas:
- material delta
- king safety delta
- pawn structure delta

### Encoders
- sparse linear / embedding projection for board features
- GRU for history

### Output
- predict from-square or moving piece first
- then predict to-square
- apply legality mask

This is much more suitable than immediately predicting a huge flat UCI vocabulary.

---

## Design Principles to Preserve

1. **Do not confuse style imitation with engine strength**
2. **History is essential**
3. **Current board alone is insufficient**
4. **NNUE is a representation inspiration, not the final task definition**
5. **Small data requires strong inductive bias**
6. **Factorized outputs are better than giant unconstrained outputs**
7. **Real human move labels should be preserved, including imperfect play**
8. **Prefer lightweight and interpretable structure before large generic deep models**

---

## Summary in One Sentence

The best direction is:

> Build a **history-conditioned player policy model** that uses an **NNUE-inspired king-relative board encoder**, a **lightweight sequence encoder for recent move / state evolution**, and a **factorized legal-move policy head**, while preserving the player's actual move distribution including mistakes.

---

## Possible Next Implementation Step

A practical next step would be to implement:

1. board feature extractor for king-relative sparse inputs
2. recent move-event + state-delta history builder
3. lightweight GRU-based history encoder
4. factorized move head with legality masking
5. training target = actual played move

That should be treated as the default baseline direction unless later experiments clearly justify a different architecture.

---

## Decisions and Workflow Reference (Updated 2026-03-26)

This section records concrete implementation and training workflow decisions agreed after reconstruction.

### 1) Data layout (separate pretrain vs fine-tune)

- Multi-player pretraining pool:
  - `data/raw/pretrain_multi/`
  - one merged PGN per player, e.g. `data/raw/pretrain_multi/<player_slug>.pgn`
- Target-player fine-tuning pool:
  - `data/raw/finetune_players/`
  - target merged PGN: `data/raw/finetune_players/<DATASET_TAG>.pgn`

Reason:
keep broad representation learning data and target adaptation data operationally separate.

### 2) Chess.com scraping storage modes

`script_chesscom_db_bulk_download.py` supports:

- `storage_mode = "pretrain"` (default)
  - writes page PGNs to `data/raw/pretrain_multi/<player_slug>/db_pages/`
  - writes merged PGN to `data/raw/pretrain_multi/<player_slug>.pgn`
- `storage_mode = "finetune"`
  - writes page PGNs to `data/raw/finetune_players/<finetune_tag>/db_pages/`
  - writes merged PGN to `data/raw/finetune_players/<finetune_tag>.pgn`

### 3) Training strategy (kept)

Use two-stage training:

1. Representation pretraining on many players.
2. Player-specific adaptation (fine-tuning) on target player.

Do not replace adaptation with only weighted mixed pretraining.
If weighting is used, still perform a target-only fine-tune stage.

### 4) Honest evaluation requirement (implemented)

Fine-tuning now uses game-level splits:

- train
- valid (for model selection)
- test (honest holdout)

`script4_finetune_history_policy.py` stores split IDs to:
- `models/<DATASET_TAG>/honest_split_game_ids.json`

and reports final honest test metrics in:
- `models/<DATASET_TAG>/history_policy_metrics.json`

### 5) Target isolation control in pretraining

`script3_pretrain_history_policy.py` has:

- `strict_target_isolation = True/False`
- `excluded_player_ids = [...]`

If strict isolation is enabled, target player samples are removed from pretraining data.

### 6) Pretraining + adaptation pipeline commands

- Multi-player pretraining:
  - `python scripts/run_pretrain_pipeline.py`
- Target-player adaptation:
  - `python scripts/run_pipeline.py`

### 7) Current implementation notes

- State encoder: HalfKP-style sparse features + dense state summaries.
- History encoder: GRU over structured move-event and state-delta inputs.
- Policy head: factorized with legality constraints (`from -> to -> promotion`).
- Training speed improved by mini-batching + `EmbeddingBag` + progress/ETA logs.

### 8) Practical data policy

- Including some target games in pretraining is allowed.
- Preferred evaluation protocol: keep final target test games unseen by fine-tune, and ideally unseen by pretraining if strict honesty is required.

### 9) Pretrain data ingestion rule (merge-first)

Pretraining now assumes many source PGNs are merged into one canonical file first:

- merge script: `script0_merge_pretrain_pgns.py`
- merged file: `data/raw/pretrain_multi/pretrain_multi_merged.pgn`
- parser input: `script1_parse_multi_player_positions.py` reads this merged file directly

This supports the workflow: collect many PGNs quickly, then one merge, then pretrain.

# Reconstruction Notes

## Reused with minor cleanup

- `scripts/pipeline_config.py`
  - Kept path helpers and dataset-tag workflow.
  - Added `HISTORY_PLIES` for sequence windows.
- `scripts/script0_merge_player_pgns.py`
  - Reused stage-0 merge behavior.
  - Simplified output logging and validation.
- `scripts/script_chesscom_db_bulk_download.py`
  - Main Chess.com download workflow for both pretrain and finetune storage modes.

## Rewritten to match AGENTS.md

- `scripts/script1_parse_pgn_to_positions.py`
  - Old: static position extraction for candidate ranking.
  - New: emits per-target-move rows with structured history:
    - move-event fields (piece/from/to/capture/check/castle/promotion)
    - state deltas (material / king safety proxy / pawn-structure proxy)
- `scripts/script2_encode_policy_samples.py`
  - Old stage-2/3 expected legal-candidate row expansion.
  - New: directly encodes policy samples with:
    - HalfKP-style sparse active indices
    - dense state summary features
    - padded history tensors + mask
    - legal-from mask and legal-to map
- `scripts/script4_finetune_history_policy.py`
  - Old pipeline ended at candidate-ranking trainer.
  - New: history-conditioned factorized policy model adaptation:
    - state encoder (HalfKP sparse + dense)
    - GRU history encoder (events + deltas)
    - factorized legal-masked heads (`from -> to -> promotion`)

## Not carried over (for now)

- Candidate expansion/ranking scripts (`script2_build_candidate_rows.py`, `script3_encode_candidates_nnue.py`, `script4_train_hcmlp_policy.py`).
- Legacy analysis and GUI scripts tied to candidate-ranking checkpoints.

These can be reintroduced later after we adapt them to the new factorized policy checkpoint format.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Reply in the most concise form possible. Skip pleasantries,preambles, and recaps of my question. No phrases like
"I'd be happy to", "Great question", or "Let me explain".Drop articles and filler words wherever the meaning stays clear.
Prefer short declarative sentences. If a tool call is needed,run it first and show only the result. Do not narrate your steps.

## What this is

A tennis match-outcome prediction pipeline. **There are two generations of this pipeline in the
repo; use v2 for anything pre-match/pre-tournament (see below) — the v1 files described later in
this document are legacy, kept for reference, and are NOT pre-match-safe (see
`MODEL_V2_REPORT.md` section 1).**

There is no package manager config (no `requirements.txt`/`pyproject.toml`) — dependencies
(`numpy`, `pandas`, `torch`, `scikit-learn`, `xgboost`, `matplotlib`) must already be available in
the active Python environment. On this machine that's specifically **`/usr/local/bin/python3.12`**
(the default `python3`/`python3.13`/`python3.14` on PATH do NOT have `xgboost` installed) — run all
v2 scripts with that interpreter explicitly.

## v2 pipeline (current, pre-match + pre-tournament) — see MODEL_V2_REPORT.md for full detail

- `data_pipeline_v2.py` — walk-forward, leak-free feature table builder from `tennis_atp/` only
  (ATP/men; no WTA archive exists in this repo, see report sect. 2.3). `PlayerStateTracker` is the
  single source of truth for Elo/rolling-form/H2H/rest-days, snapshotted strictly *before* each
  match updates it. Run once to produce `atp_matches_pretrain.csv` (already generated, 196K rows,
  1968-2026).
- `model_v2.py` — shared feature engineering (`engineer_features`), the PRIMARY model `TennisEmbeddingNet`
  (since v2.3 shipped with `emb_dim=0`: numeric-feature MLP, swap augmentation — the player-identity
  embedding overfit out-of-time, see report sect. 9), and calibration utilities. Imported by all three
  scripts below so training and inference can never silently drift apart (unlike v1's
  `runner.py`/`predict.py` split).
- `train_v2.py` — trains Elo-only / logistic regression / XGBoost / `TennisEmbeddingNet` on a
  strict temporal split (train `<2022`, val `2022`, calib `2023`, test `>=2024`); calibration
  method (raw/isotonic/Platt) per model is *selected* by validation-fold ECE, not hardcoded.
  Artifacts land in `models_v2/`. **`--final` does the two-stage production refit** (stage 1:
  early-stop on 2025-H2 to pick the epoch count; stage 2: retrain on ALL data through Apr 2026
  for that count) into `models_v2_final/` — the benchmark artifact must NOT be used for real
  predictions (its embeddings are frozen pre-2022; see report sect. 7).
- `evaluate_v2.py` — regenerates the reliability diagram + final blind-test metrics from saved
  artifacts.
- `predict_v2.py` — single future-match prediction by player name; replays full history once to
  get current player state, then scores the match with the calibrated NN (deterministic point
  estimate; MC-dropout only for the confidence interval). Auto-prefers `models_v2_final/` if
  present.
- `fetch_bracket.py` — scrape a live draw from diretta.it (Playwright headless) into bracket JSON, re-anchored to only the matches still to be played.
- `rank_bracket.py` — synthetic bracket from the top-N ATP ranking (`--exclude` for withdrawals).
- `app_v2.py` — Streamlit UI over everything above (`/usr/local/bin/python3.12 -m streamlit run app_v2.py`): train + loss/metric plots, predict with error bars + bet verdict from odds, tournament with CI bars, fetch/rank bracket.
- `tournament_v2.py` — Monte Carlo bracket simulator: pre-tournament title/round-advancement
  probabilities from a JSON bracket, with round-by-round update via `results_so_far`.

```bash
/usr/local/bin/python3.12 train_v2.py --data atp_matches_pretrain.csv --out-dir models_v2          # benchmark
/usr/local/bin/python3.12 train_v2.py --data atp_matches_pretrain.csv --out-dir models_v2_final --final  # production
/usr/local/bin/python3.12 evaluate_v2.py --data atp_matches_pretrain.csv --out-dir models_v2
/usr/local/bin/python3.12 predict_v2.py --p1 "Carlos Alcaraz" --p2 "Jannik Sinner" --surface Clay --best-of 3
/usr/local/bin/python3.12 predict_v2.py --p1 "Carlos Alcaraz" --p2 "Jannik Sinner" --matrix  # surface x format grid
/usr/local/bin/python3.12 tournament_v2.py --bracket my_bracket.json --sims 20000
```

---

## v1 pipeline (legacy — NOT pre-match-safe, kept for reference)

A BiLSTM/Transformer neural net blended with a hand-engineered "stat-profile Elo" signal, trained
on the Match Charting Project's point-by-point stats. Dependencies (`numpy`, `pandas`, `torch`,
`scikit-learn`, `matplotlib`) must already be available in the active Python environment.

Two data sources are used, both pulled in as git submodules (see `.gitmodules`):
- `tennis_atp/` — Jeff Sackmann's ATP results/rankings archive (one CSV per year, no shot-level
  stats). Used for Elo computation.
- `tennis_MatchChartingProject/` — point-by-point charted match stats (serve/rally/return
  detail). Used for training the neural net and the stat-profile Elo. Downloaded on demand from
  GitHub if missing locally (see `download_files`/`_ensure_files` in each script).

## Commands

There is no build/lint/test suite — this is a script-driven ML pipeline. Typical workflow:

```bash
# 1. Compute Elo ratings from one of the two data sources
python compute_elo.py --data-dir tennis_atp --out player_elo_atp.json
python compute_elo.py --data-dir tennis_MatchChartingProject --out player_elo.json
python compute_elo.py --data-dir tennis_atp --top 20 --surface Clay
python compute_elo.py --data-dir tennis_atp --player "Carlos Alcaraz"

# 2. Train the neural net (needs charting data; Elo file is optional but recommended)
python runner.py --epochs 40 --elo player_elo.json --data-dir tennis_MatchChartingProject
python runner.py --epochs 40   # stat-Elo only, no external Elo blended in

# 3. Predict a single match (loads tennis_winner_model.pt + player_elo.json)
python predict.py --p1 "Carlos Alcaraz" --p2 "Jannik Sinner" --surface Clay --best-of 3
python predict.py --list-players --surface Clay

# 4. Backtest / evaluate the blended predictor against real results
python live_eval.py --data-dir tennis_MatchChartingProject
python live_eval.py --model tennis_winner_model.pt --elo player_elo.json --alpha 0.25
python live_eval.py --surface Clay --min-matches 5 --out-dir eval_out/
```

`live_eval.py` writes calibration/ROC/confusion-matrix PNGs, a per-match prediction CSV, and a
summary JSON into `--out-dir` (default `eval_out/`).

## Architecture

### The 57-dim feature vector (shared contract across `runner.py` and `predict.py`)

Both files independently define the *same* `FEATURE_DIM = 57` layout — there is no shared module,
so if you change the feature engineering in one file you must mirror it in the other:

```
[0:14]   P1 per-set stats (14 raw stats)         [14:28]  P2 per-set stats
[28:34]  Key stat differentials P1-P2             [34]    Set progress
[35]     1st-serve-won delta (momentum proxy)     [36]    Surface id
[37]     Best-of-5 flag                           [38:40] Running set counts / SEQ_LEN
[40:48]  External Elo features (from player_elo.json — career/surface/recent Elo, z-scored)
[48:57]  Stat-profile Elo features (dominance score, sigmoid win prob, per-stat z-diffs)
```

Sequences are `SEQ_LEN = 5` sets long; the model is a BiLSTM feeding a Transformer encoder with a
CLS token (`TennisMatchNet` in `runner.py`), trained to predict P1-win/P2-win.

### Two independent Elo systems, both feeding the model

1. **`compute_elo.py`** — standalone chronological Elo computer (career / surface / recent /
   blended), auto-detects `tennis_atp` vs Match Charting Project format from `--data-dir`
   contents, and writes a `player_elo_*.json` file. This file is loaded as an *external* signal
   (features 40-47) by both `runner.py` (training) and `predict.py` (inference) — it is not
   recomputed live.
2. **Stat-profile Elo** (features 48-56) — computed live, inline, from per-set charted stats
   (serve/return/rally detail) using surface-specific weighted z-score "dominance" scores fed
   through an Elo-style sigmoid. Defined independently in `runner.py` (`stat_elo_features`) and
   `predict.py`. In `predict.py` this block is recomputed *inside every Monte Carlo sample* so
   noise perturbs it along with the NN output — it's not a static feature.

### Final prediction blend (`predict.py`)

`predict.py` combines three signals rather than relying on the NN alone:
- **A.** NN output, Monte Carlo mean over `--sims` (default 500) noisy forward passes — weight α=0.20
- **B.** Stat-profile Elo → sigmoid — weight β=0.45
- **C.** External Elo file (`player_elo.json`) — weight γ=0.35

The NN's contribution is down-weighted or excluded via `NN_SIGMA_CUTOFF`/`NN_MIN_MATCHES` when its
Monte Carlo variance is high or a player has too few surface matches — the NN is treated as the
least trusted of the three signals by design. **In practice this cutoff excludes the NN from
essentially every prediction** (see `eval_out/eval_summary.json`: `"blended"` metrics are
byte-for-byte identical to `"stat_elo"` metrics) because `build_feature_sequence` feeds it
synthetic noise-perturbed static profiles at inference instead of the real in-match set sequences
it was trained on — a train/inference distribution mismatch, not a tunable threshold issue. Full
analysis: `MODEL_V2_REPORT.md` section 1.1.

Live/updated ratings can be layered on top of the static Elo file with `--elo-live` (merged, live
values take priority) in `runner.py`, `predict.py`, and `live_eval.py` alike.

### `runner.py` contains dead code — read carefully before editing

`runner.py` defines every core function/class twice (`download_files`, `load_and_merge`,
`compute_population_stats`, `stat_elo_features`, `TennisMatchNet`, `train_model`, `main`, etc. all
appear once around lines 120-840 and again ~844-1633), each followed by its own
`if __name__ == "__main__": main()`. Because Python module execution is sequential, running the
script as `__main__` executes *both* `main()` bodies back-to-back — the first `main()` runs using
the first definitions, then everything gets redefined, then the second `main()` runs using the
final definitions.

Practically: the **second (bottom) block's `main()` is what actually determines training
behavior** for a normal `python runner.py` invocation, since it's called last with the final
(overwritten) function bindings. The ATP-repo-aware loader added in the second block
(`_ensure_files` injecting an `__ATP__`/`__atp_dir__` sentinel, `_load_matches_from_atp`) is
**currently dead code**: the live `main()` calls the plain `download_files`/`load_and_merge`
(MCP-only, no ATP branch), and only the *first* block's `load_and_merge` (overwritten and thus
unused) checks for the `__ATP__` sentinel. If you need runner.py to train from `tennis_atp` data
directly, this dead path needs to be reconnected rather than assumed to already work. Treat this
file as needing a de-duplication pass — don't assume both halves are equivalent when editing one.

### Checkpoint format

`save_checkpoint`/`load_checkpoint` in `runner.py` persist not just model weights but everything
needed to reproduce the exact feature vector at inference time: the `StandardScaler` mean/scale,
per-surface population stats (mean/std) for stat-Elo z-scoring, the stat-Elo weight tables, and
Elo normalization params (mean/std). `predict.py` and `live_eval.py` reconstruct their feature
pipeline entirely from this checkpoint plus the Elo JSON — they do not recompute population stats
from scratch.

#!/usr/bin/env python3
"""
data_pipeline_v2.py — Leak-free pre-match feature builder (ATP)
==================================================================
Builds one row per historical match with ONLY information available
strictly BEFORE that match was played (walk-forward / expanding window).

Fixes, relative to compute_elo.py / runner.py:
  - Players are keyed by `player_id` (unique int), never by name string.
    (19 ATP player_ids have >1 distinct name spelling in the raw data —
    name-keying silently fragments a player's history across "ghost"
    identities and corrupts Elo/rolling stats. See CLAUDE.md / report.)
  - Elo, rolling form, H2H, and "days since last match" are all computed
    in a single chronological pass and recorded as their value
    *immediately before* the row's match — never updated with that
    match's own outcome before being stored.
  - The label is the actual match winner (not a proxy stat like
    1st_won_pct), with a randomized P1/P2 side assignment so the label
    is not trivially recoverable from row order or column identity.
  - Every column in the output is something you could know the morning
    of the match: rank, ranking points, age, height, hand, prior Elo,
    prior rolling win rates, prior H2H, rest days, surface/best_of/slam
    metadata. No in-match statistic (serve %, winners, etc.) is used —
    those don't exist before the match is played.

Domain rule for best_of / is_slam (verified on the full 1968-2026 ATP
history, see MODEL_V2_REPORT.md):
  - is_slam (tourney_level == 'G') implies best_of == 5 in 99.0% of
    cases; the exceptions are 1977 US Open early rounds (a real
    historical scheduling quirk, not a data error).
  - best_of == 5 does NOT imply is_slam: Davis Cup live rubbers (level
    'D') were best-of-5 through 2018, and some Tour Finals ('F')
    matches are best-of-5. So best_of and is_slam are kept as two
    independent features, not collapsed into one.
  - There is no full WTA match archive in this repository (tennis_atp
    is ATP/men only; tennis_MatchChartingProject has a small charted
    subset for women with no pre-match ranking data). The schema below
    is gender-aware (a `tour` column) so a WTA source can be plugged in
    later, but today's training set is ATP-only — this is a dataset
    limitation, not a modeling choice, and is documented rather than
    silently ignored.

Usage:
  python data_pipeline_v2.py --data-dir tennis_atp --out atp_matches_pretrain.parquet
"""

import argparse
import glob as _glob
import math
import os
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
ELO_START = 1500.0
RECENT_LAMBDA = 0.7          # same time-decay constant as compute_elo.py, reused for consistency


def kovalchik_k(n_matches: int) -> float:
    return 250.0 / ((n_matches + 5) ** 0.4)


def _normalize_surface(s) -> str:
    s = str(s).strip().title()
    return {
        "Hard Court": "Hard", "Indoor": "Hard", "Hardcourt": "Hard",
        "Acrylic": "Hard", "Outdoor": "Hard",
        "Indoor Hard": "Hard", "Outdoor Hard": "Hard",
    }.get(s, s if s in SURFACES else "Hard")


def _elo_expected(r_i: float, r_j: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))


def _days_between(d1: int, d2: int) -> float:
    try:
        a = datetime.strptime(str(int(d1)), "%Y%m%d")
        b = datetime.strptime(str(int(d2)), "%Y%m%d")
        return (b - a).days
    except (ValueError, TypeError):
        return np.nan


ATP_LEVEL_WEIGHTS = {"G": 1.00, "M": 0.85, "F": 0.90, "A": 0.75, "D": 0.70, "C": 0.55, "S": 0.50, "O": 0.65}

# ── In-match performance stats, used ONLY as rolling pre-match "current
# form / condition" features (mean over the player's PRIOR N matches).
# Never the current match's own values — that would be target leakage.
# Columns exist in tennis_atp from ~1991 onward; older/missing rows yield
# NaN naturally, which XGBoost handles natively (no imputation needed).
STAT_KEYS = [
    "1st_in_pct", "1st_won_pct", "2nd_won_pct", "ace_rate", "df_rate",
    "bp_saved_pct", "sv_pts_won_pct", "ret_pts_won_pct",
]
STAT_WINDOW = 20   # rolling window (in matches) for "current condition"


def _safe_div(num, den):
    try:
        num, den = float(num), float(den)
        return num / den if den > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def _server_stats(svpt, in1, won1, won2, ace, df, bp_saved, bp_faced) -> Dict[str, float]:
    svpt, in1, won1, won2 = float(svpt) if pd.notna(svpt) else np.nan, \
        float(in1) if pd.notna(in1) else np.nan, \
        float(won1) if pd.notna(won1) else np.nan, \
        float(won2) if pd.notna(won2) else np.nan
    return {
        "1st_in_pct":     _safe_div(in1, svpt),
        "1st_won_pct":    _safe_div(won1, in1),
        "2nd_won_pct":    _safe_div(won2, svpt - in1),
        "ace_rate":       _safe_div(ace, svpt),
        "df_rate":        _safe_div(df, svpt),
        "bp_saved_pct":   _safe_div(bp_saved, bp_faced),
        "sv_pts_won_pct": _safe_div(won1 + won2, svpt),
    }


def per_match_player_stats(m) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (winner_stats, loser_stats) for ONE match, each including
    the player's own serve rates plus their return-points-won rate
    (derived from the opponent's serve breakdown)."""
    w = _server_stats(m.get("w_svpt"), m.get("w_1stIn"), m.get("w_1stWon"), m.get("w_2ndWon"),
                       m.get("w_ace"), m.get("w_df"), m.get("w_bpSaved"), m.get("w_bpFaced"))
    l = _server_stats(m.get("l_svpt"), m.get("l_1stIn"), m.get("l_1stWon"), m.get("l_2ndWon"),
                       m.get("l_ace"), m.get("l_df"), m.get("l_bpSaved"), m.get("l_bpFaced"))
    # returner's points won = opponent's serve points NOT won by the opponent
    def _ret_pct(opp_svpt, opp_won1, opp_won2):
        if pd.notna(opp_svpt) and pd.notna(opp_won1) and pd.notna(opp_won2):
            return _safe_div(opp_svpt - opp_won1 - opp_won2, opp_svpt)
        return np.nan
    w["ret_pts_won_pct"] = _ret_pct(m.get("l_svpt"), m.get("l_1stWon"), m.get("l_2ndWon"))
    l["ret_pts_won_pct"] = _ret_pct(m.get("w_svpt"), m.get("w_1stWon"), m.get("w_2ndWon"))
    return w, l


def _rolling_mean(dq: deque) -> Dict[str, float]:
    if not dq:
        return {k: np.nan for k in STAT_KEYS}
    out = {}
    for k in STAT_KEYS:
        vals = [d[k] for d in dq if pd.notna(d.get(k))]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def load_raw_atp(data_dir: str) -> pd.DataFrame:
    files = sorted(_glob.glob(os.path.join(data_dir, "atp_matches_????.csv")))
    if not files:
        raise RuntimeError(f"No atp_matches_YYYY.csv found in '{data_dir}'.")
    chunks = []
    for fp in files:
        try:
            chunks.append(pd.read_csv(fp, on_bad_lines="skip", low_memory=False))
        except Exception as e:
            print(f"  skipping {os.path.basename(fp)}: {e}")
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()

    # ── Data-quality filters (documented, not silent) ──────────────────────
    before = len(df)
    df = df.dropna(subset=["winner_id", "loser_id", "tourney_date", "surface"])
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce")
    df = df[df["best_of"].isin([3, 5])]                      # drop bo=1 exhibition noise (36 rows)
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df["tourney_date"] = df["tourney_date"].astype(int)
    df["surface"] = df["surface"].apply(_normalize_surface)
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    df["is_slam"] = (df["tourney_level"] == "G").astype(int)
    df["is_walkover"] = df["score"].astype(str).str.contains(r"W/O|WEA|DEF", case=False, regex=True).astype(int)
    df["_tw"] = df["tourney_level"].map(ATP_LEVEL_WEIGHTS).fillna(0.65)
    # match_num is per-tournament; sort chronologically, ties broken by match_num for a stable walk-forward order
    df["match_num"] = pd.to_numeric(df.get("match_num", 0), errors="coerce").fillna(0)
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"], kind="mergesort").reset_index(drop=True)
    print(f"  Loaded {before:,} raw rows -> {len(df):,} after quality filters "
          f"(dropped {before - len(df):,}: missing ids/date/surface or best_of not in "
          f"{{3,5}}).")
    return df


class PlayerStateTracker:
    """Single source of truth for walk-forward player state (Elo, rolling
    form, H2H, rest days, last-known bio). Used by BOTH the training-table
    builder below AND predict_v2.py at inference time, so "current state
    of a player" is computed identically in training and in production —
    unlike the old runner.py, which duplicated its feature logic and let
    the two copies drift apart.
    """

    def __init__(self):
        self.career_elo: Dict[int, float] = defaultdict(lambda: ELO_START)
        self.surf_elo: Dict[int, Dict[str, float]] = defaultdict(lambda: {s: ELO_START for s in SURFACES})
        self.n_matches: Dict[int, int] = defaultdict(int)
        self.n_matches_surf: Dict[Tuple[int, str], int] = defaultdict(int)
        self.last_date: Dict[int, int] = {}
        self.recent_all: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self.recent_surf: Dict[Tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=25))
        self.h2h: Dict[Tuple[int, int], int] = defaultdict(int)   # (lo_id, hi_id) -> wins_by_lo_id - wins_by_hi_id
        self.h2h_surf: Dict[Tuple[int, int, str], int] = defaultdict(int)  # same, but per surface
        self.form_stats: Dict[int, deque] = defaultdict(lambda: deque(maxlen=STAT_WINDOW))
        self.last_known_bio: Dict[int, dict] = {}   # most recent rank/points/age/ht/hand seen for a player

    # Pickle support: defaultdicts with lambda factories don't pickle, so
    # freeze to plain dicts on save and re-wrap on load. Lets export_state.py
    # ship the fully-replayed state as one small file for the Streamlit
    # deployment (no tennis_atp/ replay needed at app start).
    def __getstate__(self):
        return {k: dict(v) for k, v in self.__dict__.items()}

    def __setstate__(self, st):
        self.__init__()
        for k, v in st.items():
            getattr(self, k).update(v)

    def snapshot(self, pid: int, surface: str, date: int) -> dict:
        rec = self.recent_all[pid]
        rec_s = self.recent_surf[(pid, surface)]
        rest = date - self.last_date[pid] if pid in self.last_date else np.nan
        snap = dict(
            elo=self.career_elo[pid],
            elo_surf=self.surf_elo[pid][surface],
            n_matches=self.n_matches[pid],
            n_matches_surf=self.n_matches_surf[(pid, surface)],
            winrate_recent=float(np.mean(rec)) if rec else np.nan,
            winrate_recent_surf=float(np.mean(rec_s)) if rec_s else np.nan,
            rest_days=rest,
            stats_n=len(self.form_stats[pid]),
        )
        snap.update({f"form_{k}": v for k, v in _rolling_mean(self.form_stats[pid]).items()})
        return snap

    def h2h_diff(self, pid_a: int, pid_b: int) -> int:
        lo, hi = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
        d = self.h2h[(lo, hi)]
        return d if pid_a == lo else -d

    def h2h_surf_diff(self, pid_a: int, pid_b: int, surface: str) -> int:
        lo, hi = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
        d = self.h2h_surf[(lo, hi, surface)]
        return d if pid_a == lo else -d

    def update(self, w_id: int, l_id: int, surface: str, date: int, tw: float, is_walkover: bool,
               w_stats: Optional[dict] = None, l_stats: Optional[dict] = None):
        rw, rl = self.career_elo[w_id], self.career_elo[l_id]
        ew = _elo_expected(rw, rl)
        kw = kovalchik_k(self.n_matches[w_id]) * tw
        kl = kovalchik_k(self.n_matches[l_id]) * tw
        self.career_elo[w_id] += kw * (1.0 - ew)
        self.career_elo[l_id] += kl * (0.0 - (1.0 - ew))

        sw, sl = self.surf_elo[w_id][surface], self.surf_elo[l_id][surface]
        esw = _elo_expected(sw, sl)
        self.surf_elo[w_id][surface] += kw * (1.0 - esw)
        self.surf_elo[l_id][surface] += kl * (0.0 - (1.0 - esw))

        self.n_matches[w_id] += 1
        self.n_matches[l_id] += 1
        self.n_matches_surf[(w_id, surface)] += 1
        self.n_matches_surf[(l_id, surface)] += 1
        self.last_date[w_id] = date
        self.last_date[l_id] = date
        self.recent_all[w_id].append(1); self.recent_all[l_id].append(0)
        self.recent_surf[(w_id, surface)].append(1); self.recent_surf[(l_id, surface)].append(0)
        lo, hi = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        self.h2h[(lo, hi)] += 1 if w_id == lo else -1
        self.h2h_surf[(lo, hi, surface)] += 1 if w_id == lo else -1

        if not is_walkover and w_stats is not None and l_stats is not None:
            self.form_stats[w_id].append(w_stats)
            self.form_stats[l_id].append(l_stats)

    def set_bio(self, pid: int, rank, rank_points, age, ht, hand):
        self.last_known_bio[pid] = dict(rank=rank, rank_points=rank_points, age=age, ht=ht, hand=hand)

    def bio(self, pid: int) -> dict:
        return self.last_known_bio.get(pid, dict(rank=np.nan, rank_points=np.nan, age=np.nan, ht=np.nan, hand="U"))


def build_pretrain_table(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Single chronological pass: for every match, snapshot each player's
    pre-match state (via PlayerStateTracker), THEN update that state with
    the match outcome."""
    rng = np.random.default_rng(seed)
    tracker = PlayerStateTracker()

    rows = []
    for _, m in df.iterrows():
        w_id, l_id = int(m["winner_id"]), int(m["loser_id"])
        surface = m["surface"]
        date = int(m["tourney_date"])

        # ── snapshot pre-match state for both players (BEFORE any update) ──
        snap_w, snap_l = tracker.snapshot(w_id, surface, date), tracker.snapshot(l_id, surface, date)

        # rank / static bio info (already "as of tourney_date" per tennis_atp README -> pre-match safe)
        w_rank = m.get("winner_rank", np.nan)
        l_rank = m.get("loser_rank", np.nan)
        w_pts = m.get("winner_rank_points", np.nan)
        l_pts = m.get("loser_rank_points", np.nan)
        w_age = m.get("winner_age", np.nan)
        l_age = m.get("loser_age", np.nan)
        w_ht = m.get("winner_ht", np.nan)
        l_ht = m.get("loser_ht", np.nan)
        w_hand = m.get("winner_hand", "U")
        l_hand = m.get("loser_hand", "U")

        h2h_diff_w = tracker.h2h_diff(w_id, l_id)
        h2h_surf_diff_w = tracker.h2h_surf_diff(w_id, l_id, surface)

        # ── randomized side assignment: p1/p2 is NOT winner/loser ──────────
        # This makes y the true match outcome, unrecoverable from row
        # position/column identity alone (unlike the old runner.py pipeline).
        w_is_p1 = bool(rng.integers(0, 2))
        if w_is_p1:
            p1_id, p2_id = w_id, l_id
            snap_p1, snap_p2 = snap_w, snap_l
            p1_rank, p2_rank = w_rank, l_rank
            p1_pts, p2_pts = w_pts, l_pts
            p1_age, p2_age = w_age, l_age
            p1_ht, p2_ht = w_ht, l_ht
            p1_hand, p2_hand = w_hand, l_hand
            h2h_diff_p1 = h2h_diff_w
            h2h_surf_diff_p1 = h2h_surf_diff_w
            y = 1
        else:
            p1_id, p2_id = l_id, w_id
            snap_p1, snap_p2 = snap_l, snap_w
            p1_rank, p2_rank = l_rank, w_rank
            p1_pts, p2_pts = l_pts, w_pts
            p1_age, p2_age = l_age, w_age
            p1_ht, p2_ht = l_ht, w_ht
            p1_hand, p2_hand = l_hand, w_hand
            h2h_diff_p1 = -h2h_diff_w
            h2h_surf_diff_p1 = -h2h_surf_diff_w
            y = 0

        row = dict(
            match_id=f"{m['tourney_id']}_{m['match_num']}",
            date=date,
            surface=surface,
            best_of=int(m["best_of"]),
            is_bo5=int(m["best_of"] == 5),
            is_slam=int(m["is_slam"]),
            tour="ATP",
            tourney_level=m["tourney_level"],
            is_walkover=int(m["is_walkover"]),
            p1_id=p1_id, p2_id=p2_id,
            p1_elo=snap_p1["elo"], p2_elo=snap_p2["elo"],
            p1_elo_surf=snap_p1["elo_surf"], p2_elo_surf=snap_p2["elo_surf"],
            p1_matches=snap_p1["n_matches"], p2_matches=snap_p2["n_matches"],
            p1_matches_surf=snap_p1["n_matches_surf"], p2_matches_surf=snap_p2["n_matches_surf"],
            p1_winrate_recent=snap_p1["winrate_recent"], p2_winrate_recent=snap_p2["winrate_recent"],
            p1_winrate_recent_surf=snap_p1["winrate_recent_surf"], p2_winrate_recent_surf=snap_p2["winrate_recent_surf"],
            p1_rest_days=snap_p1["rest_days"], p2_rest_days=snap_p2["rest_days"],
            p1_rank=p1_rank, p2_rank=p2_rank,
            p1_rank_points=p1_pts, p2_rank_points=p2_pts,
            p1_age=p1_age, p2_age=p2_age,
            p1_ht=p1_ht, p2_ht=p2_ht,
            p1_hand=p1_hand, p2_hand=p2_hand,
            h2h_diff_p1=h2h_diff_p1,
            h2h_surf_diff_p1=h2h_surf_diff_p1,
            p1_form_n=snap_p1["stats_n"], p2_form_n=snap_p2["stats_n"],
            y=y,
        )
        # rolling pre-match "current condition" stats (mean of the player's
        # PRIOR STAT_WINDOW matches' serve/return performance) — never this
        # match's own values.
        for k in STAT_KEYS:
            row[f"p1_form_{k}"] = snap_p1[f"form_{k}"]
            row[f"p2_form_{k}"] = snap_p2[f"form_{k}"]
        rows.append(row)

        # ── update state AFTER snapshotting (walk-forward, no leakage) ─────
        w_stats = l_stats = None
        if not m["is_walkover"]:
            # only real, played matches update "current form" stats — a
            # walkover/retirement's serve numbers (if any) don't reflect
            # actual on-court performance.
            w_stats, l_stats = per_match_player_stats(m)
        tracker.update(w_id, l_id, surface, date, float(m["_tw"]), bool(m["is_walkover"]), w_stats, l_stats)
        tracker.set_bio(w_id, w_rank, w_pts, w_age, w_ht, w_hand)
        tracker.set_bio(l_id, l_rank, l_pts, l_age, l_ht, l_hand)

    out = pd.DataFrame(rows)
    return out, tracker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out", type=str, default="atp_matches_pretrain.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading raw ATP match files...")
    df = load_raw_atp(args.data_dir)

    print("Building walk-forward pre-match feature table (single chronological pass)...")
    out, _tracker = build_pretrain_table(df, seed=args.seed)
    print(f"  Built {len(out):,} match rows | P1-win rate: {out['y'].mean():.4f} "
          f"(should be ~0.50 given randomized side assignment)")

    out.to_csv(args.out, index=False)
    print(f"  Saved -> {args.out}")


if __name__ == "__main__":
    main()

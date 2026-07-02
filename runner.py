#!/usr/bin/env python3
"""
runner.py — Tennis Match Winner Prediction  (BiLSTM + Transformer, fully integrated)
======================================================================================
Feature vector layout per sequence step  (FEATURE_DIM = 57)
─────────────────────────────────────────────────────────────
 [0:14]   P1 per-set stats (14 raw stats)
 [14:28]  P2 per-set stats (14 raw stats)
 [28:34]  Key stat differentials P1−P2  (6 dims)
 [34]     Set progress  (s+1)/SEQ_LEN
 [35]     1st-serve-won delta (within-set momentum proxy)
 [36]     Surface id (normalised)
 [37]     Best-of-5 flag
 [38]     Running P1 set count / SEQ_LEN
 [39]     Running P2 set count / SEQ_LEN

 ── Elo block (constant across all set steps) ──────────────
 [40]     career_elo_p1  (z-scored vs population)
 [41]     career_elo_p2
 [42]     career_elo_diff
 [43]     career_elo_win_prob − 0.5  (centred Elo sigmoid)
 [44]     surface_blended_elo_p1
 [45]     surface_blended_elo_p2
 [46]     surface_elo_diff
 [47]     recent_elo_diff  (last-12-months decay)

 ── Stat-profile Elo block (constant across all set steps) ──
 [48]     weighted dominance score D  (= Σ wj·Δj)
 [49]     stat-Elo win probability  Pstat = sigmoid(D·k)
 [50]     D clipped to [−3, 3]  (robust version)
 [51:57]  per-stat z-score diffs for the 6 highest-weight stats
          (1st_won, bp_save, ret_pts, pressure, 2nd_won, winners)

All three signal families are present in the same feature vector so
the transformer can learn their joint interaction — e.g. "player A has
a +0.8σ 1st-serve-won advantage on grass, but the Elo gap is small,
and recent form reverses it" — rather than combining them as a post-hoc
weighted average.

Prerequisites:
  python compute_elo.py --data-dir <dir>   # produces player_elo.json

Usage:
  python runner.py --epochs 40 --elo player_elo.json \\
                   --data-dir tennis_MatchChartingProject
  python runner.py --epochs 40             # stat-Elo only (no external Elo file)
"""

import argparse
import glob as _glob
import json
import math
import os
import ssl as _ssl
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://github.com/JeffSackmann/tennis_MatchChartingProject"
    "/raw/refs/heads/master/"
)
FILES = [
    "charting-m-matches.csv",
    "charting-m-stats-Overview.csv",
    "charting-m-stats-ServeBasics.csv",
    "charting-m-stats-SnV.csv",
    "charting-m-stats-SvBreakTotal.csv",
]
SURFACES    = ["Hard", "Clay", "Grass", "Carpet", "Indoor Hard", "Outdoor Hard"]
SEQ_LEN     = 5
D_MODEL     = 128

# ── Feature layout ─────────────────────────────────────────────────────────
STAT_DIM        = 14      # raw stats per player
BASE_DIM        = 40      # [0:40]  raw stats + diffs + meta + running sets
ELO_OFFSET      = 40      # [40:48] external Elo file features
ELO_DIM         = 8
STATELO_OFFSET  = 48      # [48:57] stat-profile Elo features
STATELO_DIM     = 9
FEATURE_DIM     = BASE_DIM + ELO_DIM + STATELO_DIM   # 57

# ── Stat-profile Elo pipeline constants ────────────────────────────────────
# Stats used for the z-score dominance score (subset of STAT_COLS, same order)
# Indices into the 14-stat STAT_COLS vector:
#   0  1st_in_pct   1  1st_won_pct   2  2nd_won_pct   3  ace_rate
#   4  df_rate      5  bp_save_pct   6  ret_pts_pct   7  winners_rate
#   8  uf_rate      9  short_rally  10  forced_err    11  snv_win_pct
#  12  snv_rate    13  pressure_win_pct
ELO_STAT_IDX = [1, 5, 6, 13, 2, 7, 8, 4, 10]   # indices into STAT_COLS (9 stats)
ELO_STAT_NAMES = [
    "1st_won_pct", "bp_save_pct", "ret_pts_pct", "pressure_win_pct",
    "2nd_won_pct", "winners_rate", "uf_rate", "df_rate", "forced_err_rate",
]
LOWER_IS_BETTER_IDX = {6, 7, 8}   # positions within ELO_STAT_IDX (uf, df, ferr)
K_SCALE = 0.4

# Surface-specific importance weights (columns = ELO_STAT_NAMES order)
#                      1stwon  bpsave  retpts  press  2ndwon  win    uf     df    ferr
_W = {
    "Hard":   np.array([0.22,   0.14,   0.14,  0.12,  0.10,  0.10,  0.10,  0.04, 0.04]),
    "Clay":   np.array([0.18,   0.16,   0.16,  0.14,  0.10,  0.08,  0.10,  0.04, 0.04]),
    "Grass":  np.array([0.25,   0.12,   0.12,  0.10,  0.10,  0.14,  0.10,  0.04, 0.03]),
    "Carpet": np.array([0.23,   0.13,   0.13,  0.11,  0.10,  0.12,  0.10,  0.04, 0.04]),
}
STAT_WEIGHTS = {s: w / w.sum() for s, w in _W.items()}
STAT_WEIGHTS["Indoor Hard"]  = STAT_WEIGHTS["Hard"]
STAT_WEIGHTS["Outdoor Hard"] = STAT_WEIGHTS["Hard"]


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def download_files(data_dir: str = ".") -> Dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for fname in FILES:
        local = os.path.join(data_dir, fname)
        if not os.path.exists(local):
            print(f"  Downloading {fname} …")
            urllib.request.urlretrieve(BASE_URL + fname, local)
        else:
            print(f"  Cached {fname}")
        paths[fname] = local
    return paths


# ── SAFE HELPERS ──────────────────────────────────────────────────────────────

def _pct(num, den):
    return np.where(den > 0, num / den, np.nan)

def _safe_best_of(val) -> int:
    try:
        v = int(float(str(val).strip()))
        return v if v in (3, 5) else 3
    except (ValueError, TypeError):
        return 3

def _normalize_surface(s: str) -> str:
    s = str(s).strip().title()
    return {"Hard Court": "Hard", "Indoor": "Hard", "Hardcourt": "Hard",
            "Acrylic": "Hard", "Outdoor": "Hard",
            "Indoor Hard": "Hard", "Outdoor Hard": "Hard"}.get(s, s)


# ── ELO FILE LOADER ───────────────────────────────────────────────────────────

def load_elo(path: Optional[str]) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        if path:
            print(f"  ⚠ '{path}' not found — training without external Elo features.")
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded Elo: {len(data)} players from '{path}'")
    return data

def elo_population_stats(elo_data: Dict) -> Tuple[float, float]:
    vals = [v["career_elo"] for v in elo_data.values()
            if isinstance(v.get("career_elo"), (int, float))]
    return (float(np.mean(vals)), max(float(np.std(vals)), 1.0)) if vals else (1500., 150.)


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def load_and_merge(paths: Dict[str, str]) -> pd.DataFrame:
    if paths.get("charting-m-matches.csv") == "__ATP__":
        matches = _load_matches_from_atp(paths["__atp_dir__"])
        matches = matches.rename(columns={"Player 1": "p1", "Player 2": "p2"})
    else:
        matches = pd.read_csv(paths["charting-m-matches.csv"], on_bad_lines="skip")
        matches.columns = matches.columns.str.strip()
        matches = matches.rename(columns={
            "Player 1": "p1", "Player 2": "p2",
            "Best of": "best_of", "Surface": "surface",
        })
    matches["match_id"] = matches["match_id"].astype(str)
    matches["best_of"]  = matches["best_of"].apply(_safe_best_of)
    matches["surface"]  = (matches["surface"].fillna("Hard").astype(str)
                            .apply(_normalize_surface))
    matches = matches[["match_id","p1","p2","best_of","surface"]].copy()

    ov = pd.read_csv(paths["charting-m-stats-Overview.csv"], on_bad_lines="skip")
    ov.columns = ov.columns.str.strip()
    ov = ov[ov["set"] != "Total"].copy()
    ov["set"] = pd.to_numeric(ov["set"], errors="coerce")
    ov = ov.dropna(subset=["set"]); ov["set"] = ov["set"].astype(int)
    ov["match_id"] = ov["match_id"].astype(str)
    for col in ["serve_pts","aces","dfs","first_in","first_won","second_in",
                "second_won","bk_pts","bp_saved","return_pts","return_pts_won",
                "winners","unforced"]:
        ov[col] = pd.to_numeric(ov[col], errors="coerce").fillna(0)
    ov["1st_in_pct"]   = _pct(ov["first_in"],     ov["serve_pts"])
    ov["1st_won_pct"]  = _pct(ov["first_won"],     ov["first_in"])
    ov["2nd_won_pct"]  = _pct(ov["second_won"],    ov["second_in"])
    ov["ace_rate"]     = _pct(ov["aces"],           ov["serve_pts"])
    ov["df_rate"]      = _pct(ov["dfs"],            ov["serve_pts"])
    ov["bp_save_pct"]  = _pct(ov["bp_saved"],       ov["bk_pts"])
    ov["ret_pts_pct"]  = _pct(ov["return_pts_won"], ov["return_pts"])
    ov["winners_rate"] = _pct(ov["winners"],        ov["serve_pts"]+ov["return_pts"])
    ov["uf_rate"]      = _pct(ov["unforced"],       ov["serve_pts"]+ov["return_pts"])

    sb = pd.read_csv(paths["charting-m-stats-ServeBasics.csv"], on_bad_lines="skip")
    sb.columns = sb.columns.str.strip()
    sb["match_id"] = sb["match_id"].astype(str)
    # Detect column names defensively (header varies across repo snapshots)
    _row_col  = next((c for c in sb.columns if c.lower() == "row"), None)
    _pts_base = next((c for c in sb.columns if c.lower() in ("pts", "points")), None)
    _pts_col  = next((c for c in sb.columns
                      if "lte_3" in c.lower() or ("short" in c.lower() and "pts" in c.lower())
                      or "won_lte" in c.lower()), None)
    _ferr_col = next((c for c in sb.columns
                      if "forced_err" in c.lower() or c.lower() == "forced_err"), None)
    if _row_col:
        sb = sb[sb[_row_col].isin(["1st", "2nd"])].copy()
    sb_grp2 = sb.groupby(["match_id", "player"]).size().reset_index(name="_n2")
    if _pts_col and _pts_base:
        for _c in [_pts_base, _pts_col]:
            sb[_c] = pd.to_numeric(sb[_c], errors="coerce").fillna(0)
        _stmp = sb.groupby(["match_id","player"])[[_pts_base, _pts_col]].sum().reset_index()
        _stmp["short_rally_pct"] = _pct(_stmp[_pts_col], _stmp[_pts_base])
        sb_grp2 = sb_grp2.merge(_stmp[["match_id","player","short_rally_pct"]],
                                  on=["match_id","player"], how="left")
    else:
        sb_grp2["short_rally_pct"] = np.nan
    if _ferr_col and _pts_base:
        sb[_ferr_col] = pd.to_numeric(sb[_ferr_col], errors="coerce").fillna(0)
        _ftmp = sb.groupby(["match_id","player"])[[_pts_base, _ferr_col]].sum().reset_index()
        _ftmp["forced_err_rate"] = _pct(_ftmp[_ferr_col], _ftmp[_pts_base])
        sb_grp2 = sb_grp2.merge(_ftmp[["match_id","player","forced_err_rate"]],
                                  on=["match_id","player"], how="left")
    else:
        sb_grp2["forced_err_rate"] = np.nan
    sb_agg = sb_grp2[["match_id","player","short_rally_pct","forced_err_rate"]].copy()

    snv = pd.read_csv(paths["charting-m-stats-SnV.csv"], on_bad_lines="skip")
    snv.columns = snv.columns.str.strip()
    snv = snv[snv["row"] == "SnV"].copy()
    snv["match_id"] = snv["match_id"].astype(str)
    for col in ["snv_pts","pts_won"]:
        snv[col] = pd.to_numeric(snv[col], errors="coerce").fillna(0)
    snv["snv_win_pct"] = _pct(snv["pts_won"], snv["snv_pts"])
    snv["snv_rate"]    = snv["snv_pts"].clip(0)
    snv_agg = snv[["match_id","player","snv_win_pct","snv_rate"]].copy()

    svbk = pd.read_csv(paths["charting-m-stats-SvBreakTotal.csv"], on_bad_lines="skip")
    svbk.columns = svbk.columns.str.strip()
    svbk["match_id"] = svbk["match_id"].astype(str)
    svbk_d = svbk[svbk["row"] == "d"].copy()
    for col in ["pts","pts_won"]:
        svbk_d[col] = pd.to_numeric(svbk_d[col], errors="coerce").fillna(0)
    svbk_d["pressure_win_pct"] = _pct(svbk_d["pts_won"], svbk_d["pts"])
    svbk_agg = svbk_d[["match_id","player","pressure_win_pct"]].copy()

    ov_feats = ["1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
                "bp_save_pct","ret_pts_pct","winners_rate","uf_rate"]
    df = ov[["match_id","player","set"]+ov_feats].copy()
    df = df.merge(sb_agg,   on=["match_id","player"], how="left")
    df = df.merge(snv_agg,  on=["match_id","player"], how="left")
    df = df.merge(svbk_agg, on=["match_id","player"], how="left")
    df = df.merge(matches,  on="match_id",            how="inner")
    return df


# ── POPULATION STATS FOR z-SCORE ─────────────────────────────────────────────

def compute_population_stats(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-stat population mean and std for each surface.
    Uses per-match player means (not per-set) so each player-match pair
    contributes equally regardless of how many sets were played.
    Returns dict: surface → (mean_9, std_9) for ELO_STAT_NAMES.
    """
    STAT_COLS_14 = [
        "1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
        "bp_save_pct","ret_pts_pct","winners_rate","uf_rate",
        "short_rally_pct","forced_err_rate","snv_win_pct","snv_rate","pressure_win_pct",
    ]
    pop = {}
    for surf in SURFACES:
        sub = df[df["surface"] == surf]
        if sub.empty:
            pop[surf] = (np.zeros(9, dtype=np.float32),
                         np.ones(9,  dtype=np.float32))
            continue
        # per-match player mean
        pm = (sub.groupby(["match_id","player"])[STAT_COLS_14]
              .mean().reset_index())
        # extract the 9 Elo stats by index
        vals = pm[[STAT_COLS_14[i] for i in ELO_STAT_IDX]].values.astype(np.float32)
        mean = np.nanmean(vals, axis=0)
        std  = np.nanstd(vals,  axis=0)
        std  = np.maximum(std, 1e-4)
        pop[surf] = (mean.astype(np.float32), std.astype(np.float32))
    return pop


# ── STAT-PROFILE ELO FEATURES ─────────────────────────────────────────────────

def stat_elo_features(
    r1_14: np.ndarray,   # P1 stats for this set  (14-dim)
    r2_14: np.ndarray,   # P2 stats for this set  (14-dim)
    pop_mean: np.ndarray, pop_std: np.ndarray,  # 9-dim population stats
    surface: str,
) -> np.ndarray:
    """
    Compute the 9-dim stat-profile Elo feature block for a single set step.

    Returns:
      [0]   D score (weighted dominance)
      [1]   Pstat  (Elo sigmoid of D)
      [2]   D clipped to [−3, 3]  (robust version seen by model)
      [3:9] per-stat z-score diffs for 6 highest-weight stats
    """
    surf_key = _normalize_surface(surface)
    w = STAT_WEIGHTS.get(surf_key, STAT_WEIGHTS["Hard"])

    # Extract the 9 Elo stats from the 14-dim vectors
    v1 = r1_14[ELO_STAT_IDX]   # shape (9,)
    v2 = r2_14[ELO_STAT_IDX]

    # Step 2: per-stat z-score diff
    delta = np.zeros(9, dtype=np.float32)
    for j in range(9):
        d = (v1[j] - v2[j]) / pop_std[j]
        delta[j] = -d if j in LOWER_IS_BETTER_IDX else d

    # Step 3: weighted dominance score
    D = float(np.nansum(w * delta))

    # Step 4: Elo sigmoid
    Pstat = float(np.clip(1.0 / (1.0 + 10.0 ** (-D * K_SCALE)), 0.01, 0.99))

    feat = np.zeros(STATELO_DIM, dtype=np.float32)
    feat[0] = D
    feat[1] = Pstat
    feat[2] = float(np.clip(D, -3.0, 3.0))
    feat[3:9] = delta[:6]   # top-6 stats by weight order
    return feat


# ── EXTERNAL ELO FEATURES ─────────────────────────────────────────────────────

def external_elo_features(
    p1: str, p2: str, surface: str,
    elo_data: Optional[Dict],
    elo_mean: float, elo_std: float,
) -> np.ndarray:
    """8-dim external Elo feature block. Zeros if player unknown."""
    feat = np.zeros(ELO_DIM, dtype=np.float32)
    if elo_data is None or p1 not in elo_data or p2 not in elo_data:
        return feat
    surf_key = _normalize_surface(surface)
    d1, d2 = elo_data[p1], elo_data[p2]
    c1 = float(d1.get("career_elo", elo_mean))
    c2 = float(d2.get("career_elo", elo_mean))
    feat[0] = (c1 - elo_mean) / elo_std
    feat[1] = (c2 - elo_mean) / elo_std
    feat[2] = (c1 - c2)       / elo_std
    feat[3] = 1.0 / (1.0 + 10.0 ** ((c2 - c1) / 400.0)) - 0.5
    s1 = float(d1.get("blended_elo", {}).get(surf_key, c1))
    s2 = float(d2.get("blended_elo", {}).get(surf_key, c2))
    feat[4] = (s1 - elo_mean) / elo_std
    feat[5] = (s2 - elo_mean) / elo_std
    feat[6] = (s1 - s2)       / elo_std
    r1 = float(d1.get("recent_elo", c1))
    r2 = float(d2.get("recent_elo", c2))
    feat[7] = (r1 - r2) / elo_std
    return feat


# ── SEQUENCE BUILDER ─────────────────────────────────────────────────────────

def build_match_sequences(
    df: pd.DataFrame,
    pop_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    elo_data:  Optional[Dict] = None,
    elo_mean:  float = 1500.0,
    elo_std:   float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays — one sample per match.

    X : (n_matches, SEQ_LEN, FEATURE_DIM=57)
    y : (n_matches,)   0=P1 wins   1=P2 wins

    All three signal families are embedded in the feature vector:
      [0:40]  raw stats + diffs + meta (original runner.py layout)
      [40:48] external Elo features (from player_elo.json)
      [48:57] stat-profile Elo features (computed from population stats)
    """
    STAT_COLS = [
        "1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
        "bp_save_pct","ret_pts_pct","winners_rate","uf_rate",
        "short_rally_pct","forced_err_rate","snv_win_pct","snv_rate","pressure_win_pct",
    ]
    N           = len(STAT_COLS)   # 14
    surface_map = {s: i for i, s in enumerate(SURFACES)}
    use_elo     = elo_data is not None
    n_elo_found = 0

    X_list, y_list = [], []

    for match_id, mdf in df.groupby("match_id"):
        meta    = mdf.iloc[0]
        p1      = str(meta["p1"])
        p2      = str(meta["p2"])
        surface = str(meta.get("surface","Hard")).strip()
        surf_id = surface_map.get(surface, 0)
        best_of = _safe_best_of(meta.get("best_of", 3))

        sp1 = mdf[mdf["player"] == p1].sort_values("set")
        sp2 = mdf[mdf["player"] == p2].sort_values("set")
        if sp1.empty or sp2.empty:
            continue
        n_sets = min(len(sp1), len(sp2), SEQ_LEN)
        if n_sets < 2:
            continue

        # ── Constant blocks (same for every set step) ────────────────
        # External Elo
        ext_elo = external_elo_features(p1, p2, surface,
                                         elo_data, elo_mean, elo_std)
        if use_elo and np.any(ext_elo != 0):
            n_elo_found += 1

        # Population stats for this surface
        pop_m, pop_s = pop_stats.get(surface,
                                      pop_stats.get("Hard",
                                      (np.zeros(9,dtype=np.float32),
                                       np.ones(9, dtype=np.float32))))

        seq      = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        p1_sets  = p2_sets = 0

        for s in range(n_sets):
            r1   = sp1.iloc[s][STAT_COLS].values.astype(np.float32)
            r2   = sp2.iloc[s][STAT_COLS].values.astype(np.float32)
            diff = r1 - r2

            if r1[1] > r2[1]: p1_sets += 1
            else:              p2_sets += 1

            feat = np.zeros(FEATURE_DIM, dtype=np.float32)

            # ── [0:40] base stats + meta ──────────────────────────────
            feat[:N]     = r1
            feat[N:2*N]  = r2
            feat[28:34]  = diff[:6]
            feat[34]     = (s+1) / SEQ_LEN
            feat[35]     = np.clip(r1[1]-r2[1], -1, 1)
            feat[36]     = surf_id / max(len(SURFACES)-1, 1)
            feat[37]     = float(best_of == 5)
            feat[38]     = p1_sets / SEQ_LEN
            feat[39]     = p2_sets / SEQ_LEN

            # ── [40:48] external Elo ──────────────────────────────────
            feat[ELO_OFFSET:ELO_OFFSET+ELO_DIM] = ext_elo

            # ── [48:57] stat-profile Elo (live, per set step) ─────────
            # Uses the cumulative per-set stats available at this point
            # in the match — the model learns to update its dominance
            # estimate as more sets are played.
            stat_elo_feat = stat_elo_features(
                np.nan_to_num(r1), np.nan_to_num(r2),
                pop_m, pop_s, surface,
            )
            feat[STATELO_OFFSET:STATELO_OFFSET+STATELO_DIM] = stat_elo_feat

            np.nan_to_num(feat, copy=False)
            seq[s] = feat

        p1_avg = float(np.nanmean(sp1["1st_won_pct"].values))
        p2_avg = float(np.nanmean(sp2["1st_won_pct"].values))
        y_list.append(0 if p1_avg >= p2_avg else 1)
        X_list.append(seq)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    elo_cov = (f"{n_elo_found}/{len(X_list)} "
               f"({100*n_elo_found/max(len(X_list),1):.1f}%)"
               if use_elo else "disabled")
    print(f"  Built {len(X):,} match samples | "
          f"P1-label rate: {y.mean():.3f} | Elo coverage: {elo_cov}")
    return X, y


# ── DATASET ───────────────────────────────────────────────────────────────────

class TennisMatchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── MODEL ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TennisMatchNet(nn.Module):
    """
    Hybrid BiLSTM + Transformer for tennis match winner prediction.

    With FEATURE_DIM=57 the model jointly sees:
      - Per-set raw stats (sequential dynamics, momentum, comebacks)
      - External Elo ratings (pre-match strength prior)
      - Stat-profile Elo score (interpretable dominance signal)

    The transformer's attention can learn to weight these signals
    differently depending on how the match is unfolding — for example,
    trusting Elo more in the first set and relying on live stats in the
    fifth.
    """
    def __init__(
        self,
        feature_dim: int   = FEATURE_DIM,
        d_model:     int   = D_MODEL,
        lstm_layers: int   = 2,
        nhead:       int   = 4,
        tf_layers:   int   = 2,
        dropout:     float = 0.25,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model     = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            d_model, d_model//2, lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*4, dropout, "gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=tf_layers)
        self.gate = nn.Linear(d_model*2, d_model*2)
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.input_proj(x)
        _, (hn, _) = self.lstm(h)
        lstm_out = torch.cat([hn[-2], hn[-1]], dim=-1)
        cls      = self.cls_token.expand(B, -1, -1)
        t_in     = self.pos_enc(torch.cat([cls, h], dim=1))
        cls_out  = self.transformer(t_in)[:, 0]
        combined = torch.cat([lstm_out, cls_out], dim=-1)
        return self.head(combined * torch.sigmoid(self.gate(combined)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray, y: np.ndarray,
    epochs: int = 40, batch_size: int = 32,
    lr: float = 1e-3, device: str = "cpu",
) -> Tuple:
    B, T, F = X.shape
    idx_tr, idx_val, y_tr, y_val = train_test_split(
        np.arange(B), y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X[idx_tr].reshape(-1, F))

    def scale(arr):
        b, t, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(b, t, f).astype(np.float32)

    train_dl = DataLoader(TennisMatchDataset(scale(X[idx_tr]), y_tr),
                          batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TennisMatchDataset(scale(X[idx_val]), y_val),
                          batch_size=batch_size)

    model    = TennisMatchNet(feature_dim=F).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}  |  feature_dim={F}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history   = {"train_loss":[], "val_loss":[], "val_acc":[]}
    best_acc  = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total += loss.item() * len(xb)

        model.eval(); vl = vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item() * len(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(yb)
        tl = total / len(train_dl.dataset)
        vl = vl / len(val_dl.dataset)
        va = vc / vt
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        best_acc = max(best_acc, va)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  acc={va:.3f}")

    print(f"\n  ✓ Best val acc : {best_acc:.3f}  (final: {history['val_acc'][-1]:.3f})")
    return model, history, scaler


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model, scaler,
    pop_stats: Dict,
    elo_mean:  float = 1500.0,
    elo_std:   float = 150.0,
    elo_enabled: bool = False,
    path: str = "tennis_winner_model.pt",
):
    """
    Save everything predict.py needs to reproduce the exact same feature vector:
      - model weights + architecture params
      - scaler mean/std
      - population stats per surface  (for stat-Elo z-scoring)
      - Elo normalisation params
    """
    # Serialise pop_stats: dict of surface → (mean_list, std_list)
    pop_serial = {
        surf: {"mean": m.tolist(), "std": s.tolist()}
        for surf, (m, s) in pop_stats.items()
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim":      model.feature_dim,
        "d_model":          model.d_model,
        "scaler_mean":      scaler.mean_,
        "scaler_scale":     scaler.scale_,
        # Stat-Elo pipeline
        "pop_stats":        pop_serial,
        "elo_stat_idx":     ELO_STAT_IDX,
        "stat_weights":     {s: w.tolist() for s, w in STAT_WEIGHTS.items()},
        "k_scale":          K_SCALE,
        "statelo_offset":   STATELO_OFFSET,
        "statelo_dim":      STATELO_DIM,
        # External Elo
        "elo_enabled":      elo_enabled,
        "elo_mean":         elo_mean,
        "elo_std":          elo_std,
        "elo_offset":       ELO_OFFSET,
        "elo_dim":          ELO_DIM,
    }, path)
    print(f"  Saved → {path}")


def load_checkpoint(path: str = "tennis_winner_model.pt", device: str = "cpu"):
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    model  = TennisMatchNet(
        feature_dim=ckpt.get("feature_dim", FEATURE_DIM),
        d_model    =ckpt.get("d_model",     D_MODEL),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    scaler        = StandardScaler()
    scaler.mean_  = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    # Restore pop_stats
    pop_stats = {}
    for surf, v in ckpt.get("pop_stats", {}).items():
        pop_stats[surf] = (np.array(v["mean"], dtype=np.float32),
                           np.array(v["std"],  dtype=np.float32))
    return model, scaler, ckpt, pop_stats


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--epochs",     type=int,   default=40)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--device",     type=str,   default="cpu")
    ap.add_argument("--data-dir",   type=str,   default=".")
    ap.add_argument("--elo",        type=str,   default="player_elo.json",
                    help="Static Elo file from compute_elo.py (pass '' to disable)")
    ap.add_argument("--elo-live",   type=str,   default=None,
                    help="Live/updated Elo .json (same structure as --elo). "
                         "Merged on top of --elo; live values take priority.")
    ap.add_argument("--out",        type=str,   default="tennis_winner_model.pt")
    args = ap.parse_args()

    elo_path = args.elo if args.elo else None

    print(f"\n{'='*62}")
    print("  Tennis Match Winner Prediction — BiLSTM + Transformer")
    print(f"  FEATURE_DIM : {FEATURE_DIM}  "
          f"(40 base + 8 ext-Elo + 9 stat-Elo)")
    print(f"  Device : {args.device}  Epochs: {args.epochs}  "
          f"Batch: {args.batch_size}")
    print(f"  Elo file : {elo_path or 'DISABLED'}")
    if getattr(args, 'elo_live', None):
        print(f"  Elo-live : {args.elo_live}")
    print(f"{'='*62}\n")

    print("[1/5] Downloading data …")
    paths = download_files(args.data_dir)

    print("\n[2/5] Loading Elo ratings …")
    elo_data = load_elo(elo_path)
    elo_mean, elo_std = (elo_population_stats(elo_data)
                         if elo_data else (1500., 150.))
    if elo_data:
        print(f"  Population Elo — mean={elo_mean:.1f}  std={elo_std:.1f}")

    # ── Optional live Elo override ─────────────────────────────────────
    _lp = getattr(args, 'elo_live', None)
    if _lp:
        if not os.path.exists(_lp):
            print(f"  ⚠ --elo-live '{_lp}' not found — ignored.")
        elif not _lp.lower().endswith('.json'):
            print(f"  ⚠ --elo-live: only .json supported (got '{_lp}') — ignored.")
        else:
            with open(_lp) as _f:
                _live = json.load(_f)
            if elo_data is None:
                elo_data = _live
            else:
                elo_data.update(_live)  # live takes priority
            _vals = [v['career_elo'] for v in elo_data.values()
                     if isinstance(v.get('career_elo'), (int, float))]
            if _vals:
                elo_mean = float(np.mean(_vals))
                elo_std  = max(float(np.std(_vals)), 1.0)
            print(f"  Elo-live: {len(_live)} players merged "
                  f"(total {len(elo_data)}, mean={elo_mean:.0f} std={elo_std:.0f})")

    print("\n[3/5] Engineering features …")
    df = load_and_merge(paths)
    print(f"  {len(df):,} player×set rows | "
          f"{df['match_id'].nunique():,} matches")

    print("\n[4/5] Computing population stats for stat-Elo z-scoring …")
    pop_stats = compute_population_stats(df)
    for surf, (m, s) in pop_stats.items():
        if m.sum() > 0:
            print(f"  {surf:<14s}  mean 1st_won={m[0]:.3f}  "
                  f"std 1st_won={s[0]:.4f}")

    print("\n[5/5] Building match sequences …")
    X, y = build_match_sequences(df, pop_stats,
                                  elo_data=elo_data,
                                  elo_mean=elo_mean,
                                  elo_std=elo_std)

    print("\n[6/6] Training …")
    model, history, scaler = train_model(
        X, y, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    save_checkpoint(
        model, scaler, pop_stats,
        elo_mean=elo_mean, elo_std=elo_std,
        elo_enabled=(elo_data is not None),
        path=args.out,
    )
    with open("training_history.json","w") as f:
        json.dump(history, f, indent=2)
    print("  History → training_history.json")

    # Quick demo
    print("\n[Demo] Predictions on first 5 validation samples:")
    val_idx = int(0.8 * len(X))
    x_demo  = torch.tensor(X[val_idx:val_idx+5],
                            dtype=torch.float32).to(args.device)
    probs   = model.predict_proba(x_demo)
    for prob, lbl in zip(probs, y[val_idx:val_idx+5]):
        pred  = "P1" if prob[0] > 0.5 else "P2"
        truth = "P1" if lbl == 0 else "P2"
        mark  = "✓" if pred == truth else "✗"
        print(f"  [{mark}] P1={prob[0]:.3f}  P2={prob[1]:.3f}  "
              f"pred={pred}  truth={truth}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()


def _ensure_files(data_dir: str) -> Dict[str, str]:
    """Resolve stat-file paths.

    • If *data_dir* contains ``atp_matches_YYYY.csv`` files the directory is
      treated as a Jeff Sackmann **tennis_atp** clone.  The charting shot-level
      CSVs (Overview, ServeBasics, SnV, SvBreakTotal) are still needed for per-
      set stats; only the matches list is sourced from the ATP files instead of
      ``charting-m-matches.csv``.  An ``__atp_dir__`` sentinel is injected so
      ``load_player_stats`` / ``build_sequences`` can call
      ``_load_matches_from_atp()`` instead of reading the charting matches CSV.

    • Otherwise the original MCP flow is used with an SSL-bypass download
      (workaround for macOS Python 3.12 certificate issues).
    """
    os.makedirs(data_dir, exist_ok=True)
    atp_files = sorted(_glob.glob(os.path.join(data_dir, "atp_matches_????.csv")))
    paths: Dict[str, str] = {}

    for fname in STAT_FILES:
        local = os.path.join(data_dir, fname)
        if fname == "charting-m-matches.csv" and atp_files:
            # ATP repo: skip downloading the MCP matches file; we'll build it
            # on-the-fly from atp_matches_*.csv via _load_matches_from_atp().
            paths[fname] = "__ATP__"
            continue
        if not os.path.exists(local):
            print(f"  Downloading {fname} …")
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(BASE_URL + fname, local)
        paths[fname] = local

    if atp_files:
        paths["__atp_dir__"] = data_dir   # sentinel consumed by loaders
        print(f"  ATP mode: {len(atp_files)} atp_matches_*.csv files detected.")
    return paths


def _load_matches_from_atp(data_dir: str) -> "pd.DataFrame":
    """Load winner/loser/surface/date from all atp_matches_YYYY.csv files and
    return a DataFrame with the same column names used by the MCP matches CSV:
    match_id, p1 (winner), p2 (loser), surface, best_of, Date.
    """
    atp_files = sorted(_glob.glob(os.path.join(data_dir, "atp_matches_????.csv")))
    chunks = []
    for fp in atp_files:
        try:
            chunks.append(pd.read_csv(fp, on_bad_lines="skip", low_memory=False))
        except Exception:
            pass
    if not chunks:
        raise RuntimeError(f"No atp_matches_*.csv files readable in '{data_dir}'.")
    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "winner_name": "p1", "loser_name":   "p2",
        "tourney_date": "Date", "best_of": "Best of",
        "surface": "Surface", "tourney_id": "_tid",
        "match_num": "_mnum",
    })
    tid  = df.get("_tid",  pd.Series(range(len(df)), dtype=str))
    mnum = df.get("_mnum", pd.Series(range(len(df)), dtype=str)).astype(str)
    df["match_id"] = tid.astype(str) + "_" + mnum
    df["Date"]     = pd.to_numeric(df["Date"], errors="coerce")
    df["best_of"]  = df.get("Best of", pd.Series(3)).apply(_safe_best_of)
    df["surface"]  = df.get("Surface", pd.Series("Hard")).fillna("Hard").astype(str).apply(_normalize_surface)
    df             = df.rename(columns={"p1": "Player 1", "p2": "Player 2"})
    df             = df.dropna(subset=["Player 1", "Player 2", "Date"])
    df             = df[df["Player 1"].str.strip() != ""]
    df             = df[df["Player 2"].str.strip() != ""]
    return df.sort_values("Date", ascending=True).reset_index(drop=True)

def download_files(data_dir: str = ".") -> Dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for fname in FILES:
        local = os.path.join(data_dir, fname)
        if not os.path.exists(local):
            print(f"  Downloading {fname} …")
            urllib.request.urlretrieve(BASE_URL + fname, local)
        else:
            print(f"  Cached {fname}")
        paths[fname] = local
    return paths


# ── SAFE HELPERS ──────────────────────────────────────────────────────────────

def _pct(num, den):
    return np.where(den > 0, num / den, np.nan)

def _safe_best_of(val) -> int:
    try:
        v = int(float(str(val).strip()))
        return v if v in (3, 5) else 3
    except (ValueError, TypeError):
        return 3

def _normalize_surface(s: str) -> str:
    s = str(s).strip().title()
    return {"Hard Court": "Hard", "Indoor": "Hard", "Hardcourt": "Hard",
            "Acrylic": "Hard", "Outdoor": "Hard",
            "Indoor Hard": "Hard", "Outdoor Hard": "Hard"}.get(s, s)


# ── ELO FILE LOADER ───────────────────────────────────────────────────────────

def load_elo(path: Optional[str]) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        if path:
            print(f"  ⚠ '{path}' not found — training without external Elo features.")
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded Elo: {len(data)} players from '{path}'")
    return data

def elo_population_stats(elo_data: Dict) -> Tuple[float, float]:
    vals = [v["career_elo"] for v in elo_data.values()
            if isinstance(v.get("career_elo"), (int, float))]
    return (float(np.mean(vals)), max(float(np.std(vals)), 1.0)) if vals else (1500., 150.)


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def load_and_merge(paths: Dict[str, str]) -> pd.DataFrame:
    matches = pd.read_csv(paths["charting-m-matches.csv"], on_bad_lines="skip")
    matches.columns = matches.columns.str.strip()
    matches = matches.rename(columns={
        "Player 1": "p1", "Player 2": "p2",
        "Best of": "best_of", "Surface": "surface",
    })
    matches["match_id"] = matches["match_id"].astype(str)
    matches["best_of"]  = matches["best_of"].apply(_safe_best_of)
    matches["surface"]  = (matches["surface"].fillna("Hard").astype(str)
                            .apply(_normalize_surface))
    matches = matches[["match_id","p1","p2","best_of","surface"]].copy()

    ov = pd.read_csv(paths["charting-m-stats-Overview.csv"], on_bad_lines="skip")
    ov.columns = ov.columns.str.strip()
    ov = ov[ov["set"] != "Total"].copy()
    ov["set"] = pd.to_numeric(ov["set"], errors="coerce")
    ov = ov.dropna(subset=["set"]); ov["set"] = ov["set"].astype(int)
    ov["match_id"] = ov["match_id"].astype(str)
    for col in ["serve_pts","aces","dfs","first_in","first_won","second_in",
                "second_won","bk_pts","bp_saved","return_pts","return_pts_won",
                "winners","unforced"]:
        ov[col] = pd.to_numeric(ov[col], errors="coerce").fillna(0)
    ov["1st_in_pct"]   = _pct(ov["first_in"],     ov["serve_pts"])
    ov["1st_won_pct"]  = _pct(ov["first_won"],     ov["first_in"])
    ov["2nd_won_pct"]  = _pct(ov["second_won"],    ov["second_in"])
    ov["ace_rate"]     = _pct(ov["aces"],           ov["serve_pts"])
    ov["df_rate"]      = _pct(ov["dfs"],            ov["serve_pts"])
    ov["bp_save_pct"]  = _pct(ov["bp_saved"],       ov["bk_pts"])
    ov["ret_pts_pct"]  = _pct(ov["return_pts_won"], ov["return_pts"])
    ov["winners_rate"] = _pct(ov["winners"],        ov["serve_pts"]+ov["return_pts"])
    ov["uf_rate"]      = _pct(ov["unforced"],       ov["serve_pts"]+ov["return_pts"])

    sb = pd.read_csv(paths["charting-m-stats-ServeBasics.csv"], on_bad_lines="skip")
    sb.columns = sb.columns.str.strip()
    sb["match_id"] = sb["match_id"].astype(str)
    # Detect column names defensively (header varies across repo snapshots)
    _row_col  = next((c for c in sb.columns if c.lower() == "row"), None)
    _pts_base = next((c for c in sb.columns if c.lower() in ("pts", "points")), None)
    _pts_col  = next((c for c in sb.columns
                      if "lte_3" in c.lower() or ("short" in c.lower() and "pts" in c.lower())
                      or "won_lte" in c.lower()), None)
    _ferr_col = next((c for c in sb.columns
                      if "forced_err" in c.lower() or c.lower() == "forced_err"), None)
    if _row_col:
        sb = sb[sb[_row_col].isin(["1st", "2nd"])].copy()
    sb_grp2 = sb.groupby(["match_id", "player"]).size().reset_index(name="_n2")
    if _pts_col and _pts_base:
        for _c in [_pts_base, _pts_col]:
            sb[_c] = pd.to_numeric(sb[_c], errors="coerce").fillna(0)
        _stmp = sb.groupby(["match_id","player"])[[_pts_base, _pts_col]].sum().reset_index()
        _stmp["short_rally_pct"] = _pct(_stmp[_pts_col], _stmp[_pts_base])
        sb_grp2 = sb_grp2.merge(_stmp[["match_id","player","short_rally_pct"]],
                                  on=["match_id","player"], how="left")
    else:
        sb_grp2["short_rally_pct"] = np.nan
    if _ferr_col and _pts_base:
        sb[_ferr_col] = pd.to_numeric(sb[_ferr_col], errors="coerce").fillna(0)
        _ftmp = sb.groupby(["match_id","player"])[[_pts_base, _ferr_col]].sum().reset_index()
        _ftmp["forced_err_rate"] = _pct(_ftmp[_ferr_col], _ftmp[_pts_base])
        sb_grp2 = sb_grp2.merge(_ftmp[["match_id","player","forced_err_rate"]],
                                  on=["match_id","player"], how="left")
    else:
        sb_grp2["forced_err_rate"] = np.nan
    sb_agg = sb_grp2[["match_id","player","short_rally_pct","forced_err_rate"]].copy()

    snv = pd.read_csv(paths["charting-m-stats-SnV.csv"], on_bad_lines="skip")
    snv.columns = snv.columns.str.strip()
    snv = snv[snv["row"] == "SnV"].copy()
    snv["match_id"] = snv["match_id"].astype(str)
    for col in ["snv_pts","pts_won"]:
        snv[col] = pd.to_numeric(snv[col], errors="coerce").fillna(0)
    snv["snv_win_pct"] = _pct(snv["pts_won"], snv["snv_pts"])
    snv["snv_rate"]    = snv["snv_pts"].clip(0)
    snv_agg = snv[["match_id","player","snv_win_pct","snv_rate"]].copy()

    svbk = pd.read_csv(paths["charting-m-stats-SvBreakTotal.csv"], on_bad_lines="skip")
    svbk.columns = svbk.columns.str.strip()
    svbk["match_id"] = svbk["match_id"].astype(str)
    svbk_d = svbk[svbk["row"] == "d"].copy()
    for col in ["pts","pts_won"]:
        svbk_d[col] = pd.to_numeric(svbk_d[col], errors="coerce").fillna(0)
    svbk_d["pressure_win_pct"] = _pct(svbk_d["pts_won"], svbk_d["pts"])
    svbk_agg = svbk_d[["match_id","player","pressure_win_pct"]].copy()

    ov_feats = ["1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
                "bp_save_pct","ret_pts_pct","winners_rate","uf_rate"]
    df = ov[["match_id","player","set"]+ov_feats].copy()
    df = df.merge(sb_agg,   on=["match_id","player"], how="left")
    df = df.merge(snv_agg,  on=["match_id","player"], how="left")
    df = df.merge(svbk_agg, on=["match_id","player"], how="left")
    df = df.merge(matches,  on="match_id",            how="inner")
    return df


# ── POPULATION STATS FOR z-SCORE ─────────────────────────────────────────────

def compute_population_stats(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-stat population mean and std for each surface.
    Uses per-match player means (not per-set) so each player-match pair
    contributes equally regardless of how many sets were played.
    Returns dict: surface → (mean_9, std_9) for ELO_STAT_NAMES.
    """
    STAT_COLS_14 = [
        "1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
        "bp_save_pct","ret_pts_pct","winners_rate","uf_rate",
        "short_rally_pct","forced_err_rate","snv_win_pct","snv_rate","pressure_win_pct",
    ]
    pop = {}
    for surf in SURFACES:
        sub = df[df["surface"] == surf]
        if sub.empty:
            pop[surf] = (np.zeros(9, dtype=np.float32),
                         np.ones(9,  dtype=np.float32))
            continue
        # per-match player mean
        pm = (sub.groupby(["match_id","player"])[STAT_COLS_14]
              .mean().reset_index())
        # extract the 9 Elo stats by index
        vals = pm[[STAT_COLS_14[i] for i in ELO_STAT_IDX]].values.astype(np.float32)
        mean = np.nanmean(vals, axis=0)
        std  = np.nanstd(vals,  axis=0)
        std  = np.maximum(std, 1e-4)
        pop[surf] = (mean.astype(np.float32), std.astype(np.float32))
    return pop


# ── STAT-PROFILE ELO FEATURES ─────────────────────────────────────────────────

def stat_elo_features(
    r1_14: np.ndarray,   # P1 stats for this set  (14-dim)
    r2_14: np.ndarray,   # P2 stats for this set  (14-dim)
    pop_mean: np.ndarray, pop_std: np.ndarray,  # 9-dim population stats
    surface: str,
) -> np.ndarray:
    """
    Compute the 9-dim stat-profile Elo feature block for a single set step.

    Returns:
      [0]   D score (weighted dominance)
      [1]   Pstat  (Elo sigmoid of D)
      [2]   D clipped to [−3, 3]  (robust version seen by model)
      [3:9] per-stat z-score diffs for 6 highest-weight stats
    """
    surf_key = _normalize_surface(surface)
    w = STAT_WEIGHTS.get(surf_key, STAT_WEIGHTS["Hard"])

    # Extract the 9 Elo stats from the 14-dim vectors
    v1 = r1_14[ELO_STAT_IDX]   # shape (9,)
    v2 = r2_14[ELO_STAT_IDX]

    # Step 2: per-stat z-score diff
    delta = np.zeros(9, dtype=np.float32)
    for j in range(9):
        d = (v1[j] - v2[j]) / pop_std[j]
        delta[j] = -d if j in LOWER_IS_BETTER_IDX else d

    # Step 3: weighted dominance score
    D = float(np.nansum(w * delta))

    # Step 4: Elo sigmoid
    Pstat = float(np.clip(1.0 / (1.0 + 10.0 ** (-D * K_SCALE)), 0.01, 0.99))

    feat = np.zeros(STATELO_DIM, dtype=np.float32)
    feat[0] = D
    feat[1] = Pstat
    feat[2] = float(np.clip(D, -3.0, 3.0))
    feat[3:9] = delta[:6]   # top-6 stats by weight order
    return feat


# ── EXTERNAL ELO FEATURES ─────────────────────────────────────────────────────

def external_elo_features(
    p1: str, p2: str, surface: str,
    elo_data: Optional[Dict],
    elo_mean: float, elo_std: float,
) -> np.ndarray:
    """8-dim external Elo feature block. Zeros if player unknown."""
    feat = np.zeros(ELO_DIM, dtype=np.float32)
    if elo_data is None or p1 not in elo_data or p2 not in elo_data:
        return feat
    surf_key = _normalize_surface(surface)
    d1, d2 = elo_data[p1], elo_data[p2]
    c1 = float(d1.get("career_elo", elo_mean))
    c2 = float(d2.get("career_elo", elo_mean))
    feat[0] = (c1 - elo_mean) / elo_std
    feat[1] = (c2 - elo_mean) / elo_std
    feat[2] = (c1 - c2)       / elo_std
    feat[3] = 1.0 / (1.0 + 10.0 ** ((c2 - c1) / 400.0)) - 0.5
    s1 = float(d1.get("blended_elo", {}).get(surf_key, c1))
    s2 = float(d2.get("blended_elo", {}).get(surf_key, c2))
    feat[4] = (s1 - elo_mean) / elo_std
    feat[5] = (s2 - elo_mean) / elo_std
    feat[6] = (s1 - s2)       / elo_std
    r1 = float(d1.get("recent_elo", c1))
    r2 = float(d2.get("recent_elo", c2))
    feat[7] = (r1 - r2) / elo_std
    return feat


# ── SEQUENCE BUILDER ─────────────────────────────────────────────────────────

def build_match_sequences(
    df: pd.DataFrame,
    pop_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    elo_data:  Optional[Dict] = None,
    elo_mean:  float = 1500.0,
    elo_std:   float = 150.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays — one sample per match.

    X : (n_matches, SEQ_LEN, FEATURE_DIM=57)
    y : (n_matches,)   0=P1 wins   1=P2 wins

    All three signal families are embedded in the feature vector:
      [0:40]  raw stats + diffs + meta (original runner.py layout)
      [40:48] external Elo features (from player_elo.json)
      [48:57] stat-profile Elo features (computed from population stats)
    """
    STAT_COLS = [
        "1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
        "bp_save_pct","ret_pts_pct","winners_rate","uf_rate",
        "short_rally_pct","forced_err_rate","snv_win_pct","snv_rate","pressure_win_pct",
    ]
    N           = len(STAT_COLS)   # 14
    surface_map = {s: i for i, s in enumerate(SURFACES)}
    use_elo     = elo_data is not None
    n_elo_found = 0

    X_list, y_list = [], []

    for match_id, mdf in df.groupby("match_id"):
        meta    = mdf.iloc[0]
        p1      = str(meta["p1"])
        p2      = str(meta["p2"])
        surface = str(meta.get("surface","Hard")).strip()
        surf_id = surface_map.get(surface, 0)
        best_of = _safe_best_of(meta.get("best_of", 3))

        sp1 = mdf[mdf["player"] == p1].sort_values("set")
        sp2 = mdf[mdf["player"] == p2].sort_values("set")
        if sp1.empty or sp2.empty:
            continue
        n_sets = min(len(sp1), len(sp2), SEQ_LEN)
        if n_sets < 2:
            continue

        # ── Constant blocks (same for every set step) ────────────────
        # External Elo
        ext_elo = external_elo_features(p1, p2, surface,
                                         elo_data, elo_mean, elo_std)
        if use_elo and np.any(ext_elo != 0):
            n_elo_found += 1

        # Population stats for this surface
        pop_m, pop_s = pop_stats.get(surface,
                                      pop_stats.get("Hard",
                                      (np.zeros(9,dtype=np.float32),
                                       np.ones(9, dtype=np.float32))))

        seq      = np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        p1_sets  = p2_sets = 0

        for s in range(n_sets):
            r1   = sp1.iloc[s][STAT_COLS].values.astype(np.float32)
            r2   = sp2.iloc[s][STAT_COLS].values.astype(np.float32)
            diff = r1 - r2

            if r1[1] > r2[1]: p1_sets += 1
            else:              p2_sets += 1

            feat = np.zeros(FEATURE_DIM, dtype=np.float32)

            # ── [0:40] base stats + meta ──────────────────────────────
            feat[:N]     = r1
            feat[N:2*N]  = r2
            feat[28:34]  = diff[:6]
            feat[34]     = (s+1) / SEQ_LEN
            feat[35]     = np.clip(r1[1]-r2[1], -1, 1)
            feat[36]     = surf_id / max(len(SURFACES)-1, 1)
            feat[37]     = float(best_of == 5)
            feat[38]     = p1_sets / SEQ_LEN
            feat[39]     = p2_sets / SEQ_LEN

            # ── [40:48] external Elo ──────────────────────────────────
            feat[ELO_OFFSET:ELO_OFFSET+ELO_DIM] = ext_elo

            # ── [48:57] stat-profile Elo (live, per set step) ─────────
            # Uses the cumulative per-set stats available at this point
            # in the match — the model learns to update its dominance
            # estimate as more sets are played.
            stat_elo_feat = stat_elo_features(
                np.nan_to_num(r1), np.nan_to_num(r2),
                pop_m, pop_s, surface,
            )
            feat[STATELO_OFFSET:STATELO_OFFSET+STATELO_DIM] = stat_elo_feat

            np.nan_to_num(feat, copy=False)
            seq[s] = feat

        p1_avg = float(np.nanmean(sp1["1st_won_pct"].values))
        p2_avg = float(np.nanmean(sp2["1st_won_pct"].values))
        y_list.append(0 if p1_avg >= p2_avg else 1)
        X_list.append(seq)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    elo_cov = (f"{n_elo_found}/{len(X_list)} "
               f"({100*n_elo_found/max(len(X_list),1):.1f}%)"
               if use_elo else "disabled")
    print(f"  Built {len(X):,} match samples | "
          f"P1-label rate: {y.mean():.3f} | Elo coverage: {elo_cov}")
    return X, y


# ── DATASET ───────────────────────────────────────────────────────────────────

class TennisMatchDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── MODEL ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TennisMatchNet(nn.Module):
    """
    Hybrid BiLSTM + Transformer for tennis match winner prediction.

    With FEATURE_DIM=57 the model jointly sees:
      - Per-set raw stats (sequential dynamics, momentum, comebacks)
      - External Elo ratings (pre-match strength prior)
      - Stat-profile Elo score (interpretable dominance signal)

    The transformer's attention can learn to weight these signals
    differently depending on how the match is unfolding — for example,
    trusting Elo more in the first set and relying on live stats in the
    fifth.
    """
    def __init__(
        self,
        feature_dim: int   = FEATURE_DIM,
        d_model:     int   = D_MODEL,
        lstm_layers: int   = 2,
        nhead:       int   = 4,
        tf_layers:   int   = 2,
        dropout:     float = 0.25,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model     = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            d_model, d_model//2, lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*4, dropout, "gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=tf_layers)
        self.gate = nn.Linear(d_model*2, d_model*2)
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.input_proj(x)
        _, (hn, _) = self.lstm(h)
        lstm_out = torch.cat([hn[-2], hn[-1]], dim=-1)
        cls      = self.cls_token.expand(B, -1, -1)
        t_in     = self.pos_enc(torch.cat([cls, h], dim=1))
        cls_out  = self.transformer(t_in)[:, 0]
        combined = torch.cat([lstm_out, cls_out], dim=-1)
        return self.head(combined * torch.sigmoid(self.gate(combined)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray, y: np.ndarray,
    epochs: int = 40, batch_size: int = 32,
    lr: float = 1e-3, device: str = "cpu",
) -> Tuple:
    B, T, F = X.shape
    idx_tr, idx_val, y_tr, y_val = train_test_split(
        np.arange(B), y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X[idx_tr].reshape(-1, F))

    def scale(arr):
        b, t, f = arr.shape
        return scaler.transform(arr.reshape(-1, f)).reshape(b, t, f).astype(np.float32)

    train_dl = DataLoader(TennisMatchDataset(scale(X[idx_tr]), y_tr),
                          batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TennisMatchDataset(scale(X[idx_val]), y_val),
                          batch_size=batch_size)

    model    = TennisMatchNet(feature_dim=F).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}  |  feature_dim={F}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history   = {"train_loss":[], "val_loss":[], "val_acc":[]}
    best_acc  = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total += loss.item() * len(xb)

        model.eval(); vl = vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item() * len(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(yb)
        tl = total / len(train_dl.dataset)
        vl = vl / len(val_dl.dataset)
        va = vc / vt
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        best_acc = max(best_acc, va)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  acc={va:.3f}")

    print(f"\n  ✓ Best val acc : {best_acc:.3f}  (final: {history['val_acc'][-1]:.3f})")
    return model, history, scaler


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model, scaler,
    pop_stats: Dict,
    elo_mean:  float = 1500.0,
    elo_std:   float = 150.0,
    elo_enabled: bool = False,
    path: str = "tennis_winner_model.pt",
):
    """
    Save everything predict.py needs to reproduce the exact same feature vector:
      - model weights + architecture params
      - scaler mean/std
      - population stats per surface  (for stat-Elo z-scoring)
      - Elo normalisation params
    """
    # Serialise pop_stats: dict of surface → (mean_list, std_list)
    pop_serial = {
        surf: {"mean": m.tolist(), "std": s.tolist()}
        for surf, (m, s) in pop_stats.items()
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim":      model.feature_dim,
        "d_model":          model.d_model,
        "scaler_mean":      scaler.mean_,
        "scaler_scale":     scaler.scale_,
        # Stat-Elo pipeline
        "pop_stats":        pop_serial,
        "elo_stat_idx":     ELO_STAT_IDX,
        "stat_weights":     {s: w.tolist() for s, w in STAT_WEIGHTS.items()},
        "k_scale":          K_SCALE,
        "statelo_offset":   STATELO_OFFSET,
        "statelo_dim":      STATELO_DIM,
        # External Elo
        "elo_enabled":      elo_enabled,
        "elo_mean":         elo_mean,
        "elo_std":          elo_std,
        "elo_offset":       ELO_OFFSET,
        "elo_dim":          ELO_DIM,
    }, path)
    print(f"  Saved → {path}")


def load_checkpoint(path: str = "tennis_winner_model.pt", device: str = "cpu"):
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    model  = TennisMatchNet(
        feature_dim=ckpt.get("feature_dim", FEATURE_DIM),
        d_model    =ckpt.get("d_model",     D_MODEL),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    scaler        = StandardScaler()
    scaler.mean_  = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    # Restore pop_stats
    pop_stats = {}
    for surf, v in ckpt.get("pop_stats", {}).items():
        pop_stats[surf] = (np.array(v["mean"], dtype=np.float32),
                           np.array(v["std"],  dtype=np.float32))
    return model, scaler, ckpt, pop_stats


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--epochs",     type=int,   default=40)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--device",     type=str,   default="cpu")
    ap.add_argument("--data-dir",   type=str,   default=".")
    ap.add_argument("--elo",        type=str,   default="player_elo.json",
                    help="Static Elo file from compute_elo.py (pass '' to disable)")
    ap.add_argument("--elo-live",   type=str,   default=None,
                    help="Live/updated Elo .json (same structure as --elo). "
                         "Merged on top of --elo; live values take priority.")
    ap.add_argument("--out",        type=str,   default="tennis_winner_model.pt")
    args = ap.parse_args()

    elo_path = args.elo if args.elo else None

    print(f"\n{'='*62}")
    print("  Tennis Match Winner Prediction — BiLSTM + Transformer")
    print(f"  FEATURE_DIM : {FEATURE_DIM}  "
          f"(40 base + 8 ext-Elo + 9 stat-Elo)")
    print(f"  Device : {args.device}  Epochs: {args.epochs}  "
          f"Batch: {args.batch_size}")
    print(f"  Elo file : {elo_path or 'DISABLED'}")
    if getattr(args, 'elo_live', None):
        print(f"  Elo-live : {args.elo_live}")
    print(f"{'='*62}\n")

    print("[1/5] Downloading data …")
    paths = download_files(args.data_dir)

    print("\n[2/5] Loading Elo ratings …")
    elo_data = load_elo(elo_path)
    elo_mean, elo_std = (elo_population_stats(elo_data)
                         if elo_data else (1500., 150.))
    if elo_data:
        print(f"  Population Elo — mean={elo_mean:.1f}  std={elo_std:.1f}")

    # ── Optional live Elo override ─────────────────────────────────────
    _lp = getattr(args, 'elo_live', None)
    if _lp:
        if not os.path.exists(_lp):
            print(f"  ⚠ --elo-live '{_lp}' not found — ignored.")
        elif not _lp.lower().endswith('.json'):
            print(f"  ⚠ --elo-live: only .json supported (got '{_lp}') — ignored.")
        else:
            with open(_lp) as _f:
                _live = json.load(_f)
            if elo_data is None:
                elo_data = _live
            else:
                elo_data.update(_live)  # live takes priority
            _vals = [v['career_elo'] for v in elo_data.values()
                     if isinstance(v.get('career_elo'), (int, float))]
            if _vals:
                elo_mean = float(np.mean(_vals))
                elo_std  = max(float(np.std(_vals)), 1.0)
            print(f"  Elo-live: {len(_live)} players merged "
                  f"(total {len(elo_data)}, mean={elo_mean:.0f} std={elo_std:.0f})")

    print("\n[3/5] Engineering features …")
    df = load_and_merge(paths)
    print(f"  {len(df):,} player×set rows | "
          f"{df['match_id'].nunique():,} matches")

    print("\n[4/5] Computing population stats for stat-Elo z-scoring …")
    pop_stats = compute_population_stats(df)
    for surf, (m, s) in pop_stats.items():
        if m.sum() > 0:
            print(f"  {surf:<14s}  mean 1st_won={m[0]:.3f}  "
                  f"std 1st_won={s[0]:.4f}")

    print("\n[5/5] Building match sequences …")
    X, y = build_match_sequences(df, pop_stats,
                                  elo_data=elo_data,
                                  elo_mean=elo_mean,
                                  elo_std=elo_std)

    print("\n[6/6] Training …")
    model, history, scaler = train_model(
        X, y, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    save_checkpoint(
        model, scaler, pop_stats,
        elo_mean=elo_mean, elo_std=elo_std,
        elo_enabled=(elo_data is not None),
        path=args.out,
    )
    with open("training_history.json","w") as f:
        json.dump(history, f, indent=2)
    print("  History → training_history.json")

    # Quick demo
    print("\n[Demo] Predictions on first 5 validation samples:")
    val_idx = int(0.8 * len(X))
    x_demo  = torch.tensor(X[val_idx:val_idx+5],
                            dtype=torch.float32).to(args.device)
    probs   = model.predict_proba(x_demo)
    for prob, lbl in zip(probs, y[val_idx:val_idx+5]):
        pred  = "P1" if prob[0] > 0.5 else "P2"
        truth = "P1" if lbl == 0 else "P2"
        mark  = "✓" if pred == truth else "✗"
        print(f"  [{mark}] P1={prob[0]:.3f}  P2={prob[1]:.3f}  "
              f"pred={pred}  truth={truth}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()

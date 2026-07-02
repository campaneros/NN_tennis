#!/usr/bin/env python3
"""
predict.py — Tennis Match Winner Predictor
============================================
Feature vector layout (FEATURE_DIM = 57, mirrors runner.py exactly):
  [0:14]   P1 per-set stats
  [14:28]  P2 per-set stats
  [28:34]  Key stat differentials P1−P2
  [34]     Set progress
  [35]     1st-serve-won delta (momentum proxy)
  [36]     Surface id (normalised)
  [37]     Best-of-5 flag
  [38]     Running P1 set count / SEQ_LEN
  [39]     Running P2 set count / SEQ_LEN
  [40:48]  External Elo features (from player_elo.json)
  [48:57]  Stat-profile Elo features — computed LIVE inside each MC step

The stat-profile Elo block [48:57] is recomputed per Monte-Carlo simulation
with the same Gaussian noise applied to r1/r2, so the MC ensemble captures
uncertainty on the stat-Elo signal too, not just the NN output.

Final win probability = blend of three signals:
  A. NN (Monte-Carlo mean over 500 noise samples)   α = 0.20
  B. Stat-profile Elo  (D-score → sigmoid)          β = 0.45
  C. External Elo file                               γ = 0.35

Usage:
  python predict.py --p1 "Carlos Alcaraz" --p2 "Jannik Sinner" \\
                    --surface Clay --best-of 3 --elo player_elo.json
  python predict.py --p1 "Djokovic" --p2 "Alcaraz" --surface Hard
  python predict.py --list-players --surface Clay
"""

import argparse, json, math, os, urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://github.com/JeffSackmann/tennis_MatchChartingProject"
    "/raw/refs/heads/master/"
)
STAT_FILES = [
    "charting-m-matches.csv",
    "charting-m-stats-Overview.csv",
    "charting-m-stats-ServeBasics.csv",
    "charting-m-stats-SnV.csv",
    "charting-m-stats-SvBreakTotal.csv",
]
SURFACES    = ["Hard", "Clay", "Grass", "Carpet", "Indoor Hard", "Outdoor Hard"]
SEQ_LEN     = 5
D_MODEL     = 128

# Feature layout (must match runner.py)
N_STATS        = 14
BASE_DIM       = 40
ELO_OFFSET     = 40;  ELO_DIM     = 8
STATELO_OFFSET = 48;  STATELO_DIM = 9
FEATURE_DIM    = BASE_DIM + ELO_DIM + STATELO_DIM   # 57

# Stat-profile Elo pipeline
K_SCALE     = 0.4
# Indices into STAT_COLS for the 9-stat Elo subset:
#  1st_won(1) bp_save(5) ret_pts(6) pressure(13) 2nd_won(2)
#  winners(7) uf_rate(8) df_rate(4) forced_err(10)
ELO_STAT_IDX            = [1, 5, 6, 13, 2, 7, 8, 4, 10]
LOWER_IS_BETTER_STATELO = {6, 7, 8}   # positions within ELO_STAT_IDX

# Surface weights for the 9-stat dominance score
_W9 = {
    "Hard":   np.array([0.22, 0.14, 0.14, 0.12, 0.10, 0.10, 0.10, 0.04, 0.04]),
    "Clay":   np.array([0.18, 0.16, 0.16, 0.14, 0.10, 0.08, 0.10, 0.04, 0.04]),
    "Grass":  np.array([0.25, 0.12, 0.12, 0.10, 0.10, 0.14, 0.10, 0.04, 0.03]),
    "Carpet": np.array([0.23, 0.13, 0.13, 0.11, 0.10, 0.12, 0.10, 0.04, 0.04]),
}
_W9["Indoor Hard"]  = _W9["Hard"]
_W9["Outdoor Hard"] = _W9["Hard"]
STATELO_WEIGHTS = {s: w / w.sum() for s, w in _W9.items()}

# Full-14-stat surface weights (for stat_elo_prob standalone)
STAT_COLS = [
    "1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
    "bp_save_pct","ret_pts_pct","winners_rate","uf_rate",
    "short_rally_pct","forced_err_rate","snv_win_pct","snv_rate","pressure_win_pct",
]
STAT_LABELS = [
    "1st serve %","1st srv won %","2nd srv won %","Ace rate","DF rate",
    "BP save %","Return pts %","Winners rate","Unforced err",
    "Short rally %","Forced err rate","SnV win %","SnV rate","Pressure win %",
]
LOWER_IS_BETTER = {"DF rate", "Unforced err", "Forced err rate"}
LOWER_IDX       = {4, 8, 10}   # df_rate, uf_rate, forced_err_rate in STAT_COLS

STAT_WEIGHTS_14: Dict[str, np.ndarray] = {
    "Clay":   np.array([0.06,0.14,0.10,0.02,0.05,0.14,0.14,0.07,0.08,0.04,0.05,0.03,0.02,0.14]),
    "Hard":   np.array([0.07,0.18,0.10,0.04,0.05,0.12,0.12,0.09,0.07,0.06,0.04,0.04,0.02,0.12]),
    "Grass":  np.array([0.06,0.22,0.09,0.07,0.04,0.10,0.10,0.12,0.06,0.08,0.03,0.06,0.02,0.09]),
    "Carpet": np.array([0.06,0.20,0.09,0.06,0.04,0.11,0.11,0.11,0.06,0.07,0.03,0.05,0.02,0.10]),
}
STAT_WEIGHTS_14["Indoor Hard"]  = STAT_WEIGHTS_14["Hard"]
STAT_WEIGHTS_14["Outdoor Hard"] = STAT_WEIGHTS_14["Hard"]

# Final blend weights
ALPHA_NN   = 0.20
ALPHA_STAT = 0.45
ALPHA_ELO  = 0.35

# NN reliability thresholds
NN_SIGMA_CUTOFF  = 0.20   # σ ≥ 20pp → exclude NN from blend entirely
NN_MIN_MATCHES   = 20     # fewer matches on surface → extra data penalty


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _ensure_files(data_dir: str) -> Dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for fname in STAT_FILES:
        local = os.path.join(data_dir, fname)
        if not os.path.exists(local):
            print(f"  Downloading {fname} …")
            urllib.request.urlretrieve(BASE_URL + fname, local)
        paths[fname] = local
    return paths

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
    return {"Hard Court":"Hard","Indoor":"Hard","Hardcourt":"Hard",
            "Acrylic":"Hard","Outdoor":"Hard",
            "Indoor Hard":"Hard","Outdoor Hard":"Hard"}.get(s, s)

def _interactive_pick(name: str, candidates: List[str]) -> str:
    print(f"\n  Ambiguous name '{name}'. Multiple matches:")
    for i, c in enumerate(candidates, 1):
        print(f"    [{i}] {c}")
    while True:
        try:
            idx = int(input(f"  Pick (1-{len(candidates)}): ").strip()) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except (ValueError, EOFError):
            pass
        print("  Invalid — enter a number.")

def _fuzzy_player(name: str, known: List[str]) -> str:
    name_lower = name.strip().lower()
    for k in known:
        if k.lower() == name_lower:
            return k
    matches = [k for k in known if name_lower in k.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        last  = name_lower.split()[-1]
        exact = [k for k in matches if k.lower().split()[-1] == last]
        if len(exact) == 1:
            return exact[0]
        return _interactive_pick(name, sorted(matches))
    words = name_lower.split()
    sug   = sorted({k for k in known if any(w in k.lower() for w in words)})
    msg   = f"Player '{name}' not found."
    if sug:
        msg += "\n  Did you mean:\n" + "\n".join(f"    {s}" for s in sug[:10])
    raise ValueError(msg)


# ── STAT LOADER ───────────────────────────────────────────────────────────────

def load_player_stats(
    paths: Dict[str, str], player: str, surface: str,
    best_of: int, n_recent: int = 50,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load per-set stat rows for `player` on `surface`.
    Returns (matrix, recency_weights, n_matches).
    Falls back to all surfaces if fewer than 10 rows found on the target surface.
    """
    MIN_ROWS = 10

    matches = pd.read_csv(paths["charting-m-matches.csv"], on_bad_lines="skip")
    matches.columns = matches.columns.str.strip()
    matches = matches.rename(columns={"Player 1":"p1","Player 2":"p2",
                                       "Best of":"best_of","Surface":"surface"})
    matches["match_id"] = matches["match_id"].astype(str)
    matches["best_of"]  = matches["best_of"].apply(_safe_best_of)
    matches["surface"]  = matches["surface"].fillna("Hard").astype(str).apply(_normalize_surface)
    matches["Date"]     = pd.to_numeric(matches["Date"], errors="coerce")

    # Overview stats
    ov = pd.read_csv(paths["charting-m-stats-Overview.csv"], on_bad_lines="skip")
    ov.columns = ov.columns.str.strip()
    ov = ov[ov["set"] != "Total"].copy()
    ov["set"]      = pd.to_numeric(ov["set"], errors="coerce")
    ov             = ov.dropna(subset=["set"]); ov["set"] = ov["set"].astype(int)
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

    # Serve basics
    sb = pd.read_csv(paths["charting-m-stats-ServeBasics.csv"], on_bad_lines="skip")
    sb.columns = sb.columns.str.strip()
    sb["match_id"] = sb["match_id"].astype(str)
    pts_col  = next((c for c in sb.columns if "lte_3" in c.lower() or "short" in c.lower()), None)
    ferr_col = next((c for c in sb.columns if "forced" in c.lower()), None)
    pts_base = next((c for c in sb.columns if c.lower() in ("pts","points")), None)
    sb_grp = sb.groupby(["match_id","player"]).size().reset_index(name="_n")
    # Compute short_rally_pct
    if pts_col and pts_base:
        sb[pts_base] = pd.to_numeric(sb[pts_base], errors="coerce").fillna(0)
        sb[pts_col]  = pd.to_numeric(sb[pts_col],  errors="coerce").fillna(0)
        _tmp = sb.groupby(["match_id","player"])[[pts_base, pts_col]].sum().reset_index()
        _tmp["short_rally_pct"] = _pct(_tmp[pts_col], _tmp[pts_base])
        sb_grp = sb_grp.merge(_tmp[["match_id","player","short_rally_pct"]],
                               on=["match_id","player"], how="left")
    else:
        sb_grp["short_rally_pct"] = np.nan
    # Compute forced_err_rate
    if ferr_col and pts_base:
        sb[ferr_col] = pd.to_numeric(sb[ferr_col], errors="coerce").fillna(0)
        _tmp2 = sb.groupby(["match_id","player"])[[pts_base, ferr_col]].sum().reset_index()
        _tmp2["forced_err_rate"] = _pct(_tmp2[ferr_col], _tmp2[pts_base])
        sb_grp = sb_grp.merge(_tmp2[["match_id","player","forced_err_rate"]],
                               on=["match_id","player"], how="left")
    else:
        sb_grp["forced_err_rate"] = np.nan

    # SnV
    snv = pd.read_csv(paths["charting-m-stats-SnV.csv"], on_bad_lines="skip")
    snv.columns = snv.columns.str.strip()
    snv["match_id"] = snv["match_id"].astype(str)
    snv_pts_col = next((c for c in snv.columns if "snv_pts" in c.lower()
                        or ("snv" in c.lower() and "pt" in c.lower())), None)
    won_col     = next((c for c in snv.columns if "pts_won" in c.lower()), None)
    snv_row_col = next((c for c in snv.columns if "row" in c.lower()), None)
    if snv_row_col:
        snv = snv[snv[snv_row_col] == "SnV"].copy()
    if snv_pts_col and won_col:
        snv[snv_pts_col] = pd.to_numeric(snv[snv_pts_col], errors="coerce").fillna(0)
        snv[won_col]     = pd.to_numeric(snv[won_col],     errors="coerce").fillna(0)
        snv_grp = snv.groupby(["match_id","player"])[[snv_pts_col, won_col]].sum().reset_index()
        snv_grp["snv_win_pct"] = _pct(snv_grp[won_col], snv_grp[snv_pts_col])
        snv_grp["snv_rate"]    = snv_grp[snv_pts_col]
    else:
        snv_grp = snv.groupby(["match_id","player"]).size().reset_index(name="_n")
        snv_grp["snv_win_pct"] = np.nan; snv_grp["snv_rate"] = np.nan

    # Pressure (SvBreakTotal, row=="d")
    svbk = pd.read_csv(paths["charting-m-stats-SvBreakTotal.csv"], on_bad_lines="skip")
    svbk.columns = svbk.columns.str.strip()
    svbk["match_id"] = svbk["match_id"].astype(str)
    row_col = next((c for c in svbk.columns if "row" in c.lower()), None)
    if row_col:
        svbk = svbk[svbk[row_col] == "d"].copy()
    sv_pts = next((c for c in svbk.columns if c.lower() in ("pts","points")), None)
    sv_won = next((c for c in svbk.columns if "pts_won" in c.lower()), None)
    if sv_pts and sv_won:
        svbk[sv_pts] = pd.to_numeric(svbk[sv_pts], errors="coerce").fillna(0)
        svbk[sv_won] = pd.to_numeric(svbk[sv_won], errors="coerce").fillna(0)
        svbk_grp = svbk.groupby(["match_id","player"])[[sv_pts, sv_won]].sum().reset_index()
        svbk_grp["pressure_win_pct"] = _pct(svbk_grp[sv_won], svbk_grp[sv_pts])
    else:
        svbk_grp = svbk.groupby(["match_id","player"]).size().reset_index(name="_n")
        svbk_grp["pressure_win_pct"] = np.nan

    # Merge everything
    base_cols = ["1st_in_pct","1st_won_pct","2nd_won_pct","ace_rate","df_rate",
                 "bp_save_pct","ret_pts_pct","winners_rate","uf_rate"]
    player_ov = ov[ov["player"] == _fuzzy_player(player, ov["player"].unique().tolist())]
    if player_ov.empty:
        raise ValueError(f"No stats found for '{player}'.")
    pname = player_ov["player"].iloc[0]

    df = player_ov[["match_id","set"]+base_cols].copy()
    df = df.merge(matches[["match_id","surface","best_of","Date"]], on="match_id", how="left")
    df = df.merge(sb_grp[["match_id","player","short_rally_pct","forced_err_rate"]],
                  left_on=["match_id"], right_on=["match_id"],
                  how="left", suffixes=("","_sb"))
    df = df[df["player"] == pname] if "player" in df.columns else df
    df = df.merge(snv_grp[["match_id","player","snv_win_pct","snv_rate"]],
                  on=["match_id"], how="left", suffixes=("","_snv"))
    df = df[df["player"] == pname] if "player" in df.columns else df
    df = df.merge(svbk_grp[["match_id","player","pressure_win_pct"]],
                  on=["match_id"], how="left", suffixes=("","_sv"))
    if "player" in df.columns:
        df = df[df["player"] == pname]
    df["surface"] = df["surface"].fillna("Hard").apply(_normalize_surface)

    # Filter by surface, fall back to all if too few rows
    surf_norm = _normalize_surface(surface)
    sub = df[df["surface"] == surf_norm]
    if len(sub) < MIN_ROWS:
        print(f"  ⚠ Only {len(sub)} rows on {surf_norm} for '{pname}' — using all surfaces.")
        sub = df

    # Sort by date, keep most recent n_recent matches
    sub = sub.sort_values("Date", ascending=True, na_position="first")
    match_ids = sub["match_id"].unique()
    if len(match_ids) > n_recent:
        match_ids = match_ids[-n_recent:]
    sub = sub[sub["match_id"].isin(match_ids)]

    mat = sub[STAT_COLS].values.astype(np.float32)
    mat = np.nan_to_num(mat, nan=np.nanmedian(mat, axis=0) if mat.shape[0] > 0 else 0.0)

    # Recency weights (exponential decay, more recent = higher weight)
    n = len(mat)
    wts = np.exp(np.linspace(-1, 0, n)).astype(np.float32) if n > 0 else np.ones(1, np.float32)
    wts /= wts.sum()

    return mat, wts, sub["match_id"].nunique()


def player_profile(mat: np.ndarray, wts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-safe weighted mean and std of per-set stats.
    For each column, only rows where the value is finite contribute.
    """
    if len(mat) == 0:
        return np.zeros(N_STATS, np.float32), np.ones(N_STATS, np.float32) * 0.1
    mat  = np.asarray(mat,  dtype=np.float32)
    wts  = np.asarray(wts,  dtype=np.float32)
    mean = np.zeros(mat.shape[1], dtype=np.float32)
    std  = np.full(mat.shape[1],  0.1, dtype=np.float32)
    for j in range(mat.shape[1]):
        col  = mat[:, j]
        mask = np.isfinite(col)
        if mask.sum() == 0:
            continue  # leave mean=0, std=0.1 as fallback
        w_j  = wts[mask]
        w_j  = w_j / w_j.sum()
        c_j  = col[mask]
        mean[j] = float(np.dot(w_j, c_j))
        var_j   = float(np.dot(w_j, (c_j - mean[j])**2))
        std[j]  = float(np.sqrt(max(var_j, 1e-6)))
    return mean, std


# ── POPULATION STATS ──────────────────────────────────────────────────────────

def compute_population_stats(
    paths: Dict[str, str], surface: str, min_matches: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runtime fallback: compute population mean/std from Overview CSV.
    Used only when checkpoint has no embedded pop_stats (old checkpoints).
    For new checkpoints, runner.py embeds the training-time population.
    """
    surf = _normalize_surface(surface)
    matches = pd.read_csv(paths["charting-m-matches.csv"], on_bad_lines="skip")
    matches.columns = matches.columns.str.strip()
    matches = matches.rename(columns={"Surface":"surface"})
    matches["match_id"] = matches["match_id"].astype(str)
    matches["surface"]  = matches["surface"].fillna("Hard").astype(str).apply(_normalize_surface)
    surf_ids = set(matches[matches["surface"] == surf]["match_id"])

    ov = pd.read_csv(paths["charting-m-stats-Overview.csv"], on_bad_lines="skip")
    ov.columns = ov.columns.str.strip()
    ov = ov[ov["set"] != "Total"].copy()
    ov["match_id"] = ov["match_id"].astype(str)
    ov = ov[ov["match_id"].isin(surf_ids)]
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

    # 9-stat subset used by stat-Elo (indices 0..8 of STAT_COLS)
    sub9 = STAT_COLS[:9]
    pm   = ov.groupby(["match_id","player"])[sub9].mean().reset_index()
    vals = pm[sub9].values.astype(np.float32)
    mean = np.nanmean(vals, axis=0)
    std  = np.maximum(np.nanstd(vals,  axis=0), 1e-4)
    # Pad to N_STATS for compatibility with stat_elo_prob (14 stats)
    mean = np.concatenate([mean, np.full(N_STATS - 9, 0.3, np.float32)])
    std  = np.concatenate([std,  np.full(N_STATS - 9, 0.1, np.float32)])
    return mean.astype(np.float32), std.astype(np.float32)


# ── STAT-PROFILE ELO ──────────────────────────────────────────────────────────

def _stat_elo_feat_for_step(
    r1: np.ndarray, r2: np.ndarray,
    pop_mean: np.ndarray, pop_std: np.ndarray,
    surface: str,
) -> np.ndarray:
    """
    9-dim stat-profile Elo feature block for one sequence step.
    Called inside the MC loop so each simulation gets its own noisy D score.
    Mirrors runner.py::stat_elo_features() exactly.

    Layout:
      [0]   D  — weighted dominance score
      [1]   Pstat  — Elo sigmoid(D * K_SCALE)
      [2]   D clipped to [-3, 3]
      [3:9] per-stat z-score diffs for top-6 stats
    """
    surf_key = _normalize_surface(surface)
    w = STATELO_WEIGHTS.get(surf_key, STATELO_WEIGHTS["Hard"])

    v1 = r1[ELO_STAT_IDX]
    v2 = r2[ELO_STAT_IDX]

    delta = np.zeros(9, dtype=np.float32)
    for j in range(9):
        vv1, vv2 = float(v1[j]), float(v2[j])
        if not (np.isfinite(vv1) and np.isfinite(vv2)):
            delta[j] = 0.0  # missing stat → neutral
            continue
        d = (vv1 - vv2) / max(float(pop_std[j]), 1e-4)
        delta[j] = -d if j in LOWER_IS_BETTER_STATELO else d

    D     = float(np.dot(w, delta))  # all finite now
    Pstat = float(np.clip(1.0 / (1.0 + 10.0 ** (-D * K_SCALE)), 0.01, 0.99))

    feat = np.zeros(STATELO_DIM, dtype=np.float32)
    feat[0] = D
    feat[1] = Pstat
    feat[2] = float(np.clip(D, -3.0, 3.0))
    feat[3:9] = delta[:6]
    return feat


def stat_elo_prob(
    prof_p1: np.ndarray, prof_p2: np.ndarray,
    pop_std: np.ndarray, surface: str,
) -> float:
    """
    Scalar win probability from stat profiles (all 14 stats, full weights).
    Used in the final blend as Signal B — complements the NN.
    """
    surf_key = _normalize_surface(surface)
    w = STAT_WEIGHTS_14.get(surf_key, STAT_WEIGHTS_14["Hard"])
    D    = 0.0
    w_sum = 0.0
    n_st = min(N_STATS, len(prof_p1), len(prof_p2), len(pop_std), len(w))
    for j in range(n_st):
        v1, v2 = float(prof_p1[j]), float(prof_p2[j])
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue  # skip stats with missing data
        delta = (v1 - v2) / max(float(pop_std[j]), 1e-4)
        if j in LOWER_IDX:
            delta = -delta
        D     += w[j] * delta
        w_sum += w[j]
    if w_sum > 0:
        D = D / w_sum  # re-normalise in case some stats were skipped
    return float(np.clip(1.0 / (1.0 + 10.0 ** (-D * K_SCALE)), 0.02, 0.98))


# ── ELO FILE ──────────────────────────────────────────────────────────────────

def elo_file_prob(
    elo_data: Optional[Dict], p1: str, p2: str, surface: str,
) -> Optional[float]:
    if elo_data is None:
        return None
    if p1 not in elo_data or p2 not in elo_data:
        missing = [p for p in [p1, p2] if p not in elo_data]
        print(f"  ⚠ Elo file missing: {missing} — Elo signal skipped.")
        return None
    surf_key = _normalize_surface(surface)
    r1 = elo_data[p1].get("blended_elo", {}).get(surf_key, elo_data[p1]["career_elo"])
    r2 = elo_data[p2].get("blended_elo", {}).get(surf_key, elo_data[p2]["career_elo"])
    return float(np.clip(1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0)), 0.02, 0.98))


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
    def __init__(self, feature_dim=FEATURE_DIM, d_model=D_MODEL,
                 lstm_layers=2, nhead=4, tf_layers=2, dropout=0.25):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model     = d_model
        self.input_proj  = nn.Sequential(
            nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(d_model, d_model//2, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout,
                                          "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=tf_layers)
        self.gate = nn.Linear(d_model*2, d_model*2)
        self.head = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 32),        nn.GELU(),
            nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.size(0)
        h   = self.input_proj(x)
        _, (hn, _) = self.lstm(h)
        lstm_out   = torch.cat([hn[-2], hn[-1]], dim=-1)
        cls        = self.cls_token.expand(B, -1, -1)
        t_in       = self.pos_enc(torch.cat([cls, h], dim=1))
        cls_out    = self.transformer(t_in)[:, 0]
        combined   = torch.cat([lstm_out, cls_out], dim=-1)
        return self.head(combined * torch.sigmoid(self.gate(combined)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def load_checkpoint(
    path: str, device: str = "cpu"
) -> Tuple["TennisMatchNet", StandardScaler, dict, dict]:
    """
    Load model, scaler, raw checkpoint dict, and population stats.
    Returns 4-tuple: (model, scaler, ckpt_dict, pop_stats).
    pop_stats: dict surface → (mean_9, std_9) — same population as training.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: '{path}'")
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = TennisMatchNet(
        feature_dim=ckpt.get("feature_dim", FEATURE_DIM),
        d_model    =ckpt.get("d_model",     D_MODEL),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    scaler        = StandardScaler()
    scaler.mean_  = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    # Restore population stats saved by runner.py
    pop_stats_raw = ckpt.get("pop_stats", {})
    pop_stats = {
        surf: (np.array(v["mean"], dtype=np.float32),
               np.array(v["std"],  dtype=np.float32))
        for surf, v in pop_stats_raw.items()
    }
    return model, scaler, ckpt, pop_stats


# ── ELO FEATURE BLOCK (external Elo file) ─────────────────────────────────────

def _elo_features_pred(
    p1: str, p2: str, surface: str,
    elo_data: dict, elo_mean: float, elo_std: float,
) -> np.ndarray:
    """8-dim external Elo feature block. Mirrors runner.py::external_elo_features."""
    feat = np.zeros(ELO_DIM, dtype=np.float32)
    if elo_data is None or p1 not in elo_data or p2 not in elo_data:
        return feat
    surf_key = _normalize_surface(surface)
    d1, d2   = elo_data[p1], elo_data[p2]
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
    r1_ = float(d1.get("recent_elo", c1))
    r2_ = float(d2.get("recent_elo", c2))
    feat[7] = (r1_ - r2_) / elo_std
    return feat


# ── FEATURE SEQUENCE BUILDER ──────────────────────────────────────────────────

def build_feature_sequence(
    prof_p1: np.ndarray, prof_p2: np.ndarray,
    std_p1:  np.ndarray, std_p2:  np.ndarray,
    surface: str, best_of: int,
    elo_data:    Optional[Dict] = None,
    elo_mean:    float = 1500.0,
    elo_std:     float = 150.0,
    p1_name:     str   = "",
    p2_name:     str   = "",
    feat_dim:    int   = FEATURE_DIM,
    pop_mean:    Optional[np.ndarray] = None,
    pop_std_arr: Optional[np.ndarray] = None,
    seed:        int   = 42,
    noise_scale: float = 0.15,
) -> np.ndarray:
    """
    Build one (SEQ_LEN, feat_dim) Monte-Carlo sample.

    Feature layout (feat_dim=57, mirrors runner.py):
      [0:14]   P1 per-set stats  (with MC Gaussian noise)
      [14:28]  P2 per-set stats  (with MC Gaussian noise)
      [28:34]  Key stat differentials P1−P2
      [34]     Set progress
      [35]     1st-serve-won delta
      [36]     Surface id (normalised)
      [37]     Best-of-5 flag
      [38]     Running P1 set count / SEQ_LEN
      [39]     Running P2 set count / SEQ_LEN
      [40:48]  External Elo  (constant across steps, no noise)
      [48:57]  Stat-profile Elo  ← recomputed per step from noisy r1/r2
                                    so MC variance propagates to D and Pstat too
    """
    rng     = np.random.default_rng(seed)
    surf_id = {s: i for i, s in enumerate(SURFACES)}.get(_normalize_surface(surface), 0)

    # Population stats: prefer training-time values, fall back to neutral defaults
    _pop_mean = pop_mean    if pop_mean    is not None else np.full(9, 0.65, np.float32)
    _pop_std  = pop_std_arr if pop_std_arr is not None else np.full(9, 0.10, np.float32)

    # External Elo block — constant across all set steps
    elo_feat = np.zeros(ELO_DIM, dtype=np.float32)
    if feat_dim >= ELO_OFFSET + ELO_DIM and elo_data is not None:
        elo_feat = _elo_features_pred(p1_name, p2_name, surface,
                                       elo_data, elo_mean, elo_std)

    seq = np.zeros((SEQ_LEN, feat_dim), dtype=np.float32)
    p1_sets = p2_sets = 0

    for s in range(SEQ_LEN):
        # Perturb profiles with Gaussian noise (MC sampling)
        r1   = np.clip(prof_p1 + rng.normal(0, std_p1 * noise_scale), 0, 1).astype(np.float32)
        r2   = np.clip(prof_p2 + rng.normal(0, std_p2 * noise_scale), 0, 1).astype(np.float32)
        diff = r1 - r2

        if r1[1] > r2[1]: p1_sets += 1
        else:              p2_sets += 1

        feat = np.zeros(feat_dim, dtype=np.float32)

        # ── [0:40] base stats + meta ─────────────────────────────────────
        feat[:N_STATS]          = r1
        feat[N_STATS:2*N_STATS] = r2
        feat[28:34]             = diff[:6]
        feat[34] = (s + 1) / SEQ_LEN
        feat[35] = float(np.clip(r1[1] - r2[1], -1, 1))
        feat[36] = surf_id / max(len(SURFACES) - 1, 1)
        feat[37] = float(best_of == 5)
        feat[38] = p1_sets / SEQ_LEN
        feat[39] = p2_sets / SEQ_LEN

        # ── [40:48] external Elo ─────────────────────────────────────────
        if feat_dim >= ELO_OFFSET + ELO_DIM:
            feat[ELO_OFFSET:ELO_OFFSET + ELO_DIM] = elo_feat

        # ── [48:57] stat-profile Elo — live, per MC step ─────────────────
        # r1/r2 already have noise → D and Pstat vary across simulations
        # → MC CI reflects uncertainty on stat-Elo signal too
        if feat_dim >= STATELO_OFFSET + STATELO_DIM:
            feat[STATELO_OFFSET:STATELO_OFFSET + STATELO_DIM] = \
                _stat_elo_feat_for_step(
                    np.nan_to_num(r1), np.nan_to_num(r2),
                    _pop_mean, _pop_std, surface,
                )

        np.nan_to_num(feat, copy=False)
        seq[s] = feat

    return seq


# ── MAIN PREDICTION ───────────────────────────────────────────────────────────

def predict_match(
    p1_name: str, p2_name: str,
    surface: str, best_of: int,
    model: "TennisMatchNet", scaler: StandardScaler,
    paths: Dict[str, str],
    elo_data:       Optional[Dict] = None,
    elo_mean:       float = 1500.0,
    elo_std:        float = 150.0,
    feature_dim:    int   = FEATURE_DIM,
    device:         str   = "cpu",
    n_sims:         int   = 500,
    n_recent:       int   = 50,
    noise_scale:    float = 0.15,
    ckpt_pop_stats: Optional[Dict] = None,
) -> dict:
    """
    Full prediction pipeline:
      1. Load player stat profiles from charting CSVs
      2. Resolve population stats (checkpoint-embedded > runtime-computed)
      3. Run N Monte-Carlo simulations through the NN
      4. Compute stat-Elo and Elo-file signals
      5. Blend into final probability with adaptive alpha
    """
    mat_p1, wts_p1, n_m1 = load_player_stats(paths, p1_name, surface, best_of, n_recent)
    mat_p2, wts_p2, n_m2 = load_player_stats(paths, p2_name, surface, best_of, n_recent)
    prof_p1, std_p1 = player_profile(mat_p1, wts_p1)
    prof_p2, std_p2 = player_profile(mat_p2, wts_p2)

    # ── Population stats for z-scoring ───────────────────────────────────────
    # Prefer checkpoint-embedded population (identical to training-time data).
    # Fall back to runtime computation for backward compat with old checkpoints.
    surf_key = _normalize_surface(surface)
    if ckpt_pop_stats and surf_key in ckpt_pop_stats:
        pop_mean, pop_std = ckpt_pop_stats[surf_key]
        pop_mean = np.asarray(pop_mean, dtype=np.float32)
        pop_std  = np.asarray(pop_std,  dtype=np.float32)
        # Checkpoint may have been saved with only 9 stat-Elo elements;
        # pad to N_STATS=14 so stat_elo_prob() can index the full range.
        if len(pop_mean) < N_STATS:
            pop_mean = np.concatenate([pop_mean,
                np.full(N_STATS - len(pop_mean), 0.3, np.float32)])
        if len(pop_std) < N_STATS:
            pop_std  = np.concatenate([pop_std,
                np.full(N_STATS - len(pop_std),  0.1, np.float32)])
        # Replace any NaN/inf in population stats with safe defaults
        pop_mean = np.where(np.isfinite(pop_mean), pop_mean, 0.3)
        pop_std  = np.where(np.isfinite(pop_std)  & (pop_std > 0), pop_std, 0.1)
    else:
        pop_mean, pop_std = compute_population_stats(paths, surface)
        pop_mean = np.where(np.isfinite(pop_mean), pop_mean, 0.3)
        pop_std  = np.where(np.isfinite(pop_std)  & (pop_std > 0), pop_std, 0.1)

    # ── Signal A: Neural network (Monte Carlo ensemble) ───────────────────────
    model.train()   # keep dropout active for MC
    nn_probs = []
    for seed in range(n_sims):
        seq = build_feature_sequence(
            prof_p1, prof_p2, std_p1, std_p2,
            surface, best_of,
            elo_data=elo_data, elo_mean=elo_mean, elo_std=elo_std,
            p1_name=p1_name, p2_name=p2_name,
            feat_dim=feature_dim,
            pop_mean=pop_mean,
            pop_std_arr=pop_std,
            seed=seed, noise_scale=noise_scale,
        )
        seq_s = scaler.transform(
            seq.reshape(-1, feature_dim)
        ).reshape(1, SEQ_LEN, feature_dim).astype(np.float32)
        x = torch.tensor(seq_s, dtype=torch.float32).to(device)
        with torch.no_grad():
            nn_probs.append(F.softmax(model(x), dim=-1)[0, 0].item())

    nn_probs = np.array(nn_probs)
    p_nn     = float(nn_probs.mean())
    ci_lo, ci_hi = float(np.percentile(nn_probs, 5)), float(np.percentile(nn_probs, 95))
    nn_std   = float(nn_probs.std())

    # ── NN reliability: down-weight or exclude based on uncertainty + data ────
    # Fix A: penalise both extremity AND high MC variance
    nn_extremity   = 2.0 * abs(p_nn - 0.5)        # 0=neutral, 1=extreme
    nn_uncertainty = min(nn_std / 0.5, 1.0)        # 0=certain, 1=max noise (σ=50pp)
    # Fix B: low data → additional uncertainty penalty
    data_factor    = min(n_m1, n_m2) / NN_MIN_MATCHES
    data_factor    = float(np.clip(data_factor, 0.0, 1.0))
    alpha_nn_adj   = ALPHA_NN * max(0.05,
        (1.0 - nn_extremity) * (1.0 - nn_uncertainty) * data_factor)
    # Fix C: hard exclusion when MC variance too high (NN = noise)
    nn_reliable    = nn_std < NN_SIGMA_CUTOFF
    if not nn_reliable:
        alpha_nn_adj = 0.0   # exclude NN entirely from blend
    # ── Signal B: Stat-profile Elo (full-14-stat version) ─────────────────────
    p_stat_raw = stat_elo_prob(prof_p1, prof_p2, pop_std, surface)
    p_stat = None if not np.isfinite(p_stat_raw) else p_stat_raw

    # ── Signal C: External Elo file ───────────────────────────────────────────
    p_elo = elo_file_prob(elo_data, p1_name, p2_name, surface)

    # ── Blend (only include signals that are available and reliable) ──────────
    blend_w  = 0.0
    p_final  = 0.0
    if nn_reliable and alpha_nn_adj > 0:
        blend_w += alpha_nn_adj
        p_final += alpha_nn_adj * p_nn
    if p_stat is not None:
        blend_w += ALPHA_STAT
        p_final += ALPHA_STAT * p_stat
    if p_elo is not None:
        blend_w += ALPHA_ELO
        p_final += ALPHA_ELO * p_elo
    if blend_w == 0.0:
        p_final = 0.5  # no signal at all
    else:
        p_final = p_final / blend_w
    signals  = {
        "nn":          round(p_nn, 4),
        "stat_elo":    round(p_stat_raw, 4) if np.isfinite(p_stat_raw) else float("nan"),
        "elo_file":    round(p_elo, 4) if p_elo is not None else None,
        "nn_reliable": nn_reliable,
        "nn_weight":   round(alpha_nn_adj, 4),
    }

    # p_final is guaranteed finite from blend above
    p_final = float(np.clip(p_final, 0.02, 0.98))

    # ── Stat edges ────────────────────────────────────────────────────────────
    p1_edges = p2_edges = 0
    for i, label in enumerate(STAT_LABELS):
        v1, v2 = prof_p1[i], prof_p2[i]
        if abs(v1 - v2) < 1e-4:
            continue
        if label in LOWER_IS_BETTER:
            (p1_edges if v1 < v2 else p2_edges).__class__   # no-op trick
            if v1 < v2: p1_edges += 1
            else:       p2_edges += 1
        else:
            if v1 > v2: p1_edges += 1
            else:       p2_edges += 1

    return {
        "p1": p1_name, "p2": p2_name,
        "surface": surface, "best_of": best_of,
        "p_final":     round(p_final, 4),
        "p1_win_prob": round(p_final, 4),
        "p2_win_prob": round(1.0 - p_final, 4),
        "signals":     signals,
        "ci_90_nn":    (round(ci_lo, 4), round(ci_hi, 4)),
        "nn_std":      round(nn_std, 4),
        "n_sims":      n_sims,
        "p1_matches":  n_m1, "p2_matches": n_m2,
        "p1_profile":  prof_p1.tolist(),
        "p2_profile":  prof_p2.tolist(),
        "p1_std":      std_p1.tolist(),
        "p2_std":      std_p2.tolist(),
        "p1_edges":    p1_edges,
        "p2_edges":    p2_edges,
        "stat_labels": STAT_LABELS,
    }


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(r: dict):
    sep  = "─" * 66
    p1, p2 = r["p1"], r["p2"]
    p1p, p2p = r["p1_win_prob"], r["p2_win_prob"]
    winner = p1 if p1p > p2p else p2
    margin = abs(p1p - p2p)
    # Guard against NaN (should not happen after upstream fixes, but just in case)
    p1p = 0.5 if not np.isfinite(p1p) else p1p
    p2p = 1.0 - p1p
    BAR = 44; b1 = int(p1p * BAR); b2 = BAR - b1

    print(f"\n{'='*66}")
    print(f"  Tennis Match Prediction")
    print(f"  {p1}  vs  {p2}")
    print(f"  Surface: {r['surface']}  |  Best of {r['best_of']}")
    print(f"{'='*66}")
    print(f"\n  {'█'*b1}{'░'*b2}  ← P1 / P2")
    print(f"  {p1:<28s}  {p1p*100:>5.1f}%")
    print(f"  {p2:<28s}  {p2p*100:>5.1f}%")
    verdict = ("Slight edge" if margin < 0.08 else
               "Clear edge"  if margin < 0.18 else
               "Strong favourite")
    print(f"\n  Verdict: {winner} — {verdict}")
    print(f"\n  {sep}")
    print(f"  Signal breakdown")
    print(f"  {sep}")
    s = r["signals"]
    print(f"  {'Signal':<22s}  {'P1 prob':>8s}  {'Weight':>7s}")
    print(f"  {sep}")
    _nn_wt  = s.get("nn_weight", 0.0)
    _nn_rel = s.get("nn_reliable", True)
    _nn_wt_str = f"{_nn_wt*100:.0f}%" if _nn_rel else "excl."
    _nn_val_str = f"{s['nn']*100:>8.1f}%" if _nn_rel else "  (excl.)"
    print(f"  {'Neural Network (MC)':<22s}  {_nn_val_str:>8s}  {_nn_wt_str:>7s}")
    _pstat_str = f"{s['stat_elo']*100:>8.1f}%" if (s['stat_elo'] is not None
                  and np.isfinite(s['stat_elo'])) else "     n/a"
    print(f"  {'Stat-profile Elo':<22s}  {_pstat_str:>8s}  {'45%':>7s}")
    if s["elo_file"] is not None:
        print(f"  {'Elo file':<22s}  {s['elo_file']*100:>8.1f}%  {'35%':>7s}")
    else:
        print(f"  {'Elo file':<22s}  {'N/A':>8s}  {'—':>7s}")
    ci = r["ci_90_nn"]
    _nn_rel2 = r["signals"].get("nn_reliable", True)
    _rel_tag  = "" if _nn_rel2 else "  ⚠ HIGH VARIANCE — excluded from blend"
    print(f"\n  NN 90% CI : [{ci[0]*100:.1f}%,  {ci[1]*100:.1f}%]"
          f"   σ={r['nn_std']*100:.1f}pp   ({r['n_sims']} MC sims){_rel_tag}")
    print(f"\n  {sep}")
    print(f"  Stat edges  —  {p1}: {r['p1_edges']}/{len(r['stat_labels'])} "
          f"  |  {p2}: {r['p2_edges']}/{len(r['stat_labels'])}")
    print(f"  {sep}")
    print(f"  {'Stat':<22s}  {p1[:14]:>14s}  {p2[:14]:>14s}")
    print(f"  {sep}")
    for i, label in enumerate(r["stat_labels"]):
        v1 = r["p1_profile"][i]; v2 = r["p2_profile"][i]
        adv = ("◀" if (label in LOWER_IS_BETTER and v1 < v2) or
                      (label not in LOWER_IS_BETTER and v1 > v2)
               else ("▶" if (label in LOWER_IS_BETTER and v2 < v1) or
                            (label not in LOWER_IS_BETTER and v2 > v1)
                     else ""))
        print(f"  {label:<22s}  {v1:>13.3f}  {v2:>13.3f}  {adv}")
    print(f"\n  Data: Jeff Sackmann Match Charting Project"
          f"  |  Matches used: P1={r['p1_matches']}  P2={r['p2_matches']}")
    print(f"{'='*66}\n")


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def list_players(paths: Dict[str, str], surface: Optional[str] = None):
    matches = pd.read_csv(paths["charting-m-matches.csv"], on_bad_lines="skip")
    matches.columns = matches.columns.str.strip()
    matches = matches.rename(columns={"Player 1":"p1","Player 2":"p2","Surface":"surface"})
    matches["surface"] = matches["surface"].fillna("Hard").astype(str).apply(_normalize_surface)
    ov = pd.read_csv(paths["charting-m-stats-Overview.csv"], on_bad_lines="skip")
    ov.columns = ov.columns.str.strip()
    ov["match_id"] = ov["match_id"].astype(str)
    matches["match_id"] = matches["match_id"].astype(str)
    if surface:
        surf_norm = _normalize_surface(surface)
        ids = set(matches[matches["surface"] == surf_norm]["match_id"])
        ov  = ov[ov["match_id"].isin(ids)]
    counts  = ov.groupby("player")["match_id"].nunique().sort_values(ascending=False)
    surf_lbl = surface or "All surfaces"
    print(f"\n  Players with charting data ({surf_lbl}):")
    print(f"  {'Player':<30s}  {'Matches':>7s}")
    print(f"  {'─'*40}")
    for p, cnt in counts.head(50).items():
        print(f"  {p:<30s}  {cnt:>7d}")
    print(f"\n  ({len(counts)} total players)\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--p1",           type=str, default=None)
    ap.add_argument("--p2",           type=str, default=None)
    ap.add_argument("--surface",      type=str, default="Hard",
                    choices=["Hard","Clay","Grass","Carpet","Indoor Hard","Outdoor Hard"])
    ap.add_argument("--best-of",      type=int, default=3, choices=[3, 5])
    ap.add_argument("--model",        type=str, default="tennis_winner_model.pt")
    ap.add_argument("--elo",          type=str, default="player_elo.json",
                    help="Static Elo file from compute_elo.py")
    ap.add_argument("--elo-live",     type=str, default=None,
                    help="Live/updated Elo .json (same structure as --elo). "
                         "Merged on top of --elo; live values take priority.")
    ap.add_argument("--data-dir",     type=str, default=".")
    ap.add_argument("--sims",         type=int, default=500)
    ap.add_argument("--noise",        type=float, default=0.15)
    ap.add_argument("--n-recent",     type=int, default=50)
    ap.add_argument("--device",       type=str, default="cpu")
    ap.add_argument("--list-players", action="store_true")
    ap.add_argument("--json",         action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    paths = _ensure_files(args.data_dir)

    if args.list_players:
        list_players(paths, surface=args.surface if args.surface != "Hard" else None)
        return

    if not args.p1 or not args.p2:
        ap.error("--p1 and --p2 are required (or use --list-players)")

    # Load Elo
    elo_data = None; elo_mean = 1500.0; elo_std = 150.0
    if args.elo and os.path.exists(args.elo):
        with open(args.elo) as f:
            elo_data = json.load(f)
        vals = [v["career_elo"] for v in elo_data.values()
                if isinstance(v.get("career_elo"), (int, float))]
        if vals:
            elo_mean = float(np.mean(vals))
            elo_std  = max(float(np.std(vals)), 1.0)
        print(f"  Elo: {len(elo_data)} players  "
              f"(mean={elo_mean:.0f}  std={elo_std:.0f})")
    elif args.elo:
        print(f"  ⚠ '{args.elo}' not found — running without Elo file.")

    # ── Optional live Elo override ──────────────────────────────────────
    if getattr(args, 'elo_live', None):
        _lp = args.elo_live
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

    # Load model
    print(f"  Loading model: {args.model}")
    model, scaler, ckpt, ckpt_pop_stats = load_checkpoint(args.model, device=args.device)
    feat_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
    print(f"  feature_dim={feat_dim}  "
          f"pop_stats surfaces: {list(ckpt_pop_stats.keys()) or 'none (old ckpt)'}")

    # Resolve player names
    known  = pd.read_csv(paths["charting-m-stats-Overview.csv"],
                          on_bad_lines="skip")["player"].dropna().unique().tolist()
    p1_res = _fuzzy_player(args.p1, known)
    p2_res = _fuzzy_player(args.p2, known)
    if p1_res != args.p1: print(f"  Resolved '{args.p1}' → '{p1_res}'")
    if p2_res != args.p2: print(f"  Resolved '{args.p2}' → '{p2_res}'")

    # Run prediction
    result = predict_match(
        p1_name        = p1_res,
        p2_name        = p2_res,
        surface        = args.surface,
        best_of        = args.best_of,
        model          = model,
        scaler         = scaler,
        paths          = paths,
        elo_data       = elo_data,
        elo_mean       = elo_mean,
        elo_std        = elo_std,
        feature_dim    = feat_dim,
        device         = args.device,
        n_sims         = args.sims,
        n_recent       = args.n_recent,
        noise_scale    = args.noise,
        ckpt_pop_stats = ckpt_pop_stats,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()

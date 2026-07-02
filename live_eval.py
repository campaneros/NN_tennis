#!/usr/bin/env python3
"""
live_eval.py — Live Performance Test for the Stat-Profile Elo Score
=====================================================================
Implements the 6-step pipeline from the design spec:
  1. Surface-specific stat importance weights
  2. Per-stat z-score comparison  Δj = (μ1j − μ2j) / σpop,j
  3. Weighted dominance score     D  = Σ wj · Δj
  4. Elo sigmoid                  Pstat = 1 / (1 + 10^(−D·k))
  5. Blend with NN model output   Pfinal = α·Pmodel + (1−α)·Pstat
  6. Population variance computed at runtime from the charting dataset

Evaluation outputs
------------------
  • Accuracy, Brier score, log-loss, ECE (calibration) — per surface & overall
  • Calibration curve (reliability diagram) — saved as PNG
  • Confusion matrix + ROC curve — saved as PNG
  • Full match-level predictions — saved as CSV
  • Summary JSON — machine-readable for CI/CD pipelines

Usage
-----
  # Stat-Elo only (no NN model needed)
  python live_eval.py --data-dir tennis_MatchChartingProject

  # Full blend (requires trained model + Elo file)
  python live_eval.py --data-dir tennis_MatchChartingProject \\
                      --model tennis_winner_model.pt \\
                      --elo   player_elo.json \\
                      --alpha 0.25

  # Filter to a specific surface
  python live_eval.py --surface Clay --min-matches 5

  # Save to a custom output folder
  python live_eval.py --out-dir eval_results/
"""

import argparse
import json
import math
import os
import sys
import urllib.request
import ssl as _ssl
import glob as _glob
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import StandardScaler
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_OK = True
except ImportError:
    MPL_OK = False

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
SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

# ── Step 1: Surface-specific importance weights ───────────────────────────────
#
# Stats covered (9, matching the design spec table):
#   0  1st_won_pct      higher = better
#   1  bp_save_pct      higher = better
#   2  ret_pts_pct      higher = better
#   3  pressure_win_pct higher = better
#   4  2nd_won_pct      higher = better
#   5  winners_rate     higher = better
#   6  uf_rate          LOWER  = better  (inverted)
#   7  df_rate          LOWER  = better  (inverted)
#   8  forced_err_rate  LOWER  = better  (inverted)
#
STAT_COLS_ELO = [
    "1st_won_pct", "bp_save_pct", "ret_pts_pct", "pressure_win_pct",
    "2nd_won_pct", "winners_rate", "uf_rate", "df_rate", "forced_err_rate",
]
LOWER_IS_BETTER_IDX = {6, 7, 8}   # uf_rate, df_rate, forced_err_rate

STAT_WEIGHTS: Dict[str, np.ndarray] = {
    #              1stwon  bpsave  retpts  press  2ndwon  win    uf     df    ferr
    "Clay":   np.array([0.18,   0.16,   0.16,   0.14,  0.10,  0.08,  0.10,  0.04, 0.04]),
    "Hard":   np.array([0.22,   0.14,   0.14,   0.12,  0.10,  0.10,  0.10,  0.04, 0.04]),
    "Grass":  np.array([0.25,   0.12,   0.12,   0.10,  0.10,  0.14,  0.10,  0.04, 0.03]),
    "Carpet": np.array([0.23,   0.13,   0.13,   0.11,  0.10,  0.12,  0.10,  0.04, 0.04]),
}
# All weights must sum to 1 — normalise in case of rounding
for _s in STAT_WEIGHTS:
    STAT_WEIGHTS[_s] = STAT_WEIGHTS[_s] / STAT_WEIGHTS[_s].sum()

K_SCALE  = 0.4    # Elo sigmoid scaling constant (1-σ gap → ~65% win prob)
ALPHA_NN = 0.25   # weight for NN model in blend (spec: 25% model, 75% stat)

# ── HELPERS ───────────────────────────────────────────────────────────────────

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

def _download(data_dir: str) -> Dict[str, str]:
    """Resolve stat-file paths.

    • tennis_atp clone (atp_matches_*.csv present) → ATP mode, no MCP download
    • otherwise → MCP download with SSL bypass (macOS Python 3.12 fix)
    """
    os.makedirs(data_dir, exist_ok=True)
    atp_files = sorted(_glob.glob(os.path.join(data_dir, "atp_matches_????.csv")))
    paths: Dict[str, str] = {}
    for fname in FILES:
        local = os.path.join(data_dir, fname)
        if fname == "charting-m-matches.csv" and atp_files:
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
        paths["__atp_dir__"] = data_dir
        print(f"  ATP mode: {len(atp_files)} atp_matches_*.csv files detected.")
    return paths

def _load_matches_from_atp(data_dir: str) -> "pd.DataFrame":
    """Load winner/loser/surface/date from all atp_matches_YYYY.csv files."""
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
        "winner_name": "Player 1", "loser_name":   "Player 2",
        "tourney_date": "Date",    "best_of":       "Best of",
        "surface":      "Surface", "tourney_id":    "_tid",
        "match_num":    "_mnum",
    })
    tid  = df.get("_tid",  pd.Series(range(len(df)), dtype=str))
    mnum = df.get("_mnum", pd.Series(range(len(df)), dtype=str)).astype(str)
    df["match_id"] = tid.astype(str) + "_" + mnum
    df["Date"]     = pd.to_numeric(df["Date"], errors="coerce")
    df["best_of"]  = df.get("Best of", pd.Series(3)).apply(_safe_best_of)
    df["Surface"]  = df.get("Surface", pd.Series("Hard")).fillna("Hard").astype(str).apply(_normalize_surface)
    df = df.dropna(subset=["Player 1", "Player 2", "Date"])
    df = df[df["Player 1"].str.strip() != ""]
    df = df[df["Player 2"].str.strip() != ""]
    return df.sort_values("Date", ascending=True).reset_index(drop=True)



# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data(paths: Dict[str, str]) -> pd.DataFrame:
    """Load and merge all CSVs into a per-match-per-player-per-set DataFrame."""
    if paths.get("charting-m-matches.csv") == "__ATP__":
        _m = _load_matches_from_atp(paths["__atp_dir__"])
        matches = _m.rename(columns={"Player 1": "p1", "Player 2": "p2",
                                     "Surface": "surface", "Best of": "best_of"})
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
    matches = matches[["match_id", "p1", "p2", "best_of", "surface"]].copy()

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
    ov["1st_won_pct"]  = _pct(ov["first_won"],     ov["first_in"])
    ov["2nd_won_pct"]  = _pct(ov["second_won"],    ov["second_in"])
    ov["bp_save_pct"]  = _pct(ov["bp_saved"],       ov["bk_pts"])
    ov["ret_pts_pct"]  = _pct(ov["return_pts_won"], ov["return_pts"])
    ov["winners_rate"] = _pct(ov["winners"],
                               ov["serve_pts"] + ov["return_pts"])
    ov["uf_rate"]      = _pct(ov["unforced"],
                               ov["serve_pts"] + ov["return_pts"])
    ov["df_rate"]      = _pct(ov["dfs"],            ov["serve_pts"])
    ov["ace_rate"]     = _pct(ov["aces"],            ov["serve_pts"])

    sb = pd.read_csv(paths["charting-m-stats-ServeBasics.csv"], on_bad_lines="skip")
    sb.columns = sb.columns.str.strip()
    sb = sb[sb["row"].isin(["1st", "2nd"])].copy()
    sb["match_id"] = sb["match_id"].astype(str)
    for col in ["pts", "pts_won_lte_3_shots", "forced_err"]:
        sb[col] = pd.to_numeric(sb[col], errors="coerce").fillna(0)
    sb["forced_err_rate"] = _pct(sb["forced_err"], sb["pts"])
    sb_agg = sb.groupby(["match_id","player"])[["forced_err_rate"]].mean().reset_index()

    svbk = pd.read_csv(paths["charting-m-stats-SvBreakTotal.csv"], on_bad_lines="skip")
    svbk.columns = svbk.columns.str.strip()
    svbk["match_id"] = svbk["match_id"].astype(str)
    svbk_d = svbk[svbk["row"] == "d"].copy()
    for col in ["pts", "pts_won"]:
        svbk_d[col] = pd.to_numeric(svbk_d[col], errors="coerce").fillna(0)
    svbk_d["pressure_win_pct"] = _pct(svbk_d["pts_won"], svbk_d["pts"])
    svbk_agg = svbk_d[["match_id","player","pressure_win_pct"]].copy()

    ov_feats = ["1st_won_pct","2nd_won_pct","bp_save_pct","ret_pts_pct",
                "winners_rate","uf_rate","df_rate","ace_rate"]
    df = ov[["match_id","player","set"] + ov_feats].copy()
    df = df.merge(sb_agg,   on=["match_id","player"], how="left")
    df = df.merge(svbk_agg, on=["match_id","player"], how="left")
    df = df.merge(matches,  on="match_id",            how="inner")
    return df


# ── POPULATION STATS (Step 6) ─────────────────────────────────────────────────

def compute_pop_stats(df: pd.DataFrame, surface: str
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-stat population mean and std across ALL players on surface.
    Uses per-match means (not set rows) to avoid giving high-match-count
    players too much weight in the variance estimate.
    """
    sub = df[df["surface"] == surface].copy()
    pm  = (sub.groupby(["match_id","player"])[STAT_COLS_ELO]
              .mean().reset_index())
    vals = pm[STAT_COLS_ELO].values.astype(np.float32)
    mean = np.nanmean(vals, axis=0)
    std  = np.nanstd(vals,  axis=0)
    std  = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


# ── PLAYER PROFILE (recent-weighted mean per stat) ────────────────────────────

def player_profile(df: pd.DataFrame, player: str, surface: str,
                   n_recent: int = 50) -> Tuple[np.ndarray, int]:
    """
    Recency-weighted (0.95^rank) mean of STAT_COLS_ELO for `player` on `surface`.
    Falls back to all surfaces if < 5 surface-specific matches found.
    """
    sub = df[(df["player"] == player) & (df["surface"] == surface)].copy()
    if sub["match_id"].nunique() < 5:
        sub = df[df["player"] == player].copy()

    # Keep n_recent most recent match_ids (proxy: match_id alphabetical desc)
    recent_ids = list(sub["match_id"].unique())[-n_recent:]
    sub = sub[sub["match_id"].isin(recent_ids)].copy()
    n_m = sub["match_id"].nunique()
    if n_m == 0:
        return np.full(len(STAT_COLS_ELO), np.nan, dtype=np.float32), 0

    mid_order = {mid: i for i, mid in enumerate(sorted(sub["match_id"].unique()))}
    sub = sub.copy()
    sub["_rank"] = sub["match_id"].map(mid_order).fillna(0)
    sub["_w"]    = (0.95 ** (n_m - 1 - sub["_rank"])).astype(np.float32)

    profile = np.empty(len(STAT_COLS_ELO), dtype=np.float32)
    for j, col in enumerate(STAT_COLS_ELO):
        col_vals = sub[col].values.astype(np.float32)
        wts      = sub["_w"].values.astype(np.float32)
        ok       = np.isfinite(col_vals)
        if ok.sum() < 2:
            profile[j] = np.nan; continue
        w = wts[ok]; w /= w.sum()
        profile[j] = float(np.dot(w, col_vals[ok]))
    return profile, n_m


# ── STEPS 2–4: Stat-Elo win probability ──────────────────────────────────────

def stat_elo_prob(
    prof_p1: np.ndarray, prof_p2: np.ndarray,
    pop_std: np.ndarray,
    surface: str,
    k: float = K_SCALE,
) -> float:
    """
    Step 2: per-stat z-score diff  Δj = (μ1j − μ2j) / σpop,j
    Step 3: weighted dominance     D  = Σ wj · Δj  (sign-flipped for inv stats)
    Step 4: Elo sigmoid            P  = 1 / (1 + 10^(−D·k))
    """
    surf_key = _normalize_surface(surface)
    w = STAT_WEIGHTS.get(surf_key, STAT_WEIGHTS["Hard"])

    D = 0.0
    for j in range(len(STAT_COLS_ELO)):
        v1, v2 = prof_p1[j], prof_p2[j]
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        delta = (v1 - v2) / pop_std[j]
        if j in LOWER_IS_BETTER_IDX:
            delta = -delta
        D += w[j] * delta

    return float(np.clip(1.0 / (1.0 + 10.0 ** (-D * k)), 0.01, 0.99))


# ── EVALUATION HELPERS ────────────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # y_true: 0=P1 wins; y_prob: P(P1 wins)  →  indicator = 1 - y_true
    p1_ind = (1 - y_true).astype(np.float32)
    return float(np.mean((y_prob - p1_ind) ** 2))

def log_loss(y_true: np.ndarray, y_prob: np.ndarray,
             eps: float = 1e-7) -> float:
    # y_true: 0=P1 wins; y_prob: P(P1 wins)  →  P1 indicator = 1 - y_true
    p   = np.clip(y_prob, eps, 1 - eps)
    p1  = (1 - y_true).astype(np.float32)
    return float(-np.mean(p1 * np.log(p) + (1 - p1) * np.log(1 - p)))

def accuracy(y_true: np.ndarray, y_prob: np.ndarray,
             threshold: float = 0.5) -> float:
    # y_true: 0=P1 wins, 1=P2 wins;  y_prob: P(P1 wins)
    pred_p1   = (y_prob >= threshold).astype(int)  # 1 => predict P1
    actual_p1 = (1 - y_true).astype(int)           # 1 => P1 actually won
    return float(np.mean(pred_p1 == actual_p1))

def ece(y_true: np.ndarray, y_prob: np.ndarray,
        n_bins: int = 10) -> float:
    """ECE. y_true: 0=P1 wins; y_prob: P(P1 wins)."""
    bins  = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    p1_ind  = (1 - y_true).astype(np.float32)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        frac_pos  = p1_ind[mask].mean()  # actual P1 win rate in bin
        mean_conf = y_prob[mask].mean()   # mean predicted P(P1 wins)
        ece_val  += mask.sum() * abs(frac_pos - mean_conf)
    return float(ece_val / max(len(y_true), 1))

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC via Wilcoxon-Mann-Whitney (vectorised).
    y_true: 0=P1 wins, 1=P2 wins;  y_prob: P(P1 wins).
    Positive class = P1 wins (y_true==0), scored by y_prob."""
    pos_scores = y_prob[y_true == 0]  # P1 won → high score is correct
    neg_scores = y_prob[y_true == 1]  # P2 won → low  score is correct
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    diff = pos_scores[:, None] - neg_scores[None, :]  # (n_pos, n_neg)
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (n_pos * n_neg))


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                       n_bins: int = 10):
    """Reliability diagram. y_true: 0=P1 wins; y_prob: P(P1 wins)."""
    bins   = np.linspace(0, 1, n_bins + 1)
    frac_pos, mean_conf, counts = [], [], []
    p1_ind = (1 - y_true).astype(np.float32)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() < 3:
            continue
        frac_pos.append(float(p1_ind[mask].mean()))  # actual P1 win rate
        mean_conf.append(float(y_prob[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(mean_conf), np.array(frac_pos), np.array(counts)


# ── NN MODEL (minimal re-implementation for loading) ─────────────────────────

if TORCH_OK:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=16, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])

    class TennisMatchNet(nn.Module):
        def __init__(self, feature_dim=48, d_model=128,
                     lstm_layers=2, nhead=4, tf_layers=2, dropout=0.25):
            super().__init__()
            self.feature_dim = feature_dim; self.d_model = d_model
            self.input_proj = nn.Sequential(
                nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Dropout(dropout))
            self.lstm = nn.LSTM(d_model, d_model//2, lstm_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if lstm_layers > 1 else 0.)
            self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model, nhead, d_model*4, dropout, "gelu",
                batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(enc, num_layers=tf_layers)
            self.gate = nn.Linear(d_model*2, d_model*2)
            self.head = nn.Sequential(
                nn.Linear(d_model*2, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, 32),
                nn.GELU(), nn.Linear(32, 2))
        def forward(self, x):
            B = x.size(0); h = self.input_proj(x)
            _, (hn, _) = self.lstm(h)
            lstm_out = torch.cat([hn[-2], hn[-1]], dim=-1)
            cls  = self.cls_token.expand(B, -1, -1)
            t_in = self.pos_enc(torch.cat([cls, h], dim=1))
            cls_out = self.transformer(t_in)[:, 0]
            combined = torch.cat([lstm_out, cls_out], dim=-1)
            return self.head(combined * torch.sigmoid(self.gate(combined)))
        @torch.no_grad()
        def predict_proba(self, x):
            self.eval()
            return F.softmax(self.forward(x), dim=-1)

    def load_nn(path: str, device: str = "cpu"):
        ckpt   = torch.load(path, map_location=device, weights_only=False)
        fdim   = ckpt.get("feature_dim", 48)
        model  = TennisMatchNet(feature_dim=fdim).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        sc     = StandardScaler()
        sc.mean_  = ckpt["scaler_mean"]
        sc.scale_ = ckpt["scaler_scale"]
        return model, sc, ckpt


def nn_stat_features(
    prof_p1: np.ndarray, prof_p2: np.ndarray,
    surface: str, best_of: int,
    elo_feats: np.ndarray,
    feature_dim: int = 48,
    seq_len: int = 5,
) -> np.ndarray:
    """Build a single (seq_len, feature_dim) sequence using player profiles."""
    N = len(STAT_COLS_ELO)
    SURFACES_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
    surf_id = SURFACES_MAP.get(_normalize_surface(surface), 0)

    seq = np.zeros((seq_len, feature_dim), dtype=np.float32)
    # For eval we use profile means directly (no Monte Carlo noise)
    r1 = np.nan_to_num(prof_p1[:N]).astype(np.float32)
    r2 = np.nan_to_num(prof_p2[:N]).astype(np.float32)
    diff = r1 - r2

    for s in range(seq_len):
        feat = np.zeros(feature_dim, dtype=np.float32)
        feat[:N]       = r1
        feat[N:2*N]    = r2
        feat[28:34]    = diff[:6]
        feat[34]       = (s+1) / seq_len
        feat[35]       = np.clip(r1[1] - r2[1], -1, 1)
        feat[36]       = surf_id / max(len(SURFACES_MAP)-1, 1)
        feat[37]       = float(best_of == 5)
        if feature_dim >= 48:
            feat[40:48] = elo_feats
        np.nan_to_num(feat, copy=False)
        seq[s] = feat
    return seq


# ── MAIN EVALUATION LOOP ─────────────────────────────────────────────────────

def run_evaluation(
    paths: Dict[str, str],
    model_path: Optional[str],
    elo_path: Optional[str],
    surface_filter: Optional[str],
    min_matches: int,
    alpha_nn: float,
    n_recent: int,
    out_dir: str,
    device: str,
    elo_live_path: Optional[str] = None,
) -> dict:

    os.makedirs(out_dir, exist_ok=True)

    print("\n  Loading data …")
    df = load_data(paths)
    print(f"  {len(df):,} rows | {df['match_id'].nunique():,} matches | "
          f"{df['player'].nunique():,} players")

    # Load NN model if available
    nn_model = nn_scaler = nn_ckpt = None
    nn_feature_dim = 48
    if model_path and os.path.exists(model_path) and TORCH_OK:
        nn_model, nn_scaler, nn_ckpt = load_nn(model_path, device)
        nn_feature_dim = int(nn_ckpt.get("feature_dim", 48))
        print(f"  NN model loaded: feature_dim={nn_feature_dim}")
    else:
        if model_path:
            print(f"  ⚠ Model '{model_path}' not found — stat-Elo only mode.")
        else:
            print("  Running in stat-Elo only mode (no --model specified).")

    # Load Elo file if available
    elo_data = None
    elo_mean, elo_std = 1500.0, 150.0
    if elo_path and os.path.exists(elo_path):
        with open(elo_path) as f:
            elo_data = json.load(f)
        vals = [v["career_elo"] for v in elo_data.values()
                if isinstance(v.get("career_elo"), (int, float))]
        if vals:
            elo_mean = float(np.mean(vals))
            elo_std  = max(float(np.std(vals)), 1.0)
        print(f"  Elo file loaded: {len(elo_data)} players | "
              f"mean={elo_mean:.1f} std={elo_std:.1f}")
    if elo_live_path and os.path.exists(elo_live_path):
        with open(elo_live_path) as f:
            _live = json.load(f)
        if elo_data is None:
            elo_data = _live
        else:
            elo_data.update(_live)
        print(f"  Elo-live: {len(_live)} players merged (total {len(elo_data)})")

    # ── Build per-match ground-truth labels + stat profiles ──────────────────
    surfaces_to_eval = ([_normalize_surface(surface_filter)]
                        if surface_filter else SURFACES)

    all_results = []

    for surface in surfaces_to_eval:
        print(f"\n  ── Surface: {surface} ──")
        pop_mean, pop_std = compute_pop_stats(df, surface)

        surf_df   = df[df["surface"] == surface]
        match_ids = surf_df["match_id"].unique()
        print(f"     {len(match_ids):,} matches on {surface}")

        rows = []
        skipped_low_data = skipped_no_both = 0

        for mid in match_ids:
            mdf  = surf_df[surf_df["match_id"] == mid]
            meta = mdf.iloc[0]
            p1, p2 = str(meta["p1"]), str(meta["p2"])
            best_of = _safe_best_of(meta.get("best_of", 3))

            # Ground truth: player with higher mean 1st-serve-won % wins
            sp1 = mdf[mdf["player"] == p1]
            sp2 = mdf[mdf["player"] == p2]
            if sp1.empty or sp2.empty:
                skipped_no_both += 1; continue
            p1_avg = float(np.nanmean(sp1["1st_won_pct"].values))
            p2_avg = float(np.nanmean(sp2["1st_won_pct"].values))
            # Convention: 0=P1 wins, 1=P2 wins (consistent with runner.py)
            # p_stat = P(P1 wins), so correct pred when p>=0.5 and y_true==0
            y_true = 0 if p1_avg >= p2_avg else 1   # 0=P1 wins, 1=P2 wins

            # Player profiles (history EXCLUDING this match to avoid leakage)
            df_excl = df[df["match_id"] != mid]
            prof_p1, n1 = player_profile(df_excl, p1, surface, n_recent)
            prof_p2, n2 = player_profile(df_excl, p2, surface, n_recent)

            if min(n1, n2) < min_matches:
                skipped_low_data += 1; continue

            # ── Step 2-4: Stat-Elo probability ──────────────────────────
            p_stat = stat_elo_prob(prof_p1, prof_p2, pop_std, surface)

            # ── NN probability (if model loaded) ──────────────────────
            p_nn   = None
            if nn_model is not None:
                # Build Elo feature block for this pair
                elo_block = np.zeros(8, dtype=np.float32)
                if elo_data and p1 in elo_data and p2 in elo_data:
                    surf_key = _normalize_surface(surface)
                    d1, d2 = elo_data[p1], elo_data[p2]
                    c1 = float(d1.get("career_elo", elo_mean))
                    c2 = float(d2.get("career_elo", elo_mean))
                    s1 = float(d1.get("blended_elo", {}).get(surf_key, c1))
                    s2 = float(d2.get("blended_elo", {}).get(surf_key, c2))
                    r1e = float(d1.get("recent_elo", c1))
                    r2e = float(d2.get("recent_elo", c2))
                    elo_block[0] = (c1-elo_mean)/elo_std
                    elo_block[1] = (c2-elo_mean)/elo_std
                    elo_block[2] = (c1-c2)/elo_std
                    elo_block[3] = 1./(1.+10.**((c2-c1)/400.))-0.5
                    elo_block[4] = (s1-elo_mean)/elo_std
                    elo_block[5] = (s2-elo_mean)/elo_std
                    elo_block[6] = (s1-s2)/elo_std
                    elo_block[7] = (r1e-r2e)/elo_std

                # Use stat profile cols to match training order (14 stats):
                STAT14 = ["1st_won_pct","2nd_won_pct","bp_save_pct","ret_pts_pct",
                          "winners_rate","uf_rate","df_rate","ace_rate",
                          "forced_err_rate",
                          "pressure_win_pct","1st_won_pct","2nd_won_pct",
                          "uf_rate","df_rate"]
                prof14_p1 = np.array([prof_p1[STAT_COLS_ELO.index(c)]
                                       if c in STAT_COLS_ELO else 0.
                                       for c in STAT14], dtype=np.float32)
                prof14_p2 = np.array([prof_p2[STAT_COLS_ELO.index(c)]
                                       if c in STAT_COLS_ELO else 0.
                                       for c in STAT14], dtype=np.float32)

                seq  = nn_stat_features(prof14_p1, prof14_p2,
                                         surface, best_of, elo_block,
                                         nn_feature_dim)
                seq_s = nn_scaler.transform(
                    seq.reshape(-1, nn_feature_dim)
                ).reshape(1, 5, nn_feature_dim).astype(np.float32)
                x_t  = torch.tensor(seq_s, dtype=torch.float32).to(device)
                with torch.no_grad():
                    p_nn = float(nn_model.predict_proba(x_t)[0, 0].item())

            # ── Step 5: Blend ───────────────────────────────────────────
            if p_nn is not None:
                nn_extremity = 2 * abs(p_nn - 0.5)
                alpha_adj    = alpha_nn * max(0.05, 1.0 - nn_extremity)
                total_w      = alpha_adj + (1.0 - alpha_nn)
                p_final      = (alpha_adj * p_nn +
                                (1.0 - alpha_nn) * p_stat) / total_w
            else:
                p_final = p_stat

            rows.append({
                "match_id":  mid,
                "surface":   surface,
                "p1": p1, "p2": p2,
                "y_true":    y_true,
                "p_stat":    round(p_stat, 4),
                "p_nn":      round(p_nn, 4) if p_nn is not None else None,
                "p_final":   round(float(np.clip(p_final, 0.01, 0.99)), 4),
                "n1": n1, "n2": n2,
            })

        print(f"     Evaluated : {len(rows):,} matches")
        print(f"     Skipped (low data): {skipped_low_data:,}  "
              f"(no both players): {skipped_no_both:,}")

        if len(rows) < 10:
            print(f"     ⚠ Too few matches to report metrics — skipping.")
            continue

        all_results.extend(rows)

        # ── Per-surface metrics ──────────────────────────────────────────
        y_t  = np.array([r["y_true"]  for r in rows], dtype=np.float32)
        p_s  = np.array([r["p_stat"]  for r in rows], dtype=np.float32)
        p_f  = np.array([r["p_final"] for r in rows], dtype=np.float32)

        print(f"\n     {'Metric':<22s}  {'Stat-Elo':>9s}  {'Blended':>9s}")
        print(f"     {'─'*22}  {'─'*9}  {'─'*9}")
        print(f"     {'Accuracy':<22s}  "
              f"  {accuracy(y_t,p_s)*100:>7.2f}%  {accuracy(y_t,p_f)*100:>7.2f}%")
        metrics = {}
        for label, probs in [("stat_elo", p_s), ("blended", p_f)]:
            metrics[label] = {
                "accuracy":    round(accuracy(y_t, probs), 4),
                "brier":       round(brier_score(y_t, probs), 4),
                "log_loss":    round(log_loss(y_t, probs), 4),
                "ece":         round(ece(y_t, probs), 4),
                "roc_auc":     round(roc_auc(y_t, probs), 4),
                "n":           len(rows),
            }
            print(f"     {'─'*48}")
            print(f"     [{label}]")
            for k, v in metrics[label].items():
                if k != "n":
                    print(f"       {k:<14s}: {v:.4f}")

        # ── Calibration plot ──────────────────────────────────────────
        if MPL_OK and len(rows) >= 30:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f"Live Performance Test — {surface}  "
                         f"(n={len(rows)})", fontsize=13, fontweight="bold")

            # Reliability diagram
            ax = axes[0]
            for label, probs, color in [
                ("Stat-Elo", p_s, "#1a6fa3"),
                ("Blended",  p_f, "#e07b39"),
            ]:
                mc, fp, cnt = calibration_curve(y_t, probs)
                if len(mc) > 1:
                    ax.plot(mc, fp, "o-", color=color, lw=2, label=label,
                            markersize=5)
            ax.plot([0,1],[0,1],"--", color="#999", lw=1.5, label="Perfect")
            ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction positive")
            ax.set_title("Calibration (Reliability Diagram)")
            ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.grid(alpha=0.3)

            # Accuracy bar
            ax = axes[1]
            labels_bar = ["Stat-Elo", "Blended"]
            accs = [accuracy(y_t, p_s)*100, accuracy(y_t, p_f)*100]
            bars = ax.bar(labels_bar, accs, color=["#1a6fa3","#e07b39"],
                          width=0.5, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, accs):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_ylim(45, min(max(accs)+8, 100))
            ax.axhline(50, color="#ccc", linestyle="--", lw=1)
            ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy by Signal")
            ax.grid(axis="y", alpha=0.3)

            # Brier & ECE bars
            ax = axes[2]
            x  = np.arange(2)
            w  = 0.3
            briers = [brier_score(y_t, p_s), brier_score(y_t, p_f)]
            eces   = [ece(y_t, p_s), ece(y_t, p_f)]
            b1 = ax.bar(x - w/2, briers, w, label="Brier ↓",
                        color=["#1a6fa3","#e07b39"], edgecolor="white")
            b2 = ax.bar(x + w/2, eces,   w, label="ECE ↓",
                        color=["#1a6fa3","#e07b39"], alpha=0.55, edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(["Stat-Elo","Blended"])
            ax.set_ylabel("Score (lower = better)")
            ax.set_title("Brier Score & ECE"); ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(out_dir, f"eval_{surface.lower()}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n     Plot → {plot_path}")

    # ── Overall metrics ───────────────────────────────────────────────────────
    if len(all_results) < 10:
        print("\n  ⚠ Not enough results to compute overall metrics.")
        return {}

    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS  ({len(all_results):,} matches)")
    print(f"{'='*60}")

    y_t  = np.array([r["y_true"]  for r in all_results], dtype=np.float32)
    p_s  = np.array([r["p_stat"]  for r in all_results], dtype=np.float32)
    p_f  = np.array([r["p_final"] for r in all_results], dtype=np.float32)

    overall = {}
    for label, probs in [("stat_elo", p_s), ("blended", p_f)]:
        overall[label] = {
            "accuracy":  round(accuracy(y_t, probs), 4),
            "brier":     round(brier_score(y_t, probs), 4),
            "log_loss":  round(log_loss(y_t, probs), 4),
            "ece":       round(ece(y_t, probs), 4),
            "roc_auc":   round(roc_auc(y_t, probs), 4),
            "n":         len(all_results),
        }
        print(f"\n  [{label}]")
        for k, v in overall[label].items():
            if k != "n":
                print(f"    {k:<14s}: {v:.4f}")

    # Overall calibration + ROC plot
    if MPL_OK and len(all_results) >= 30:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Overall Evaluation  (n={len(all_results):,})",
                     fontsize=13, fontweight="bold")

        # Reliability diagram
        ax = axes[0]
        for label, probs, color in [
            ("Stat-Elo", p_s, "#1a6fa3"),
            ("Blended",  p_f, "#e07b39"),
        ]:
            mc, fp, cnt = calibration_curve(y_t, probs)
            if len(mc) > 1:
                ax.plot(mc, fp, "o-", color=color, lw=2, label=label, ms=5)
        ax.plot([0,1],[0,1],"--",color="#999",lw=1.5,label="Perfect")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Calibration Curve"); ax.legend()
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)

        # Metric comparison
        ax = axes[1]
        met_labels = ["Accuracy","Brier↓","LogLoss↓","ECE↓","AUC"]
        met_s = [overall["stat_elo"]["accuracy"],
                 1-overall["stat_elo"]["brier"],
                 1-overall["stat_elo"]["log_loss"]/2,
                 1-overall["stat_elo"]["ece"],
                 overall["stat_elo"]["roc_auc"]]
        met_f = [overall["blended"]["accuracy"],
                 1-overall["blended"]["brier"],
                 1-overall["blended"]["log_loss"]/2,
                 1-overall["blended"]["ece"],
                 overall["blended"]["roc_auc"]]
        x = np.arange(len(met_labels)); w = 0.3
        ax.bar(x-w/2, met_s, w, label="Stat-Elo", color="#1a6fa3")
        ax.bar(x+w/2, met_f, w, label="Blended",  color="#e07b39")
        ax.set_xticks(x); ax.set_xticklabels(met_labels, fontsize=9)
        ax.set_ylim(0.4, 1.0); ax.set_title("Metric Comparison")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "eval_overall.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Overall plot → {plot_path}")

    # Save CSVs and JSON
    pd.DataFrame(all_results).to_csv(
        os.path.join(out_dir, "eval_predictions.csv"), index=False)
    summary = {
        "n_matches": len(all_results),
        "surfaces":  surfaces_to_eval,
        "alpha_nn":  alpha_nn,
        "k_scale":   K_SCALE,
        "metrics":   overall,
        "stat_weights": {s: w.tolist() for s, w in STAT_WEIGHTS.items()},
        "stat_cols": STAT_COLS_ELO,
    }
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Predictions → {out_dir}/eval_predictions.csv")
    print(f"  Summary     → {out_dir}/eval_summary.json")

    return summary


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--data-dir",  type=str, default=".",
                    help="Directory with cached CSV files")
    ap.add_argument("--model",     type=str, default=None,
                    help="Path to tennis_winner_model.pt (optional)")
    ap.add_argument("--elo",       type=str, default="player_elo.json",
                    help="Path to player_elo.json (optional)")
    ap.add_argument("--elo-live",  type=str, default=None, dest="elo_live",
                    help="Live/updated Elo JSON (merged on top of --elo; live values take priority)")
    ap.add_argument("--surface",   type=str, default=None,
                    choices=["Hard","Clay","Grass","Carpet"],
                    help="Evaluate only this surface")
    ap.add_argument("--min-matches", type=int, default=5,
                    help="Min historical matches required per player")
    ap.add_argument("--alpha",     type=float, default=ALPHA_NN,
                    help="NN blend weight (0=stat-only, 0.25=spec default)")
    ap.add_argument("--recent",    type=int,   default=50,
                    help="Max recent matches for player profile")
    ap.add_argument("--out-dir",   type=str,   default="eval_out",
                    help="Output directory for plots and CSVs")
    ap.add_argument("--device",    type=str,   default="cpu")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(" Live Performance Test — Stat-Profile Elo Evaluation")
    print(f"{'='*60}")
    print(f"  Pipeline steps: z-score diff → weighted D → Elo sigmoid → blend")
    print(f"  k={K_SCALE}  alpha_nn={args.alpha}  min_matches={args.min_matches}")
    print(f"  Output dir: {args.out_dir}")

    paths = _download(args.data_dir)

    run_evaluation(
        paths         = paths,
        model_path    = args.model,
        elo_path      = args.elo,
        surface_filter= args.surface,
        min_matches   = args.min_matches,
        alpha_nn      = args.alpha,
        n_recent      = args.recent,
        out_dir       = args.out_dir,
        device        = args.device,
        elo_live_path  = getattr(args, "elo_live", None),
    )

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()

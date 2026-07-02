#!/usr/bin/env python3
"""
model_v2.py — Shared feature engineering, NN architecture, calibration
=========================================================================
Single source of truth imported by train_v2.py, predict_v2.py and
tournament_v2.py. `runner.py`/`predict.py` (v1) duplicated feature-building
logic between training and inference and let the two copies silently drift
apart (see CLAUDE.md). This module exists so that never happens again for
v2: the exact same `engineer_features()` builds the numeric matrix in
training, in single-match inference, and inside the tournament simulator.

Primary model: TennisEmbeddingNet (see MODEL_V2_REPORT.md, section 4, for
the full mathematical justification). Player identity is folded in through
a shared embedding table rather than one-hot/name features, which is what
lets the model transfer statistical strength from data-rich players to
data-poor ones (rookies, journeymen, old/rare match-ups) instead of just
memorizing or ignoring them.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from data_pipeline_v2 import STAT_KEYS, SURFACES  # single source, see data_pipeline_v2.py

OOV_IDX = 0  # reserved embedding row for unknown / never-seen-in-training players


# ── FEATURE ENGINEERING (shared by training + both inference paths) ────────

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Numeric feature matrix. Every column here must be computable strictly
    pre-match (see data_pipeline_v2.py header for the walk-forward contract).
    Player identity is NOT included here — it goes through the embedding
    table in TennisEmbeddingNet instead (see build_player_vocab / encode_ids)."""
    X = pd.DataFrame(index=df.index)

    for s in SURFACES:
        X[f"surface_{s}"] = (df["surface"] == s).astype(int)
    X["is_bo5"] = df["is_bo5"].astype(int)
    X["is_slam"] = df["is_slam"].astype(int)

    X["elo_diff"] = df["p1_elo"] - df["p2_elo"]
    X["elo_surf_diff"] = df["p1_elo_surf"] - df["p2_elo_surf"]
    X["p1_elo"] = df["p1_elo"]; X["p2_elo"] = df["p2_elo"]

    X["rank_points_log_diff"] = np.log1p(df["p1_rank_points"]) - np.log1p(df["p2_rank_points"])
    X["rank_log_diff"] = np.log1p(df["p2_rank"]) - np.log1p(df["p1_rank"])

    X["age_diff"] = df["p1_age"] - df["p2_age"]
    X["ht_diff"] = df["p1_ht"] - df["p2_ht"]
    X["p1_lefty"] = (df["p1_hand"] == "L").astype(int)
    X["p2_lefty"] = (df["p2_hand"] == "L").astype(int)

    X["winrate_recent_diff"] = df["p1_winrate_recent"] - df["p2_winrate_recent"]
    X["winrate_recent_surf_diff"] = df["p1_winrate_recent_surf"] - df["p2_winrate_recent_surf"]
    X["matches_surf_diff"] = df["p1_matches_surf"] - df["p2_matches_surf"]
    X["experience_diff"] = np.log1p(df["p1_matches"]) - np.log1p(df["p2_matches"])

    X["rest_days_diff"] = (df["p1_rest_days"] - df["p2_rest_days"]).clip(-60, 60)

    X["h2h_diff"] = df["h2h_diff_p1"]

    for k in STAT_KEYS:
        X[f"form_{k}_diff"] = df[f"p1_form_{k}"] - df[f"p2_form_{k}"]
        X[f"p1_form_{k}"] = df[f"p1_form_{k}"]
        X[f"p2_form_{k}"] = df[f"p2_form_{k}"]

    feature_cols = list(X.columns)
    return X, feature_cols


def temporal_split(df: pd.DataFrame):
    tr = df["date"] < 20220101
    va = (df["date"] >= 20220101) & (df["date"] < 20230101)
    ca = (df["date"] >= 20230101) & (df["date"] < 20240101)
    te = df["date"] >= 20240101
    return tr, va, ca, te


# ── PLAYER VOCABULARY (fit on train split ONLY — never on val/calib/test) ──

def build_player_vocab(df_train: pd.DataFrame) -> Dict[int, int]:
    """player_id -> embedding row index. Index 0 is reserved for OOV
    (players never seen in the training window — new/young pros at
    inference time). Fitting this on train-only mirrors the imputer/scaler
    discipline: a player who debuts in 2023 must be OOV to a model whose
    training window ends in 2021, exactly as it would be in production."""
    ids = sorted(pd.concat([df_train["p1_id"], df_train["p2_id"]]).unique().tolist())
    return {int(pid): i + 1 for i, pid in enumerate(ids)}


def encode_ids(df: pd.DataFrame, vocab: Dict[int, int], col: str) -> np.ndarray:
    return df[col].map(lambda pid: vocab.get(int(pid), OOV_IDX)).values.astype(np.int64)


# ── METRICS ──────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, p, n_bins=10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def eval_metrics(y_true, p, name: str) -> Dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "n": int(len(y_true)),
        "accuracy": float(((p >= 0.5).astype(int) == y_true).mean()),
        "log_loss": float(log_loss(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ece": expected_calibration_error(np.asarray(y_true), np.asarray(p)),
    }


# ── CALIBRATION (Platt scaling; see MODEL_V2_REPORT.md sect. 4.5) ──────────

def platt_fit(p_raw: np.ndarray, y: np.ndarray) -> LogisticRegression:
    logit = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
    return LogisticRegression().fit(logit.reshape(-1, 1), y)


def platt_apply(lr: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
    logit = np.log(p_raw / (1 - p_raw))
    return lr.predict_proba(logit.reshape(-1, 1))[:, 1]


def choose_calibration(y_val: np.ndarray, p_val_raw: np.ndarray,
                        iso: "IsotonicRegression", platt: LogisticRegression) -> str:
    """Pick raw / isotonic / platt by ECE on the VALIDATION fold — not the
    calib fold the calibrators were fit on (that would be circular: a
    calibrator fit on calib is trivially "best" on calib), and not the test
    fold (that would leak the model-selection decision into the number
    we're supposed to report as a blind estimate). Val was already spent
    on early stopping, so reusing it here is a second, still-legitimate
    hyperparameter-selection use, not new leakage."""
    candidates = {
        "raw": p_val_raw,
        "isotonic": iso.transform(p_val_raw),
        "platt": platt_apply(platt, p_val_raw),
    }
    eces = {name: expected_calibration_error(y_val, p) for name, p in candidates.items()}
    return min(eces, key=eces.get)


def apply_calibration(method: str, p_raw: np.ndarray, iso: "IsotonicRegression",
                       platt: LogisticRegression) -> np.ndarray:
    if method == "raw":
        return p_raw
    if method == "isotonic":
        return iso.transform(p_raw)
    if method == "platt":
        return platt_apply(platt, p_raw)
    raise ValueError(f"unknown calibration method: {method}")


# ── PRIMARY MODEL: embedding-based pairwise-comparison network ────────────

class TennisEmbeddingNet(nn.Module):
    """
    f(p1, p2, x) = sigma( MLP([e(p1) - e(p2), e(p1) + e(p2), x]) )

    where e(.) is a shared player embedding (same weight matrix looks up
    p1 and p2 — a Siamese design) and x is the numeric pre-match feature
    diff vector from engineer_features(). See MODEL_V2_REPORT.md section 4
    for the full derivation; in short:
      - emb_diff = e(p1)-e(p2) is antisymmetric under swapping p1<->p2,
        mirroring the antisymmetry already built into the numeric diff
        features and into the label itself (P(p1 wins | swap) = 1 - P(...)).
        This is the same structural idea as a Bradley-Terry / Elo model,
        generalized from a scalar rating to a learned vector.
      - emb_sum = e(p1)+e(p2) is symmetric (order-invariant) and lets the
        network model swap-invariant context, e.g. "two big servers" style
        interactions that a pure rating difference cannot express.
      - LayerNorm (not BatchNorm) is used deliberately so a single-sample
        forward pass at inference (batch size 1, for predict_v2.py) behaves
        identically to a batched training forward pass — BatchNorm's
        batch-statistics dependence was a latent bug risk in the old
        TennisMatchNet (runner.py) for single-match inference.
    """

    def __init__(self, n_players: int, num_dim: int, emb_dim: int = 24,
                 hidden: Tuple[int, int] = (96, 48), dropout: float = 0.35):
        super().__init__()
        self.emb_dim = emb_dim
        self.player_emb = nn.Embedding(n_players + 1, emb_dim, padding_idx=OOV_IDX)
        nn.init.normal_(self.player_emb.weight, std=0.05)
        with torch.no_grad():
            self.player_emb.weight[OOV_IDX].zero_()

        trunk_in = 2 * emb_dim + num_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden[0]), nn.LayerNorm(hidden[0]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.LayerNorm(hidden[1]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, p1_idx: torch.Tensor, p2_idx: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        e1 = self.player_emb(p1_idx)
        e2 = self.player_emb(p2_idx)
        h = torch.cat([e1 - e2, e1 + e2, x_num], dim=-1)
        return self.trunk(h).squeeze(-1)


def train_embedding_net(model: TennisEmbeddingNet, p1_tr, p2_tr, Xtr, ytr,
                         p1_va, p2_va, Xva, yva, device="cpu",
                         epochs=200, patience=15, lr=1e-3, weight_decay=1e-4,
                         batch_size=512, seed=42) -> Dict:
    torch.manual_seed(seed)
    """weight_decay is applied to ALL parameters including the embedding
    table: an L2 penalty on an embedding row is equivalent (MAP estimation)
    to a Gaussian prior N(0, 1/(2*n*weight_decay)) on that player's latent
    vector. Rare players receive few gradient updates, so their row is
    pulled back toward the OOV zero-vector by this prior between updates —
    i.e. towards "no special information beyond the numeric features",
    which is exactly the desired cold-start fallback. This is the same
    regularization mechanism used for latent-factor shrinkage in
    collaborative filtering (Koren, Bell & Volinsky, 2009)."""
    to_t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    p1_tr_t, p2_tr_t, Xtr_t, ytr_t = to_t(p1_tr, torch.long), to_t(p2_tr, torch.long), to_t(Xtr, torch.float32), to_t(ytr, torch.float32)
    p1_va_t, p2_va_t, Xva_t, yva_t = to_t(p1_va, torch.long), to_t(p2_va, torch.long), to_t(Xva, torch.float32), to_t(yva, torch.float32)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    n = len(Xtr_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(p1_tr_t[idx], p2_tr_t[idx], Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(p1_va_t, p2_va_t, Xva_t), yva_t).item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return history


@torch.no_grad()
def embedding_net_predict(model: TennisEmbeddingNet, p1_idx, p2_idx, x_num, device="cpu") -> np.ndarray:
    model.eval()
    to_t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    logits = model(to_t(p1_idx, torch.long), to_t(p2_idx, torch.long), to_t(x_num, torch.float32))
    return torch.sigmoid(logits).cpu().numpy()


@torch.no_grad()
def embedding_net_mc_predict(model: TennisEmbeddingNet, p1_idx, p2_idx, x_num,
                              n_samples: int = 200, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo dropout (Gal & Ghahramani, 2016): keep dropout ACTIVE at
    inference and repeat the forward pass. The spread across samples is an
    approximate epistemic-uncertainty interval — useful for single-match
    prediction (predict_v2.py), where "how confident is this probability
    itself" matters for bet sizing, not just the point estimate."""
    model.train()  # keep dropout active; LayerNorm makes batch size 1 safe
    to_t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    p1_t, p2_t, x_t = to_t(p1_idx, torch.long), to_t(p2_idx, torch.long), to_t(x_num, torch.float32)
    samples = np.stack([
        torch.sigmoid(model(p1_t, p2_t, x_t)).cpu().numpy() for _ in range(n_samples)
    ], axis=0)
    model.eval()
    return samples.mean(axis=0), samples


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)

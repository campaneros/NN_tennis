#!/usr/bin/env python3
"""
train_v2.py — Pre-match ATP win-probability model: training + benchmark
=========================================================================
Trains and compares, on the SAME leak-free, walk-forward feature table
produced by data_pipeline_v2.py:

  0. Elo-only baseline      — zero-parameter Elo sigmoid (career Elo diff)
  1. Logistic Regression    — linear interpretable baseline
  2. XGBoost (GBDT)         — strong non-neural benchmark
  3. TennisEmbeddingNet     — PRIMARY model: player-embedding neural net
                              (see model_v2.py + MODEL_V2_REPORT.md sect. 4)

The neural net is the model under study; XGBoost/LogReg/Elo are kept as
benchmarks it must beat (or, if it doesn't, that must be reported honestly
— see MODEL_V2_REPORT.md sect. 5).

Strictly temporal split (no shuffling across time):
  train  : date <  2022-01-01
  val    : 2022-01-01 <= date < 2023-01-01   (early stopping)
  calib  : 2023-01-01 <= date < 2024-01-01   (post-hoc probability calibration)
  test   : date >= 2024-01-01                (final blind holdout, never touched
                                               for fitting/tuning/calibration)

All feature engineering (imputation medians, scaler stats, player vocab) is
fit on `train` ONLY and applied to val/calib/test — no lookahead.

Usage:
  python train_v2.py --data atp_matches_pretrain.csv --out-dir models_v2
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_v2 import (
    TennisEmbeddingNet, apply_calibration, build_player_vocab, choose_calibration,
    encode_ids, engineer_features, eval_metrics, expected_calibration_error,
    platt_apply, platt_fit, temporal_split, train_embedding_net, embedding_net_predict,
)


def elo_only_prob(df: pd.DataFrame) -> np.ndarray:
    diff = df["p1_elo"] - df["p2_elo"]
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default="atp_matches_pretrain.csv")
    ap.add_argument("--out-dir", type=str, default="models_v2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--emb-dim", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading pre-match feature table...")
    df = pd.read_csv(args.data)
    print(f"  {len(df):,} matches | date range {df.date.min()}-{df.date.max()}")

    X, feature_cols = engineer_features(df)
    y = df["y"].values.astype(int)

    tr, va, ca, te = temporal_split(df)
    print(f"  train={tr.sum():,}  val={va.sum():,}  calib={ca.sum():,}  test={te.sum():,}")

    Xtr, ytr = X[tr].values, y[tr]
    Xva, yva = X[va].values, y[va]
    Xca, yca = X[ca].values, y[ca]
    Xte, yte = X[te].values, y[te]

    results = []

    # ── 0. Elo-only baseline (zero-parameter) ──────────────────────────
    print("\n[0/4] Elo-only baseline (unfitted sigmoid)...")
    p_elo_test = elo_only_prob(df[te])
    results.append(eval_metrics(yte, p_elo_test.values, "elo_only"))

    # ── 1. Logistic Regression baseline ─────────────────────────────────
    print("[1/4] Logistic Regression...")
    imputer = SimpleImputer(strategy="median").fit(Xtr)
    scaler = StandardScaler().fit(imputer.transform(Xtr))

    def prep(Xarr):
        return scaler.transform(imputer.transform(Xarr))

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(prep(Xtr), ytr)
    p_lr_test = lr.predict_proba(prep(Xte))[:, 1]
    results.append(eval_metrics(yte, p_lr_test, "logistic_regression"))

    # ── 2. XGBoost (strong non-neural benchmark) ────────────────────────
    print("[2/4] XGBoost...")
    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feature_cols, missing=np.nan)
    dval = xgb.DMatrix(Xva, label=yva, feature_names=feature_cols, missing=np.nan)
    dcalib = xgb.DMatrix(Xca, feature_names=feature_cols, missing=np.nan)
    dtest = xgb.DMatrix(Xte, feature_names=feature_cols, missing=np.nan)

    xgb_params = dict(
        objective="binary:logistic", eval_metric="logloss",
        max_depth=4, eta=0.03, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=20, reg_lambda=2.0, reg_alpha=0.5,
        tree_method="hist", seed=42,
    )
    booster = xgb.train(
        xgb_params, dtrain, num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50, verbose_eval=False,
    )
    print(f"  best_iteration={booster.best_iteration}  best_val_logloss={booster.best_score:.4f}")

    p_xgb_val_raw = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    p_xgb_calib_raw = booster.predict(dcalib, iteration_range=(0, booster.best_iteration + 1))
    p_xgb_test_raw = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    results.append(eval_metrics(yte, p_xgb_test_raw, "xgboost_raw"))

    iso_xgb = IsotonicRegression(out_of_bounds="clip").fit(p_xgb_calib_raw, yca)
    results.append(eval_metrics(yte, iso_xgb.transform(p_xgb_test_raw), "xgboost_isotonic"))
    platt_xgb = platt_fit(p_xgb_calib_raw, yca)
    results.append(eval_metrics(yte, platt_apply(platt_xgb, p_xgb_test_raw), "xgboost_platt"))

    xgb_calib_method = choose_calibration(yva, p_xgb_val_raw, iso_xgb, platt_xgb)
    results.append(eval_metrics(
        yte, apply_calibration(xgb_calib_method, p_xgb_test_raw, iso_xgb, platt_xgb),
        f"xgboost_SHIPPED[{xgb_calib_method}]"))

    importance = booster.get_score(importance_type="gain")
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:20])

    # ── 3. TennisEmbeddingNet (PRIMARY model) ───────────────────────────
    print("[3/4] TennisEmbeddingNet (player-embedding NN, primary model)...")
    vocab = build_player_vocab(df[tr])
    n_players = len(vocab)
    print(f"  player vocab size (train-only) = {n_players:,}")

    p1_tr, p2_tr = encode_ids(df[tr], vocab, "p1_id"), encode_ids(df[tr], vocab, "p2_id")
    p1_va, p2_va = encode_ids(df[va], vocab, "p1_id"), encode_ids(df[va], vocab, "p2_id")
    p1_ca, p2_ca = encode_ids(df[ca], vocab, "p1_id"), encode_ids(df[ca], vocab, "p2_id")
    p1_te, p2_te = encode_ids(df[te], vocab, "p1_id"), encode_ids(df[te], vocab, "p2_id")
    oov_rate_test = float(((p1_te == 0) | (p2_te == 0)).mean())
    print(f"  test-set OOV-player match rate = {oov_rate_test:.3f} (never-before-seen players)")

    Xtr_p, Xva_p, Xca_p, Xte_p = prep(Xtr), prep(Xva), prep(Xca), prep(Xte)

    net = TennisEmbeddingNet(n_players=n_players, num_dim=Xtr_p.shape[1], emb_dim=args.emb_dim)
    nn_history = train_embedding_net(
        net, p1_tr, p2_tr, Xtr_p, ytr.astype(np.float32),
        p1_va, p2_va, Xva_p, yva.astype(np.float32), device=args.device,
    )
    p_nn_val_raw = embedding_net_predict(net, p1_va, p2_va, Xva_p, device=args.device)
    p_nn_calib_raw = embedding_net_predict(net, p1_ca, p2_ca, Xca_p, device=args.device)
    p_nn_test_raw = embedding_net_predict(net, p1_te, p2_te, Xte_p, device=args.device)
    results.append(eval_metrics(yte, p_nn_test_raw, "embedding_nn_raw"))

    iso_nn = IsotonicRegression(out_of_bounds="clip").fit(p_nn_calib_raw, yca)
    results.append(eval_metrics(yte, iso_nn.transform(p_nn_test_raw), "embedding_nn_isotonic"))
    platt_nn = platt_fit(p_nn_calib_raw, yca)
    results.append(eval_metrics(yte, platt_apply(platt_nn, p_nn_test_raw), "embedding_nn_platt"))

    nn_calib_method = choose_calibration(yva, p_nn_val_raw, iso_nn, platt_nn)
    results.append(eval_metrics(
        yte, apply_calibration(nn_calib_method, p_nn_test_raw, iso_nn, platt_nn),
        f"embedding_nn_SHIPPED[{nn_calib_method}]"))
    print(f"  calibration method selected by validation-fold ECE: {nn_calib_method}")

    # ── Report ────────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 78)
    print(res_df.to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 78)

    # ── Save artifacts ───────────────────────────────────────────────────
    booster.save_model(os.path.join(args.out_dir, "xgb_model.json"))
    torch.save(net.state_dict(), os.path.join(args.out_dir, "embedding_nn.pt"))
    with open(os.path.join(args.out_dir, "player_vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(args.out_dir, "preprocessing.pkl"), "wb") as f:
        pickle.dump({
            "imputer": imputer, "scaler": scaler,
            "iso_xgb": iso_xgb, "platt_xgb": platt_xgb,
            "iso_nn": iso_nn, "platt_nn": platt_nn,
            "nn_calib_method": nn_calib_method,
            "xgb_calib_method": xgb_calib_method,
            "feature_cols": feature_cols,
            "num_dim": Xtr_p.shape[1],
            "emb_dim": args.emb_dim,
            "n_players": n_players,
        }, f)
    with open(os.path.join(args.out_dir, "logreg_model.pkl"), "wb") as f:
        pickle.dump(lr, f)
    res_df.to_csv(os.path.join(args.out_dir, "benchmark_results.csv"))
    with open(os.path.join(args.out_dir, "feature_importance_xgb.json"), "w") as f:
        json.dump(importance, f, indent=2)
    with open(os.path.join(args.out_dir, "nn_history.json"), "w") as f:
        json.dump(nn_history, f, indent=2)
    with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
        json.dump({
            "train_n": int(tr.sum()), "val_n": int(va.sum()),
            "calib_n": int(ca.sum()), "test_n": int(te.sum()),
            "train_end": "2022-01-01", "val_end": "2023-01-01", "calib_end": "2024-01-01",
            "test_oov_player_rate": oov_rate_test,
        }, f, indent=2)

    print(f"\nSaved model artifacts -> {args.out_dir}/")
    print("Top-15 XGBoost feature importances (gain):")
    for k, v in list(importance.items())[:15]:
        print(f"  {k:<28} {v:.1f}")


if __name__ == "__main__":
    main()

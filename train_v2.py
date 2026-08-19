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
    swap_sides,
    TennisEmbeddingNet, apply_calibration, build_player_vocab, choose_calibration,
    encode_ids, engineer_features, eval_metrics, expected_calibration_error,
    platt_apply, platt_fit, temporal_split, train_embedding_net, embedding_net_predict,
)


def elo_only_prob(df: pd.DataFrame) -> np.ndarray:
    diff = df["p1_elo"] - df["p2_elo"]
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def training_arrays(df_tr, X_tr, y_tr, vocab, prep, args):
    """(p1_idx, p2_idx, X_prepped, y) for the NN, optionally doubled with
    the mirrored rows (swap_sides): teaches P(p1 wins|swap)=1-P directly."""
    p1, p2 = encode_ids(df_tr, vocab, "p1_id"), encode_ids(df_tr, vocab, "p2_id")
    Xp, yy = prep(X_tr), y_tr.astype(np.float32)
    if args.no_swap_aug:
        return p1, p2, Xp, yy
    Xs, _ = engineer_features(swap_sides(df_tr))
    return (np.concatenate([p1, p2]), np.concatenate([p2, p1]),
            np.vstack([Xp, prep(Xs.values)]), np.concatenate([yy, 1 - yy]))


def make_net(n_players, num_dim, args):
    return TennisEmbeddingNet(n_players=n_players, num_dim=num_dim, emb_dim=args.emb_dim,
                              player_dropout=args.player_dropout)


def train_final(df, X, feature_cols, y, args):
    """PRODUCTION refit (--final), two stages:

    Stage 1 (model selection): train on data < 2025-07-01, early-stop on
      2025-H2 val, fit calibrators there, pick calibration method on the
      2026 slice. This discovers the ONE hyperparameter that can't be fixed
      a priori — how many epochs to train — plus the calibration choice,
      exactly as in the benchmark protocol.
    Stage 2 (shipped model): retrain from scratch on ALL rows (through the
      end of the data file) for exactly the epoch count stage 1 found.
      Standard train-on-all-after-selection refit (Hastie et al., ESL
      §7.10): the most recent matches — the most informative ones for
      predicting upcoming matches — DO contribute gradient updates to the
      shipped weights; nothing is held back from final training.

    Honest-metrics note: reported performance numbers still come ONLY from
    the benchmark protocol run (train<2022 / blind test >=2024). Stage 1's
    2025-H2/2026 slices are model-selection tools here, not blind estimates,
    and stage 2 by construction has no held-out data at all.
    """
    tr = df["date"] < 20250701
    va = (df["date"] >= 20250701) & (df["date"] < 20260101)
    se = df["date"] >= 20260101
    print(f"  FINAL refit stage 1 (selection): train={tr.sum():,}  val={va.sum():,}  select={se.sum():,}")

    Xtr, ytr = X[tr].values, y[tr]
    Xva, yva = X[va].values, y[va]
    Xse, yse = X[se].values, y[se]

    # Stage-1 preprocessing/vocab fit on its own training window only
    imputer1 = SimpleImputer(strategy="median").fit(Xtr)
    scaler1 = StandardScaler().fit(imputer1.transform(Xtr))
    prep1 = lambda A: scaler1.transform(imputer1.transform(A))
    vocab1 = build_player_vocab(df[tr])
    p1_tr, p2_tr = encode_ids(df[tr], vocab1, "p1_id"), encode_ids(df[tr], vocab1, "p2_id")
    p1_va, p2_va = encode_ids(df[va], vocab1, "p1_id"), encode_ids(df[va], vocab1, "p2_id")
    p1_se, p2_se = encode_ids(df[se], vocab1, "p1_id"), encode_ids(df[se], vocab1, "p2_id")
    Xtr_p, Xva_p, Xse_p = prep1(Xtr), prep1(Xva), prep1(Xse)

    net1 = make_net(len(vocab1), Xtr_p.shape[1], args)
    hist1 = train_embedding_net(
        net1, *training_arrays(df[tr], Xtr, ytr, vocab1, prep1, args),
        p1_va, p2_va, Xva_p, yva.astype(np.float32), device=args.device)
    best_epoch = int(np.argmin(hist1["val_loss"])) + 1
    print(f"  stage 1: best epoch by val log-loss = {best_epoch} (of {len(hist1['val_loss'])} run)")

    p_va = embedding_net_predict(net1, p1_va, p2_va, Xva_p, device=args.device)
    stage1_entropy_floor = float(binary_entropy(p_va).mean())
    p_se = embedding_net_predict(net1, p1_se, p2_se, Xse_p, device=args.device)
    iso_nn = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)
    platt_nn = platt_fit(p_va, yva)
    nn_calib_method = choose_calibration(yse, p_se, iso_nn, platt_nn)
    print(f"  calibration method selected on 2026 slice: {nn_calib_method}")
    res = [eval_metrics(yse, apply_calibration(m, p_se, iso_nn, platt_nn), m)
           for m in ["raw", "isotonic", "platt"]]
    print(pd.DataFrame(res).set_index("model").to_string(float_format=lambda v: f"{v:.4f}"))
    print("  (2026-slice numbers above are stage-1 diagnostics, NOT a blind benchmark)")

    # ── Stage 2: retrain on EVERYTHING for the discovered epoch count ──────
    print(f"  FINAL refit stage 2: retraining on all {len(df):,} rows for {best_epoch} epochs...")
    imputer = SimpleImputer(strategy="median").fit(X.values)
    scaler = StandardScaler().fit(imputer.transform(X.values))
    prep = lambda A: scaler.transform(imputer.transform(A))
    vocab = build_player_vocab(df)
    print(f"  player vocab size (full data) = {len(vocab):,}")
    p1_all, p2_all = encode_ids(df, vocab, "p1_id"), encode_ids(df, vocab, "p2_id")
    Xall_p = prep(X.values)

    net = make_net(len(vocab), Xall_p.shape[1], args)
    nn_history = train_embedding_net(
        net, *training_arrays(df, X.values, y, vocab, prep, args),
        epochs=best_epoch, device=args.device)
    nn_history["stage1_val_loss"] = hist1["val_loss"]
    nn_history["stage1_best_epoch"] = best_epoch
    nn_history["val_entropy_floor"] = stage1_entropy_floor

    # Calibrators shipped are the stage-1 ones (fit on genuinely held-out
    # 2025-H2 predictions). When the selected method is "raw" — the case in
    # every run so far — they are inert. If a future retrain selects
    # isotonic/platt, note the mild approximation of applying a
    # stage-1-fitted calibrator to stage-2 outputs (documented trade-off:
    # the alternative, calibrating on stage-2 training-set predictions,
    # would be fit on overconfident in-sample outputs — strictly worse).
    torch.save(net.state_dict(), os.path.join(args.out_dir, "embedding_nn.pt"))
    with open(os.path.join(args.out_dir, "player_vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(args.out_dir, "preprocessing.pkl"), "wb") as f:
        pickle.dump({
            "imputer": imputer, "scaler": scaler,
            "iso_nn": iso_nn, "platt_nn": platt_nn,
            "nn_calib_method": nn_calib_method,
            "feature_cols": feature_cols,
            "num_dim": Xall_p.shape[1],
            "emb_dim": args.emb_dim,
            "n_players": len(vocab),
        }, f)
    with open(os.path.join(args.out_dir, "nn_history.json"), "w") as f:
        json.dump(nn_history, f, indent=2)
    with open(os.path.join(args.out_dir, "split_info.json"), "w") as f:
        json.dump({
            "mode": "final_production_refit_two_stage",
            "stage1_train_n": int(tr.sum()), "stage1_val_n": int(va.sum()), "stage1_select_n": int(se.sum()),
            "stage1_best_epoch": best_epoch,
            "stage2_train_n": int(len(df)),
            "note": "stage 2 trains on ALL data for the stage-1-discovered epoch count; "
                    "no blind test fold — performance estimates come from the benchmark protocol run",
        }, f, indent=2)
    print(f"  Saved production artifacts -> {args.out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default="atp_matches_pretrain.csv")
    ap.add_argument("--out-dir", type=str, default="models_v2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--emb-dim", type=int, default=0,
                    help="player-identity embedding size; 0 (default) = numeric features only, "
                         "which is what wins out-of-time (report sect. 9)")
    ap.add_argument("--player-dropout", type=float, default=0.0,
                    help="only with --emb-dim>0: prob. of replacing a player id with OOV during training")
    ap.add_argument("--no-swap-aug", action="store_true",
                    help="disable p1<->p2 mirroring augmentation of the training rows")
    ap.add_argument("--final", action="store_true",
                    help="production refit on data through 2025-06 (see train_final); "
                         "use --out-dir models_v2_final to keep benchmark artifacts intact")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading pre-match feature table...")
    df = pd.read_csv(args.data)
    print(f"  {len(df):,} matches | date range {df.date.min()}-{df.date.max()}")

    X, feature_cols = engineer_features(df)
    y = df["y"].values.astype(int)

    if args.final:
        train_final(df, X, feature_cols, y, args)
        return

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

    net = make_net(n_players, Xtr_p.shape[1], args)
    nn_history = train_embedding_net(
        net, *training_arrays(df[tr], Xtr, ytr, vocab, prep, args),
        p1_va, p2_va, Xva_p, yva.astype(np.float32), device=args.device,
    )
    p_nn_val_raw = embedding_net_predict(net, p1_va, p2_va, Xva_p, device=args.device)
    # Reference floor for the training plot: if the model's val probabilities
    # were EXACTLY the true probabilities, expected log-loss would equal the
    # mean binary entropy of those probabilities. val_loss - this = the
    # calibration gap; this - 0 = the irreducible part at the model's current
    # sharpness (only new information can lower it).
    nn_history["val_entropy_floor"] = float(binary_entropy(p_nn_val_raw).mean())
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
    print(f"  calibration method selected by validation-fold log-loss: {nn_calib_method}")

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

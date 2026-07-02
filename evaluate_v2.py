#!/usr/bin/env python3
"""
evaluate_v2.py — Reliability diagrams + final metrics table for MODEL_V2_REPORT.md
======================================================================================
Reloads the artifacts saved by train_v2.py and regenerates, on the blind
test split (date >= 2024-01-01) only, the calibration curves needed to
judge whether predicted probabilities are trustworthy enough for betting
sizing (not just whether the argmax class is correct).

Usage:
  python evaluate_v2.py --data atp_matches_pretrain.csv --out-dir models_v2
"""
import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from model_v2 import (
    TennisEmbeddingNet, apply_calibration, encode_ids, engineer_features, eval_metrics,
    platt_apply, temporal_split, embedding_net_predict, load_json,
)


def reliability_curve(y_true, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf.append(p[mask].mean())
        acc.append(y_true[mask].mean())
        cnt.append(mask.sum())
    return np.array(conf), np.array(acc), np.array(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="atp_matches_pretrain.csv")
    ap.add_argument("--out-dir", default="models_v2")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X, feature_cols = engineer_features(df)
    y = df["y"].values.astype(int)
    tr, va, ca, te = temporal_split(df)

    with open(f"{args.out_dir}/preprocessing.pkl", "rb") as f:
        prep_art = pickle.load(f)
    imputer, scaler = prep_art["imputer"], prep_art["scaler"]
    prep = lambda A: scaler.transform(imputer.transform(A))

    Xte_p = prep(X[te].values)
    yte = y[te]

    booster = xgb.Booster()
    booster.load_model(f"{args.out_dir}/xgb_model.json")
    dtest = xgb.DMatrix(Xte_p if False else X[te].values, feature_names=feature_cols, missing=np.nan)
    p_xgb = booster.predict(dtest)

    vocab = load_json(f"{args.out_dir}/player_vocab.json")
    vocab = {int(k): v for k, v in vocab.items()}
    p1_te, p2_te = encode_ids(df[te], vocab, "p1_id"), encode_ids(df[te], vocab, "p2_id")
    net = TennisEmbeddingNet(n_players=prep_art["n_players"], num_dim=prep_art["num_dim"], emb_dim=prep_art["emb_dim"])
    net.load_state_dict(torch.load(f"{args.out_dir}/embedding_nn.pt", map_location="cpu"))
    p_nn = embedding_net_predict(net, p1_te, p2_te, Xte_p)
    nn_method = prep_art.get("nn_calib_method", "platt")
    p_nn_shipped = apply_calibration(nn_method, p_nn, prep_art["iso_nn"], prep_art["platt_nn"])

    diff = df.loc[te, "p1_elo"] - df.loc[te, "p2_elo"]
    p_elo = (1.0 / (1.0 + 10.0 ** (-diff / 400.0))).values

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for name, p, style in [
        ("Elo-only", p_elo, "o-"),
        ("XGBoost (raw)", p_xgb, "s-"),
        ("EmbeddingNet (raw)", p_nn, "^-"),
        (f"EmbeddingNet (shipped: {nn_method})", p_nn_shipped, "d-"),
    ]:
        conf, acc, cnt = reliability_curve(yte, p)
        ax.plot(conf, acc, style, label=name, markersize=5)
    ax.set_xlabel("Predicted P(P1 wins)")
    ax.set_ylabel("Observed P1 win rate")
    ax.set_title("Reliability diagram — test set (matches from 2024-01-01+)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/reliability_diagram.png", dpi=140)
    print(f"Saved -> {args.out_dir}/reliability_diagram.png")

    final = {
        "elo_only": eval_metrics(yte, p_elo, "elo_only"),
        "xgboost_raw": eval_metrics(yte, p_xgb, "xgboost_raw"),
        "embedding_nn_raw": eval_metrics(yte, p_nn, "embedding_nn_raw"),
        f"embedding_nn_shipped[{nn_method}]": eval_metrics(yte, p_nn_shipped, f"embedding_nn_shipped[{nn_method}]"),
    }
    with open(f"{args.out_dir}/final_test_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

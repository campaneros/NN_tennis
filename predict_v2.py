#!/usr/bin/env python3
"""
predict_v2.py — Pre-match win-probability prediction for one future match
=============================================================================
Reconstructs each player's current (as-of-today) state by replaying the
FULL walk-forward history once through PlayerStateTracker (the exact same
class used to build the training table in data_pipeline_v2.py — no
separate "inference feature builder" to drift out of sync with training,
unlike v1's runner.py/predict.py split), then scores the hypothetical
match with the calibrated primary model (TennisEmbeddingNet).

All inputs are things you know before a ball is hit: two player names,
surface, best_of, whether the tournament is a Slam. Per the ATP domain
rule (see data_pipeline_v2.py docstring): men's Slams are best-of-5, all
other ATP tour-level matches are best-of-3 (Davis Cup was historically an
exception; flagged as a warning, not silently overridden).

Output: P(p1 wins), P(p2 wins), plus a Monte-Carlo-dropout confidence
interval (epistemic uncertainty on the model's own probability estimate —
see model_v2.embedding_net_mc_predict / Gal & Ghahramani 2016).

Usage:
  python predict_v2.py --p1 "Carlos Alcaraz" --p2 "Jannik Sinner" \\
      --surface Clay --best-of 3
  python predict_v2.py --p1 "Novak Djokovic" --p2 "Carlos Alcaraz" \\
      --surface Grass --best-of 5 --slam
  python predict_v2.py --list-players "alcaraz"
"""
import argparse
import glob
import pickle
import sys
import unicodedata
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from data_pipeline_v2 import build_pretrain_table, load_raw_atp
from model_v2 import (
    TennisEmbeddingNet, apply_calibration, embedding_net_mc_predict, embedding_net_predict,
    encode_ids, engineer_features, load_json, platt_apply,
)

_CACHE: Dict[str, object] = {}


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip().lower()


def build_name_index(data_dir: str) -> Dict[str, int]:
    files = sorted(glob.glob(f"{data_dir}/atp_matches_????.csv"))
    idx: Dict[str, int] = {}
    for fp in files:
        try:
            d = pd.read_csv(fp, usecols=["winner_id", "winner_name", "loser_id", "loser_name"], low_memory=False)
        except Exception:
            continue
        for _, r in d.iterrows():
            if pd.notna(r["winner_id"]) and pd.notna(r["winner_name"]):
                idx[_norm_name(r["winner_name"])] = int(r["winner_id"])
            if pd.notna(r["loser_id"]) and pd.notna(r["loser_name"]):
                idx[_norm_name(r["loser_name"])] = int(r["loser_id"])
    return idx


def resolve_player(name: str, name_index: Dict[str, int]) -> Optional[int]:
    key = _norm_name(name)
    if key in name_index:
        return name_index[key]
    candidates = [k for k in name_index if key in k]
    if len(candidates) == 1:
        return name_index[candidates[0]]
    if len(candidates) > 1:
        print(f"Ambiguous name '{name}', candidates: {candidates[:10]}", file=sys.stderr)
    return None


def load_state(data_dir: str, out_dir: str):
    """Replay full history once (walk-forward) to get each player's CURRENT
    state, and load the trained model artifacts. Cached in-process so a
    tournament simulation (tournament_v2.py) calling this repeatedly for
    many pairs pays the replay cost once."""
    if "tracker" in _CACHE:
        return _CACHE

    print("Replaying full match history to build current player state (one-time)...", file=sys.stderr)
    df_raw = load_raw_atp(data_dir)
    _, tracker = build_pretrain_table(df_raw)
    name_index = build_name_index(data_dir)

    with open(f"{out_dir}/preprocessing.pkl", "rb") as f:
        prep_art = pickle.load(f)
    vocab = {int(k): v for k, v in load_json(f"{out_dir}/player_vocab.json").items()}
    net = TennisEmbeddingNet(n_players=prep_art["n_players"], num_dim=prep_art["num_dim"], emb_dim=prep_art["emb_dim"])
    net.load_state_dict(torch.load(f"{out_dir}/embedding_nn.pt", map_location="cpu"))
    net.eval()

    _CACHE.update(dict(
        tracker=tracker, name_index=name_index, prep_art=prep_art,
        vocab=vocab, net=net, last_date=int(df_raw["tourney_date"].max()),
    ))
    return _CACHE


def build_match_row(pid1: int, pid2: int, surface: str, best_of: int, is_slam: bool, state) -> pd.DataFrame:
    tracker = state["tracker"]
    snap1 = tracker.snapshot(pid1, surface, state["last_date"])
    snap2 = tracker.snapshot(pid2, surface, state["last_date"])
    bio1, bio2 = tracker.bio(pid1), tracker.bio(pid2)
    h2h = tracker.h2h_diff(pid1, pid2)

    row = dict(
        date=state["last_date"], surface=surface, best_of=best_of,
        is_bo5=int(best_of == 5), is_slam=int(is_slam),
        p1_id=pid1, p2_id=pid2,
        p1_elo=snap1["elo"], p2_elo=snap2["elo"],
        p1_elo_surf=snap1["elo_surf"], p2_elo_surf=snap2["elo_surf"],
        p1_matches=snap1["n_matches"], p2_matches=snap2["n_matches"],
        p1_matches_surf=snap1["n_matches_surf"], p2_matches_surf=snap2["n_matches_surf"],
        p1_winrate_recent=snap1["winrate_recent"], p2_winrate_recent=snap2["winrate_recent"],
        p1_winrate_recent_surf=snap1["winrate_recent_surf"], p2_winrate_recent_surf=snap2["winrate_recent_surf"],
        p1_rest_days=np.nan, p2_rest_days=np.nan,  # unknown for a not-yet-scheduled future match
        p1_rank=bio1["rank"], p2_rank=bio2["rank"],
        p1_rank_points=bio1["rank_points"], p2_rank_points=bio2["rank_points"],
        p1_age=bio1["age"], p2_age=bio2["age"],
        p1_ht=bio1["ht"], p2_ht=bio2["ht"],
        p1_hand=bio1["hand"], p2_hand=bio2["hand"],
        h2h_diff_p1=h2h,
    )
    for k in ["1st_in_pct", "1st_won_pct", "2nd_won_pct", "ace_rate", "df_rate",
              "bp_saved_pct", "sv_pts_won_pct", "ret_pts_won_pct"]:
        row[f"p1_form_{k}"] = snap1[f"form_{k}"]
        row[f"p2_form_{k}"] = snap2[f"form_{k}"]
    return pd.DataFrame([row])


def predict_match_prob(pid1: int, pid2: int, surface: str, best_of: int, is_slam: bool,
                        data_dir: str = "tennis_atp", out_dir: str = "models_v2",
                        n_mc: int = 200) -> Dict:
    state = load_state(data_dir, out_dir)
    row = build_match_row(pid1, pid2, surface, best_of, is_slam, state)
    X, _ = engineer_features(row)
    prep_art = state["prep_art"]
    Xp = prep_art["scaler"].transform(prep_art["imputer"].transform(X.values))

    vocab = state["vocab"]
    p1_idx = np.array([vocab.get(pid1, 0)])
    p2_idx = np.array([vocab.get(pid2, 0)])

    # Calibration method (raw / isotonic / platt) was picked at training time
    # by validation-fold ECE, not hardcoded — see model_v2.choose_calibration
    # and MODEL_V2_REPORT.md sect. 4.5/5.2 (isotonic/Platt do not always beat
    # an already well-calibrated raw probability on a small calib fold).
    calib_fn = lambda p: apply_calibration(prep_art["nn_calib_method"], p, prep_art["iso_nn"], prep_art["platt_nn"])

    if n_mc <= 1:
        # Deterministic path (dropout off): used by tournament_v2.py to build
        # a stable, reproducible pairwise probability matrix — a fresh noisy
        # MC sample per pair would make Monte Carlo bracket simulation
        # results non-reproducible run-to-run for a REASON unrelated to the
        # bracket randomness itself.
        raw = embedding_net_predict(state["net"], p1_idx, p2_idx, Xp)
        p_point = float(calib_fn(raw)[0])
        lo = hi = p_point
    else:
        mean_raw, samples = embedding_net_mc_predict(state["net"], p1_idx, p2_idx, Xp, n_samples=n_mc)
        p_cal_samples = calib_fn(samples.reshape(-1)).reshape(samples.shape)
        p_point = float(calib_fn(mean_raw)[0])
        lo, hi = np.percentile(p_cal_samples[:, 0], [10, 90])

    return dict(
        p1_win_prob=p_point,
        p2_win_prob=1.0 - p_point,
        p1_win_ci90=(float(lo), float(hi)),
        p1_oov=bool(p1_idx[0] == 0),
        p2_oov=bool(p2_idx[0] == 0),
        p1_elo=float(row["p1_elo"].iloc[0]), p2_elo=float(row["p2_elo"].iloc[0]),
        p1_elo_surf=float(row["p1_elo_surf"].iloc[0]), p2_elo_surf=float(row["p2_elo_surf"].iloc[0]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", type=str)
    ap.add_argument("--p2", type=str)
    ap.add_argument("--surface", type=str, default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--slam", action="store_true")
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out-dir", type=str, default="models_v2")
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--list-players", type=str, default=None)
    args = ap.parse_args()

    if args.slam and args.best_of != 5:
        print("WARNING: Slam tournaments are best-of-5 for ATP men's singles; "
              "--best-of 3 with --slam is historically rare (pre-Open-era / weather "
              "shortened play) — double-check the inputs.", file=sys.stderr)
    if args.best_of == 5 and not args.slam:
        print("NOTE: best_of=5 without --slam is valid (Davis Cup live rubbers, some "
              "Tour Finals matches) but is NOT the men's Slam case; is_slam is being "
              "recorded as 0, as requested.", file=sys.stderr)

    if args.list_players is not None:
        name_index = build_name_index(args.data_dir)
        key = _norm_name(args.list_players)
        matches = sorted({k for k in name_index if key in k})
        print("\n".join(matches[:40]) or "No matches.")
        return

    if not args.p1 or not args.p2:
        ap.error("--p1 and --p2 are required (or use --list-players)")

    state = load_state(args.data_dir, args.out_dir)
    pid1 = resolve_player(args.p1, state["name_index"])
    pid2 = resolve_player(args.p2, state["name_index"])
    if pid1 is None or pid2 is None:
        print(f"Could not resolve player(s): p1={args.p1!r} -> {pid1}, p2={args.p2!r} -> {pid2}. "
              f"Try --list-players to search.", file=sys.stderr)
        sys.exit(1)

    result = predict_match_prob(pid1, pid2, args.surface, args.best_of, args.slam,
                                 args.data_dir, args.out_dir, args.n_mc)

    print(f"\n{args.p1} vs {args.p2}  |  surface={args.surface}  best_of={args.best_of}  "
          f"slam={args.slam}")
    print(f"  Career Elo:      {args.p1}={result['p1_elo']:.0f}   {args.p2}={result['p2_elo']:.0f}")
    print(f"  Surface Elo:     {args.p1}={result['p1_elo_surf']:.0f}   {args.p2}={result['p2_elo_surf']:.0f}")
    if result["p1_oov"] or result["p2_oov"]:
        print("  NOTE: one or both players had zero/near-zero training-set history "
              "(new/rare player) — prediction relies more heavily on Elo/rank/bio "
              "features than on learned player identity.")
    print(f"\n  P({args.p1} wins) = {result['p1_win_prob']:.3f}  "
          f"(90% MC-dropout interval: {result['p1_win_ci90'][0]:.3f}-{result['p1_win_ci90'][1]:.3f})")
    print(f"  P({args.p2} wins) = {result['p2_win_prob']:.3f}")


if __name__ == "__main__":
    main()

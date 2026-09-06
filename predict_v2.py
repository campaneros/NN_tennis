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
import os
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
STATE_SNAPSHOT = "deploy/state.pkl"  # written by export_state.py


def has_raw_data(data_dir: str = "tennis_atp") -> bool:
    """True only with the real tennis_atp/ checkout. (A git submodule that was
    not initialised — e.g. on Streamlit Cloud — still exists as an EMPTY dir,
    so os.path.isdir is not enough.)"""
    return os.path.exists(os.path.join(data_dir, "atp_rankings_current.csv"))


def resolve_out_dir(out_dir: Optional[str]) -> str:
    """Prefer the production refit (models_v2_final, trained through 2025 —
    see train_v2.train_final) over the benchmark artifacts (models_v2,
    trained only on pre-2022 data for honest out-of-time evaluation; its
    player embeddings are frozen at each player's pre-2022 self and should
    not be used for real predictions)."""
    if out_dir:
        return out_dir
    chosen = "models_v2_final" if os.path.isdir("models_v2_final") else "models_v2"
    print(f"Using model artifacts from {chosen}/", file=sys.stderr)
    if chosen == "models_v2":
        print("WARNING: models_v2_final/ not found — falling back to the BENCHMARK "
              "model (training window ends 2022; stale player embeddings). Run "
              "`train_v2.py --final --out-dir models_v2_final` for production use.",
              file=sys.stderr)
    return chosen


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


def load_state(data_dir: str, out_dir: Optional[str] = None):
    """Replay full history once (walk-forward) to get each player's CURRENT
    state, and load the trained model artifacts. Cached in-process so a
    tournament simulation (tournament_v2.py) calling this repeatedly for
    many pairs pays the replay cost once."""
    if "tracker" in _CACHE:
        return _CACHE

    out_dir = resolve_out_dir(out_dir)
    if os.path.exists(STATE_SNAPSHOT):
        # Pre-replayed state exported by export_state.py (Streamlit deploy path:
        # no tennis_atp/ data and no 90 s replay needed — loads in <1 s).
        with open(STATE_SNAPSHOT, "rb") as f:
            snap = pickle.load(f)
        tracker, name_index, last_date = snap["tracker"], snap["name_index"], snap["last_date"]
        _CACHE["last_active"] = snap.get("last_active", {})
        _CACHE["full_names"] = snap.get("full_names", {})
        _CACHE["minutes"] = snap.get("minutes", {})
        print(f"Loaded player state snapshot {STATE_SNAPSHOT} (data through {last_date})", file=sys.stderr)
    else:
        print("Replaying full match history to build current player state (one-time)...", file=sys.stderr)
        df_raw = load_raw_atp(data_dir)
        _, tracker = build_pretrain_table(df_raw)
        name_index = build_name_index(data_dir)
        last_date = int(df_raw["tourney_date"].max())
        _CACHE["last_active"] = tracker.last_date   # pid -> last match date (for name-pick ordering)

    if os.path.isdir("data_updated"):
        # Fresh results beyond the archive/snapshot cutoff (see live_data.py):
        # keeps Elo/form/H2H current without waiting for a tennis_atp release.
        try:
            from live_data import extend_tracker
            if "full_names" not in _CACHE:
                from fetch_bracket import load_full_names_and_last_active
                name_index, _CACHE["full_names"], _CACHE["last_active"] = load_full_names_and_last_active(data_dir)
            last_date = extend_tracker(tracker, last_date, name_index,
                                       _CACHE["full_names"], _CACHE["last_active"])
        except Exception as e:
            print(f"live_data extension skipped: {e}", file=sys.stderr)

    with open(f"{out_dir}/preprocessing.pkl", "rb") as f:
        prep_art = pickle.load(f)
    vocab = {int(k): v for k, v in load_json(f"{out_dir}/player_vocab.json").items()}
    net = TennisEmbeddingNet(n_players=prep_art["n_players"], num_dim=prep_art["num_dim"], emb_dim=prep_art["emb_dim"])
    net.load_state_dict(torch.load(f"{out_dir}/embedding_nn.pt", map_location="cpu"))
    net.eval()

    _CACHE.update(dict(
        tracker=tracker, name_index=name_index, prep_art=prep_art,
        vocab=vocab, net=net, last_date=last_date,
    ))
    return _CACHE


def build_match_row(pid1: int, pid2: int, surface: str, best_of: int, is_slam: bool, state) -> pd.DataFrame:
    tracker = state["tracker"]
    snap1 = tracker.snapshot(pid1, surface, state["last_date"])
    snap2 = tracker.snapshot(pid2, surface, state["last_date"])
    bio1, bio2 = tracker.bio(pid1), tracker.bio(pid2)
    h2h = tracker.h2h_diff(pid1, pid2)
    h2h_surf = tracker.h2h_surf_diff(pid1, pid2, surface)

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
        h2h_surf_diff_p1=h2h_surf,
    )
    for k in ["1st_in_pct", "1st_won_pct", "2nd_won_pct", "ace_rate", "df_rate",
              "bp_saved_pct", "sv_pts_won_pct", "ret_pts_won_pct"]:
        row[f"p1_form_{k}"] = snap1[f"form_{k}"]
        row[f"p2_form_{k}"] = snap2[f"form_{k}"]
    return pd.DataFrame([row])


def predict_match_prob(pid1: int, pid2: int, surface: str, best_of: int, is_slam: bool,
                        data_dir: str = "tennis_atp", out_dir: Optional[str] = None,
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
    # by validation-fold log-loss, not hardcoded — see model_v2.choose_calibration
    # and MODEL_V2_REPORT.md sect. 4.5/5.2 (isotonic/Platt do not always beat
    # an already well-calibrated raw probability on a small calib fold).
    calib_fn = lambda p: apply_calibration(prep_art["nn_calib_method"], p, prep_art["iso_nn"], prep_art["platt_nn"])

    # Point estimate is ALWAYS the deterministic forward pass (dropout off):
    # repeat runs of the same query must return the same probability, and
    # this is also what the model was calibrated/evaluated with. MC-dropout
    # (Gal & Ghahramani, 2016) is used ONLY for the uncertainty interval.
    # Symmetrized scoring: average f(p1,p2) with 1-f(p2,p1) so the output is
    # EXACTLY antisymmetric under swapping the two players (the network is
    # trained with swap augmentation to be nearly so; this removes the
    # residual order dependence entirely — same idea as test-time
    # augmentation over a known invariance).
    Xs, _ = engineer_features(build_match_row(pid2, pid1, surface, best_of, is_slam, state))
    Xsp = prep_art["scaler"].transform(prep_art["imputer"].transform(Xs.values))
    X2 = np.vstack([Xp, Xsp]); i1 = np.concatenate([p1_idx, p2_idx]); i2 = np.concatenate([p2_idx, p1_idx])
    sym = lambda p: 0.5 * (p[..., 0] + 1.0 - p[..., 1])

    raw = embedding_net_predict(state["net"], i1, i2, X2)
    p_point = float(sym(calib_fn(raw)))
    if n_mc <= 1:
        lo = hi = p_point
        p1_win_samples = np.array([p_point])
    else:
        _, samples = embedding_net_mc_predict(state["net"], i1, i2, X2, n_samples=n_mc)
        p_cal_samples = calib_fn(samples.reshape(-1)).reshape(samples.shape)
        p1_win_samples = sym(p_cal_samples)
        lo, hi = np.percentile(p1_win_samples, [10, 90])

    return dict(
        p1_win_prob=p_point,
        p2_win_prob=1.0 - p_point,
        p1_win_ci90=(float(lo), float(hi)),
        p1_win_samples=p1_win_samples,  # calibrated MC-dropout draws — reused by tournament_v2.py
        p1_oov=bool(p1_idx[0] == 0),
        p2_oov=bool(p2_idx[0] == 0),
        p1_elo=float(row["p1_elo"].iloc[0]), p2_elo=float(row["p2_elo"].iloc[0]),
        p1_elo_surf=float(row["p1_elo_surf"].iloc[0]), p2_elo_surf=float(row["p2_elo_surf"].iloc[0]),
    )


def value_bet_analysis(label: str, model_p: float, ci90: "tuple[float, float]", decimal_odds: float) -> str:
    """Single-bet framing: what you win/lose on THIS one bet, how likely each
    outcome is, how wrong the model could plausibly be (its 90% CI), and
    whether the OFFERED price overpays or underpays that risk (fair odds =
    1/p). A positive verdict never means "you will win": it means the price
    is better than the risk — you still lose (1-p) of the time."""
    implied_p = 1.0 / decimal_odds
    lo, hi = ci90
    fair = 1.0 / model_p if model_p > 0 else float("inf")
    win_amt = 100 * (decimal_odds - 1)
    lines = [
        f"\n  --- Single bet: 100 on {label} @ {decimal_odds:.2f} ---",
        f"  This one bet: WIN +{win_amt:.0f} with probability {model_p:.0%} | LOSE -100 with probability {1-model_p:.0%}",
        f"  How wrong could the model be: win probability between {lo:.0%} and {hi:.0%} (90% interval)",
        f"  Fair odds for this risk: {fair:.2f} — offered {decimal_odds:.2f} "
        f"({'the book overpays you' if decimal_odds > fair else 'the book underpays you'})",
    ]
    if lo > implied_p:
        lines.append(f"  => VALUE BET: the price overpays the risk even in the model's pessimistic case "
                      f"(worst-case P {lo:.0%} still above the {implied_p:.0%} the price charges). "
                      f"You can still lose this single bet ({1-model_p:.0%} chance) — the price is right, "
                      f"the outcome is not guaranteed.")
    elif model_p > implied_p:
        lines.append(f"  => MARGINAL: at the model's central estimate the price slightly overpays the risk, "
                      f"but if the model is at the pessimistic end ({lo:.0%}) you're being underpaid. "
                      f"The edge is smaller than the model's own uncertainty — skip, or bet small.")
    else:
        lines.append(f"  => NO BET: the price underpays the risk — you'd risk 100 to win {win_amt:.0f} "
                      f"on a {model_p:.0%} chance, and that trade is priced against you.")
    return "\n".join(lines)


def optimal_allocation(p1: float, p2_prob: float, o1: float, o2: float, kelly_fraction: float = 0.25):
    """Growth-optimal stake split across BOTH sides of a 2-outcome market.

    Betting f1 on player 1 and f2 on player 2 (fractions of bankroll), the
    log-growth to maximize is
        p*log(1 + f1*(o1-1) - f2) + (1-p)*log(1 - f1 + f2*(o2-1))
    i.e. multi-outcome Kelly (Kelly 1956; Thorp 2006). Two regimes fall out
    of it automatically, which is why this is solved numerically rather
    than with the single-bet formula:
      - 1/o1 + 1/o2 >= 1 (the normal case, the book has a margin): the
        optimum puts money on AT MOST one side — hedging both sides of a
        margin-carrying market strictly loses. Betting "both at 2.00 to
        break even" is only break-even, never profitable.
      - 1/o1 + 1/o2 < 1 (arbitrage; happens across two different books):
        the optimum backs both sides and locks a risk-free profit,
        independent of the model.
    Returns (f1, f2) already scaled by `kelly_fraction` (default quarter-
    Kelly, the standard haircut for an estimated rather than known p).
    """
    from scipy.optimize import minimize
    p = float(np.clip(p1, 1e-6, 1 - 1e-6))
    neg_growth = lambda f: -(p * np.log(max(1e-9, 1 + f[0] * (o1 - 1) - f[1]))
                             + (1 - p) * np.log(max(1e-9, 1 - f[0] + f[1] * (o2 - 1))))
    best = min((minimize(neg_growth, x0, bounds=[(0, 0.95), (0, 0.95)], method="L-BFGS-B")
                for x0 in ([0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1])), key=lambda r: r.fun)
    f1, f2 = (max(0.0, v) * kelly_fraction for v in best.x)
    return (0.0 if f1 < 1e-4 else f1), (0.0 if f2 < 1e-4 else f2)


def staking_plan(n1: str, n2: str, p1: float, ci1, o1: float, o2: float,
                 kelly_fraction: float = 0.25) -> str:
    """One combined recommendation given odds on BOTH players: who to back,
    how much, and the same CI-robustness check value_bet_analysis applies
    per side (an edge that vanishes at the pessimistic end of the model's
    own 90% interval is flagged, not silently staked)."""
    p2 = 1.0 - p1
    lo, hi = ci1
    f1, f2 = optimal_allocation(p1, p2, o1, o2, kelly_fraction)
    overround = 1 / o1 + 1 / o2
    out = [f"\n  --- Which side, if any ---",
           f"  Odds {n1} {o1:.2f} (implied {1/o1:.3f}) | {n2} {o2:.2f} (implied {1/o2:.3f}) "
           f"| book overround {overround:.3f}"]
    if overround < 1:
        # Stake in proportion to implied probabilities -> identical payout
        # whoever wins: guaranteed return 1/overround per unit staked.
        w1, w2 = (1 / o1) / overround, (1 / o2) / overround
        out.append(f"  ARBITRAGE: implied probabilities sum to {overround:.3f} < 1 — split ANY "
                   f"stake {w1:.1%} on {n1} / {w2:.1%} on {n2} for a risk-free "
                   f"{1 / overround - 1:+.2%} return, model-independent.")
        return "\n".join(out)
    # Single-bet framing per side: what this one bet wins or loses, the
    # probability of each, and the fair odds for that risk vs the offered odds.
    for name, p_side, ci_lo, odds in ((n1, p1, lo, o1), (n2, p2, 1 - hi, o2)):
        fair = 1.0 / p_side if p_side > 0 else float("inf")
        out.append(f"  {name:<22} bet 100: win +{100*(odds-1):.0f} ({p_side:.0%}) / lose -100 ({1-p_side:.0%}) "
                   f"| fair odds {fair:.2f} vs offered {odds:.2f}")
    if f1 == 0 and f2 == 0:
        out.append(f"  => NO BET: neither side beats its break-even probability.")
        return "\n".join(out)
    for name, f, p_side, ci_lo, odds in ((n1, f1, p1, lo, o1), (n2, f2, p2, 1 - hi, o2)):
        if f == 0:
            continue
        robust = ci_lo > 1 / odds
        out.append(f"  => BET on {name} @ {odds:.2f}: the offered price beats the fair price for the risk "
                   f"— {'even in the pessimistic case' if robust else 'but NOT in the pessimistic case (MARGINAL)'}. "
                   f"You still lose this single bet {1-p_side:.0%} of the time.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", type=str)
    ap.add_argument("--p2", type=str)
    ap.add_argument("--surface", type=str, default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    ap.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    ap.add_argument("--slam", action="store_true")
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="model artifacts dir (default: models_v2_final if present, else models_v2)")
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--list-players", type=str, default=None)
    ap.add_argument("--matrix", action="store_true",
                    help="print the full surface x format probability grid for the pair "
                         "(Bo3 non-Slam vs Bo5 Slam on Hard/Clay/Grass) instead of a single prediction")
    ap.add_argument("--odds-p1", type=float, default=None,
                    help="bookmaker decimal odds on --p1 to win (e.g. 2.10) — prints a value-bet check")
    ap.add_argument("--kelly", type=float, default=0.25,
                    help="Kelly fraction for stake sizing (0.25 = quarter-Kelly, the default haircut)")
    ap.add_argument("--news", action="store_true",
                    help="print recent injury/withdrawal headlines for both players (Google News RSS, English)")
    ap.add_argument("--odds-p2", type=float, default=None,
                    help="bookmaker decimal odds on --p2 to win — prints a value-bet check")
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

    if args.matrix:
        # Full surface x format grid. Format changes the answer two ways:
        # (a) best-of-5 lowers outcome variance, amplifying the favorite
        # (classic longer-series effect), and (b) is_slam carries learned
        # context beyond match length. Both flags are explicit model inputs.
        # "Bo5 Slam" is the ATP men's Slam configuration; Carpet omitted
        # (no tour-level carpet events since 2009).
        print(f"\n{args.p1} vs {args.p2} — P({args.p1} wins)  (90% MC-dropout interval in brackets)")
        print(f"{'Surface':<10} {'Bo3 non-Slam':>24} {'Bo5 Slam':>24}")
        for surf in ["Hard", "Clay", "Grass"]:
            r3 = predict_match_prob(pid1, pid2, surf, 3, False, args.data_dir, args.out_dir, n_mc=args.n_mc)
            r5 = predict_match_prob(pid1, pid2, surf, 5, True, args.data_dir, args.out_dir, n_mc=args.n_mc)
            c3, c5 = r3["p1_win_ci90"], r5["p1_win_ci90"]
            print(f"{surf:<10} {r3['p1_win_prob']:>8.3f} [{c3[0]:.3f}-{c3[1]:.3f}] "
                  f"{r5['p1_win_prob']:>8.3f} [{c5[0]:.3f}-{c5[1]:.3f}]")
        return

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

    if args.odds_p1 is not None:
        print(value_bet_analysis(args.p1, result["p1_win_prob"], result["p1_win_ci90"], args.odds_p1))
    if args.odds_p2 is not None:
        p2_ci90 = (1.0 - result["p1_win_ci90"][1], 1.0 - result["p1_win_ci90"][0])
        print(value_bet_analysis(args.p2, result["p2_win_prob"], p2_ci90, args.odds_p2))
    if args.news:
        from news_v2 import fetch_news, format_news
        for name in (args.p1, args.p2):
            print(format_news(name, fetch_news(name)))
        print("  (headlines are context the model does NOT see — e.g. a taped wrist — weigh them before staking)")
    if args.odds_p1 and args.odds_p2:
        print(staking_plan(args.p1, args.p2, result["p1_win_prob"], result["p1_win_ci90"],
                           args.odds_p1, args.odds_p2, args.kelly))


if __name__ == "__main__":
    main()

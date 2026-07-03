#!/usr/bin/env python3
"""
tournament_v2.py — Pre-tournament bracket simulation
=========================================================================
Takes a tournament bracket (players in initial-round slot order, i.e. the
standard single-elimination seeding order) and Monte-Carlo simulates the
whole draw using the SAME calibrated pre-match model as predict_v2.py, to
estimate, for every player:
  - probability of reaching each round,
  - probability of winning the title.

Round-by-round update (no future leakage):
  Pass --results-so-far with the ACTUAL winners of already-completed
  rounds. Those rounds are then treated as fixed/observed (probability
  1.0 for the real winner) instead of simulated, and only the remaining,
  not-yet-played rounds are drawn stochastically. This is the correct way
  to "update after each phase": nothing about future rounds is used to
  decide who advanced in past rounds, and the simulation for the
  unresolved part of the draw is unconditional on anything not yet known.

Why Monte Carlo (rather than closed-form round-probability recursion):
  A player's probability of REACHING round k depends on who they are
  paired against, which itself depends on the outcomes of the OTHER
  half's earlier matches. This is a probabilistic dependency structure
  (see MODEL_V2_REPORT.md sect. 4.6) most cleanly evaluated by sampling
  complete draws end-to-end (standard approach in bracket forecasting,
  e.g. FiveThirtyEight's tennis Elo tournament forecasts) rather than by
  a per-round independence approximation, which would ignore the
  opponent-identity dependency across rounds and bias title probabilities
  for top seeds downward (their round-2 opponent's strength is not drawn
  independently of round-1 results elsewhere in the bracket).

Bracket JSON schema:
{
  "surface": "Hard", "best_of": 5, "is_slam": true,
  "players": ["Carlos Alcaraz", "Jannik Sinner", "Novak Djokovic", ...],
  "results_so_far": [["Carlos Alcaraz", "Novak Djokovic", ...]]   // optional,
     // one list of winners per COMPLETED round, in bracket order; omit or
     // leave empty for a pre-tournament (round-0) forecast.
}
`players` length must be a power of 2 (byes should be modeled as an
explicit "BYE" placeholder player that always loses, or by pre-filling
round 1 into results_so_far).

Usage:
  python tournament_v2.py --bracket wimbledon_2026.json --sims 20000
"""
import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from predict_v2 import build_name_index, load_state, predict_match_prob, resolve_player


def _next_pow2_check(n: int):
    if n & (n - 1) != 0 or n < 2:
        raise ValueError(f"players list length must be a power of 2 (got {n}). "
                          f"Model byes as an explicit 'BYE' player or pre-fill round 1 "
                          f"in results_so_far.")


def match_prob_matrix(players: List[str], pids: List[Optional[int]], surface: str, best_of: int,
                       is_slam: bool, data_dir: str, out_dir: str, n_mc_model: int = 30) -> Dict:
    """Pre-compute, for every pair, both a point estimate P[i,j] (for
    display) and `n_mc_model` MC-dropout draws P_samples[i,j,:] of P(i
    beats j) (Gal & Ghahramani, 2016 — same mechanism predict_v2.py uses
    for its single-match confidence interval), so the Monte-Carlo bracket
    loop below can draw a fresh, differently-uncertain matchup probability
    on every simulated trial instead of reusing one fixed number for all
    20,000 draws. This is what makes the tournament's reach/title
    probabilities carry an error bar that reflects genuine MODEL
    uncertainty, not just bracket-structure sampling noise: two players
    with identical point-estimate matchup probabilities but very different
    confidence should NOT produce equally sharp tournament forecasts.

    Players with pid=None (not found in the ATP dataset — no history to
    score them on) are treated as an automatic loss against any resolvable
    opponent, so the rest of the bracket's probabilities are computed AS IF
    that player weren't a threat, rather than crashing the whole run. Two
    unresolvable players facing each other (only possible pre-tournament,
    since results_so_far already fixes real winners for completed rounds)
    falls back to a 50/50 coin flip."""
    n = len(players)
    P = np.full((n, n), np.nan)
    P_samples = np.full((n, n, n_mc_model), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isnan(P[i, j]):
                continue
            if pids[i] is None or pids[j] is None:
                if pids[i] is None and pids[j] is None:
                    pij = 0.5
                elif pids[i] is None:
                    pij = 0.0
                else:
                    pij = 1.0
                P[i, j], P[j, i] = pij, 1.0 - pij
                P_samples[i, j, :], P_samples[j, i, :] = pij, 1.0 - pij
                continue
            r = predict_match_prob(pids[i], pids[j], surface, best_of, is_slam,
                                    data_dir, out_dir, n_mc=n_mc_model)
            P[i, j] = r["p1_win_prob"]
            P[j, i] = 1.0 - r["p1_win_prob"]
            P_samples[i, j, :] = r["p1_win_samples"]
            P_samples[j, i, :] = 1.0 - r["p1_win_samples"]
    return dict(P=P, P_samples=P_samples)


def simulate(players: List[str], P_samples: np.ndarray, results_so_far: Optional[List[List[str]]],
             n_sims: int, seed: int = 42) -> Dict:
    n = len(players)
    n_mc_model = P_samples.shape[2]
    n_rounds = int(np.log2(n))
    name_to_idx = {p: i for i, p in enumerate(players)}
    results_so_far = results_so_far or []

    reach_counts = np.zeros((n, n_rounds + 1), dtype=np.int64)  # reach_counts[i, r] = reached round r (0=round1 entrant)
    title_counts = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for _ in range(n_sims):
        alive = list(range(n))  # current round's participants, in bracket order
        for i in alive:
            reach_counts[i, 0] += 1
        for rnd in range(n_rounds):
            fixed_round = results_so_far[rnd] if rnd < len(results_so_far) else None
            winners = []
            for m in range(len(alive) // 2):
                a, b = alive[2 * m], alive[2 * m + 1]
                if fixed_round is not None:
                    w_name = fixed_round[m]
                    w = name_to_idx[w_name] if w_name in name_to_idx else (a if players[a] == w_name else b)
                else:
                    # Draw a fresh MC-dropout sample of P(a beats b) for THIS
                    # trial (rather than reusing one fixed number every
                    # time) — this is what propagates model epistemic
                    # uncertainty into the final title/reach probabilities,
                    # on top of the bracket's own structural randomness.
                    p_a = P_samples[a, b, rng.integers(n_mc_model)]
                    w = a if rng.random() < p_a else b
                winners.append(w)
            alive = winners
            for i in alive:
                reach_counts[i, rnd + 1] += 1
        title_counts[alive[0]] += 1

    reach_prob = reach_counts / n_sims
    title_prob = title_counts / n_sims
    # Each of the n_sims trials independently redraws BOTH the bracket
    # structure and the per-match model-uncertainty sample, so the trials
    # are i.i.d. Bernoulli(reach_prob) — a standard normal-approximation
    # binomial standard error is therefore a valid 90% CI (z=1.645), and it
    # is wider than pure sampling noise alone precisely because it also
    # carries the model's own uncertainty about each matchup.
    se = np.sqrt(np.clip(reach_prob * (1 - reach_prob), 0, None) / n_sims)
    reach_ci90 = np.clip(reach_prob[:, :, None] + np.array([-1.645, 1.645]) * se[:, :, None], 0.0, 1.0)
    title_se = np.sqrt(np.clip(title_prob * (1 - title_prob), 0, None) / n_sims)
    title_ci90 = np.clip(np.stack([title_prob - 1.645 * title_se, title_prob + 1.645 * title_se], axis=1), 0.0, 1.0)
    return dict(reach_prob=reach_prob, reach_ci90=reach_ci90, title_prob=title_prob,
                title_ci90=title_ci90, n_rounds=n_rounds)


ROUND_NAMES = {0: "Entered", 1: "R1 won", 2: "R2 won", 3: "R3 won", 4: "R4 won",
               5: "QF won", 6: "SF won", 7: "F won (champion)"}


def round_name(n_rounds: int, r: int) -> str:
    # map generically for any draw size: last round label = champion
    if r == n_rounds:
        return "CHAMPION"
    if r == 0:
        return "Round-1 entrant"
    return f"Reached round {r + 1}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bracket", type=str, required=True)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--n-mc-model", type=int, default=30,
                    help="MC-dropout samples per pairwise matchup, used to propagate model "
                         "uncertainty into the tournament's reach/title probability error bars")
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="model artifacts dir (default: models_v2_final if present, else models_v2)")
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    with open(args.bracket) as f:
        cfg = json.load(f)

    players = cfg["players"]
    _next_pow2_check(len(players))
    surface = cfg.get("surface", "Hard")
    best_of = int(cfg.get("best_of", 3))
    is_slam = bool(cfg.get("is_slam", False))
    results_so_far = cfg.get("results_so_far", [])

    print(f"Bracket: {len(players)} players | surface={surface} best_of={best_of} is_slam={is_slam} "
          f"| {len(results_so_far)} round(s) already fixed", file=sys.stderr)

    state = load_state(args.data_dir, args.out_dir)
    pids: List[Optional[int]] = []
    n_unresolved = 0
    for p in players:
        pid = resolve_player(p, state["name_index"])
        if pid is None:
            n_unresolved += 1
            print(f"WARNING: could not resolve player '{p}' (no ATP tour-level history in the "
                  f"dataset) — check spelling with predict_v2.py --list-players. Treating as an "
                  f"automatic loss against any resolved opponent; probabilities for the rest of "
                  f"the bracket are computed without this player.", file=sys.stderr)
        pids.append(pid)

    if n_unresolved:
        print(f"\n{n_unresolved} of {len(players)} player(s) unresolved — excluded from win-probability "
              f"scoring as described above.\n", file=sys.stderr)

    print("Scoring all pairwise matchups with the calibrated model "
          f"({args.n_mc_model} MC-dropout samples/pair)...", file=sys.stderr)
    mat = match_prob_matrix(players, pids, surface, best_of, is_slam, args.data_dir, args.out_dir,
                             n_mc_model=args.n_mc_model)

    print(f"Running {args.sims:,} Monte Carlo tournament simulations...", file=sys.stderr)
    sim = simulate(players, mat["P_samples"], results_so_far, args.sims)

    n_rounds = sim["n_rounds"]
    order = np.argsort(-sim["title_prob"])
    print(f"\nReach probabilities (90% CI in brackets; CI reflects both bracket-draw randomness "
          f"AND the model's own MC-dropout uncertainty about each matchup):")
    print(f"{'Player':<28} " + " ".join(f"{round_name(n_rounds, r):>26}" for r in range(1, n_rounds + 1)))
    for i in order:
        row = " ".join(
            f"{sim['reach_prob'][i, r]:6.3f} [{sim['reach_ci90'][i, r, 0]:.3f}-{sim['reach_ci90'][i, r, 1]:.3f}]"
            for r in range(1, n_rounds + 1)
        )
        print(f"{players[i]:<28} {row}")

    print(f"\nTitle probabilities (sorted, 90% CI in brackets):")
    for i in order:
        lo, hi = sim["title_ci90"][i]
        print(f"  {players[i]:<28} {sim['title_prob'][i]:.4f}  [{lo:.4f}-{hi:.4f}]")

    if args.out_json:
        out = {
            "surface": surface, "best_of": best_of, "is_slam": is_slam,
            "n_sims": args.sims, "n_mc_model": args.n_mc_model,
            "players": [
                {
                    "name": players[i],
                    "title_prob": float(sim["title_prob"][i]),
                    "title_ci90": [float(sim["title_ci90"][i, 0]), float(sim["title_ci90"][i, 1])],
                    "round_reach_prob": {round_name(n_rounds, r): float(sim["reach_prob"][i, r])
                                          for r in range(1, n_rounds + 1)},
                    "round_reach_ci90": {round_name(n_rounds, r): [float(sim["reach_ci90"][i, r, 0]),
                                                                    float(sim["reach_ci90"][i, r, 1])]
                                         for r in range(1, n_rounds + 1)},
                }
                for i in order
            ],
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()

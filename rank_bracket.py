#!/usr/bin/env python3
"""
rank_bracket.py — Synthesize a pre-tournament bracket from the ATP ranking
============================================================================
For "what if the top N ranked players played a tournament" style forecasts,
where no real draw exists (yet) to scrape with fetch_bracket.py. Reads the
latest ranking snapshot from tennis_atp/atp_rankings_current.csv, takes the
top --top-n players (skipping any named with --exclude, e.g. injury
withdrawals), pads up to the next power of two by continuing down the
ranking list (single-elimination brackets need a power-of-2 field), seeds
them with the standard tournament seeding order (1 vs last, 2 vs second-
last, ... so top seeds are maximally spread apart and can only meet in
later rounds — the same logic real tour draws use), and writes a bracket
JSON with an empty results_so_far (nothing has been played yet) ready for
tournament_v2.py.

Usage:
  python3.12 rank_bracket.py --top-n 30 --surface Hard --best-of 3 --out top30_hard.json
  python3.12 rank_bracket.py --top-n 30 --exclude "Carlos Alcaraz,Jannik Sinner" \\
      --surface Grass --best-of 5 --slam --out top30_no_alcaraz_sinner.json
"""
import argparse
import glob
import json
import sys
from typing import Dict, List, Optional

import pandas as pd

from fetch_bracket import load_full_names_and_last_active
from predict_v2 import resolve_player


def latest_ranking(data_dir: str) -> pd.DataFrame:
    fp = f"{data_dir}/atp_rankings_current.csv"
    d = pd.read_csv(fp)
    latest_date = d["ranking_date"].max()
    d = d[d["ranking_date"] == latest_date].sort_values("rank")
    print(f"Using ranking snapshot {latest_date} from {fp}", file=sys.stderr)
    return d


def seed_order(n: int) -> List[int]:
    """Standard single-elimination seeding sequence (1-indexed seeds), e.g.
    n=8 -> [1, 8, 4, 5, 2, 7, 3, 6]: seed 1 and seed 2 can only meet in the
    final, seeds 1-4 can only meet from the semifinals on, etc."""
    if n & (n - 1) != 0 or n < 1:
        raise ValueError(f"n must be a power of 2, got {n}")
    if n == 1:
        return [1]
    prev = seed_order(n // 2)
    out = []
    for s in prev:
        out.append(s)
        out.append(n + 1 - s)
    return out


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def build_rank_bracket(ranking, top_n, exclude, name_index, surface, best_of, slam):
    """ranking: list of (rank, pid, name) sorted by rank. exclude: list of names."""
    ex = set()
    for raw in exclude:
        pid = resolve_player(raw, name_index)
        if pid is None:
            print(f"WARNING: --exclude name '{raw}' did not resolve — ignored.", file=sys.stderr)
        else:
            ex.add(pid)
    size = next_pow2(top_n)
    picked = [(pid, name) for _, pid, name in ranking if pid not in ex][:size]
    if len(picked) < size:
        raise SystemExit(f"Only {len(picked)} eligible ranked players, need {size}.")
    if size != top_n:
        print(f"top-n {top_n} padded to {size} with: {', '.join(n for _, n in picked[top_n:])}", file=sys.stderr)
    players = [picked[s - 1][1] for s in seed_order(size)]
    return {"surface": surface, "best_of": best_of, "is_slam": bool(slam), "players": players, "results_so_far": []}


def ranking_from_snapshot():
    import pickle
    snap = pickle.load(open("deploy/state.pkl", "rb"))
    return snap["ranking"], snap["name_index"]


def ranking_from_csv(data_dir):
    name_index, full_names, _ = load_full_names_and_last_active(data_dir)
    d = latest_ranking(data_dir)
    return [(int(r["rank"]), int(r.player), full_names[int(r.player)]) for _, r in d.iterrows()
            if int(r.player) in full_names], name_index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-n", type=int, required=True)
    ap.add_argument("--exclude", type=str, default="", help="comma-separated names to leave out (withdrawals)")
    ap.add_argument("--surface", type=str, default="Hard")
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--slam", action="store_true")
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    import os
    ranking, name_index = (ranking_from_csv(args.data_dir) if os.path.isdir(args.data_dir)
                           else ranking_from_snapshot())
    bracket = build_rank_bracket(ranking, args.top_n, [s.strip() for s in args.exclude.split(",") if s.strip()],
                                 name_index, args.surface, args.best_of, args.slam)
    with open(args.out, "w") as f:
        json.dump(bracket, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.out} ({len(bracket['players'])} players, seed 1 = {bracket['players'][0]})", file=sys.stderr)


if __name__ == "__main__":
    main()

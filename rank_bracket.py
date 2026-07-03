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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-n", type=int, required=True, help="how many ranked players to include")
    ap.add_argument("--exclude", type=str, default="", help="comma-separated player names to leave out (withdrawals)")
    ap.add_argument("--surface", type=str, default="Hard")
    ap.add_argument("--best-of", type=int, default=3)
    ap.add_argument("--slam", action="store_true", help="set is_slam=true")
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    name_index, full_names, last_active = load_full_names_and_last_active(args.data_dir)

    exclude_ids = set()
    for raw in [s.strip() for s in args.exclude.split(",") if s.strip()]:
        pid = resolve_player(raw, name_index)
        if pid is None:
            print(f"WARNING: --exclude name '{raw}' did not resolve to any ATP player — ignored.",
                  file=sys.stderr)
        else:
            print(f"Excluding {full_names.get(pid, raw)} (requested via --exclude)", file=sys.stderr)
            exclude_ids.add(pid)

    ranking = latest_ranking(args.data_dir)
    field_size = next_pow2(args.top_n)
    if field_size != args.top_n:
        print(f"--top-n {args.top_n} is not a power of 2 — padding to {field_size} with the "
              f"next best-ranked available players (single-elimination needs a power-of-2 field).",
              file=sys.stderr)

    picked: List[int] = []
    padded_in = []
    for _, row in ranking.iterrows():
        if len(picked) >= field_size:
            break
        pid = int(row["player"])
        if pid in exclude_ids or pid in picked:
            continue
        if pid not in full_names:
            continue  # ranking row with no matches in atp_matches_*.csv (shouldn't normally happen)
        if len(picked) >= args.top_n:
            padded_in.append(full_names[pid])
        picked.append(pid)

    if len(picked) < field_size:
        raise SystemExit(f"Only found {len(picked)} eligible ranked players, need {field_size}. "
                          f"Check --exclude spelling or --top-n value.")
    if padded_in:
        print(f"Padded slots filled by (in rank order): {', '.join(padded_in)}", file=sys.stderr)

    seeds = seed_order(field_size)  # seeds[i] = seed number (1=best) placed at bracket slot i
    players = [full_names[picked[seed - 1]] for seed in seeds]

    bracket = {
        "surface": args.surface,
        "best_of": args.best_of,
        "is_slam": bool(args.slam),
        "players": players,
        "results_so_far": [],
    }
    with open(args.out, "w") as f:
        json.dump(bracket, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.out} ({len(players)} players, seed 1 = {players[0]})", file=sys.stderr)


if __name__ == "__main__":
    main()

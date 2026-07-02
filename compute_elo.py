#!/usr/bin/env python3
"""
compute_elo.py — Compute Tennis Elo Ratings
============================================
Supports two data sources:
  1. Jeff Sackmann tennis_atp repo  (atp_matches_YYYY.csv files)
  2. Match Charting Project          (charting-m-matches.csv)

Source is auto-detected from --data-dir contents.

Usage:
  python compute_elo.py --data-dir tennis_atp   --out player_elo_atp.json
  python compute_elo.py --data-dir tennis_MatchChartingProject --out player_elo.json
  python compute_elo.py --top 20 --surface Clay
  python compute_elo.py --player "Carlos Alcaraz"
"""

import argparse
import glob as _glob
import json
import math
import os
import ssl
import urllib.request
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_URL = (
    "https://github.com/JeffSackmann/tennis_MatchChartingProject"
    "/raw/refs/heads/master/"
)
MATCHES_FILE = "charting-m-matches.csv"

ELO_START     = 1500.0
SURFACES      = ["Hard", "Clay", "Grass", "Carpet"]
RECENT_LAMBDA = 0.7
BLEND_LAMBDA  = 0.5


def kovalchik_k(n_matches: int) -> float:
    return 250.0 / ((n_matches + 5) ** 0.4)


TOURNEY_WEIGHTS = {
    "australian_open": 1.00, "roland_garros": 1.00, "wimbledon": 1.00,
    "us_open": 1.00, "french_open": 1.00,
    "indian_wells": 0.85, "miami": 0.85, "monte_carlo": 0.85,
    "madrid": 0.85, "rome": 0.85, "canada": 0.85, "montreal": 0.85,
    "toronto": 0.85, "cincinnati": 0.85, "shanghai": 0.85,
    "paris": 0.85, "bercy": 0.85,
    "atp_finals": 0.90, "nitto": 0.90, "tour_finals": 0.90,
    "nextgen": 0.80, "olympics": 0.80,
}
DEFAULT_TOURNEY_WEIGHT = 0.65

ROUND_WEIGHTS = {
    "F": 1.00, "SF": 0.90, "QF": 0.85,
    "R16": 0.80, "4R": 0.80,
    "R32": 0.78, "3R": 0.78,
    "R64": 0.75, "2R": 0.75,
    "R128": 0.72, "1R": 0.72,
    "RR": 0.80, "BR": 0.80,
}
DEFAULT_ROUND_WEIGHT = 0.75

# ATP round codes → our round keys
ATP_ROUND_MAP = {
    "F": "F", "SF": "SF", "QF": "QF",
    "R16": "R16", "R32": "R32", "R64": "R64", "R128": "R128",
    "RR": "RR", "BR": "BR",
}

# ATP tourney_level → tournament weight
ATP_LEVEL_WEIGHTS = {
    "G": 1.00,   # Grand Slam
    "M": 0.85,   # Masters 1000
    "A": 0.75,   # ATP 500/250
    "F": 0.90,   # ATP Finals
    "D": 0.70,   # Davis Cup
    "C": 0.55,   # Challenger
    "S": 0.50,   # Satellite/ITF
}


def _normalize_surface(s: str) -> str:
    s = str(s).strip().title()
    return {
        "Hard Court": "Hard", "Indoor": "Hard", "Hardcourt": "Hard",
        "Acrylic": "Hard", "Outdoor": "Hard",
        "Indoor Hard": "Hard", "Outdoor Hard": "Hard",
    }.get(s, s)


def _safe_best_of(val) -> int:
    try:
        v = int(float(str(val).strip()))
        return v if v in (3, 5) else 3
    except (ValueError, TypeError):
        return 3


def _tournament_weight(tournament: str) -> float:
    t = str(tournament).lower().replace(" ", "_").replace("-", "_")
    for key, w in TOURNEY_WEIGHTS.items():
        if key in t:
            return w
    return DEFAULT_TOURNEY_WEIGHT


def _round_weight(rnd: str) -> float:
    return ROUND_WEIGHTS.get(str(rnd).strip().upper(), DEFAULT_ROUND_WEIGHT)


def _elo_expected(r_i: float, r_j: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_j - r_i) / 400.0))


def _days_ago(date_int: int, reference: int) -> float:
    try:
        d = datetime.strptime(str(int(date_int)), "%Y%m%d")
        r = datetime.strptime(str(int(reference)), "%Y%m%d")
        return max((r - d).days, 0)
    except (ValueError, TypeError):
        return 0.0


def _recent_decay(days: float) -> float:
    return max(math.exp(-RECENT_LAMBDA * days / 365.0), 0.05)


# ── DATA LOADER ───────────────────────────────────────────────────────────────

def load_matches(data_dir: str) -> pd.DataFrame:
    """
    Auto-detect data source:
      • If data_dir contains atp_matches_YYYY.csv  → load Jeff Sackmann tennis_atp
      • Otherwise                                  → load Match Charting Project
    """

    # ── Branch A: tennis_atp (Jeff Sackmann ATP repo) ─────────────────────────
    atp_files = sorted(_glob.glob(os.path.join(data_dir, "atp_matches_????.csv")))
    if atp_files:
        print(f"  Found {len(atp_files)} ATP files in '{data_dir}' — reading locally …")
        chunks = []
        for fp in atp_files:
            try:
                chunks.append(pd.read_csv(fp, on_bad_lines="skip", low_memory=False))
            except Exception as e:
                print(f"  ⚠ skipping {os.path.basename(fp)}: {e}")
        if not chunks:
            raise RuntimeError("No ATP match files could be read.")
        df = pd.concat(chunks, ignore_index=True)
        df.columns = df.columns.str.strip()

        # Map ATP column names → internal names
        df = df.rename(columns={
            "winner_name":  "p1",       # winner = p1 in our convention
            "loser_name":   "p2",
            "tourney_date": "Date",
            "tourney_name": "tournament",
            "match_num":    "match_num",
            "round":        "round",
            "best_of":      "best_of",
            "surface":      "surface",
            "tourney_level":"tourney_level",
            "tourney_id":   "tourney_id",
        })

        # Unique match_id
        tid = df.get("tourney_id", df["Date"].astype(str))
        mnum = df.get("match_num", pd.Series(range(len(df)))).astype(str)
        df["match_id"] = tid.astype(str) + "_" + mnum

        df["Date"]    = pd.to_numeric(df["Date"], errors="coerce")
        df["best_of"] = df["best_of"].apply(_safe_best_of)
        df["surface"] = df["surface"].fillna("Hard").astype(str).apply(_normalize_surface)
        df["round"]   = df["round"].fillna("R32").astype(str).str.strip().map(
            lambda r: ATP_ROUND_MAP.get(r.upper(), r.upper()))
        df["tournament"] = df["tournament"].fillna("").astype(str)

        # Tournament weight from tourney_level (ATP has explicit level field)
        if "tourney_level" in df.columns:
            df["_tw"] = df["tourney_level"].map(ATP_LEVEL_WEIGHTS).fillna(DEFAULT_TOURNEY_WEIGHT)
        else:
            df["_tw"] = df["tournament"].apply(_tournament_weight)

        df["winner"] = df["p1"]
        df["loser"]  = df["p2"]

        df = df.dropna(subset=["p1", "p2", "Date"])
        df = df[df["p1"].str.strip() != ""]
        df = df[df["p2"].str.strip() != ""]
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)

        yr_min = int(df["Date"].min() // 10000)
        yr_max = int(df["Date"].max() // 10000)
        print(f"  ATP matches loaded: {len(df):,}  ({yr_min}–{yr_max})  "
              f"players: {df['p1'].nunique() + df['p2'].nunique()}")
        return df

    # ── Branch B: Match Charting Project ──────────────────────────────────────
    local = os.path.join(data_dir, MATCHES_FILE)
    if not os.path.exists(local):
        print(f"  Downloading {MATCHES_FILE} …")
        # macOS Python 3.12 SSL workaround
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(BASE_URL + MATCHES_FILE, local)

    df = pd.read_csv(local, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Player 1": "p1", "Player 2": "p2",
        "Best of": "best_of", "Surface": "surface",
        "Round": "round", "Tournament": "tournament",
    })
    df["match_id"]   = df["match_id"].astype(str)
    df["Date"]       = pd.to_numeric(df["Date"], errors="coerce")
    df["best_of"]    = df["best_of"].apply(_safe_best_of)
    df["surface"]    = df["surface"].fillna("Hard").apply(_normalize_surface)
    df["round"]      = df["round"].fillna("R32").astype(str).str.strip()
    df["tournament"] = df["tournament"].fillna("").astype(str)
    df["_tw"]        = df["tournament"].apply(_tournament_weight)
    df["winner"]     = df["p1"]
    df["loser"]      = df["p2"]
    df = df.dropna(subset=["p1", "p2", "Date"])
    df = df[df["p1"].str.strip() != ""]
    df = df[df["p2"].str.strip() != ""]
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    return df


# ── ELO COMPUTATION ───────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame) -> Dict:
    reference_date = int(df["Date"].max())

    career_elo:  Dict[str, float]             = defaultdict(lambda: ELO_START)
    surface_elo: Dict[str, Dict[str, float]]  = defaultdict(
        lambda: {s: ELO_START for s in SURFACES})
    recent_elo:  Dict[str, float]             = defaultdict(lambda: ELO_START)
    n_matches:   Dict[str, int]               = defaultdict(int)
    last_date:   Dict[str, int]               = defaultdict(int)

    print(f"  Processing {len(df):,} matches chronologically …")

    has_tw = "_tw" in df.columns

    for _, row in df.iterrows():
        winner  = str(row["winner"]).strip()
        loser   = str(row["loser"]).strip()
        surface = row["surface"]
        rnd     = row["round"]
        tourney = row["tournament"]
        date_int = int(row["Date"])

        tw = float(row["_tw"]) if has_tw else _tournament_weight(tourney)
        rw = _round_weight(rnd)
        match_w = tw * rw

        days_w = _days_ago(date_int, reference_date)
        decay  = _recent_decay(days_w)

        # Career Elo
        r_w = career_elo[winner]; r_l = career_elo[loser]
        e_w = _elo_expected(r_w, r_l); e_l = 1.0 - e_w
        k_w = kovalchik_k(n_matches[winner]) * match_w
        k_l = kovalchik_k(n_matches[loser])  * match_w
        career_elo[winner] += k_w * (1.0 - e_w)
        career_elo[loser]  += k_l * (0.0 - e_l)

        # Surface Elo
        if surface in SURFACES:
            sr_w = surface_elo[winner][surface]
            sr_l = surface_elo[loser][surface]
            se_w = _elo_expected(sr_w, sr_l); se_l = 1.0 - se_w
            surface_elo[winner][surface] += k_w * (1.0 - se_w)
            surface_elo[loser][surface]  += k_l * (0.0 - se_l)

        # Recent Elo
        rr_w = recent_elo[winner]; rr_l = recent_elo[loser]
        re_w = _elo_expected(rr_w, rr_l); re_l = 1.0 - re_w
        k_w_r = kovalchik_k(n_matches[winner]) * match_w * decay
        k_l_r = kovalchik_k(n_matches[loser])  * match_w * decay
        recent_elo[winner] += k_w_r * (1.0 - re_w)
        recent_elo[loser]  += k_l_r * (0.0 - re_l)

        n_matches[winner] += 1
        n_matches[loser]  += 1
        last_date[winner]  = max(last_date[winner], date_int)
        last_date[loser]   = max(last_date[loser],  date_int)

    all_players = set(career_elo) | set(recent_elo)
    result = {}
    for player in all_players:
        ce = career_elo[player]
        se = surface_elo[player]
        blended = {
            surf: BLEND_LAMBDA * se[surf] + (1 - BLEND_LAMBDA) * ce
            for surf in SURFACES
        }
        result[player] = {
            "career_elo":  round(ce, 2),
            "surface_elo": {s: round(se[s], 2) for s in SURFACES},
            "blended_elo": {s: round(blended[s], 2) for s in SURFACES},
            "recent_elo":  round(recent_elo[player], 2),
            "n_matches":   n_matches[player],
            "last_date":   last_date[player],
        }
    return result


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def print_top(result: Dict, surface: Optional[str] = None,
              elo_type: str = "career_elo", n: int = 20):
    if surface and elo_type in ("surface_elo", "blended_elo"):
        scores = {p: v[elo_type][surface]
                  for p, v in result.items() if v["n_matches"] >= 10}
        label = f"{elo_type} ({surface})"
    else:
        scores = {p: v[elo_type]
                  for p, v in result.items() if v["n_matches"] >= 10}
        label = elo_type
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {n} by {label}:")
    print(f"  {'Rank':<5} {'Player':<28} {'Elo':>8} {'Matches':>7} {'Last match'}")
    print(f"  {'─'*5} {'─'*28} {'─'*8} {'─'*7} {'─'*10}")
    for i, (player, score) in enumerate(ranked[:n], 1):
        nm = result[player]["n_matches"]
        ld = str(result[player]["last_date"])
        ld_fmt = f"{ld[:4]}-{ld[4:6]}-{ld[6:]}" if len(ld) == 8 else ld
        print(f"  {i:<5} {player:<28} {score:>8.1f} {nm:>7} {ld_fmt}")


def print_player(result: Dict, name: str):
    if name not in result:
        # fuzzy fallback
        matches = [p for p in result if name.lower() in p.lower()]
        if len(matches) == 1:
            name = matches[0]
        elif len(matches) > 1:
            print(f"  Ambiguous '{name}': {matches[:5]}")
            return
        else:
            print(f"  Player '{name}' not found.")
            return
    v = result[name]
    ld = str(v["last_date"])
    ld_fmt = f"{ld[:4]}-{ld[4:6]}-{ld[6:]}" if len(ld) == 8 else ld
    print(f"\n  ── {name} ──")
    print(f"  Matches played : {v['n_matches']}")
    print(f"  Last match     : {ld_fmt}")
    print(f"  Career Elo     : {v['career_elo']:.1f}")
    print(f"  Recent Elo     : {v['recent_elo']:.1f}")
    print(f"\n  {'Surface':<12} {'Surface Elo':>12} {'Blended Elo':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12}")
    for s in SURFACES:
        print(f"  {s:<12} {v['surface_elo'][s]:>12.1f} {v['blended_elo'][s]:>12.1f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compute tennis Elo (ATP repo or Match Charting Project).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_elo.py --data-dir tennis_atp --out player_elo_atp.json
  python compute_elo.py --data-dir tennis_MatchChartingProject --out player_elo.json
  python compute_elo.py --data-dir tennis_atp --top 20 --surface Clay
  python compute_elo.py --data-dir tennis_atp --player "Carlos Alcaraz"
""",
    )
    ap.add_argument("--data-dir", type=str, default=".",
                    help="Directory with atp_matches_*.csv OR charting-m-matches.csv")
    ap.add_argument("--out", "--output", dest="output", type=str,
                    default="player_elo.json",
                    help="Output JSON file (default: player_elo.json)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--surface", type=str, default=None, choices=SURFACES)
    ap.add_argument("--player", type=str, default=None)
    args = ap.parse_args()

    print("  Loading match data …")
    df = load_matches(args.data_dir)

    result = compute_elo(df)
    print(f"\n  Computed Elo for {len(result):,} players.")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {args.output}")

    print_top(result,
              surface=args.surface,
              elo_type="blended_elo" if args.surface else "career_elo",
              n=args.top)

    if args.player:
        print_player(result, args.player)

    print_top(result, elo_type="recent_elo", n=10)


if __name__ == "__main__":
    main()

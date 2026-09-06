#!/usr/bin/env python3
"""
live_data.py — bridge from data_updated/*.csv (tennisdata-style season files,
Flashscore ids, "Surname I." names, bookmaker odds, SCHEDULED rows) to the v2
pipeline (ATP player ids, walk-forward PlayerStateTracker).

Used for two things:
  1. extend_tracker(): walk the FINISHED matches played AFTER the tennis_atp
     archive's cutoff through the tracker, so predictions use up-to-date
     Elo/form/H2H (tennis_atp trails reality by weeks-months).
  2. upcoming(): the SCHEDULED ATP-Tour matches with odds — the "prediction
     cards" input.

Flashscore ids are mapped to ATP ids by name (surname+initial resolver from
fetch_bracket, most-recently-active tie-break). Unresolved players (deep
challenger fields with no ATP tour/chall history) are skipped for state and
flagged on cards.
"""
import glob
import sys
from typing import Dict, Optional

import pandas as pd

from fetch_bracket import resolve_flashscore_name

SLAM_WORDS = ("us open", "wimbledon", "french open", "roland garros", "australian open")


def load_live(live_dir: str = "data_updated") -> pd.DataFrame:
    files = sorted(glob.glob(f"{live_dir}/*-atp-season.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date_timestamp"], unit="s")
    df["yyyymmdd"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["surface_norm"] = df["surface"].str.title().replace({"Indoor": "Hard"}).fillna("Hard")
    df.loc[~df["surface_norm"].isin(["Hard", "Clay", "Grass", "Carpet"]), "surface_norm"] = "Hard"
    df["is_slam"] = df["tournament"].str.lower().str.contains("|".join(SLAM_WORDS)) & \
                    (df["tour_type_human"] == "ATP Tour") & (df["round"] != "Qualifier")
    df["tw"] = df["tour_type_human"].map({"ATP Tour": 0.75, "ATP Chall": 0.55}).fillna(0.5)
    return df


def build_id_map(df: pd.DataFrame, name_index: Dict[str, int], full_names: Dict[int, str],
                 last_active: Dict[int, int]) -> Dict[int, Optional[int]]:
    """Flashscore player id -> ATP player id (or None). Name-based, cached per id."""
    pairs = pd.concat([df[["home_id", "home_name"]].rename(columns={"home_id": "fid", "home_name": "nm"}),
                       df[["away_id", "away_name"]].rename(columns={"away_id": "fid", "away_name": "nm"})]
                      ).drop_duplicates("fid")
    out = {}
    rev = {v: k for k, v in full_names.items()}
    for fid, nm in zip(pairs.fid, pairs.nm):
        full = resolve_flashscore_name(str(nm), name_index, full_names, last_active)
        out[int(fid)] = rev.get(full) if full else None
    return out


def extend_tracker(tracker, cutoff: int, name_index, full_names, last_active,
                   live_dir: str = "data_updated") -> int:
    """Walk FINISHED matches with yyyymmdd > cutoff through the tracker, in
    chronological order. Returns the new last date (or cutoff if nothing)."""
    df = load_live(live_dir)
    if df.empty:
        return cutoff
    idmap = build_id_map(df, name_index, full_names, last_active)
    rows = df[(df.status == "FINISHED") & (df.yyyymmdd > cutoff)].sort_values("date_timestamp")
    n_used = 0
    for r in rows.itertuples():
        h, a = idmap.get(int(r.home_id)), idmap.get(int(r.away_id))
        if h is None or a is None or r.winner_code not in (1, 2):
            continue
        w, l = (h, a) if r.winner_code == 1 else (a, h)
        walkover = str(r.status_extra) in ("WALKOVER", "RETIRED")
        # per-set stats not in tennis_atp format -> no form-stat update (Elo,
        # winrate, H2H, rest days still update); form features age gracefully
        tracker.update(w, l, r.surface_norm, int(r.yyyymmdd), float(r.tw), walkover, None, None)
        n_used += 1
    new_last = int(rows.yyyymmdd.max()) if len(rows) else cutoff
    print(f"live_data: extended state with {n_used:,} matches ({cutoff} -> {new_last})", file=sys.stderr)
    return new_last


def season_record(live_dir: str = "data_updated") -> Dict[int, "tuple[int, int]"]:
    """Flashscore id -> (wins, losses) in the current (= latest) season file."""
    df = load_live(live_dir)
    if df.empty:
        return {}
    df = df[(df.season_year == df.season_year.max()) & (df.status == "FINISHED") & df.winner_code.isin([1, 2])]
    rec: Dict[int, list] = {}
    for r in df.itertuples():
        w, l_ = (int(r.home_id), int(r.away_id)) if r.winner_code == 1 else (int(r.away_id), int(r.home_id))
        rec.setdefault(w, [0, 0])[0] += 1
        rec.setdefault(l_, [0, 0])[1] += 1
    return {k: (v[0], v[1]) for k, v in rec.items()}


def upcoming(name_index, full_names, last_active, live_dir: str = "data_updated",
             tour_only: bool = True) -> pd.DataFrame:
    """SCHEDULED matches with resolved ATP names + odds, soonest first."""
    df = load_live(live_dir)
    if df.empty:
        return df
    up = df[df.status == "SCHEDULED"].copy()
    if tour_only:
        up = up[up.tour_type_human == "ATP Tour"]
    idmap = build_id_map(up, name_index, full_names, last_active)
    for side in ("home", "away"):
        up[f"{side}_pid"] = up[f"{side}_id"].map(lambda i: idmap.get(int(i)))
        up[f"{side}_atp_name"] = up[f"{side}_pid"].map(lambda p: full_names.get(p) if p else None)
    return up.sort_values("date_timestamp")

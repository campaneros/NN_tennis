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


STAT_COLS = {"aces": "Aces (avg/match)", "double_faults": "Double faults (avg/match)",
             "service_points_won_perc": "Service points won %", "return_points_won_perc": "Return points won %",
             "break_points_won_perc": "Break points converted %", "break_points_saved_perc": "Break points saved %"}


def season_stats(live_dir: str = "data_updated") -> "pd.DataFrame":
    """Per-player current-season averages of the in-match stats (fid-indexed)."""
    df = load_live(live_dir)
    if df.empty:
        return pd.DataFrame()
    df = df[(df.season_year == df.season_year.max()) & (df.status == "FINISHED")]
    parts = []
    for side in ("home", "away"):
        cols = {f"{side}_{c}": c for c in STAT_COLS}
        p = df[[f"{side}_id"] + list(cols)].rename(columns={f"{side}_id": "fid", **cols})
        parts.append(p)
    return pd.concat(parts).groupby("fid").mean(numeric_only=True)


def h2h_list(fid1: int, fid2: int, live_dir: str = "data_updated") -> "pd.DataFrame":
    """All FINISHED meetings between the two (any season in the folder), newest first."""
    df = load_live(live_dir)
    if df.empty:
        return df
    m = df[(df.status == "FINISHED") &
           (((df.home_id == fid1) & (df.away_id == fid2)) | ((df.home_id == fid2) & (df.away_id == fid1)))]
    return m.sort_values("date_timestamp", ascending=False)


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


# ── LIVE source: diretta.it/Flashscore daily feed + per-bookmaker odds ──────
# (no CSV needed: schedule of today/tomorrow + odds comparison, both from the
# same feeds the website itself uses; x-fsign is embedded in the public page)
import json as _json
import statistics as _stats
import urllib.request as _rq

import ssl as _ssl
try:
    import certifi as _certifi
    _SSLCTX = _ssl.create_default_context(cafile=_certifi.where())
except ImportError:
    _SSLCTX = _ssl.create_default_context()

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
       "Referer": "https://www.diretta.it/", "Origin": "https://www.diretta.it"}
_SURF_IT = {"cemento": "Hard", "terra": "Clay", "erba": "Grass", "sintetico": "Hard"}
_SLAM_IT = ("us open", "wimbledon", "french open", "roland garros", "australian open")
_SIGN = {"v": "SW9D1eZo"}


def _refresh_sign():
    import re as _re
    html = _rq.urlopen(_rq.Request("https://www.diretta.it/tennis/", headers=_UA),
                       timeout=15, context=_SSLCTX).read().decode("utf-8", "replace")
    m = _re.search(r'feed_sign":"([A-Za-z0-9]+)"', html)
    if m:
        _SIGN["v"] = m.group(1)


def _get(url: str, headers: dict, tries: int = 3) -> bytes:
    import time as _t
    last = None
    for i in range(tries):
        try:
            return _rq.urlopen(_rq.Request(url, headers=headers), timeout=20, context=_SSLCTX).read()
        except Exception as e:  # transient RemoteDisconnected/timeouts happen on this CDN
            last = e
            _t.sleep(0.8 * (i + 1))
    raise last


def _feed(name: str) -> str:
    url = f"https://400.flashscore.ninja/400/x/feed/{name}"
    try:
        out = _get(url, {**_UA, "x-fsign": _SIGN["v"]}).decode("utf-8", "replace")
    except Exception:
        _refresh_sign()   # signature may have rotated: re-scrape it from the public page
        out = _get(url, {**_UA, "x-fsign": _SIGN["v"]}).decode("utf-8", "replace")
    if "401 Unauthorized" in out[:400]:
        _refresh_sign()
        out = _get(url, {**_UA, "x-fsign": _SIGN["v"]}).decode("utf-8", "replace")
    return out


def diretta_daily(days=(0, 1)) -> pd.DataFrame:
    """Scheduled ATP-singles TOUR matches for today(+tomorrow) from the daily
    feed. Columns mirror what the Upcoming cards need."""
    rows = []
    for day in days:
        raw = _feed(f"f_2_{day}_3_it_1")
        tournament = surface = None
        is_tour = False
        for seg in raw.split("~"):
            d = dict(f.split("÷", 1) for f in seg.split("¬") if "÷" in f)
            if "ZA" in d:  # section header, e.g. 'ATP - SINGOLARE: US Open (USA), cemento'
                hdr = d["ZA"]
                is_tour = hdr.startswith("ATP - SINGOLARE:") and "qualificazione" not in hdr.lower()
                if is_tour:
                    body = hdr.split(":", 1)[1].strip()
                    tournament = body.split("(")[0].strip()
                    surface = next((v for k, v in _SURF_IT.items() if k in body.lower()), "Hard")
            elif is_tour and "AA" in d and d.get("AB") == "1":  # scheduled match
                rows.append(dict(
                    match_id=d["AA"], date_timestamp=int(d["AD"]),
                    tournament=tournament, surface_norm=surface,
                    is_slam=any(s in tournament.lower() for s in _SLAM_IT),
                    home_name=d.get("AE", "?"), away_name=d.get("AF", "?"),
                    home_slug=d.get("WU"), away_slug=d.get("WV"),
                    url=f"https://www.diretta.it/partita/tennis/{d.get('WU','')}-{d.get('JA','')}/"
                        f"{d.get('WV','')}-{d.get('JB','')}/?mid={d['AA']}",
                ))
    df = pd.DataFrame(rows)
    if len(df):
        df["date_human"] = pd.to_datetime(df.date_timestamp, unit="s").dt.strftime("%d %b %Y %H:%M")
    return df


def diretta_odds(match_id: str) -> "tuple[Optional[float], Optional[float], int]":
    """(median home odds, median away odds, n bookmakers) for match winner,
    from the public odds-comparison endpoint (same one the match page uses)."""
    url = (f"https://global.ds.lsapp.eu/odds/pq_graphql?_hash=oce&eventId={match_id}"
           f"&projectId=400&geoIpCode=IT&geoIpSubdivisionCode=IT-62")
    try:
        d = _json.loads(_get(url, _UA))
        entries = [o for o in d["data"]["findOddsByEventId"]["odds"]
                   if o.get("bettingType") == "HOME_AWAY" and o.get("bettingScope") == "FULL_TIME"]
        hs, as_ = [], []
        for o in entries:
            v = o.get("odds", [])
            if len(v) == 2 and v[0].get("value") and v[1].get("value"):
                hs.append(float(v[0]["value"])); as_.append(float(v[1]["value"]))
        if not hs:
            return None, None, 0
        return _stats.median(hs), _stats.median(as_), len(hs)
    except Exception:
        return None, None, 0


def upcoming_live(name_index, full_names, last_active, days=(0, 1)) -> pd.DataFrame:
    """diretta.it schedule + odds, with players resolved to ATP ids (via the
    feed's full-name slugs, same resolver as fetch_bracket)."""
    from fetch_bracket import make_resolver
    df = diretta_daily(days)
    if df.empty:
        return df
    resolve = make_resolver(name_index, full_names, last_active)
    rev = {v: k for k, v in full_names.items()}
    for side in ("home", "away"):
        df[f"{side}_atp_name"] = [resolve(nm, sl) for nm, sl in zip(df[f"{side}_name"], df[f"{side}_slug"])]
        df[f"{side}_pid"] = df[f"{side}_atp_name"].map(lambda n: rev.get(n))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        odds = list(ex.map(diretta_odds, df.match_id))
    df["home_odds_match_winner"] = [o[0] for o in odds]
    df["away_odds_match_winner"] = [o[1] for o in odds]
    df["n_bookmakers"] = [o[2] for o in odds]
    df["round"] = ""
    return df.sort_values("date_timestamp")

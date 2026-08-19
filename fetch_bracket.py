#!/usr/bin/env python3
"""
fetch_bracket.py — download an ATP draw from diretta.it (Flashscore) into the
bracket JSON used by tournament_v2.py. No browser needed: the tournament page's
static HTML carries the tournament/stage ids and the feed signature, and the
draw feed (`.../feed/dr_<tournamentId>_<stageId>`) is plain text.

  python3.12 fetch_bracket.py --tournament cincinnati --out cincinnati.json
  python3.12 fetch_bracket.py --url https://www.diretta.it/tennis/atp-singolare/us-open/ --out usopen.json

Output = ONLY the part of the draw still to be played: completed leading rounds
are collapsed into `players` (the survivors, in bracket order); a round that is
partially played is kept with its known winners in results_so_far (None for
matches not yet played). Byes are resolved (the non-bye side advances).
Player names are mapped to the ATP dataset via the feed's full-name slugs
(e.g. `alcaraz-garfia-carlos`), falling back to surname+initial matching.
"""
import argparse
import json
import re
import sys
import unicodedata
import urllib.request
from typing import Dict, List, Optional

import ssl
try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://www.diretta.it/tennis/atp-singolare/"
SLAMS = {"wimbledon", "french-open", "roland-garros", "us-open", "australian-open"}
GRASS = {"wimbledon", "halle", "queens", "queen-s-club", "eastbourne", "stuttgart", "s-hertogenbosch", "mallorca", "newport"}
CLAY = {"french-open", "roland-garros", "roma", "rome", "madrid", "monte-carlo", "montecarlo", "barcellona", "barcelona",
        "amburgo", "hamburg", "bastad", "gstaad", "kitzbuhel", "umag", "estoril", "houston", "buenos-aires", "rio-de-janeiro",
        "santiago", "cordoba", "marrakech", "munich", "monaco-di-baviera", "geneva", "ginevra", "lyon", "bucharest", "belgrado"}


def _get(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20, context=_SSL).read().decode("utf-8", "replace")


def resolve_tournament(slug_or_url: str) -> Dict:
    """diretta slug ('cincinnati') or any diretta tournament URL -> ids + feed sign."""
    url = slug_or_url if slug_or_url.startswith("http") else BASE + slug_or_url.strip("/") + "/"
    url = re.sub(r"(atp-singolare/[^/]+/).*", r"\1tabellone/", url)
    html = _get(url)
    g = lambda pat: (re.search(pat, html) or [None, None])[1]
    tid, sid, sign = g(r'tournamentId":"([A-Za-z0-9]{8})'), g(r'tournamentStageId: "([A-Za-z0-9]{8})'), g(r'feed_sign":"([A-Za-z0-9]+)')
    if not (tid and sid and sign):
        raise SystemExit(f"Could not find tournament ids on {url} (tid={tid} sid={sid} sign={sign})")
    slug = re.search(r"atp-singolare/([^/]+)/", url).group(1)
    return dict(url=url, slug=slug, tournament_id=tid, stage_id=sid, feed_sign=sign, project=g(r'projectId":(\d+)') or "400")


def fetch_draw(t: Dict) -> Dict:
    """-> {'names': {idx: 'Sinner J.'}, 'slugs': {idx: 'sinner-jannik'}, 'rounds': [[match,...],...]}
    match = {'home': idx|None, 'away': idx|None, 'winner': idx|None, 'played': bool}"""
    raw = urllib.request.urlopen(urllib.request.Request(
        f"https://{t['project']}.flashscore.ninja/{t['project']}/x/feed/dr_{t['tournament_id']}_{t['stage_id']}",
        headers={**UA, "x-fsign": t["feed_sign"]}), timeout=20, context=_SSL).read().decode("utf-8", "replace")
    segs = [s for s in raw.split("~") if s]
    kv = lambda s: dict(f.split("÷", 1) for f in s.split("¬") if "÷" in f)
    names = {int(x.split("_", 1)[0]): x.split("_", 1)[1] for x in kv(segs[0])["PA"].split("|")}
    slugs, rounds = {}, []
    for s in segs:
        d = kv(s)
        if "RI" in d:
            rounds.append([])
        if rounds and ("HP" in d or "AP" in d):
            h, a = (int(d["HP"]) if d.get("HP") not in (None, "") else None), (int(d["AP"]) if d.get("AP") not in (None, "") else None)
            if h is not None and not names.get(h): h = None   # bye slot has empty name
            if a is not None and not names.get(a): a = None
            if "RQ" in d:  # 'id;hp;ap;ts;score;winner;home-slug;away-slug;...'
                p = d["RQ"].split(";")
                if len(p) > 7:
                    if h is not None and p[6]: slugs[h] = p[6]
                    if a is not None and p[7]: slugs[a] = p[7]
            wi = d.get("WI")
            winner = int(wi) if wi not in (None, "", "-1") else None
            if winner is None and (h is None) != (a is None):   # bye
                winner = h if h is not None else a
            rounds[-1].append(dict(home=h, away=a, winner=winner, played=winner is not None))
    return dict(names=names, slugs=slugs, rounds=rounds)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.replace("-", " ")).strip().lower()


def make_resolver(name_index: Dict[str, int], full_names: Dict[int, str], last_active: Dict[int, int]):
    """Returns f(display_name, slug) -> ATP full name or None."""
    by_tokens: Dict[frozenset, List[int]] = {}
    for key, pid in name_index.items():
        by_tokens.setdefault(frozenset(key.split()), []).append(pid)

    def resolve(display: str, slug: Optional[str]) -> Optional[str]:
        if slug:
            toks = _norm(slug).split()
            cands = by_tokens.get(frozenset(toks), [])
            if not cands and len(toks) > 2:   # 'o-connell-christopher' vs ATP 'Christopher Oconnell'
                cands = [p for k, ps in by_tokens.items() if k == frozenset(("".join(toks[:2]), *toks[2:])) for p in ps]
            if len(cands) == 1:
                return full_names[cands[0]]
            if len(cands) > 1:
                return full_names[max(cands, key=lambda p: last_active.get(p, 0))]
        return resolve_flashscore_name(display, name_index, full_names, last_active)
    return resolve


def resolve_flashscore_name(fs_name: str, name_index: Dict[str, int],
                             full_names: Dict[int, str],
                             last_active: Optional[Dict[int, int]] = None) -> Optional[str]:
    """fs_name like 'Davidovich Fokina A.' or 'Cerundolo J. M.' or
    'Struff J-L.' -> full ATP name, or None. Ties (two players sharing
    surname + first initial) are broken by picking whoever has the more
    recent match in the dataset (the currently-active tour player)."""
    tokens = fs_name.strip().split()
    is_initial = re.compile(r"^[A-Z](\.|-[A-Z]\.?)?$")
    split_at = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if is_initial.match(tokens[i]):
            split_at = i
        else:
            break
    if split_at == len(tokens):  # no trailing initial token found
        return None
    surname_tokens, initial_tokens = tokens[:split_at], tokens[split_at:]
    if not surname_tokens or not initial_tokens:
        return None
    surname, initial = _norm(" ".join(surname_tokens)), initial_tokens[0][0].lower()

    def find(cmp_fn):
        out = []
        for key, pid in name_index.items():
            parts = key.split()
            if len(parts) < 2:
                continue
            first, rest = parts[0], " ".join(parts[1:])
            if first[:1] == initial and cmp_fn(rest, surname):
                out.append(pid)
        return sorted(set(out))

    candidates = find(lambda rest, sn: rest == sn)
    if not candidates:
        candidates = find(lambda rest, sn: rest.endswith(sn) or rest.startswith(sn) or sn.endswith(rest))
    if not candidates:
        return None
    if len(candidates) > 1 and last_active:
        candidates.sort(key=lambda pid: last_active.get(pid, 0), reverse=True)
        print(f"  AMBIGUOUS '{fs_name}': {[full_names[c] for c in candidates]} "
              f"-> picked most recently active: {full_names[candidates[0]]}", file=sys.stderr)
    elif len(candidates) > 1:
        print(f"  AMBIGUOUS '{fs_name}': {[full_names[c] for c in candidates]}", file=sys.stderr)
        return None
    return full_names[candidates[0]]




def load_full_names_and_last_active(data_dir: str = "tennis_atp"):
    """(name_index, full_names, last_active). From deploy/state.pkl when the
    raw data isn't there (Streamlit Cloud), else from tennis_atp/ CSVs."""
    import os, pickle
    if not os.path.isdir(data_dir) and os.path.exists("deploy/state.pkl"):
        snap = pickle.load(open("deploy/state.pkl", "rb"))
        return snap["name_index"], snap["full_names"], snap["last_active"]
    import glob
    import pandas as pd
    from predict_v2 import build_name_index
    name_index = build_name_index(data_dir)
    full_names, last_active = {}, {}
    for fp in sorted(glob.glob(f"{data_dir}/atp_matches_????.csv")):
        d = pd.read_csv(fp, usecols=["tourney_date", "winner_id", "winner_name", "loser_id", "loser_name"], low_memory=False)
        for side in ("winner", "loser"):
            for pid, name, date in zip(d[f"{side}_id"], d[f"{side}_name"], d["tourney_date"]):
                if pd.notna(pid):
                    pid = int(pid); full_names[pid] = name; last_active[pid] = max(last_active.get(pid, 0), int(date))
    return name_index, full_names, last_active


def build_bracket(draw: Dict, resolve, surface: str, best_of: int, is_slam: bool) -> Dict:
    """Collapse finished rounds; emit only what is still to be played."""
    names, slugs, rounds = draw["names"], draw["slugs"], draw["rounds"]
    name_of = lambda i: None if i is None else (resolve(names[i], slugs.get(i)) or names[i])
    unresolved = set()
    def nm(i):
        n = name_of(i)
        if i is not None and resolve(names[i], slugs.get(i)) is None:
            unresolved.add(names[i])
        return n
    # first round with at least one unplayed match
    open_idx = next((k for k, r in enumerate(rounds) if any(not m["played"] for m in r)), None)
    if open_idx is None:   # tournament over
        champ = rounds[-1][0]["winner"]
        return dict(surface=surface, best_of=best_of, is_slam=is_slam, players=[], results_so_far=[],
                    finished=True, champion=name_of(champ), unresolved=[])
    rnd = rounds[open_idx]
    players = [p for m in rnd for p in (nm(m["home"]), nm(m["away"]))]
    if any(p is None for p in players):   # future round whose participants aren't set yet -> step back one
        rnd = rounds[open_idx - 1]
        players = [p for m in rnd for p in (nm(m["home"]), nm(m["away"]))]
    partial = [nm(m["winner"]) if m["played"] else None for m in rnd]
    results = [partial] if any(partial) else []
    return dict(surface=surface, best_of=best_of, is_slam=is_slam, players=players, results_so_far=results,
                finished=False, round_index=open_idx, rounds_total=len(rounds),
                unresolved=sorted(unresolved))


def infer_format(slug: str):
    slam = slug in SLAMS
    surface = "Grass" if slug in GRASS else "Clay" if slug in CLAY else "Hard"
    return surface, (5 if slam else 3), slam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tournament", help="diretta.it slug, e.g. cincinnati, us-open, wimbledon")
    ap.add_argument("--url", help="any diretta.it URL of the tournament")
    ap.add_argument("--surface"); ap.add_argument("--best-of", type=int); ap.add_argument("--slam", action="store_true")
    ap.add_argument("--data-dir", default="tennis_atp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.tournament or a.url):
        ap.error("--tournament or --url")
    t = resolve_tournament(a.url or a.tournament)
    surface, best_of, slam = infer_format(t["slug"])
    surface, best_of, slam = a.surface or surface, a.best_of or best_of, a.slam or slam
    draw = fetch_draw(t)
    resolve = make_resolver(*load_full_names_and_last_active(a.data_dir))
    b = build_bracket(draw, resolve, surface, best_of, slam)
    if b["finished"]:
        print(f"{t['slug']}: tournament finished, champion {b['champion']}", file=sys.stderr)
    else:
        print(f"{t['slug']}: round {b['round_index']+1}/{b['rounds_total']} open — {len(b['players'])} players, "
              f"{sum(1 for w in (b['results_so_far'][0] if b['results_so_far'] else []) if w)} match(es) of it already played",
              file=sys.stderr)
    if b["unresolved"]:
        print(f"  unresolved names (no ATP history; treated as auto-loss by tournament_v2): {', '.join(b['unresolved'])}", file=sys.stderr)
    json.dump({k: b[k] for k in ("surface", "best_of", "is_slam", "players", "results_so_far")}, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"Saved -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

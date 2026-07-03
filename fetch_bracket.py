#!/usr/bin/env python3
"""
fetch_bracket.py — Download an ATP tournament draw from diretta.it (Flashscore)
================================================================================
Converts a diretta.it "tabellone" (draw) page into the bracket JSON schema
consumed by tournament_v2.py.

Why scrape the rendered page instead of the underlying feed API
-----------------------------------------------------------------
diretta.it is a Flashscore-family SPA: the draw is fetched client-side from
`https://400.flashscore.ninja/400/x/feed/dr_<tournamentId>_<stageId>`, but
that request requires a signed `x-fsign` header that's computed by obfuscated
JS and re-derived per session — not worth reverse-engineering for a stable
script. Instead this script drives a real (headless) Chromium via Playwright,
lets the page render normally, and reads the already-rendered draw straight
out of the DOM (`.draw__round` / `.draw__bracket` / `.bracket__result`
elements), which is exactly what a human reading the page sees.

Why --url instead of --tournament/--year auto-lookup
--------------------------------------------------------
Flashscore's tournament-search endpoint (that would map "Wimbledon"+2026 ->
its internal tournamentId/stageId pair) was not identified within this
script's scope — the tabellone URL itself already encodes those ids
(.../tabellone/<tournamentId>/tabellone/ or .../tabellone/<stageId>/...).
Rather than guess at a fragile lookup, you supply the tabellone URL directly
(copy it from the browser address bar); everything else is automatic. A
--tournament/--year convenience mode is provided for the four Slams via a
small hardcoded slug table, but it still requires you to have visited the
page once to confirm the slug is current.

Name resolution
------------------
Flashscore renders players as "Surname(s) X." (surname + first-initial).
The ATP dataset (tennis_atp/) has full "Firstname Surname(s)" names. This
script matches by (normalized surname suffix + first-initial), reusing the
name index built the same way predict_v2.py's resolve_player() does so the
resolved names are guaranteed valid inputs to tournament_v2.py. Ambiguous or
unresolved names are printed for manual review and left as the raw
Flashscore string in the output JSON (tournament_v2.py will fail loudly on
those rather than silently mis-resolving).

Usage
-------
  python3.12 fetch_bracket.py --url "https://www.diretta.it/tennis/atp-singolare/wimbledon/tabellone/xY6rfy4l/tabellone/" \\
      --out wimbledon_2026.json

  python3.12 fetch_bracket.py --tournament wimbledon --year 2026 --out wimbledon_2026.json
"""
import argparse
import json
import re
import sys
import unicodedata
from typing import Dict, List, Optional

from predict_v2 import build_name_index

# Known tabellone URLs for the four Slams. Stage ids change every year and
# are NOT guessable from tournament+year alone (see module docstring) — fill
# these in by visiting diretta.it once per edition and copying the URL.
KNOWN_SLAM_URLS: Dict[str, Dict[int, str]] = {
    "wimbledon": {
        2026: "https://www.diretta.it/tennis/atp-singolare/wimbledon/tabellone/xY6rfy4l/tabellone/",
    },
}

SLAM_SLUGS = {"wimbledon", "roland-garros", "french-open", "us-open", "australian-open"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


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


def scrape_draw(url: str, headless: bool = True) -> List[Dict]:
    """Returns a list of rounds: [{"round": str, "matches": [...]}], each
    match = {"home": {"name","seed"}, "homeScore", "away": {...}, "awayScore"}."""
    from playwright.sync_api import sync_playwright

    extract_js = """
    () => {
      function parseRow(row) {
        if (!row) return null;
        const nameEl = row.querySelector('.wcl-participants_ASufu');
        const seedEl = Array.from(row.children).find(c => c.tagName === 'SPAN');
        return {
          name: nameEl ? nameEl.innerText.trim() : null,
          seed: seedEl ? seedEl.innerText.trim() : null
        };
      }
      const rounds = document.querySelectorAll('.draw__round');
      const out = [];
      rounds.forEach((r) => {
        const roundName = (r.querySelector('div,span') || {}).innerText || '';
        const brackets = r.querySelectorAll('.draw__bracket');
        const matches = [];
        brackets.forEach(b => {
          const home = b.querySelector('.bracket__participantRow--home');
          const away = b.querySelector('.bracket__participantRow--away');
          const homeScore = (b.querySelector('.bracket__result--home .bracket__score') || {}).innerText ?? null;
          const awayScore = (b.querySelector('.bracket__result--away .bracket__score') || {}).innerText ?? null;
          matches.push({ home: parseRow(home), homeScore, away: parseRow(away), awayScore });
        });
        out.push({ round: roundName, matches });
      });
      return out;
    }
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"))
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_selector(".draw__round", timeout=15000)
        page.wait_for_timeout(1500)  # let the draw feed request settle
        rounds = page.evaluate(extract_js)
        browser.close()
    return rounds


_RETIRED_TAG = re.compile(r"RET|W\.?O\.?|WALKOVER", re.IGNORECASE)


def match_winner_side(m: Dict) -> Optional[str]:
    """Returns 'home'/'away'/None. Handles the tied-score-at-retirement case
    (e.g. 'Bonzi B. (RET.) 2 - Diallo G. 2'): the set score alone doesn't
    determine the winner there, the (RET.)/(W.O.) tag on the loser does."""
    if not m["home"] or not m["away"] or not m["home"]["name"] or not m["away"]["name"]:
        return None
    if m["homeScore"] is None or m["awayScore"] is None:
        return None
    try:
        h, a = int(m["homeScore"]), int(m["awayScore"])
    except (TypeError, ValueError):
        return None
    if h != a:
        return "home" if h > a else "away"
    home_ret = bool(m["home"]["seed"] and _RETIRED_TAG.search(m["home"]["seed"]))
    away_ret = bool(m["away"]["seed"] and _RETIRED_TAG.search(m["away"]["seed"]))
    if home_ret and not away_ret:
        return "away"
    if away_ret and not home_ret:
        return "home"
    return None  # genuine tie / unresolvable — treat as not decided


def round_is_complete(matches: List[Dict]) -> bool:
    return all(match_winner_side(m) is not None for m in matches)


def winner_name(m: Dict) -> str:
    side = match_winner_side(m)
    return m[side]["name"]


def load_full_names_and_last_active(data_dir: str) -> "tuple[Dict[str, int], Dict[int, str], Dict[int, int]]":
    """Returns (name_index, full_names, last_active): name_index maps
    normalized 'first last' -> player_id (predict_v2's index); full_names
    maps player_id -> a nicely-cased name as it appears in the raw ATP CSVs;
    last_active maps player_id -> the tourney_date (YYYYMMDD int) of their
    most recent match, used to break surname+initial ties in favor of the
    currently-active tour player."""
    import glob
    import pandas as pd

    name_index = build_name_index(data_dir)
    full_names: Dict[int, str] = {}
    for key, pid in name_index.items():
        full_names.setdefault(pid, key)  # any normalized variant is fine; overwritten with nice casing below
    last_active: Dict[int, int] = {}
    for fp in sorted(glob.glob(f"{data_dir}/atp_matches_????.csv")):
        try:
            d = pd.read_csv(fp, usecols=["tourney_date", "winner_id", "winner_name", "loser_id", "loser_name"],
                             low_memory=False)
        except Exception:
            continue
        for _, r in d.iterrows():
            date = int(r["tourney_date"]) if pd.notna(r["tourney_date"]) else 0
            if pd.notna(r["winner_id"]):
                pid = int(r["winner_id"])
                full_names[pid] = r["winner_name"]
                last_active[pid] = max(last_active.get(pid, 0), date)
            if pd.notna(r["loser_id"]):
                pid = int(r["loser_id"])
                full_names[pid] = r["loser_name"]
                last_active[pid] = max(last_active.get(pid, 0), date)
    return name_index, full_names, last_active


def build_bracket_json(rounds: List[Dict], surface: str, best_of: int, is_slam: bool,
                        data_dir: str) -> Dict:
    if not rounds or not rounds[0]["matches"]:
        raise ValueError("No draw data found on the page (selectors may be stale).")

    name_index, full_names, last_active = load_full_names_and_last_active(data_dir)

    round1 = rounds[0]["matches"]
    players_fs, players_resolved, unresolved = [], [], []
    for m in round1:
        for side in ("home", "away"):
            fs_name = m[side]["name"] or "?"
            resolved = resolve_flashscore_name(fs_name, name_index, full_names, last_active)
            players_fs.append(fs_name)
            if resolved is None:
                unresolved.append(fs_name)
                players_resolved.append(fs_name)  # tournament_v2.py now warns + excludes rather than crashing
            else:
                players_resolved.append(resolved)

    results_so_far = []
    for rnd in rounds:
        if round_is_complete(rnd["matches"]):
            winners = []
            for m in rnd["matches"]:
                w_fs = winner_name(m)
                resolved = resolve_flashscore_name(w_fs, name_index, full_names, last_active)
                if resolved is None:
                    unresolved.append(w_fs)
                winners.append(resolved if resolved else w_fs)
            results_so_far.append(winners)
        else:
            break  # rounds must be fixed as a contiguous prefix (see tournament_v2.py)

    # Re-anchor to whatever's still open: any fully-completed leading rounds
    # are collapsed into the `players` list itself (as the survivors who
    # enter the next, not-yet-played round) instead of being carried as a
    # results_so_far prefix. The output bracket then contains ONLY the
    # matches still to be played — already-finished rounds (and any
    # already-eliminated players in them, including unresolvable
    # qualifiers/wildcards with zero ATP history — see the name-resolution
    # warnings) are dropped entirely rather than replayed by the simulator.
    if results_so_far:
        print(f"\n{len(results_so_far)} round(s) already complete — re-anchoring bracket to the "
              f"{len(results_so_far[-1])} surviving players and the matches still to be played "
              f"(dropping the original {len(players_resolved)}-player draw and finished rounds).",
              file=sys.stderr)
        players_resolved = results_so_far[-1]
        results_so_far = []

    if unresolved:
        unresolved = sorted(set(unresolved))
        print(f"\n{len(unresolved)} player name(s) could not be resolved against the ATP dataset "
              f"(no ATP tour-level history) — left as raw Flashscore strings in the output. "
              f"tournament_v2.py will print a warning and treat them as an automatic loss rather "
              f"than crash:", file=sys.stderr)
        for n in unresolved:
            print(f"  - {n}", file=sys.stderr)
        print("Fix by editing the output JSON by hand if you know the correct ATP name.\n",
              file=sys.stderr)

    return {
        "surface": surface,
        "best_of": best_of,
        "is_slam": is_slam,
        "players": players_resolved,
        "results_so_far": results_so_far,
    }


def resolve_url(args) -> str:
    if args.url:
        return args.url
    slug = args.tournament.lower()
    table = KNOWN_SLAM_URLS.get(slug)
    if not table or args.year not in table:
        raise SystemExit(
            f"No known tabellone URL for {args.tournament} {args.year}. "
            f"Visit diretta.it, open the tournament's 'Tabellone' tab, and pass "
            f"--url <that address> instead (or add it to KNOWN_SLAM_URLS)."
        )
    return table[args.year]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", type=str, help="diretta.it tabellone URL")
    ap.add_argument("--tournament", type=str, help="slug, e.g. wimbledon (requires --year, see KNOWN_SLAM_URLS)")
    ap.add_argument("--year", type=int)
    ap.add_argument("--surface", type=str, default=None, help="override auto-detected surface")
    ap.add_argument("--best-of", type=int, default=None, help="override auto-detected best-of")
    ap.add_argument("--slam", action="store_true", help="force is_slam=true")
    ap.add_argument("--data-dir", type=str, default="tennis_atp")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--show-browser", action="store_true", help="run non-headless (debugging)")
    args = ap.parse_args()

    if not args.url and not (args.tournament and args.year):
        ap.error("pass --url, or --tournament + --year")

    url = resolve_url(args)
    slug_guess = next((s for s in SLAM_SLUGS if s in url), None)
    is_slam = args.slam or (slug_guess is not None)
    surface = args.surface or ("Grass" if slug_guess == "wimbledon" else
                                "Clay" if slug_guess in ("roland-garros", "french-open") else "Hard")
    best_of = args.best_of or (5 if is_slam else 3)

    print(f"Fetching draw from {url} ...", file=sys.stderr)
    rounds = scrape_draw(url, headless=not args.show_browser)
    print(f"Got {len(rounds)} round(s): " +
          ", ".join(f"{r['round']}({len(r['matches'])})" for r in rounds), file=sys.stderr)

    bracket = build_bracket_json(rounds, surface, best_of, is_slam, args.data_dir)
    with open(args.out, "w") as f:
        json.dump(bracket, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.out} ({len(bracket['players'])} players, "
          f"{len(bracket['results_so_far'])} round(s) fixed)", file=sys.stderr)


if __name__ == "__main__":
    main()

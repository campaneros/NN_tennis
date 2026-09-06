#!/usr/bin/env python3
"""
app_v2.py — Streamlit front-end for the v2 pipeline.
  /usr/local/bin/python3.12 -m streamlit run app_v2.py
Two modes, auto-detected: LOCAL (tennis_atp/ present: everything, incl. training)
and CLOUD (Streamlit Community Cloud: predict / tournament / draw download / news,
from deploy/state.pkl + committed model artifacts).
"""
import glob
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PY = sys.executable
from predict_v2 import has_raw_data
LOCAL = has_raw_data()
st.set_page_config(page_title="NN tennis v2", layout="wide")
HAS_LIVE = os.path.isdir("data_updated")
action = st.sidebar.radio("Action", (["Upcoming matches"] if HAS_LIVE else []) +
                          ["Predict match", "Tournament", "Rank bracket (top-N ATP)"] + (["Train"] if LOCAL else []))
if not LOCAL:
    st.sidebar.caption("Cloud mode: pre-computed player state (deploy/state.pkl); training runs locally only.")


def run_cmd(args):
    """Stream a subprocess' output into the page; return exit code."""
    box = st.empty()
    lines = []
    proc = subprocess.Popen([PY] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]))
    proc.wait()
    return proc.returncode


@st.cache_resource(show_spinner="Loading model and player state...")
def state(out_dir):
    from predict_v2 import load_state
    return load_state("tennis_atp", out_dir or None)


def model_dir_picker():
    dirs = [d for d in ("models_v2_final", "models_v2") if os.path.isdir(d)]
    return st.selectbox("Model artifacts", dirs) if dirs else None


def match_player(query, players):
    """Tolerant lookup of a typed name inside a bracket.
    -> ("ok", name) | ("choose", [candidates]) | ("none", [suggestions])"""
    import difflib, unicodedata
    norm = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("-", " ")
    q = norm(query).split()
    if not q:
        return "none", []
    exact = [p for p in players if norm(p) == " ".join(q)]
    if exact:
        return "ok", exact[0]
    # every typed token is a whole token of the player's name (surname only, first name only, ...)
    tok = [p for p in players if set(q) <= set(norm(p).split())]
    if len(tok) == 1:
        return "ok", tok[0]
    if len(tok) > 1:
        return "choose", tok
    # typo: closest names by string similarity
    by_name = {norm(p): p for p in players}
    close = difflib.get_close_matches(" ".join(q), list(by_name), n=3, cutoff=0.5)
    sugg = [by_name[c] for c in close]
    # also try surname-only similarity (typo in the surname)
    for p in players:
        if p not in sugg and difflib.SequenceMatcher(None, q[-1], norm(p).split()[-1]).ratio() >= 0.7:
            sugg.append(p)
    return "none", sugg[:4]


def parse_odds_line(line):
    """'Sinner 1.9' / 'Jannik Sinner, 1.90' / 'sinner @1.9' -> (name, odds) or None."""
    import re
    m = re.search(r"[@,;:\s]*([0-9]+(?:[.,][0-9]+)?)\s*$", line)
    if not m or not line[:m.start()].strip():
        return None
    return line[:m.start()].strip(" ,;:@"), float(m.group(1).replace(",", "."))


def pick_player_ui(label, typed, s, key):
    """Resolve a typed player name against the whole ATP name index, asking the
    user to pick when it's a bare surname (several players) or a typo
    (closest names). Returns the canonical name, or None."""
    from predict_v2 import _norm_name
    idx = s["name_index"]
    if _norm_name(typed) in idx:
        return typed
    names = st.session_state.setdefault("_all_names", sorted(idx, key=lambda k: -s.get("last_active", {}).get(idx[k], 0)))
    status, res = match_player(typed, names)
    if status == "ok":
        st.caption(f"{label}: '{typed}' → **{res.title()}**")
        return res
    if status == "choose":
        pick = st.selectbox(f"{label}: '{typed}' matches several players — which one?", ["(choose)"] + [r.title() for r in res[:8]], key=key)
        return None if pick == "(choose)" else pick
    if res:
        pick = st.selectbox(f"{label}: '{typed}' not found — did you mean:", ["(choose)"] + [r.title() for r in res], key=key)
        return None if pick == "(choose)" else pick
    st.warning(f"{label}: '{typed}' not found and no similar name")
    return None


def verdict_box(txt, label):
    verdict = txt.split("=> ")[-1].split(":")[0]
    {"VALUE BET": st.success, "MARGINAL": st.warning}.get(verdict, st.error)(f"{label}: **{verdict}**")
    st.code(txt)


# ---------------------------------------------------------------- Upcoming cards
if action == "Upcoming matches":
    from predict_v2 import predict_match_prob, value_bet_analysis
    out_dir = model_dir_picker()
    n_mc = st.sidebar.slider("MC-dropout samples", 20, 300, 100, 20)
    s = state(out_dir)
    from live_data import season_record, upcoming
    SEASON = season_record()
    up = upcoming(s["name_index"], s["full_names"], s["last_active"])
    if up.empty:
        st.info("No scheduled ATP Tour matches in data_updated/.")
    else:
        tournaments = list(up.tournament.unique())
        sel = st.multiselect("Tournaments", tournaments, default=tournaments)
        up = up[up.tournament.isin(sel)]
        st.caption(f"{len(up)} scheduled match(es). Model = calibrated NN with state updated through the "
                   f"latest results in data_updated/; Win% error bars are the 90% MC-dropout interval.")
        for r in up.itertuples():
            with st.container(border=True):
                slam = bool(r.is_slam)
                bo = 5 if slam else 3
                head = f"**{r.tournament}** · {r.surface_norm} · Bo{bo}{' · Slam' if slam else ''} · {r.round} · {r.date_human}"
                st.markdown(head)
                if r.home_pid is None or r.away_pid is None:
                    st.warning(f"{r.home_name} vs {r.away_name}: player not in ATP dataset — no prediction")
                    continue
                pr = predict_match_prob(int(r.home_pid), int(r.away_pid), r.surface_norm, bo, slam,
                                        "tennis_atp", out_dir, n_mc)
                lo, hi = pr["p1_win_ci90"]
                tr = s["tracker"]
                sn_h = tr.snapshot(int(r.home_pid), r.surface_norm, s["last_date"])
                sn_a = tr.snapshot(int(r.away_pid), r.surface_norm, s["last_date"])
                rec_h = SEASON.get(int(r.home_id), (0, 0)); rec_a = SEASON.get(int(r.away_id), (0, 0))
                h2h = tr.h2h_diff(int(r.home_pid), int(r.away_pid))
                c1, c2, c3 = st.columns([3, 2, 3])
                c1.metric(r.home_atp_name, f"{pr['p1_win_prob']:.0%}", f"±{(hi-lo)/2:.0%} CI", delta_color="off")
                c1.caption(f"ATP #{int(r.home_rank) if pd.notna(r.home_rank) else '?'} · season {rec_h[0]}-{rec_h[1]} · "
                           f"Elo {sn_h['elo']:.0f} ({r.surface_norm} {sn_h['elo_surf']:.0f}) · "
                           f"last-50 win {sn_h['winrate_recent']:.0%}" if sn_h['winrate_recent'] == sn_h['winrate_recent'] else "")
                c3.metric(r.away_atp_name, f"{pr['p2_win_prob']:.0%}", f"±{(hi-lo)/2:.0%} CI", delta_color="off")
                c3.caption(f"ATP #{int(r.away_rank) if pd.notna(r.away_rank) else '?'} · season {rec_a[0]}-{rec_a[1]} · "
                           f"Elo {sn_a['elo']:.0f} ({r.surface_norm} {sn_a['elo_surf']:.0f}) · "
                           f"last-50 win {sn_a['winrate_recent']:.0%}" if sn_a['winrate_recent'] == sn_a['winrate_recent'] else "")
                if h2h:
                    c2.markdown(f"<div style='text-align:center'>H2H: <b>{'+' if h2h>0 else ''}{h2h}</b> "
                                f"{r.home_atp_name.split()[-1] if h2h>0 else r.away_atp_name.split()[-1]}</div>",
                                unsafe_allow_html=True)
                oh, oa = r.home_odds_match_winner, r.away_odds_match_winner
                if pd.notna(oh) and pd.notna(oa):
                    c2.markdown(f"<div style='text-align:center'>odds<br><b>{oh:.2f}</b> — <b>{oa:.2f}</b><br>"
                                f"implied {1/oh:.0%} — {1/oa:.0%}</div>", unsafe_allow_html=True)
                    for nm, p, ci, od in ((r.home_atp_name, pr["p1_win_prob"], (lo, hi), oh),
                                          (r.away_atp_name, pr["p2_win_prob"], (1-hi, 1-lo), oa)):
                        txt = value_bet_analysis(nm, p, ci, float(od))
                        verdict = txt.split("=> ")[-1].split(":")[0]
                        if verdict != "NO BET":
                            {"VALUE BET": st.success, "MARGINAL": st.warning}[verdict](f"{nm} @ {od:.2f}: **{verdict}**")
                            with st.expander("details"):
                                st.code(txt)

# ---------------------------------------------------------------- Predict
elif action == "Predict match":
    from predict_v2 import predict_match_prob, resolve_player, staking_plan, value_bet_analysis
    c1, c2 = st.columns(2)
    p1 = c1.text_input("Player 1", "Carlos Alcaraz")
    p2 = c2.text_input("Player 2", "Jannik Sinner")
    surface = c1.selectbox("Surface", ["Hard", "Clay", "Grass"])
    best_of = c2.selectbox("Best of", [3, 5])
    slam = c1.checkbox("Slam", value=best_of == 5)
    n_mc = c2.slider("MC-dropout samples", 50, 500, 200, 50)
    o1 = c1.number_input(f"Odds on P1 (0 = none)", 0.0, 50.0, 0.0, 0.05)
    o2 = c2.number_input(f"Odds on P2 (0 = none)", 0.0, 50.0, 0.0, 0.05)
    news = c2.checkbox("Search injury news (Google News, English)", value=True)
    out_dir = model_dir_picker()
    s = state(out_dir)
    p1 = pick_player_ui("Player 1", p1, s, "pick_p1") if p1.strip() else None
    p2 = pick_player_ui("Player 2", p2, s, "pick_p2") if p2.strip() else None
    if p1 and p2 and st.button("Run"):
        pid1, pid2 = resolve_player(p1, s["name_index"]), resolve_player(p2, s["name_index"])
        r = predict_match_prob(pid1, pid2, surface, best_of, slam, "tennis_atp", out_dir, n_mc)
        lo, hi = r["p1_win_ci90"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([p1, p2], [r["p1_win_prob"], r["p2_win_prob"]],
               yerr=[[r["p1_win_prob"] - lo, hi - r["p1_win_prob"]], [hi - r["p1_win_prob"], r["p1_win_prob"] - lo]],
               capsize=8, color=["tab:blue", "tab:orange"])
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.set_ylim(0, 1); ax.set_ylabel("P(win)  ±90% CI")
        ax.set_title(f"{surface} · Bo{best_of}{' · Slam' if slam else ''}")
        st.pyplot(fig)
        st.write(f"**P({p1}) = {r['p1_win_prob']:.3f}** [{lo:.3f}–{hi:.3f}] · "
                 f"**P({p2}) = {r['p2_win_prob']:.3f}** · Elo {r['p1_elo']:.0f} vs {r['p2_elo']:.0f}")
        for name, p, ci, odds in ((p1, r["p1_win_prob"], (lo, hi), o1),
                                  (p2, r["p2_win_prob"], (1 - hi, 1 - lo), o2)):
            if odds > 1:
                verdict_box(value_bet_analysis(name, p, ci, odds), f"{name} @ {odds:.2f}")
        if news:
            from news_v2 import fetch_news
            for name in (p1, p2):
                items = fetch_news(name)
                with st.expander(f"News: {name} — {len(items)} injury/withdrawal headlines, last 30d", expanded=bool(items)):
                    for d in items:
                        st.markdown(f"- {d['date']} [{d['title']}]({d['link']}) — {d['source']}")
                    if not items:
                        st.caption("no matching headlines")
            st.caption("Headlines are context the model does NOT see — weigh them before staking.")
        if o1 > 1 and o2 > 1:
            plan = staking_plan(p1, p2, r["p1_win_prob"], (lo, hi), o1, o2)
            (st.success if "BET" in plan and "NO BET" not in plan else st.info)(
                "\n".join(l.strip() for l in plan.splitlines() if l.strip().startswith(("=>", "ARB"))))
            st.code(plan)

# ---------------------------------------------------------------- Tournament
elif action == "Tournament":
    from predict_v2 import resolve_player, value_bet_analysis
    from tournament_v2 import match_prob_matrix, simulate
    SURF = ["Hard", "Clay", "Grass"]
    src = st.radio("Bracket source", ["Download from diretta.it", "Paste names", "Pick file", "Upload"], horizontal=True)
    cfg = None
    if src == "Download from diretta.it":
        from fetch_bracket import build_bracket, fetch_draw, infer_format, load_full_names_and_last_active, make_resolver, resolve_tournament
        slug = st.text_input("Tournament slug or diretta.it URL", "cincinnati",
                             help="as in https://www.diretta.it/tennis/atp-singolare/<slug>/ — e.g. us-open, wimbledon, cincinnati, roma")
        if slug:
            try:
                t = resolve_tournament(slug)
                s0, b0, sl0 = infer_format(t["slug"])
                c1, c2, c3 = st.columns(3)
                surface = c1.selectbox("Surface", SURF, index=SURF.index(s0))
                best_of = c2.selectbox("Best of", [3, 5], index=[3, 5].index(b0))
                slam = c3.checkbox("Slam", value=sl0)
                b = build_bracket(fetch_draw(t), make_resolver(*load_full_names_and_last_active("tennis_atp")),
                                  surface, int(best_of), slam)
                if b["finished"]:
                    st.info(f"{t['slug']}: tournament finished — champion {b['champion']}")
                else:
                    cfg = {k: b[k] for k in ("surface", "best_of", "is_slam", "players", "results_so_far")}
                    played = sum(1 for w in (b["results_so_far"][0] if b["results_so_far"] else []) if w)
                    st.success(f"{t['slug']}: round {b['round_index']+1}/{b['rounds_total']} open — "
                               f"{len(b['players'])} players still in, {played} match(es) of this round already played")
                    if b["unresolved"]:
                        st.warning(f"No ATP history (treated as auto-loss): {', '.join(b['unresolved'])}")
                    with st.expander("Bracket (next-round pairings, in order)"):
                        for i in range(0, len(cfg["players"]), 2):
                            st.write(f"{cfg['players'][i]}  vs  {cfg['players'][i+1]}")
            except Exception as e:
                st.error(f"Download failed: {e}")
    elif src == "Paste names":
        txt = st.text_area("One player per line, in bracket order (power of 2)", height=200)
        c1, c2, c3 = st.columns(3)
        names = [l.strip() for l in txt.splitlines() if l.strip()]
        if names:
            cfg = {"players": names, "surface": c1.selectbox("Surface", SURF), "best_of": c2.selectbox("Best of", [3, 5]),
                   "is_slam": c3.checkbox("Slam"), "results_so_far": []}
            if len(names) & (len(names) - 1):
                st.error(f"{len(names)} players — need a power of 2"); cfg = None
    elif src == "Pick file":
        files = sorted(glob.glob("*.json"))
        if files:
            cfg = json.load(open(st.selectbox("Bracket JSON", files)))
    else:
        up = st.file_uploader("Bracket JSON", type="json")
        cfg = json.load(up) if up else None

    sims = st.slider("Simulations", 1000, 50000, 10000, 1000)
    n_mc_model = st.slider("MC-dropout samples per pair", 5, 100, 30, 5)
    odds_txt = st.text_area("Outright (title) odds — optional, one per line: `Player name, decimal odds`",
                            placeholder="Jannik Sinner, 1.90\nZverev 6.5", height=90)
    outright = []   # [(player name in bracket, odds)] resolved interactively below
    if cfg and odds_txt.strip():
        for k, line in enumerate(l for l in odds_txt.splitlines() if l.strip()):
            parsed = parse_odds_line(line)
            if not parsed:
                st.warning(f"Line {k+1}: could not find the odds in {line!r} — write `Name, 1.90` (or `Name 1.90`)"); continue
            nm, od = parsed
            status, res = match_player(nm, cfg["players"])
            if status == "ok":
                outright.append((res, od))
                if res.lower() != nm.lower():
                    st.caption(f"Line {k+1}: '{nm}' → **{res}** @ {od:.2f}")
            elif status == "choose":
                pick = st.selectbox(f"Line {k+1}: '{nm}' matches several players — which one?", ["(skip)"] + res, key=f"odds_pick_{k}")
                if pick != "(skip)":
                    outright.append((pick, od))
            else:
                if res:
                    pick = st.selectbox(f"Line {k+1}: '{nm}' not in this bracket — did you mean:", ["(skip)"] + res, key=f"odds_sugg_{k}")
                    if pick != "(skip)":
                        outright.append((pick, od))
                else:
                    st.warning(f"Line {k+1}: '{nm}' not in this bracket and no similar name found")
    out_dir = model_dir_picker()
    if cfg and st.button("Run"):
        s = state(out_dir)
        players = cfg["players"]
        pids = [resolve_player(p, s["name_index"]) for p in players]
        bad = [p for p, i in zip(players, pids) if i is None]
        if bad:
            st.warning(f"Unresolved (treated as auto-loss): {', '.join(bad)}")
        with st.spinner(f"Scoring {len(players)}×{len(players)} matchups..."):
            mat = match_prob_matrix(players, pids, cfg.get("surface", "Hard"), int(cfg.get("best_of", 3)),
                                    bool(cfg.get("is_slam", False)), "tennis_atp", out_dir, n_mc_model)
        with st.spinner("Simulating..."):
            sim = simulate(players, mat["P_samples"], cfg.get("results_so_far", []), sims)
        order = np.argsort(-sim["title_prob"])[:15]
        fig, ax = plt.subplots(figsize=(7, 5))
        tp, ci = sim["title_prob"][order], sim["title_ci90"][order]
        ax.barh([players[i] for i in order][::-1], tp[::-1],
                xerr=[(tp - ci[:, 0])[::-1], (ci[:, 1] - tp)[::-1]], capsize=4)
        ax.set_xlabel("P(title)  ±90% CI"); ax.set_title("Top 15 title probabilities")
        st.pyplot(fig)
        for nm, od in outright:
            idx = players.index(nm)
            verdict_box(value_bet_analysis(f"{nm} (title)", float(sim["title_prob"][idx]),
                                           tuple(map(float, sim["title_ci90"][idx])), od),
                        f"{nm} to win the title @ {od:.2f}")
        n_r = sim["n_rounds"]
        df = pd.DataFrame(sim["reach_prob"][:, 1:], index=players, columns=[f"R{r}" for r in range(1, n_r)] + ["Title"])
        st.dataframe(df.sort_values("Title", ascending=False).style.format("{:.3f}"), height=500)

# ---------------------------------------------------------------- Train
elif action == "Train":
    data = st.text_input("Data CSV", "atp_matches_pretrain.csv")
    final = st.checkbox("Production refit (--final, all data)")
    out_dir = st.text_input("Out dir", "models_v2_final" if final else "models_v2")
    if st.button("Run training"):
        rc = run_cmd(["train_v2.py", "--data", data, "--out-dir", out_dir] + (["--final"] if final else []))
        st.success("Done") if rc == 0 else st.error(f"Exit {rc}")
    st.divider()
    show = st.selectbox("Show results of", [d for d in ("models_v2", "models_v2_final") if os.path.isdir(d)])
    if show:
        h = json.load(open(f"{show}/nn_history.json"))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(h["train_loss"], label="train loss")
        if h.get("val_loss"):
            ax.plot(h["val_loss"], label="val loss")
            be = int(np.argmin(h["val_loss"]))
            ax.axvline(be, c="k", ls=":", label=f"kept weights (epoch {be + 1}, early stopping)")
        if h.get("stage1_val_loss"):
            ax.plot(h["stage1_val_loss"], "--", label="stage-1 val loss")
            ax.axvline(h["stage1_best_epoch"], c="gray", ls=":", label="chosen epochs")
        if h.get("val_entropy_floor"):
            ax.axhline(h["val_entropy_floor"], c="tab:red", ls="--", lw=1,
                       label=f"calibrated-oracle floor at this sharpness ({h['val_entropy_floor']:.4f})")
            st.caption("Red dashed line: the log-loss a PERFECTLY calibrated model would score if its "
                       "probabilities were the true ones, at the model's current confidence level. "
                       "Val loss sitting on it = calibrated; the distance from it to 0 can only be "
                       "closed by sharper predictions, i.e. new information, not more training.")
        ax.set_xlabel("epoch"); ax.set_ylabel("log-loss"); ax.legend()
        st.pyplot(fig)
        if os.path.exists(f"{show}/benchmark_results.csv"):
            bench = pd.read_csv(f"{show}/benchmark_results.csv").set_index("model")
            st.dataframe(bench.style.format("{:.4f}"))
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for ax, m in zip(axes.flat, ["accuracy", "log_loss", "roc_auc", "ece"]):
                bench[m].plot.barh(ax=ax, color="tab:blue"); ax.set_title(m); ax.set_ylabel("")
                lo, hi = bench[m].min(), bench[m].max(); ax.set_xlim(lo - 0.3 * (hi - lo), hi + 0.1 * (hi - lo))
            fig.tight_layout(); st.pyplot(fig)
        if os.path.exists(f"{show}/reliability_diagram.png"):
            st.image(f"{show}/reliability_diagram.png")


# ---------------------------------------------------------------- Rank bracket
else:
    from rank_bracket import build_rank_bracket, ranking_from_csv, ranking_from_snapshot
    n = st.number_input("Top N", 4, 128, 30)
    excl = st.text_input("Exclude (comma-separated)", "")
    c1, c2 = st.columns(2)
    surface = c1.selectbox("Surface", ["Hard", "Clay", "Grass"])
    best_of = c2.selectbox("Best of", [3, 5])
    slam = st.checkbox("Slam")
    out = st.text_input("Output JSON", f"top{n}_{surface.lower()}.json")
    if st.button("Build"):
        ranking, name_index = ranking_from_csv("tennis_atp") if LOCAL else ranking_from_snapshot()
        bracket = build_rank_bracket(ranking, int(n), [x.strip() for x in excl.split(",") if x.strip()],
                                     name_index, surface, int(best_of), slam)
        json.dump(bracket, open(out, "w"), indent=2, ensure_ascii=False)
        st.success(f"Saved {out} — now Tournament → Pick file")
        st.json(bracket, expanded=False)

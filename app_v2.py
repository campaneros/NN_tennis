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
LOCAL = os.path.isdir("tennis_atp")
st.set_page_config(page_title="NN tennis v2", layout="wide")
action = st.sidebar.radio("Action", ["Predict match", "Tournament", "Rank bracket (top-N ATP)", "Train"])
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


def verdict_box(txt, label):
    verdict = txt.split("=> ")[-1].split(":")[0]
    {"VALUE BET": st.success, "MARGINAL": st.warning}.get(verdict, st.error)(f"{label}: **{verdict}**")
    st.code(txt)


# ---------------------------------------------------------------- Predict
if action == "Predict match":
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
    if st.button("Run"):
        s = state(out_dir)
        pid1, pid2 = resolve_player(p1, s["name_index"]), resolve_player(p2, s["name_index"])
        if pid1 is None or pid2 is None:
            st.error(f"Unresolved: p1→{pid1}, p2→{pid2}. Check spelling.")
            st.stop()
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
                            placeholder="Jannik Sinner, 1.90\nAlexander Zverev, 6.5", height=90)
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
        for line in [l for l in odds_txt.splitlines() if l.strip()]:
            try:
                nm, od = [x.strip() for x in line.rsplit(",", 1)]; od = float(od)
            except ValueError:
                st.warning(f"Could not parse {line!r} (expected `Name, odds`)"); continue
            idx = next((i for i, p in enumerate(players) if p.lower() == nm.lower()), None)
            if idx is None:
                pid = resolve_player(nm, s["name_index"]); idx = next((i for i, q in enumerate(pids) if q is not None and q == pid), None)
            if idx is None:
                st.warning(f"'{nm}' is not in this bracket"); continue
            verdict_box(value_bet_analysis(f"{players[idx]} (title)", float(sim["title_prob"][idx]),
                                           tuple(map(float, sim["title_ci90"][idx])), od),
                        f"{players[idx]} to win the title @ {od:.2f}")
        n_r = sim["n_rounds"]
        df = pd.DataFrame(sim["reach_prob"][:, 1:], index=players, columns=[f"R{r}" for r in range(1, n_r)] + ["Title"])
        st.dataframe(df.sort_values("Title", ascending=False).style.format("{:.3f}"), height=500)

# ---------------------------------------------------------------- Train
elif action == "Train":
    data = st.text_input("Data CSV", "atp_matches_pretrain.csv")
    final = st.checkbox("Production refit (--final, all data)")
    out_dir = st.text_input("Out dir", "models_v2_final" if final else "models_v2")
    if not LOCAL:
        st.info("Training needs the full dataset — run `train_v2.py` locally, then `export_state.py`, commit, push.")
    elif st.button("Run training"):
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

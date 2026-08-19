#!/usr/bin/env python3
"""
export_state.py — freeze the fully-replayed player state into deploy/state.pkl
so the Streamlit app (which has no tennis_atp/ submodule and ~1 GB RAM) can
predict without replaying 196K matches. Re-run after every data update:
  git submodule update --remote && python3.12 export_state.py && git commit -am "state"
"""
import os, pickle, sys
import pandas as pd
from data_pipeline_v2 import build_pretrain_table, load_raw_atp
from predict_v2 import build_name_index

data_dir = sys.argv[1] if len(sys.argv) > 1 else "tennis_atp"
df_raw = load_raw_atp(data_dir)
_, tracker = build_pretrain_table(df_raw)
name_index = build_name_index(data_dir)
from fetch_bracket import load_full_names_and_last_active
_, full_name, last_active = load_full_names_and_last_active(data_dir)
r = pd.read_csv(f"{data_dir}/atp_rankings_current.csv")
r = r[r.ranking_date == r.ranking_date.max()].sort_values("rank").head(300)
ranking = [(int(row["rank"]), int(row.player), full_name.get(int(row.player), "?"))
           for _, row in r.iterrows() if int(row.player) in full_name]
os.makedirs("deploy", exist_ok=True)
with open("deploy/state.pkl", "wb") as f:
    pickle.dump(dict(tracker=tracker, name_index=name_index, last_date=int(df_raw.tourney_date.max()),
                     ranking=ranking, ranking_date=int(r.ranking_date.max()),
                     full_names=full_name, last_active=last_active), f)
print(f"deploy/state.pkl: {os.path.getsize('deploy/state.pkl')/1e6:.1f} MB, data through {df_raw.tourney_date.max()}, "
      f"ranking {r.ranking_date.max()} top {len(ranking)}")

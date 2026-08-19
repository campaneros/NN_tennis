# Streamlit deployment

The app (`app_v2.py`) runs in two modes, detected automatically:

| | local checkout (`tennis_atp/` present) | Streamlit Cloud (this branch as-is) |
|---|---|---|
| Predict / Tournament / Rank bracket / news | replays history once (~90 s) | loads `deploy/state.pkl` (<1 s) |
| Train tab | runs `train_v2.py`, shows plots | shows saved plots only |
| Fetch bracket (diretta.it, Playwright) | yes | hidden |

## Deploy
1. share.streamlit.io → New app → repo `campaneros/NN_tennis`, branch `streamlit`, main file `app_v2.py`. Python 3.12 (`.python-version`), deps from `requirements.txt`.
2. Done. Nothing else to configure.

## Update the model / data (local, then push)
```bash
git submodule update --remote tennis_atp
/usr/local/bin/python3.12 data_pipeline_v2.py                      # -> atp_matches_pretrain.csv
/usr/local/bin/python3.12 train_v2.py --data atp_matches_pretrain.csv --out-dir models_v2          # benchmark (plots)
/usr/local/bin/python3.12 train_v2.py --data atp_matches_pretrain.csv --out-dir models_v2_final --final
/usr/local/bin/python3.12 export_state.py                           # -> deploy/state.pkl (~15 MB)
git add models_v2 models_v2_final deploy && git commit -m "refresh model + state" && git push
```
Tracked artifacts (all small): `models_v2_final/*` (model, scaler, vocab), `models_v2/*` (benchmark metrics + plots), `deploy/state.pkl` (frozen player state, ranking top-300, name index).

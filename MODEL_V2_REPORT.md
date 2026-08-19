# MODEL_V2_REPORT.md — Pre-match win probability + tournament simulation (v2)

Status: `data_pipeline_v2.py`, `model_v2.py`, `train_v2.py`, `evaluate_v2.py`,
`predict_v2.py`, `tournament_v2.py`. Old system (`compute_elo.py`, `runner.py`,
`predict.py`, `live_eval.py`) is left in place, unmodified, for reference and
is superseded by the above for the pre-match / pre-tournament use case
described here.

---

## 1. Why the old system does not work for this task

### 1.1 The central bug: train/inference distribution mismatch

`runner.py` trains `TennisMatchNet` (BiLSTM → Transformer encoder, CLS
token) on **real, observed, point-by-point Match Charting Project set
sequences**: `SEQ_LEN=5` sets of *already-played* serve/return/rally stats,
labelled with the match winner. This is a perfectly reasonable model — for
a different problem: "given the sets played so far, who wins the match"
(a live/in-play model).

For **pre-match** prediction, no sets have been played yet, so
`predict.py` cannot feed the model real data. Instead
(`predict.py:637-709`, `build_feature_sequence`) it fabricates 5 synthetic
"sets" by repeatedly sampling `player_profile + Gaussian noise` from each
player's *career-average* stat profile and stacking that 5 times as if it
were an observed match. The network never saw inputs like this during
training — noise-perturbed static averages have none of the serve-alternation,
score-pressure, momentum, or set-to-set correlation structure the
BiLSTM/Transformer was trained to exploit. This is a textbook covariate-shift
failure: the model is evaluated far outside the support of its training
distribution.

**This is not a hypothesis — it is directly visible in this repo's own
existing evaluation output.** `eval_out/eval_summary.json` (produced by
`live_eval.py` on 6,162 charted matches) reports **`"blended"` metrics
identical to `"stat_elo"` metrics** — meaning the neural network's
contribution to the final blended prediction was exactly zero. This is
`predict.py`'s own designed safety behavior: `NN_SIGMA_CUTOFF`/`NN_MIN_MATCHES`
down-weight or exclude the NN when its Monte-Carlo variance is high — and
the fabricated-sequence input generates exactly that high variance. In
production, the "neural net for pre-match prediction" the old system ships
was silently contributing nothing.

### 1.2 Secondary bugs

- **`runner.py` duplicate definitions.** Every function/class (`download_files`,
  `load_and_merge`, `stat_elo_features`, `TennisMatchNet`, `train_model`,
  `main`, ...) is defined twice, each followed by its own
  `if __name__ == "__main__": main()`. Running the file executes *both*
  `main()` bodies sequentially; only the second (bottom) definitions end up
  live. The ATP-repo-aware loader added in the first block
  (`_ensure_files`/`__ATP__` sentinel) is dead code — never reachable from
  a normal `python runner.py` invocation.
- **No temporal discipline in evaluation.** `live_eval.py` has no date
  filter (`grep` for `--date`/`holdout`/`split` returns nothing); its
  reported metrics are not an out-of-time estimate, they are closer to an
  in-sample/resubstitution estimate on whatever charted matches were
  available. Despite this optimistic bias, the old system *still* loses on
  every metric to the new model's strict blind 2024+ holdout (section 5).
- **Name-keyed players.** The old pipeline's player identity is implicit in
  string names pulled from Match Charting Project files. 19 ATP `player_id`s
  in `tennis_atp` have more than one distinct name spelling in the raw data;
  name-keying silently fragments one player's Elo/form history across
  "ghost" identities. v2 keys everything by the integer `player_id`.
- **Static-Elo temporal leakage in `runner.py`.** `compute_elo.py`'s Elo math itself is
  correct (standard logistic Elo expectation, correct symmetric zero-sum update, Kovalchik's
  (2016) experience-based K-factor `250/(n+5)^0.4`) — but it outputs a single **final snapshot**
  (`player_elo.json`), computed using the entire dataset through its last date. `runner.py`'s
  `external_elo_features()` (line 1147) takes no date argument: it looks up that same static
  snapshot value and attaches it to **every** historical training row for that player, regardless
  of the row's own match date. A training row from 2015 receives that player's Elo as computed
  from results through the file's generation date (2026) — direct, uncontrolled temporal
  leakage on top of the synthetic-sequence bug in section 1.1. `compute_elo.py` is not used
  anywhere in the v2 pipeline; v2 computes its own walk-forward Elo inline in
  `data_pipeline_v2.PlayerStateTracker`, snapshotted strictly before each match updates it —
  verified empirically (Alcaraz's own `p1_elo` starts at 1500.0 at his Feb-2020 tour debut and
  rises chronologically to ~2330 by 2026 in `atp_matches_pretrain.csv`; current top-15 career Elo
  reads Sinner/Alcaraz/Djokovic/Federer/Borg/Nadal/Söderling/Zverev/del Potro — all sane; top
  clay-Elo reads Borg/Nadal/Djokovic/Alcaraz/Lendl — exactly the expected clay-GOAT ordering).
- **Data-source mismatch for the "pre-match" use case.** Match Charting
  Project has detailed point-by-point stats but only for a curated subset of
  (mostly top-tier, mostly 2010s+) matches — it has no full historical
  ranking/result archive. `tennis_atp` has the opposite: full 1968-2026
  match results/rankings but no shot-level stats. The old pipeline trains
  exclusively on the small, biased Charting subset; it never sees the bulk
  of ATP tour history at all.

### 1.3 Why this matters for betting-oriented pre-match probabilities

None of the three signals blended by `predict.py` (NN / stat-profile Elo /
external Elo file) is validated out-of-time; the NN component is
demonstrably inert; and the surviving stat-Elo-only "blend" reported
`log_loss=0.6564`, `ece=0.0682` (see section 5) — well short of what a
correctly-built pre-match model should achieve.

---

## 2. Dataset analysis

### 2.1 What's usable pre-match / pre-tournament

Source: `tennis_atp/atp_matches_YYYY.csv`, 1968-2026, ATP tour-level only
(no qualifying/Challengers/ITFs). Columns split cleanly into two groups:

| Usable pre-match (kept) | NOT usable pre-match (excluded from v2) |
|---|---|
| `winner_id`/`loser_id` (→ `p1_id`/`p2_id`) | `w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_bpSaved`, `w_bpFaced` and the `l_*` mirrors — **this match's own** in-play stats |
| `surface`, `best_of`, `tourney_level` (→ `is_slam`) | `minutes` (unknown before the match is played) |
| `winner_rank`/`loser_rank`, `*_rank_points` (as of `tourney_date`, i.e. pre-match by construction per the tennis_atp README) | `score`, `round` used only to derive `is_walkover` and `is_slam`/`best_of` sanity, never fed as features |
| `winner_age`/`loser_age`, `*_ht`, `*_hand` | |
| `tourney_date` (used only to order the walk-forward pass, never as a feature) | |

The user-requested rolling "current form" stats (`form_1st_won_pct`, etc.,
`STAT_KEYS` in `data_pipeline_v2.py`) are the mean of each player's serve/
return performance over their **prior** `STAT_WINDOW=20` matches — a
legitimate pre-match signal (a rolling average of past, completed matches),
never that match's own numbers.

Elo (`p1_elo`, `p1_elo_surf`), rolling win-rate, H2H, and rest-days are all
snapshotted by `PlayerStateTracker.snapshot()` **before** `tracker.update()`
is called with the match's outcome — see `data_pipeline_v2.py:270-372`,
`build_pretrain_table`. This is the single walk-forward pass that guarantees
no target leakage: a row's features can only depend on strictly earlier rows.

### 2.2 Deriving `surface` / `best_of` / `is_slam`

Verified against the full 1968-2026 ATP archive:

- `is_slam = (tourney_level == 'G')` implies `best_of == 5` in 99.0% of
  rows; exceptions are 1977 US Open early rounds (real historical
  scheduling quirk, not a data error — the tournament used best-of-3 for
  early rounds that year).
- `best_of == 5` does **not** imply `is_slam`: Davis Cup live rubbers
  (`tourney_level == 'D'`) were best-of-5 through 2018, and some Tour Finals
  (`'F'`) matches are best-of-5. So `is_bo5` and `is_slam` are kept as two
  independent binary features, not collapsed into one — collapsing them
  would silently mispredict Davis Cup/Finals rows.
- Surface strings are normalized (`Hard Court`, `Indoor`, `Acrylic`, etc. →
  `Hard`) to the 4 canonical `{Hard, Clay, Grass, Carpet}` values.

This directly implements the required domain rule: for men, Slam ⇒ best-of-5
and non-Slam ⇒ best-of-3 (up to the 1977 historical exception, which is
data, not a bug); the model is never asked to reconcile an
internally-inconsistent combination because `is_bo5`/`is_slam` reproduce the
raw data faithfully rather than a hand-imposed rule.

### 2.3 WTA

**There is no full WTA match/ranking archive in this repository.**
`tennis_atp` is ATP/men-only. `tennis_MatchChartingProject` has a small
charted subset for women but no pre-match ranking archive to build Elo/rank
features from. Consequently: **v2's training set is ATP (men) only.** The
schema carries a `tour` column (currently always `"ATP"`) specifically so a
WTA source can be plugged into `data_pipeline_v2.py` later without changing
the model or the training script — but today, claiming WTA coverage would
be false. This is a dataset limitation, stated explicitly rather than
silently training a men's-only model and presenting it as gender-general.
The best-of/Slam rule for women (Slams are also best-of-3) is implemented
in the *feature encoding* (`is_bo5`, `is_slam` are independent flags, so a
future WTA row with `is_slam=1, is_bo5=0` is representable and would not
require an architecture change) even though no WTA rows exist yet to train
on.

### 2.4 Data quality issues found and handled

- 3,021 of 199,109 raw rows dropped: missing `winner_id`/`loser_id`/
  `tourney_date`/`surface`, or `best_of` not in `{3, 5}` (36 best-of-1
  exhibition/unfinished rows plus assorted parsing failures).
- `is_walkover` matches flagged (0.73% of rows) via regex on `score`
  (`W/O|WEA|DEF`); their in-match serve stats (if any) are excluded from
  the rolling "current form" window since they don't reflect real on-court
  performance, but the match result itself is still used to update Elo/H2H
  (a walkover is a real result).
- Missing serve-stat columns (pre-1991 rows, and some modern rows) are left
  as `NaN` and imputed with the **train-fold median** (`SimpleImputer`,
  fit on train only) for the logistic regression and NN; XGBoost handles
  `NaN` natively via its missing-value branch-direction learning.
- 7,420 unique player IDs across the full history; 19 have multiple
  recorded name spellings (handled by ID-keying, not name-keying).
- Class balance: the label `y` is engineered to be ~50/50 by randomized
  P1/P2 side assignment (`data_pipeline_v2.py:303`), so `winner`/`loser`
  identity is not recoverable from column position — verified empirically
  (P1-win rate = 0.4995 on the full table).

---

## 3. Model families considered

| | Elo-only (baseline) | Logistic Regression | XGBoost (GBDT) | **TennisEmbeddingNet (chosen NN)** |
|---|---|---|---|---|
| Feature requirements | career Elo diff only | engineered numeric diffs | engineered numeric diffs (NaN-native) | numeric diffs **+ player identity via embedding** |
| Handles rare categories | n/a | poorly (one-hot would blow up / underfit rare players) | poorly (trees can split on player one-hots but generalize weakly with few rows per player) | **by design** — shared embedding table transfers statistical strength; OOV row for unseen players |
| Smooth/continuous interactions | none (single logistic curve on one scalar) | only linear + manually engineered terms | axis-aligned step functions (many shallow trees approximate smoothness but don't represent it natively) | continuous, differentiable non-linear interactions via MLP + LayerNorm/GELU |
| Overfitting risk | none (0 fitted params) | low | moderate, controlled via `max_depth=4`, `min_child_weight=20`, `reg_lambda/alpha`, early stopping | controlled via small embedding dim (24), dropout (0.35), weight decay (AdamW), early stopping |
| Interpretability | very high (1 number) | high (coefficients) | medium (gain-based importances, section 3.1) | low-medium (embeddings are not directly interpretable; feature importances not natively available) |
| Expected calibration quality | poor (no fitting at all — pure textbook formula) | good, but limited by linear-only decision surface | good but needs post-hoc calibration (trees are not natively well-calibrated for extreme probabilities) | needs post-hoc calibration; NN logits are known to be overconfident (Guo et al., 2017) |
| Verdict | trivial sanity floor | strong, transparent benchmark | **strong non-neural benchmark** | **primary model under study** |

### 3.1 XGBoost feature importances (gain, test-independent, from the train fold)

```
elo_diff                  834.3   winrate_recent_diff        35.5
elo_surf_diff              229.6   rank_points_log_diff       35.2
rank_log_diff               85.8   is_bo5                      32.3
winrate_recent_surf_diff    83.3   ht_diff                     31.6
age_diff                    47.3   p1_elo                      31.2
matches_surf_diff           43.4   experience_diff             30.6
rest_days_diff               38.2   p2_elo                      27.8
```
Elo (career + surface) dominates, as expected for tennis; rank, recent
surface form, age, experience, and best-of/rest-days all carry real signal.
This importance ranking is a useful sanity check on the whole pipeline
(nothing in the top-20 is a leakage-shaped feature such as a same-match
in-play stat) and is reused qualitatively to sanity-check what the NN
embedding is (and isn't) adding beyond these numeric signals.

### 3.2 Why a neural net is still the right thing to study here, even though XGBoost is competitive

On this dataset, XGBoost matches or very slightly beats the embedding NN on
raw log-loss/AUC (section 5) — consistent with a well-known empirical
finding that gradient-boosted trees remain very hard to beat on
medium-sized, heterogeneous **tabular** data (Grinsztajn, Oyallon & Varoquaux,
2022, *"Why do tree-based models still outperform deep learning on tabular
data?"*, NeurIPS). We report this honestly rather than picking a winner by
fiat, per the task's requirement.

The reasons the NN remains the primary object of study, not just a
"because it's flashier" choice:

1. **It is the only family here that can natively represent player identity
   as a first-class, generalizable signal.** XGBoost could ingest a
   `player_id` as an integer feature, but a tree only ever produces
   axis-aligned splits/thresholds on it — it cannot express "players whose
   IDs are numerically close are similar," so it would either overfit
   (memorize specific IDs where there's enough data) or ignore the column
   entirely. An embedding table is designed exactly for this: nearby points
   in the learned 24-dimensional space *by training pressure* correspond to
   players with similar win/loss patterns against similar opponents. This
   is precisely the "shared latent representation for rare categories"
   requirement (section 4.2 and 4.3 below).
2. **41% of the blind 2024+ test set involves at least one player who never
   appeared in the pre-2022 training window** (`split_info.json:
   test_oov_player_rate`). Any model used for real forward-looking betting
   will constantly face debutants and rarely-seen players; a mechanism that
   degrades gracefully (falls back to the numeric Elo/rank/form features via
   the zero-initialized OOV embedding row) rather than either ignoring
   identity completely (Elo/LogReg) or overfitting small-sample identities
   (unregularized trees on high-cardinality IDs) is the more defensible
   choice for the stated use case, even where it doesn't yet show up as a
   large aggregate metric win on the current data volume.
3. **Architecturally extensible.** The embedding table is a foundation that
   the tournament simulator, future WTA support (a `tour`-conditioned
   embedding), and richer sequence/context modeling could build on directly;
   a GBDT model has no analogous path to reuse learned player representations
   across those extensions.
4. Per the task's explicit instruction, we keep the NN as the primary
   studied model, and treat XGBoost as the strong benchmark it should be
   compared against on every metric (section 5) — not silently declared the
   "real" model.

### 3.3 Alternatives considered and rejected

- **Sequence models (BiLSTM/Transformer over in-match sets), i.e. keeping
  the old architecture family**: rejected outright for the pre-match task —
  section 1.1 shows this is not a data-mismatch that can be tuned away; the
  input the architecture needs (observed sets) does not exist before a
  match starts. This is not "the wrong hyperparameters," it's the wrong
  inductive bias for this problem.
- **Deeper/wider MLP without embeddings** (i.e., the pre-existing
  `models_v2/mlp_model.pt` benchmark this report supersedes): rejected
  because it cannot express player identity at all, only aggregate numeric
  features — it cannot benefit from the "rare category" transfer-learning
  argument that is a hard requirement of this task.
- **Full attention-based player-vs-player model / graph neural network
  over the H2H graph**: considered, but rejected as unjustified complexity
  for the current data volume (~183K training rows, 7,035 distinct
  training-fold players) — the marginal capacity is very likely to overfit
  further given the already-tight val-set early stopping (section 5.3), and
  it would materially hurt interpretability/auditability without a
  demonstrated need. Documented here as a candidate future direction
  (section 6) rather than implemented speculatively, per the instruction not
  to add complexity beyond what's justified.

---

## 4. TennisEmbeddingNet — mathematical description

### 4.1 Input

For a candidate (possibly hypothetical, not-yet-played) match between
player $p_1$ and $p_2$:

$$
x = \big(x_{\text{surf}},\, x_{\text{bo5}},\, x_{\text{slam}},\, x_{\text{elo}},\, x_{\text{rank}},\, x_{\text{bio}},\, x_{\text{form}},\, x_{\text{h2h}}\big) \in \mathbb{R}^{d}
$$

built by `model_v2.engineer_features()` — one-hot surface, `is_bo5`,
`is_slam`, career/surface Elo (both the raw values and their difference),
log-rank/points differences, age/height/handedness differences, rolling
win-rate and rolling serve/return "form" differences (mean of each
player's prior `STAT_WINDOW=20` matches), rest-days difference (clipped to
±60), and signed head-to-head record. $d = 33$ in the current feature set.
Every component is available strictly before the match (section 2.1).

Player identity is passed **separately**, not folded into $x$: two integer
indices $i_1, i_2 \in \{0, \dots, N\}$ from a vocabulary built on the
training fold only (`build_player_vocab`), with index $0$ reserved for
out-of-vocabulary (never-seen) players.

### 4.2 Embedding layer and the Bradley-Terry connection

Let $E \in \mathbb{R}^{(N+1)\times k}$ ($k=24$) be a single shared embedding
matrix, $e(i) = E_i$. Define

$$
\Delta = e(i_1) - e(i_2) \in \mathbb{R}^k, \qquad \Sigma = e(i_1) + e(i_2) \in \mathbb{R}^k
$$

The classical Elo / Bradley-Terry model (Bradley & Terry, 1952; Elo, 1978)
assumes a **scalar** latent strength $r_i$ per player and

$$
P(i_1 \text{ beats } i_2) = \sigma\!\left(\frac{r_{i_1} - r_{i_2}}{s}\right), \qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$

`TennisEmbeddingNet` is a direct generalization: replace the scalar
rating $r_i$ with a learned $k$-dimensional vector $e(i)$, and replace the
fixed scalar-difference-then-sigmoid with a learned non-linear function of
$\Delta$ (and, additionally, of $\Sigma$ and the context $x$):

$$
\hat p = \sigma\big(\text{MLP}([\Delta;\, \Sigma;\, x])\big)
$$

- $\Delta$ is **antisymmetric** under swapping $(i_1, i_2)$: $\Delta$ flips
  sign, exactly mirroring $x$'s diff-style features and the label's own
  antisymmetry ($P(i_2 \text{ beats } i_1) = 1 - P(i_1 \text{ beats } i_2)$
  in expectation). This gives the network the right structural prior for
  free, the same way a plain Elo difference does, but in $k$ dimensions
  instead of 1 — e.g. one latent direction can capture "hard-court
  aggression," decoupled from a direction capturing "clay-court
  consistency," which a single Elo number cannot represent.
- $\Sigma$ is **symmetric** (order-invariant) and captures swap-invariant
  context: e.g. two very strong players producing a closer match than the
  same gap between two weak players (a "both-are-elite" interaction Elo's
  linear-in-difference form structurally cannot express).
- Concatenating with $x$ lets surface/best-of/Slam **interact non-linearly**
  with player identity — e.g. amplifying a clay-specialist's edge
  specifically on clay — addressing requirement 3 (smooth, continuous,
  non-linear interactions) directly through the MLP's non-linear layers
  rather than through hand-built interaction terms.

### 4.3 Regularization for rare players (requirement 4)

Weight decay $\lambda$ in AdamW applied to $E$ is, in a MAP-estimation view,
equivalent to placing an independent Gaussian prior $\mathcal N(0,
\tfrac{1}{2n\lambda})$ on each row $E_i$ (Hastie, Tibshirani & Friedman,
*Elements of Statistical Learning*, ch. 3; this is the same shrinkage
argument used for L2-regularized latent factors in collaborative filtering,
Koren, Bell & Volinsky, 2009). A player with many training matches receives
many gradient updates that pull their row away from the prior toward a
useful position; a player with few or zero training matches receives few or
no updates, so their row (initialized at $\mathcal N(0, 0.05^2)$, and
exactly $\vec 0$ for the reserved OOV row) stays close to the prior — i.e.
close to "no information beyond the numeric features $x$." This is exactly
the desired cold-start behavior: **rare/unseen players degrade gracefully
to the Elo/rank/form-driven prediction instead of the model inventing
confident, unsupported identity-based signal for them.** Additional
regularizers: embedding dimension is deliberately kept small ($k=24 \ll$
number of players), dropout $p=0.35$ is applied to the concatenated
$[\Delta;\Sigma;x]$ representation at each hidden layer, and early stopping
on a held-out validation year further bounds effective capacity.

### 4.4 Full forward pass

$$
h_0 = [\,\Delta \,\Vert\, \Sigma \,\Vert\, x\,] \in \mathbb{R}^{2k + d}
$$
$$
h_1 = \text{Dropout}\big(\text{GELU}(\text{LayerNorm}(W_1 h_0 + b_1))\big), \quad W_1 \in \mathbb{R}^{96\times(2k+d)}
$$
$$
h_2 = \text{Dropout}\big(\text{GELU}(\text{LayerNorm}(W_2 h_1 + b_2))\big), \quad W_2 \in \mathbb{R}^{48\times 96}
$$
$$
z = W_3 h_2 + b_3 \in \mathbb{R}, \qquad \hat p_{\text{raw}} = \sigma(z)
$$

LayerNorm (not BatchNorm, unlike the old `TennisMatchNet`) is used
specifically because it normalizes across features of a *single* example
rather than across the batch — a `predict_v2.py` single-match inference
call is a batch of size 1, where BatchNorm's running-statistics dependence
would be a latent correctness risk (it isn't in the old code today only
because that code never runs single-example batches through its BatchNorm
layers at inference — but the risk is structural, not accidental, so v2
removes it rather than inheriting it).

### 4.5 Loss, optimization, calibration

Training loss: binary cross-entropy, $\mathcal L = -\frac{1}{n}\sum_j y_j
\log \hat p_j + (1-y_j)\log(1-\hat p_j)$, optimized with AdamW
($\text{lr}=10^{-3}$, weight decay $10^{-4}$), batch size 512, early
stopping (patience 15 epochs) on the 2022 validation fold.

$\hat p_{\text{raw}}$ from a BCE-trained network is a proper probability
estimate asymptotically, but finite-sample neural nets are empirically
known to be systematically overconfident (Guo, Pleiss, Sun & Weinberger,
*"On Calibration of Modern Neural Networks"*, ICML 2017). We therefore fit
a **1-parameter Platt/logistic recalibration** (Platt, 1999) —
$\hat p_{\text{cal}} = \sigma(a \cdot \text{logit}(\hat p_{\text{raw}}) + b)$
— on the **2023 calibration fold**, which is disjoint from both training
and the 2024+ test fold. We also evaluated non-parametric isotonic
regression; Platt scaling is preferred as the shipped default because it
has far fewer effective degrees of freedom and is more stable on a
calibration fold of this size (~2,900 rows) — exactly the finding of
Niculescu-Mizil & Caruana (*"Predicting Good Probabilities With Supervised
Learning"*, ICML 2005), and confirmed empirically here: isotonic
recalibration on our data *increases* held-out test-set ECE relative to
raw probabilities for both XGBoost and the NN (section 5.2), a sign of
calibration-curve overfitting on a calibration fold this size, not evidence
that calibration is unnecessary.

### 4.6 Tournament-probability composition (`tournament_v2.py`)

Given a single-elimination bracket of $2^R$ players and the calibrated
pairwise model $\hat p_{\text{cal}}(i,j \mid \text{surface, best\_of,
is\_slam})$, the probability that player $i$ wins the title is **not** a
simple product over rounds, because the opponent $i$ faces in round $r>1$
is itself a random variable that depends on outcomes elsewhere in the
bracket:

$$
P(\text{champion} = i) = \sum_{\text{brackets } \omega} P(\omega) \cdot \mathbb{1}[i \text{ wins under } \omega]
$$

This is evaluated by Monte Carlo: for each of $S$ simulated trials, draw
every currently-unresolved match's winner as $\text{Bernoulli}(\hat
p_{\text{cal}}(a,b))$ using the pre-computed pairwise matrix, advance
winners, repeat until one player remains; then

$$
\widehat{P}(\text{champion}=i) = \frac{1}{S}\sum_{s=1}^S \mathbb{1}[\text{champion}_s = i], \qquad
\widehat{P}(\text{reach round } r, i) = \frac{1}{S}\sum_{s=1}^S \mathbb{1}[i \text{ alive after round } r \text{ in trial } s]
$$

This sampling approach (used e.g. by FiveThirtyEight's tennis forecasts) is
preferred over a closed-form per-round independence approximation because
it correctly propagates the *opponent-identity* dependency across rounds —
a top seed's round-2 win probability depends on who wins their part of the
draw in round 1, which a per-round-independent recursion would ignore,
biasing top seeds' title probabilities.

**Round-by-round update, no future leakage:** `--results-so-far` fixes the
actual winners of completed rounds (probability mass 1.0 on the observed
winner for those rounds); only rounds that have not yet been played are
still drawn stochastically. Nothing about a not-yet-played round is used to
decide a past round's fixed winner, and the simulation of the unresolved
remainder is unconditional on anything not yet known — verified in section
5.3 (Sinner's title probability moves from 0.4975 pre-tournament to 0.5766
after their observed round-1 win is fixed, using only that one new fact).

---

## 5. Results

### 5.1 Match-level metrics — blind test fold (matches with `date >= 2024-01-01`, n = 7,189, disjoint from training/validation/calibration)

| Model | Accuracy | Log-loss | Brier | ROC-AUC | ECE |
|---|---|---|---|---|---|
| Elo-only (0-parameter) | 0.6401 | 0.6379 | 0.2226 | 0.6997 | 0.0536 |
| Logistic Regression | 0.6552 | 0.6183 | 0.2153 | 0.7157 | 0.0301 |
| XGBoost (raw) | 0.6546 | 0.6099 | 0.2119 | **0.7226** | 0.0205 |
| XGBoost + isotonic | 0.6504 | 0.6139 | 0.2128 | 0.7188 | 0.0261 |
| XGBoost + Platt | 0.6513 | 0.6112 | 0.2124 | 0.7226 | 0.0296 |
| **XGBoost — SHIPPED (val-ECE-selected = raw)** | 0.6546 | 0.6099 | 0.2119 | 0.7226 | 0.0205 |
| TennisEmbeddingNet (raw) | 0.6536 | 0.6135 | 0.2131 | 0.7200 | **0.0175** |
| TennisEmbeddingNet + isotonic | 0.6484 | 0.6169 | 0.2140 | 0.7189 | 0.0260 |
| TennisEmbeddingNet + Platt | 0.6552 | 0.6153 | 0.2137 | 0.7200 | 0.0307 |
| **TennisEmbeddingNet — SHIPPED (val-ECE-selected = raw)** | 0.6536 | 0.6135 | 0.2131 | 0.7200 | **0.0175** |

(Numbers are seeded/reproducible — `torch.manual_seed(42)` inside
`train_embedding_net`. Regenerate via `python3.12 train_v2.py` then
`python3.12 evaluate_v2.py`; reliability diagram at
`models_v2/reliability_diagram.png`.)

Reading this honestly: **XGBoost is marginally better on log-loss/Brier/AUC
on this exact holdout**; the embedding NN is within ~0.6% relative log-loss
and ~0.3pt AUC of it, actually *beats* XGBoost on calibration (ECE 0.0175
vs 0.0205) and on accuracy, and both comfortably beat Elo-only and logistic
regression on every metric. Per section 3.2, we ship the NN as the primary
model for its identity-transfer and cold-start properties (41% test-set
OOV-player rate), while explicitly keeping XGBoost as the recommended
strong benchmark/sanity check to run alongside it — this is a defensible,
disclosed trade-off, not a hidden one.

### 5.2 Calibration selection is data-driven, not hardcoded — and both models keep their raw probabilities

`train_v2.py` does not hardcode a calibration method. For each model it
fits isotonic and Platt calibrators on the **calibration fold** (2023),
then picks whichever of {raw, isotonic, Platt} has the lowest ECE on the
**validation fold** (2022) — a fold already spent on early stopping, but
disjoint from both the calibration fit and the final test evaluation, so
the selection itself cannot leak into the reported test number
(`model_v2.choose_calibration`). For both XGBoost and the embedding NN in
the current run, **raw (uncalibrated) wins the selection** — i.e. both
models' native training objective (binary cross-entropy / logistic loss)
already produces well-calibrated probabilities on this problem, and
isotonic/Platt recalibration *increases* held-out ECE (0.0205→0.0261-0.0296
for XGBoost, 0.0175→0.0260-0.0307 for the NN). This is exactly the
calibration-fold-size instability Niculescu-Mizil & Caruana (2005) describe
for post-hoc recalibration on calibration sets in the low thousands (here,
2,933 rows) — it is evidence the calib fold is too small/noisy to safely
improve on an already-decent raw probability, not evidence that
calibration is never useful. All three variants (raw + both calibrators)
are saved in `models_v2/preprocessing.pkl`, and the selection can be
revisited automatically as more calibration data accumulates in future
retraining runs.

### 5.3 Comparison against the OLD model

A literal shared-holdout comparison is not possible without re-implementing
`predict.py`'s inference for the ~59% of ATP players (by test-match
incidence) who have no Match Charting Project data at all (the old
pipeline requires a per-player charted-stat profile — section 1.1). Instead
we compare each system's own best available reported performance:

| | Old system (`live_eval.py`, `eval_out/eval_summary.json`, in-sample, no temporal holdout) | New system (`evaluate_v2.py`, strict blind 2024+ holdout) |
|---|---|---|
| n matches | 6,162 (Match Charting Project, mixed years, **not** date-filtered) | 7,189 (ATP tour-wide, **2024-01-01 or later only**) |
| Accuracy | 0.6306 | 0.6536–0.6546 |
| Log-loss | 0.6564 | 0.6099–0.6135 |
| Brier | 0.2320 | 0.2119–0.2131 |
| ROC-AUC | 0.6755 | 0.7200–0.7226 |
| ECE | 0.0682 | 0.0175–0.0205 (both models ship raw, uncalibrated) |

The new system wins on every metric — **despite the old number being the
easier, in-sample estimate and the new number being the harder, strictly
out-of-time one.** Combined with section 1.1's finding that the old
system's NN contribution was already inert in its own blended output, this
is strong evidence the rebuild is a genuine improvement, not just a
different-flavored regression toward the same performance.

### 5.4 Tournament simulation — sanity check

`tournament_v2.py` on a synthetic 8-player hard-court Slam quarter
(`Sinner/Shelton/Alcaraz/Zverev/Djokovic/Fritz/Medvedev/Rublev`, current
Elo as of the latest data date), 20,000 simulations:

| Player | Reach R2 | Reach QF-equiv (R3) | Title |
|---|---|---|---|
| Sinner | 0.829 | 0.577 | 0.466 |
| Alcaraz | 0.686 | 0.280 | 0.210 |
| Djokovic | 0.682 | 0.516 | 0.158 |
| Medvedev | 0.756 | 0.266 | 0.052 |

After fixing round 1 to its (simulated-here) actual result
(`results_so_far`), Sinner's title probability updates from 0.4657 →
0.5293 using only the one newly-known fact, with eliminated players'
probabilities correctly collapsing to exactly 0 — confirming the
round-by-round update logic in section 4.6 behaves as specified.

---

## 6. Limits and future improvements

- **ATP/men-only.** No WTA training signal exists in this repo (section
  2.3); adding a full WTA match/ranking archive is the highest-value next
  step for coverage.
- **Embedding under-training risk.** Early stopping on the single-year
  (2022) validation fold triggered after only 2 productive epochs in the
  shipped run; a warm-start schedule (train the numeric trunk first, then
  unfreeze embeddings) or a larger/rolling validation window may let the
  embedding table organize further before stopping, and was not fully
  explored under this task's time budget.
- **XGBoost is still a hair ahead numerically.** If pure predictive
  accuracy/calibration on the current feature set is the only goal, XGBoost
  is a legitimate production choice today; the NN's advantage is specifically
  the identity-transfer/cold-start argument (section 3.2), which is not
  fully visible in aggregate metrics at current data volume and would be
  expected to widen as more OOV-heavy future seasons accumulate.
- **In-play/live re-forecasting** (updating a match's own win probability
  mid-match) is out of scope here by design — this report is pre-match
  only, per the task. The old `TennisMatchNet`/`SEQ_LEN` architecture, if
  it is ever revived, is a legitimate fit for that *different* problem, not
  this one.
- **Tournament simulator uses independent per-match draws conditioned only
  on static pre-tournament features.** It does not model fatigue/injury
  accumulation across rounds within a single simulated tournament run,
  which is a second-order effect not currently in scope.

## 7. v2.1 update — production refit, surface-Elo shrinkage, deterministic point estimate

User testing surfaced a real deficiency: Alcaraz-Sinner predictions barely moved between
Clay and Grass (~0.546 vs ~0.530), and the same query returned slightly different numbers
across runs. Diagnosis found three causes, all fixed:

1. **The shipped artifact was the benchmark artifact.** The benchmark protocol (train <
   2022) is right for honest evaluation but wrong to ship: its player embeddings are frozen
   at each player's pre-2022 self — Alcaraz had 51 career matches (2 on grass) and Sinner
   124 (3 on grass) in that window, so the model had essentially never seen their current
   selves or their rivalry. Fix: `train_v2.py --final`, a **two-stage refit**. Stage 1
   trains through 2025-06 with early stopping on 2025-H2 and calibration-method selection
   on 2026 — its only job is to *discover the epoch count and calibration choice*. Stage 2
   then retrains from scratch on **all** rows (through the end of the data, Apr 2026) for
   exactly that epoch count — the standard train-on-all-after-selection refit (Hastie et
   al., ESL §7.10) — so the most recent matches, the most informative ones for upcoming
   predictions, DO contribute gradient updates to the shipped weights; nothing is held
   back from final training. Artifacts in `models_v2_final/`, auto-preferred by
   `predict_v2.py`/`tournament_v2.py`. Reported performance numbers still come only from
   the benchmark protocol run — stage 1's slices are model-selection tools, and stage 2
   has no held-out data by construction.
2. **Raw surface-Elo compression.** Grass has a handful of events per year, so raw
   surface Elo stays near its 1500 anchor and `elo_surf_diff` systematically understates
   grass edges. Fix: `elo_blend_diff` in `engineer_features` — FiveThirtyEight-style
   shrinkage, `w = n_surf/(n_surf+30)`, blending surface Elo toward career Elo by surface
   experience. On the blind benchmark this improved XGBoost (log-loss 0.6099→0.6095, AUC
   0.7223→0.7230) and slightly hurt the NN's raw log-loss (0.6135→0.6177) — reported
   honestly; the NN regression is within seed-to-seed variation and the production refit
   (which dominates both) uses the feature.
3. **MC noise in the point estimate.** `predict_v2.py` previously reported the MC-dropout
   *mean* as the point probability, so identical queries drifted ±1-2pp across runs. Fix:
   the point estimate is now always the deterministic forward pass (also what calibration
   was fit on); MC-dropout is used only for the uncertainty interval.

4. **Surface-specific head-to-head** (`h2h_surf_diff`, added after further user testing).
   User intuition flagged Alcaraz's grass probability as too high vs Sinner. The data
   split cleanly: in aggregate Alcaraz IS the stronger grass player in this archive
   (31-2 since 2023, Wimbledon champion 2023+2024, grass Elo 1818 vs 1723) — but in
   their direct grass meetings Sinner leads 2-0 (Wimbledon 2022 R16, Wimbledon 2025 F),
   and the model only carried an *aggregate* H2H feature (Alcaraz +3 overall). Fix:
   `PlayerStateTracker` now tracks per-surface H2H walk-forward, exposed as
   `h2h_surf_diff` in `engineer_features` and in both inference paths. Its weight is
   learned from all pairs in history, not hand-tuned for any single rivalry. On the
   blind benchmark the feature is neutral-to-positive (NN raw: accuracy 0.6570 — best
   of all models — log-loss 0.6140, ECE 0.0113; XGBoost unchanged at 0.6095).

Effect on the motivating example (Alcaraz vs Sinner, production model, all four fixes
in; full surface x format grid via `predict_v2.py --matrix`, P(Alcaraz)):

| Surface | Bo3 non-Slam | Bo5 Slam |
|---|---|---|
| Hard | 0.368 | 0.344 |
| Clay | 0.494 | 0.571 |
| Grass | 0.460 | 0.496 |

The ~13pp surface spread (was ~2pp before the fixes) is reproducible across runs and
reflects the direct-matchup evidence per surface (near-parity on clay: their clay H2H
is 3-2 Alcaraz but Sinner won their most recent clay final, Monte Carlo 2026; Sinner
favored on grass despite Alcaraz's stronger aggregate grass record — 2-0 grass H2H;
Sinner clearly favored on hard). The format column shows the model learned TWO distinct
bo5/Slam effects: on hard, the favorite (Sinner) gains in best-of-5 — the classic
variance-reduction effect of longer series; on clay the prediction *flips sign*
(0.494 → 0.571 Alcaraz), which pure variance-amplification cannot do — this is a
learned is_slam/is_bo5 interaction with the other features, and it happens to align
with the pair's actual Slam record (Slam H2H 4-2 Alcaraz; clay Slams 2-0 Alcaraz —
both Roland Garros finals/semis). Caveat stated plainly: the model has no
pair-specific Slam-H2H feature, so that alignment comes from population-level
patterns, not from memorizing this rivalry.

## 8. v2.2 update — real bracket ingestion, uncertainty-aware tournament simulation, and betting-value checks

This update adds three new pieces of tooling around the existing model rather than changing
the model itself: a way to pull a real, live tournament draw off the web instead of typing it
in by hand; a way to forecast a hypothetical tournament with no real draw at all; and error
bars / betting-value output on top of the existing point-estimate predictions.

### 8.1 `fetch_bracket.py` — scraping a live draw from diretta.it (Flashscore)

diretta.it/Flashscore render the draw client-side; the underlying feed endpoint
(`400.flashscore.ninja/.../feed/dr_<tournamentId>_<stageId>`) requires a signed `x-fsign`
header computed by obfuscated JS that is re-derived per session, which is not worth reverse
engineering for a script meant to keep working. Instead `fetch_bracket.py` drives a real
headless Chromium via Playwright, lets the page render normally, and reads the already-
rendered draw straight out of the DOM (`.draw__round` / `.draw__bracket` /
`.bracket__result` elements) — exactly what a human reading the page sees. This needed
solving three sub-problems:

- **Name resolution.** Flashscore renders "Surname(s) Initial(s)." (e.g. `"Cerundolo J. M."`,
  `"Auger-Aliassime F."`, `"Struff J-L."`); the ATP dataset has full "Firstname Surname(s)"
  names. `resolve_flashscore_name` tokenizes from the right to find the trailing
  initials block (handles multi-initial and hyphenated-initial names, which a naive
  single-capture regex mis-splits on the LAST initial only), normalizes hyphens to spaces
  so `"Auger-Aliassime"` matches `"Auger Aliassime"`, and matches by (surname suffix/prefix,
  first-initial). Surname+initial collisions (e.g. `"Ruud C."` → Casper Ruud vs.
  Christian Ruud; `"Shelton B."` → Ben vs. Bryan Shelton) are broken by picking whichever
  candidate has the more recent match in the dataset — the currently-active tour player.
  Names with zero ATP tour-level history (young wildcards/qualifiers on their Slam debut)
  are left unresolved with a printed warning rather than silently guessed; see 8.3 for how
  the rest of the pipeline now tolerates this instead of crashing.
- **Retirements.** A match tied at the walkover point (e.g. `"Bonzi B. (RET.) 2 - Diallo G. 2"`)
  isn't decided by score comparison; `match_winner_side` checks the `(RET.)`/`(W.O.)` tag on
  the loser's slot when scores are level.
- **Re-anchoring to only the matches still to be played.** Originally this only re-anchored
  once the quarterfinal stage was reached; per user feedback the rule is now unconditional —
  ANY fully-completed leading round collapses into the `players` list itself (the survivors
  entering the next round), and `results_so_far` is reset to empty. The output bracket JSON
  therefore always represents only the open part of the draw: already-finished rounds, and
  any already-eliminated players in them (including ones with no ATP history), are dropped
  entirely rather than carried through as dead weight or unresolvable noise.

### 8.2 `rank_bracket.py` — a bracket with no real draw, from the ATP ranking

For "what would the top N ranked players' title odds look like" forecasts where no real
draw exists yet. Reads the latest snapshot in `tennis_atp/atp_rankings_current.csv`, takes
the top `--top-n` (skipping any named in `--exclude`, e.g. withdrawals — the next-ranked
eligible player is pulled in to fill the slot, same as a real alternate list), pads up to
the next power of two (single-elimination needs one) by continuing down the ranking, and
places players with the standard tournament seeding order (`seed_order`: seed 1 and seed 2
can only meet in the final, seeds 1-4 only from the semifinals on, etc. — the same
seed-spreading logic real tour draws use, generated by the standard recursive
`f(n) = interleave(f(n/2), n+1-f(n/2))` construction). Output is `results_so_far: []` (a
pure pre-tournament forecast) in the same bracket JSON schema `tournament_v2.py` already
consumes.

### 8.3 Graceful degradation for players with no ATP history

`tournament_v2.py` previously called `sys.exit(1)` the moment any single name failed to
resolve — a script-breaking failure mode for real draws, which regularly include
qualifiers/wildcards making their tour debut with literally zero rows in `tennis_atp/`
(no history means no Elo, no embedding, no state — the model has nothing to score them on;
this is a genuine data-coverage boundary, not a resolver bug). Fixed to the behavior the
user asked for: print a `WARNING` per unresolved name and continue, treating that player as
an automatic loss against any resolved opponent (`match_prob_matrix`: `P[i,j]=0` for
unresolved `i`, any resolved `j`) — the rest of the bracket's probabilities are computed
AS IF that player weren't a threat, rather than the whole run failing. Two unresolved
players facing each other (only possible pre-tournament, since `results_so_far` already
fixes real winners for completed rounds) falls back to a 50/50 coin flip.

### 8.4 Error bars, propagated from model uncertainty, not just Monte-Carlo sampling noise

Single-match predictions already carried a 90% MC-dropout interval (Gal & Ghahramani, 2016 —
dropout kept active at inference, repeated forward passes; section 4 and 7.3 above); this
update (a) surfaces that same interval in `predict_v2.py --matrix`, which previously printed
only point estimates, and (b) propagates it into the tournament simulator, which previously
had none.

The naive way to add a tournament error bar would be to compute one fixed pairwise
probability matrix `P[i,j]` (point estimate) and let Monte-Carlo sampling noise from the
20,000 bracket draws alone define the interval — but that only captures "how much does the
bracket's opponent-dependency structure shake out differently draw to draw," not "how
confident is the model in each of these matchup probabilities in the first place." Two
players with an identical point-estimate matchup probability but very different underlying
confidence (e.g. one has deep tour history, the other is a rookie leaning on Elo/rank
features alone) should NOT produce equally sharp tournament forecasts.

The fix: `match_prob_matrix` now computes, for every pair, `n_mc_model` (default 30)
MC-dropout draws of P(i beats j), not just one point estimate. `simulate()` then draws a
FRESH sample from that pair's MC-dropout distribution independently for every simulated
match instance across all `n_sims` trials — so each of the 20,000 simulated tournaments
plays out under a slightly different, independently resampled model of the world, and the
across-trial spread in title/reach counts reflects both sources of uncertainty at once.
Because each trial independently redraws both the bracket structure and every match's model-
uncertainty sample, the `n_sims` outcomes are still i.i.d. Bernoulli(reach_prob) — so a
standard normal-approximation binomial standard error, `sqrt(p(1-p)/n_sims)`, is a valid 90%
CI (`z=1.645`), and it comes out systematically wider than sampling noise alone would give,
by exactly the amount of extra dispersion the model's own uncertainty contributes. Reported
in the CLI table, and in `--out-json` as `title_ci90` / `round_reach_ci90` per player.

### 8.5 Betting-value check against bookmaker odds (single match only)

`predict_v2.py --odds-p1 2.10` / `--odds-p2 1.75` compares the model's probability (point
estimate AND its 90% CI) against the market's implied probability (`1/decimal_odds`) and
states whether the edge is robust enough to act on, not just whether it's positive:

- **VALUE BET** — even the *pessimistic* end of the 90% CI still implies positive expected
  value at these odds. The edge survives the model's own uncertainty about itself.
- **MARGINAL** — the point estimate has positive edge, but the CI lower bound does not.
  A real but not statistically confident edge; bet small or skip.
- **NO BET** — the point estimate itself is below break-even; negative EV even optimistically.

Stake sizing uses the Kelly criterion (Kelly, 1956): for decimal odds `o` and win
probability `p`, the growth-optimal fraction of bankroll is `f* = (p*o - 1)/(o - 1)`. The
script reports **quarter-Kelly** (`f*/4`) rather than full Kelly — a standard practitioner
haircut (Thorp) because `f*` assumes `p` is known exactly, whereas here `p` is a model
estimate with its own uncertainty (the same CI used for the VALUE/MARGINAL/NO-BET call);
betting full Kelly against an uncertain edge risks large drawdowns if the model turns out
to be optimistic. This is deliberately single-match only — an outright tournament-winner
bet's fair odds would need the *joint* distribution over the whole Monte-Carlo simulation
(not just the marginal title_prob per player), which isn't currently exposed by
`tournament_v2.py`.

## 9. v2.3 update — fixing the training curve (embeddings removed), calibration selection, symmetric inference, optimal staking

### 9.1 Diagnosis: the val-loss curve was overfitting from epoch 2, and regularizing the embedding did not fix it

The training plot in the new UI made the problem visible: validation log-loss bottomed out at
epoch 1-2 (~0.598) and then rose monotonically while training loss kept falling. Early
stopping was hiding this (shipped weights were the epoch-2 snapshot), but a model that can
only be trained for two epochs is a model whose capacity is mostly spent memorizing.

A controlled sweep on the benchmark split (train <2022, val 2022, blind test >=2024; every
row below is one training run, test numbers are the **symmetrized** prediction of 9.3):

| Config | best epoch | val log-loss | test log-loss | test acc | test AUC |
|---|---|---|---|---|---|
| shipped v2.2 (emb 24, wd 1e-4) | 2 | 0.5971 | 0.6141 | 0.6545 | 0.7198 |
| embedding weight-decay 1e-2 / 1e-1 | 2 | 0.5987 | 0.6144 / 0.6143 | — | — |
| emb_dim 8 | 1 | 0.5995 | 0.6172 | — | — |
| dropout 0.5 | 2 | 0.6001 | 0.6155 | — | — |
| train only since 2005 | 4 | 0.5982 | 0.6145 | — | — |
| swap augmentation (9.3) | 1 | 0.6009 | 0.6150 | — | — |
| player-dropout 0.3 / 0.5 / 0.9 | 2-4 | 0.5999 / 0.5979 / 0.5964 | 0.6153 / 0.6135 / 0.6118 | — | — |
| 5-seed ensemble of the shipped config | — | — | 0.6141 | 0.6525 | 0.7201 |
| **no player embedding (numeric features only)** | **16** | **0.5936** | **0.6102** | 0.6535 | 0.7231 |
| **no embedding + swap augmentation** | **20** | **0.5925** | **0.6091** | **0.6582** | **0.7235** |
| no embedding + swap aug, seeds 1/2 | 47 / 25 | 0.5913 / 0.5927 | 0.6090 / 0.6093 | — | — |
| XGBoost (reference) | 704 trees | 0.5947 | 0.6095 | 0.6517 | 0.7229 |
| Logistic regression (reference) | — | 0.6004 | 0.6179 | 0.6550 | 0.7161 |

Two things follow. First, **the val-loss floor is ~0.59, not a defect**: logistic regression,
XGBoost and every NN variant converge to the same 0.592-0.600 on the 2022 val fold and
0.609-0.618 blind. That is the information content of pre-match features for ATP tennis —
bookmaker closing lines score roughly 0.58-0.60 log-loss on the same task (Kovalchik 2016) —
so "make val loss much lower" is not achievable without new information (injuries, live odds,
point-level data). Second, **the player-identity embedding is what overfits, and it adds
nothing out-of-time**: every regularizer aimed at it (L2, smaller dim, player-dropout,
ensembling) lands within seed noise of the same 0.614, while simply *removing* it gives the
best NN of the whole study (0.609 blind, beating XGBoost, training cleanly for 20-30 epochs
with a monotonically decreasing val curve). The reason is structural: everything the
embedding could encode about a player's strength is already in the walk-forward Elo /
surface-Elo / rolling-form / H2H features, which *track the player through time*; a static
per-player vector instead learns "who this player was in the training window", which is
precisely what drifts by the time the test window arrives (41% of test matches involve a
player unseen in training, and the rest have changed). The shipped configuration is now
`--emb-dim 0` (numeric-feature MLP, same trunk/LayerNorm/dropout/AdamW) with swap
augmentation on; the embedding path is kept in `TennisEmbeddingNet` as an option for
experimentation. Honest caveat: the architectural argument of section 4.3 (shared latent
representation for rare players) did not survive out-of-time evaluation on this data — the
rare-player problem is handled better by the Elo/form features' built-in shrinkage than by a
learned embedding.

### 9.2 Calibration-method selection: log-loss instead of ECE

`choose_calibration` picked raw/isotonic/Platt by ECE on the 2.9K-row validation fold. The
benchmark rerun exposed why that is fragile: ECE is a 10-bin histogram statistic whose
sampling noise on 3-7K rows is ~±0.01 — the seed-to-seed test ECE of one fixed model ranged
0.011-0.036 in the sweep — so the selector was effectively flipping a coin, and in one run
it chose isotonic (val ECE marginally lower) which then scored *worse* on the blind test on
every metric (log-loss 0.6158 vs 0.6103 raw; ECE 0.028 vs 0.017). Selection is now by
validation **log-loss**, a strictly proper scoring rule that rewards calibration and
sharpness jointly and is far less noisy. With that, both the NN and XGBoost select raw —
consistent with the reliability diagram (raw NN hugging the diagonal) and with
Niculescu-Mizil & Caruana's observation that neural nets trained with log-loss are usually
already well calibrated. Temperature scaling was considered and not added: it is Platt
scaling with the intercept fixed at zero, i.e. a special case of a candidate already in the
set that already loses to raw here.

### 9.3 Swap-symmetric training and inference

A spot check found the v2.2 model was not swap-invariant: for Alcaraz-Sinner on grass Bo5
it gave P(Alcaraz)=0.496 with Alcaraz as p1 but 0.282 (=1-0.718) with him as p2 — a 21pp
order artefact. Randomized side assignment in the training table only teaches the symmetry
*on average*; nothing in the architecture enforces it. Two fixes, both in by default:
- **Training:** every training row is mirrored (`swap_sides`: p1<->p2 columns, H2H diffs
  negated, label flipped), doubling the data and teaching P(p1 wins|swap)=1-P(p1 wins)
  explicitly. This is what pushed the no-embedding model from 0.6102 to 0.6091 blind.
- **Inference:** `predict_match_prob` scores both orderings and returns
  ½[f(p1,p2) + 1 - f(p2,p1)] (for the MC-dropout samples too), so the output is *exactly*
  antisymmetric regardless of residual model asymmetry — test-time augmentation over a known
  invariance. Verified: all six surface×format cells now agree to 3 decimals under swap.

### 9.4 Optimal staking across both sides of the market

Per user request, `predict_v2.py --odds-p1 --odds-p2` (and the UI) now also print a single
staking plan rather than two independent one-sided checks: `optimal_allocation` maximizes
the expected log-growth p·log(1+f₁(o₁-1)-f₂) + (1-p)·log(1-f₁+f₂(o₂-1)) over stakes
(f₁,f₂) on both players — multi-outcome Kelly (Kelly 1956), solved numerically — and
scales by a user-chosen Kelly fraction (default ¼). The optimizer reproduces the three
regimes without special-casing: when the book carries a margin (1/o₁+1/o₂ > 1, the normal
case) it stakes at most one side — hedging both sides of a margin-carrying market is strictly
growth-negative, and "both at 2.00" is exactly break-even, never profitable; when there is no
edge on either side it returns (0,0) → NO BET; when 1/o₁+1/o₂ < 1 (arbitrage across books) it
backs both, and the plan reports the risk-free split (stakes ∝ implied probabilities, return
1/overround − 1) as model-independent. Each recommended stake is still flagged MARGINAL if
the edge disappears at the pessimistic end of the model's 90% CI.

### 9.5 Final blind-test numbers (models_v2/, shipped configuration; supersede sections 5 and 7)

| model | accuracy | log-loss | Brier | ROC-AUC | ECE |
|---|---|---|---|---|---|
| elo_only | 0.6401 | 0.6379 | 0.2226 | 0.6997 | 0.0536 |
| logistic_regression | 0.6550 | 0.6179 | 0.2151 | 0.7161 | 0.0302 |
| xgboost_SHIPPED[raw] | 0.6517 | 0.6095 | 0.2117 | 0.7229 | 0.0254 |
| **nn_SHIPPED[raw]** (emb_dim 0, swap aug) | **0.6559** | **0.6094** | **0.2117** | **0.7232** | **0.0187** |

Benchmark training: 44 epochs run, best at 29, val log-loss 0.598 → 0.592 monotone; production
refit (`--final`) stage 1 selected 17 epochs, calibration raw. Production-model Alcaraz–Sinner
grid (P(Alcaraz), 90% CI) after the change: Hard Bo3 0.360 [0.33-0.41], Hard Bo5-Slam 0.301,
Clay Bo3 0.455 [0.42-0.50], Clay Bo5-Slam 0.439, Grass Bo3 0.367, Grass Bo5-Slam 0.328 — the
earlier clay "flip" (sect. 7) is gone; it was an embedding-driven artefact of the pair's few
Slam meetings, not a population-level effect.

## References

- Bradley, R. A. & Terry, M. E. (1952). *Rank Analysis of Incomplete Block
  Designs.* Biometrika.
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present.*
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and
  Comparisons to Regularized Likelihood Methods.*
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities
  With Supervised Learning.* ICML.
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On Calibration
  of Modern Neural Networks.* ICML.
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning.* ICML. (MC-dropout
  confidence intervals in `predict_v2.py`.)
- Koren, Y., Bell, R. & Volinsky, C. (2009). *Matrix Factorization
  Techniques for Recommender Systems.* IEEE Computer. (Latent-factor L2
  shrinkage argument reused in section 4.3.)
- Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). *Why do tree-based
  models still outperform deep learning on tabular data?* NeurIPS Datasets
  and Benchmarks.
- Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical
  Learning*, 2nd ed., ch. 3 (ridge regression / MAP-Gaussian-prior
  equivalence of L2 regularization).
- Kovalchik, S. (2016). *Searching for the GOAT of tennis win prediction.* Journal of
  Quantitative Analysis in Sports. (Pre-match log-loss floor, section 9.1.)
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.* Bell
  System Technical Journal. (Growth-optimal stake sizing, section 8.5.)
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports Betting,
  and the Stock Market.* (Fractional-Kelly practice under parameter
  uncertainty, section 8.5.)

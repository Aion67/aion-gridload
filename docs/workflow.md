# Grid Load Prediction — Team Workflow & Organization Plan

This document lays out *how* the team will work, not just *what* needs to get done.
Every choice below has a reason attached, so you can reuse the pattern on future projects.

---

## 1. Core Principle: Baseline First, Then Branch Into Tracked Experiments

The brief asks for a "best model" reached through iterative, findings-driven exploration
(imputation, feature engineering, augmentation, cross-validation). That's fundamentally
different from a linear "step 1 → step 10" checklist — it's a loop:

```
understand data → establish a floor (baseline) → branch into parallel experiments
→ each experiment produces a finding → findings steer the next round of experiments
→ converge on a best model → explain it → report it
```

**Why this matters for organization:** a linear folder of "step1, step2, step3..." notebooks
breaks down the moment you're running five parallel feature-engineering ideas at once.
So the structure below is built around *experiments as first-class, trackable units*,
not sequential steps.

---

## 2. Repository Structure

```
grid-load-prediction/
├── README.md
├── data/
│   ├── raw/                     # untouched original data (never edit these files)
│   └── processed/               # cleaned / feature-engineered versions, versioned by name
├── notebooks/
│   ├── 00_data_understanding.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── experiments/
│       ├── imputation_01_forward_fill.ipynb
│       ├── imputation_02_knn_impute.ipynb
│       ├── feature_engineering_01_lag_features.ipynb
│       ├── feature_engineering_02_rolling_stats.ipynb
│       ├── feature_engineering_03_calendar_cyclical.ipynb
│       ├── augmentation_01_window_slicing.ipynb
│       ├── model_01_random_forest_basic.ipynb
│       ├── model_02_random_forest_cv.ipynb
│       ├── model_03_xgboost_cv.ipynb
│       ├── model_04_lstm.ipynb
│       └── explainability_01_shap_bestmodel.ipynb
├── src/                          # reusable functions pulled out of notebooks (see §5)
├── reports/
│   ├── figures/                  # exported charts used in the final report/slides
│   └── final_report.md
├── experiment_log.md             # the single most important file — see §4
└── best_model/                   # final chosen model, its notebook, and its metrics
```

**Why `raw/` vs `processed/` is separated:** nobody should ever have to wonder whether a
CSV has already been cleaned. Raw stays immutable; every transformation produces a new,
clearly named file in `processed/`.

**Why `experiments/` is its own subfolder, not mixed with the main sequence:** the four
notebooks in `notebooks/` root (00–02) are the *linear spine* of the project — things
everyone needs and that don't change once done. Everything exploratory and possibly
dead-ending lives in `experiments/`, so the spine stays clean and presentable on its own.

---

## 3. Notebook Naming Convention

**Pattern:** `<category>_<sequence>_<short-description>.ipynb`

Examples: `feature_engineering_01_lag_features.ipynb`, `imputation_02_knn_impute.ipynb`,
`model_03_xgboost_cv.ipynb`

**Reasoning for this exact pattern over alternatives:**

- **Category prefix (not date or author):** sorting alphabetically groups everything
  related to one question together — all feature-engineering attempts sit next to each
  other regardless of who wrote them or when. A date-based name (`2024-06-01-fe.ipynb`)
  scatters related work across the file list by whoever ran it last.
- **Zero-padded sequence number (`01`, `02`, not `1`, `2`):** keeps sort order correct
  once you pass 9 experiments — `feature_engineering_10` would otherwise sort before
  `feature_engineering_2`.
- **Short description in the filename, not just in the notebook:** you should be able to
  tell what a notebook tried *without opening it*. `feature_engineering_02_rolling_stats`
  tells you the finding category before you've read a line of code.
- **Fixed category vocabulary** — agree on a short list up front so people don't invent
  synonyms (`feat_eng` vs `feature_engineering` vs `features`). Suggested categories:
  `imputation`, `feature_engineering`, `augmentation`, `model`, `tuning`, `explainability`.

---

## 4. The Experiment Log (the piece most teams skip, and shouldn't)

Create `experiment_log.md` as a running table, one row per experiment notebook:

| ID | Notebook | Category | Hypothesis | What changed | Metric before → after | Decision | Owner |
|----|----------|----------|------------|---------------|------------------------|----------|-------|
| 01 | feature_engineering_01_lag_features | feature_engineering | Past load values predict current load | Added lag-1, lag-24, lag-168 features | RMSE 412 → 355 | Kept | — |
| 02 | imputation_02_knn_impute | imputation | KNN imputation preserves local patterns better than mean-fill | Replaced mean imputation with KNN (k=5) | RMSE 355 → 351 | Kept | — |

**Why this file matters more than any single notebook:** this is what turns "we tried a
bunch of stuff" into "we systematically explored the space and can justify our final
model." It's also *exactly* what section 6 and 8 of the final report ("model training and
validation," "conclusions and recommendations") are supposed to summarize — if you keep
the log updated as you go, the report writes itself instead of requiring someone to
reverse-engineer six weeks of notebooks at the end.

Update it the moment an experiment notebook is finished — not at the end of the project.

---

## 5. Shared Code Goes in `src/`, Not Copy-Pasted Between Notebooks

Once a cleaning function, feature-engineering transform, or evaluation helper is used in
two or more notebooks, move it into `src/` (e.g. `src/features.py`, `src/evaluate.py`) and
import it. This avoids the classic failure mode where someone fixes a bug in one notebook's
copy of a function and the other four notebooks silently keep using the buggy version.

---

## 6. Team Roles — Loosely Assigned, Not Siloed

Given the iterative/branching nature of this project, rigid single-person ownership of a
pipeline stage (like a waterfall) doesn't fit well — findings from one area (e.g. feature
engineering) directly change what's worth trying in another (e.g. which model to test
next). Instead:

- Each person has a **primary area** matching the five areas in the brief (data
  understanding, cleaning/preprocessing, visualization/EDA, modeling/evaluation, report/
  presentation) — this is who's accountable for that area's notebooks and log entries.
- But **anyone can open an experiment notebook in any category** — the naming convention
  and experiment log exist precisely so that's safe to do without stepping on others' work.
- **Rotate the "log keeper" and "spine notebook" owner weekly** so the shared/root
  notebooks (00–02, final report) don't become one person's personal file that others are
  scared to touch.

**Why not strict silos:** if only one person owns "modeling," they become a bottleneck
every time an EDA finding suggests a new feature to try, and everyone else sits idle
waiting on them.

---

## 7. Model Approaches to Cover (Basic Pass → Cross-Validated Pass)

Run the shortlist below as **basic** (single train/test split) first — this is fast and
tells you roughly who's in the running. Only take the top 2–3 performers into a proper
**cross-validated** pass with time-series-aware splitting (expanding-window or
`TimeSeriesSplit`, *not* random k-fold, since shuffling breaks temporal order in load
data and leaks future information into training).

| Family | Basic candidates | Notes |
|---|---|---|
| Naive baselines | Persistence (last value), moving average, seasonal naive | Always run these first — if a "real" model can't beat them, something's wrong |
| Linear | Linear Regression, Ridge, Lasso | Fast, interpretable, good sanity check |
| Tree-based | Random Forest, Gradient Boosting (XGBoost/LightGBM) | Usually strong on tabular load data with lag/calendar features |
| Classical time series | SARIMA, Holt-Winters exponential smoothing | Good if load has strong, stable seasonality |
| Neural | MLP, LSTM/GRU | Worth trying if you have enough data and want to capture longer temporal dependencies |

**Why basic-then-CV instead of cross-validating everything immediately:** CV with 5–10
folds across 8+ model types is expensive and slow, and most of those combinations won't
matter. Screen cheaply, then spend the expensive, rigorous evaluation only on the models
that already showed promise.

---

## 8. Techniques to Explore (mapped to notebook categories)

- **Imputation:** forward/backward fill, linear interpolation, KNN imputation, model-based
  imputation — compare against each other's downstream effect on model RMSE, not just
  "how complete the data looks."
- **Feature engineering:** lag features (t-1, t-24, t-168 for hourly data), rolling
  mean/std windows, calendar features (hour, day-of-week, holiday flag), cyclical encoding
  (sin/cos of hour and day-of-year so midnight and 11pm are numerically close).
- **Augmentation:** for time series this usually means window slicing (multiple
  overlapping training windows), small jitter/noise injection, or block bootstrapping —
  more relevant if the dataset is short, less critical if you already have a long history.
- **Cross-validation:** time-series split, not random k-fold (see §7).
- **Explainability:** SHAP values or permutation importance per model, partial dependence
  plots for the top 2–3 features — put this in its own `explainability_*` notebook run
  against the final chosen model, and pull the key charts into the report.

---

## 9. Visualization Plan

Keep all report-bound charts in `reports/figures/`, generated by the notebook that
produces them (not hand-exported ad hoc), so every chart is reproducible. Suggested set:
raw load time series with missing/outlier points highlighted, correlation heatmap,
feature importance plot for the final model, actual-vs-predicted plot over a test window,
and an error-by-hour-of-day or error-by-season breakdown (useful for the "insights and
recommendations" section, since it shows *when* the model struggles, not just its
aggregate error).

---

## 10. How This Maps to the Deliverables You Were Given

- **Clean notebook/script** → `notebooks/00`–`02` plus `best_model/`, kept tidy because
  exploratory mess stays quarantined in `experiments/`.
- **Visualizations** → `reports/figures/`, built once and reused across report and slides.
- **Final report** → largely assembled from `experiment_log.md` plus the figures — write
  it last, incrementally, rather than all at once at the deadline.
- **GitHub repository** → this exact folder structure, pushed as you go (see note below).
- **Presentation material** → pull directly from `reports/figures/` and the experiment
  log's "what changed / why it helped" story, which is naturally presentation-shaped.

**One Git note:** notebooks merge badly in Git (diffs are unreadable JSON). Since the
naming convention already guarantees each experiment lives in its own file, conflicts are
rare — each person can commit/push their own experiment notebooks directly to a shared
branch without needing feature-branch-per-person overhead. Reserve actual branches for
changes to the shared `src/` code or the root spine notebooks.

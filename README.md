# Santander Product Recommendation — what will a bank customer add next month?

For each of 929,615 bank customers, rank the products they do not hold yet by how likely they are to add each one in
June 2016. The data is the Kaggle competition
[Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation): monthly
snapshots from 2015-01 to 2016-05, with 24 product flags and a customer profile for each snapshot. The metric is MAP@7.

**The full story is in [`story.ipynb`](story.ipynb).** It covers the data problems, target and
validation design, baselines, the experiment log, the model comparison, the error analysis, explainability and the
decision to remove sensitive attributes. GitHub shows its saved charts and tables without running anything.

## Results

![MAP@7 by model: learned models around 0.89-0.90, popularity and association baselines around 0.65-0.67](docs/model_comparison.png)

| | MAP@7 |
|---|---|
| Best popularity rule (same month last year) | 0.652 |
| Item-association rule | 0.673 |
| Logistic regression, final features | 0.889 |
| **LightGBM multiclass, final model (175 features)** | **0.898** |

These are means over 4 time-based validation folds (2016-02 .. 2016-05), counting customers who add at least one
product. Seven model families land within ~0.006 of each other, about the month-to-month noise.

**Kaggle (late submission):** 0.02740 private / 0.02713 public; the winning private score was 0.03140. Kaggle
averages over *all* test customers, and non-adders score 0, so these numbers are on a different scale from the table
above. Every setting was chosen on this project's own validation folds, never from leaderboard feedback.

## Main decisions

- **Time-based validation only:** train on all months before the validation month, retrain for each fold. No random
  K-fold, because the same customer appears in many months.
- **Data quirks fixed, not hidden:** profile columns are frozen for 2015-01..07, so seniority is recomputed from the
  contract date. The 2015-07 jump in customers is explained before any model sees the data.
- **Features use information up to the end of month t−1 only.** The same code builds train and test.
- **Changes are adopted only if they beat seed noise in paired comparisons.** Most experiments did not. The one that
  did was the raw profile lags (set B): +0.0018, better in 8/8 fold × seed pairs.
- **Sex, foreigner and residence flags removed:** they cost nothing to remove (−0.0003, within noise). A
  counterfactual test showed that flipping sex alone changed the #1 product for 8.4% of customers.

## Repository layout

```
story.ipynb             the write-up (charts and tables are saved in the file)
santander/              shared code: column names, loading + MAP@7 evaluation, the final feature set,
                        and the analysis functions the notebook calls
pipeline/               data cleaning -> target -> features -> final model and submission
experiments/            scripts behind the saved results in analysis_output/ (model comparison,
                        experiment log, hyperparameter search, explainability)
analysis_output/        saved results read by the notebook
models/                 the final model, and version 1 (with sensitive columns) used by the XAI checks
```

**What you need, and what is optional:**
- **Core code:** `santander/` and `pipeline/`. They are enough to build the data, train the final model, write the
  submission and run the notebook.
- **`experiments/` is optional.** It is the record of how the saved results in `analysis_output/` were produced:
  the experiment log (§8), the model comparison (§9, incl. the current settings from the hyperparameter
  search) and the XAI comparison (§11). You do not need to
  run it to read or run the notebook. Each script trains models, and takes minutes to hours.

## Getting the data

1. Sign in to Kaggle, open the
   [competition data page](https://www.kaggle.com/competitions/santander-product-recommendation/data) and accept the
   competition rules. The rules do not allow the data to be redistributed, so it is not in this repository.
2. Download it, either from that page or with the Kaggle CLI (`pip install kaggle`, with an API token in
   `~/.kaggle/kaggle.json`):
   ```bash
   kaggle competitions download -c santander-product-recommendation
   unzip santander-product-recommendation.zip
   unzip train_ver2.csv.zip && unzip test_ver2.csv.zip      # keep sample_submission.csv.zip zipped
   ```
3. The project folder (this repository's root) should then contain `train_ver2.csv` (2.3 GB, 13.6M rows),
   `test_ver2.csv` and `sample_submission.csv.zip`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # on macOS first: brew install libomp
```

## Running the pipeline

Run each step from the repository root as a module (`python -m pipeline.make_features`), in this order.
Steps 1–4 build `processed/`, which the notebook needs.

| step | script | output |
|---|---|---|
| 1 | `pipeline/make_clean_parquet.py` | `processed/train_clean.parquet`, `processed/test_clean.parquet` (English column names via `santander/column_map.py`) |
| 2 | `pipeline/make_target_table.py` | `processed/target.parquet` (which products each customer added each month) |
| 3 | `pipeline/make_features.py` | `processed/features_{train,test}.parquet` (166 features, adder rows only for train) |
| 4 | `pipeline/make_profile_change.py` | `processed/profile_change_{train,test}.parquet` (profile lags, set B) |
| 5 | `santander/final_features.py` | not run directly: the final 175-feature list and LightGBM parameters |
| 6 | `pipeline/final_train_predict.py` | `models/final_lgbm.txt`, `submission/submission.csv` |
| 7 | `pipeline/check_submission.py` | independent checks of the submission against the raw data |

Shared code lives in `santander/`: `train_common.py` (loading, MAP@7 evaluation), `final_features.py` and
`story_analysis.py` (the analysis functions the notebook calls).

## Running the notebook

After steps 1–4, open `story.ipynb` and run all cells (under a minute). The notebook does two things:

- **Computes live from `processed/`:** the data, target and baseline tables and the SHAP / counterfactual checks.
  The SHAP and counterfactual checks use the saved models in `models/`.
- **Reads saved results from `analysis_output/` (committed):** results of model training (comparison, experiment
  log, XAI comparison). Nothing is retrained in the notebook.

**Optional: where the saved results come from.** These are the scripts in `experiments/`. Run them as modules, e.g.
`python -m experiments.run_change`.

| results | script |
|---|---|
| `analysis_output/models/feat_experiments.csv` | `run_feat.py` |
| `analysis_output/models/lag_experiments.csv` | `python -m pipeline.make_features --nlag 12`, then `run_lagsets.py` |
| `analysis_output/models/change_experiments.csv` | `run_change.py` |
| `analysis_output/models/sensitive_ablation.csv`, `month_ablation.csv` | `ablate_sensitive.py`, `ablate_month.py` |
| `analysis_output/models/extra_models.csv` | `compare_extra.py logreg` |
| `analysis_output/tuning/` | `tune.py` (driver: `experiments/run_tuning.command`), summary: `tune_summary.py` |
| `analysis_output/models/lgbm_val*_scores.npy` (error analysis) | `run_lgbm.py --val-month 13..16` |
| `analysis_output/xai/` | `xai_shap.py`, `xai_part2.py`, `train_compare_models.py`, `xai_compare.py` |
| `analysis_output/model_comparison.csv` | assembled by hand from the per-fold result files of `run_lgbm.py`, `run_extra.py`, `run_rank.py`, `run_mlp.py`, `run_nn.py`, `run_gru.py`; no single script |

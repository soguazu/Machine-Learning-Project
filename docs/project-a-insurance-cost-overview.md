# Project A: Insurance Charges Prediction — Project Overview

## 1. Problem statement

Predict an individual's annual **medical/insurance charges** (a continuous US
dollar amount) from a small set of personal and lifestyle attributes. This is a
**supervised regression** problem.

Accurately estimating expected charges helps insurers with risk-based pricing
and helps individuals understand which factors drive their costs.

## 2. Objective

Build and evaluate a model that maps personal attributes →
`charges`, then identify which factors most influence cost.

**Target variable:** `charges` (continuous, USD).

## 3. Dataset

- **Source file:** [`data/raw/insurance.csv`](../data/raw/insurance.csv)
- **Rows:** 1,338 individuals (1 exact duplicate row present — see
  [Data quality](#6-data-quality-notes)).
- **Columns:** 7 (6 features + 1 target)
- **Missing values:** none.

See the [data dictionary](#4-data-dictionary) below for column details.

## 4. Data dictionary

| Column     | Type        | Description                                  | Values / range                                            |
| ---------- | ----------- | -------------------------------------------- | --------------------------------------------------------- |
| `age`      | int         | Age of the primary beneficiary               | 18 – 64 (mean ≈ 39)                                        |
| `sex`      | categorical | Biological sex of the policyholder           | `male` (676), `female` (662)                              |
| `bmi`      | float       | Body mass index (kg/m²)                       | 15.96 – 53.13 (mean ≈ 30.7)                               |
| `children` | int         | Number of dependents covered                 | 0 – 5 (most common: 0)                                    |
| `smoker`   | categorical | Whether the person smokes                    | `no` (1,064), `yes` (274)                                 |
| `region`   | categorical | US residential region                        | `southeast` (364), `southwest` (325), `northwest` (325), `northeast` (324) |
| `charges`  | float       | **Target** — annual medical costs billed (USD) | 1,121.87 – 63,770.43 (mean ≈ 13,270)                     |

## 5. Exploratory findings (baseline)

- **`charges` is right-skewed** (mean 13,270 vs. median 9,382, max 63,770). A
  log transform of the target often stabilizes variance and improves linear
  models.
- **Smoking is the dominant driver.** Smokers form a clearly separated,
  much-higher-cost cluster — expect it to be the strongest single predictor.
- **Linear correlations with `charges`:** `age` ≈ 0.30, `bmi` ≈ 0.20,
  `children` ≈ 0.07. These are modest on their own because the biggest effect
  (smoking) is categorical and interacts with `bmi` and `age`.
- **Interactions matter** — e.g. high BMI *combined with* smoking drives costs
  far higher than either alone. Tree-based models capture this automatically;
  for linear models, add interaction terms.

> Numbers above are computed from the raw file and are meant as orientation, not
> final analysis. Reproduce them in a notebook under `notebooks/`.

## 6. Data quality notes

- **1 duplicate row** — decide whether to drop it (likely yes).
- **No missing values**, so no imputation is required.
- **`bmi` up to 53** and **`charges` up to ~63.8k** — real but extreme values;
  check whether to keep, cap, or model with a robust/log approach.
- Categorical features need encoding (one-hot for `sex`, `smoker`, `region`).

## 7. Modeling approach

1. **Split** the data (e.g. 80/20 train/test, fixed `random_state` for
   reproducibility).
2. **Preprocess** with a `scikit-learn` `Pipeline` + `ColumnTransformer`:
   - one-hot encode `sex`, `smoker`, `region`;
   - optionally scale numeric features (`age`, `bmi`, `children`);
   - optionally log-transform the target.
3. **Baseline model:** Linear Regression — interpretable reference point.
4. **Stronger models:** Ridge/Lasso (regularized linear), and tree ensembles
   (RandomForest, GradientBoosting) which capture interactions natively.
5. **Tune** hyperparameters with cross-validation (`GridSearchCV` /
   `cross_val_score`).
6. **Persist** the chosen model to [`models/`](../models/).

## 8. Evaluation

Regression metrics on the held-out test set:

| Metric   | Meaning                                                    |
| -------- | --------------------------------------------------------- |
| **RMSE** | Root mean squared error — penalizes large misses (USD).   |
| **MAE**  | Mean absolute error — typical error magnitude (USD).      |
| **R²**   | Share of variance in `charges` explained by the model.    |

Use **k-fold cross-validation** during development for stable estimates, and
report final numbers on the untouched test set. Also inspect a
**residuals plot** and **predicted vs. actual** scatter to find systematic bias
(e.g. underpricing high-cost smokers).

### Baseline vs. dummy benchmark

First result (80/20 split, `random_state=42`). The **dummy** predicts the
training mean for every row; the **baseline** is plain Linear Regression on
one-hot-encoded features.

| Metric | Dummy (mean) | Linear baseline | Improvement |
| ------ | -----------: | --------------: | ----------- |
| MAE    | $9,593       | $4,181          | −56%        |
| RMSE   | $12,466      | $5,796          | −53%        |
| R²     | −0.0009      | 0.784           | ~0 → 0.78   |

**Finding:** the linear baseline more than halves typical error and explains
~78% of the variance in `charges`, while the dummy explains essentially none
(R² ≈ 0). The features carry real signal and the split shows no leakage. Treat
the **linear baseline** — not the dummy — as the bar for stronger models
(Ridge/Lasso, tree ensembles, log-target, interaction terms).

## 9. Reproducible workflow

1. Raw data lives in `data/raw/insurance.csv` (read-only).
2. Cleaning/feature steps live in `src/` and write to `data/processed/`.
3. Exploration and model development happen in `notebooks/`.
4. Tests for `src/` logic live in `tests/`.
5. Final model artifacts go to `models/`; figures to `reports/figures/`.

```python
import pandas as pd
df = pd.read_csv("data/raw/insurance.csv")
```

## 10. Possible extensions

- Engineer interaction features (`smoker × bmi`, `smoker × age`).
- Add an `obese` flag (`bmi >= 30`) and test interactions with `smoker`.
- Compare model explainability with SHAP / permutation importance.
- Quantify prediction uncertainty (quantile regression or prediction intervals).

## Roadmap context

- **Project A — Insurance charges (regression):** this document.
- **Project B — Breast cancer (classification):**
  [`project-b-breast-cancer-overview.md`](project-b-breast-cancer-overview.md).
- **Project C — Mall Customers (unsupervised):** K-means clustering, PCA intuition.
- **Then:** portfolio-style consolidation of all three projects.

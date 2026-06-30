# Project B: Breast Cancer Classification — Project Overview

## 1. Problem statement

Predict whether a breast tumor is **benign** or **malignant** from features
computed on digitized images of a fine-needle aspirate (FNA) of a breast mass.
This is a **supervised binary classification** problem.

Reliable benign/malignant prediction supports diagnostic decision-making — the
clinically important goal is to **not miss a malignant case** (minimize false
negatives), even at the cost of some false alarms.

This is the second project in the roadmap. Project A (insurance charges) covered
the full **regression** workflow; Project B covers the full **classification**
workflow. Project C (Mall Customers) will cover **unsupervised** learning.

## 2. Objective

Build and evaluate a classifier that maps the 30 cell-nucleus measurements →
`diagnosis` (benign vs. malignant), and identify which metrics matter for a
medical screening context.

**Target variable:** `diagnosis` — binary (`malignant` = positive class,
`benign` = negative class).

## 3. Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset, available via
  `sklearn.datasets.load_breast_cancer` (no download or CSV needed).
- **Rows:** 569 tumors.
- **Features:** 30 numeric features (all continuous, no categoricals).
- **Class balance:** benign **357** (62.7%), malignant **212** (37.3%) —
  mildly imbalanced; not severe enough to need resampling, but accuracy alone
  can be misleading, so report precision/recall/F1 too.
- **Missing values:** none.

### Class label convention

`load_breast_cancer` encodes `target = 0` → **malignant**, `target = 1` →
**benign**. For medical screening, treat **malignant as the positive class** so
"recall" means "fraction of cancers caught." Be explicit about this in the
notebook (e.g. `pos_label`) — it is the single most common source of confusion
in this dataset.

## 4. Data dictionary

The 30 features are **10 base measurements**, each reported as three statistics:
the **mean**, the **standard error (`se`)**, and the **worst** (mean of the
three largest values). The 10 base measurements per cell nucleus:

| Base measurement   | Meaning                                              |
| ------------------ | ---------------------------------------------------- |
| `radius`           | Mean distance from center to perimeter points        |
| `texture`          | Std. dev. of gray-scale values                       |
| `perimeter`        | Nucleus perimeter                                    |
| `area`             | Nucleus area                                         |
| `smoothness`       | Local variation in radius lengths                    |
| `compactness`      | perimeter² / area − 1.0                              |
| `concavity`        | Severity of concave portions of the contour          |
| `concave points`   | Number of concave portions of the contour            |
| `symmetry`         | Symmetry of the nucleus                              |
| `fractal dimension`| "Coastline approximation" − 1.0                     |

So the full feature set is e.g. `mean radius`, `radius error`, `worst radius`,
… (10 × 3 = 30). `radius`, `perimeter`, and `area` are strongly correlated by
construction — worth noting for the multicollinearity discussion.

## 5. Expected findings (orientation)

- **Strong, well-separated signal.** This dataset is known to be highly
  separable — a Logistic Regression baseline typically reaches ~95–98% accuracy.
  The challenge is not "can we classify" but "how do we trade precision vs.
  recall responsibly."
- **`worst` features carry the most signal.** `worst radius`, `worst perimeter`,
  `worst area`, and `worst concave points` are usually the top discriminators.
- **Feature scaling matters** for Logistic Regression (features span very
  different ranges, e.g. `area` in the hundreds vs. `smoothness` ~0.1). Scale
  inside the pipeline.
- **Multicollinearity** among size features (`radius`/`perimeter`/`area`) inflates
  coefficient variance — fine for prediction, but interpret coefficients with care.

> These are orientation notes, not final results. Reproduce them in a notebook
> under `notebooks/`.

## 6. Data quality notes

- **No missing values**, no imputation needed.
- **All features numeric** — no encoding required (contrast with Project A).
- **Different scales across features** → standardize (`StandardScaler`) for
  linear/distance-based models.
- **Mild class imbalance (63/37)** → don't rely on accuracy alone; use the
  confusion matrix and F1. Consider `class_weight="balanced"` and inspect its
  effect on recall.

## 7. Modeling approach

1. **Frame the problem** — confirm X (30 features), y (`diagnosis`), and which
   class is positive (malignant).
2. **Split** the data (stratified 80/20, fixed `random_state`). Use
   `stratify=y` so both splits keep the 63/37 balance.
3. **Naive benchmark** — `DummyClassifier(strategy="most_frequent")` (always
   predicts benign). This is the no-skill floor: ~62.7% accuracy, but **0%
   recall on malignant** — the headline reason accuracy alone is the wrong metric.
4. **Baseline model** — `LogisticRegression` in a `Pipeline` with
   `StandardScaler`. Interpretable reference point.
5. **Stronger model (if needed)** — a tree-based classifier
   (`RandomForestClassifier` / `GradientBoosting` / `HistGradientBoosting`),
   which captures nonlinear interactions and needs no scaling.
6. **Overfitting check** — compare train vs. test metrics; large gaps signal
   overfitting (especially for the tree model).
7. **Final summary** — best model, key metrics, and which features drove it.

This mirrors the Project A progression: **naive benchmark → interpretable
baseline → improved nonlinear model → overfitting check → final summary.**

## 8. Evaluation

Because the classes are imbalanced and the cost of a missed cancer is high,
**accuracy is necessary but not sufficient.** Report all of:

| Metric              | Meaning / why it matters here                                    |
| ------------------- | --------------------------------------------------------------- |
| **Confusion matrix**| The ground truth — shows false negatives (missed cancers) directly. |
| **Accuracy**        | Overall correct rate. Misleading under imbalance — report but don't rely on it. |
| **Precision**       | Of predicted malignant, how many truly were. High precision = few false alarms. |
| **Recall (sensitivity)** | Of true malignant, how many were caught. **The priority metric** — minimize missed cancers. |
| **F1-score**        | Harmonic mean of precision and recall — single balanced summary. |
| **ROC-AUC** (optional) | Ranking quality across all thresholds; useful for threshold tuning. |

**Decision lens:** prefer the model/threshold that maximizes **recall on
malignant** while keeping precision acceptable. Discuss the
precision/recall trade-off explicitly and consider moving the decision
threshold below 0.5 to catch more cancers.

## 9. Reproducible workflow

1. Load data from `sklearn.datasets.load_breast_cancer` (no raw CSV needed); if
   you prefer a file, save a snapshot to `data/raw/breast_cancer.csv`.
2. Cleaning/feature steps (if any) live in `src/` and write to `data/processed/`.
3. Exploration and model development happen in `notebooks/`
   (e.g. `notebooks/breast-cancer-classification.ipynb`).
4. Tests for `src/` logic live in `tests/`.
5. Final model artifacts go to `models/`; figures to `reports/figures/`.

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer(as_frame=True)
df = data.frame              # 30 features + 'target'
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
```

## 10. Possible extensions

- Tune the decision threshold to hit a target recall (e.g. ≥ 99% of malignant
  caught) and report the precision cost.
- Compare `class_weight="balanced"` vs. default.
- Use `PCA` to visualize the two classes in 2D (also a bridge to Project C).
- Feature importance via coefficients (logistic) vs. permutation importance (trees).
- Calibrate predicted probabilities (`CalibratedClassifierCV`) if probabilities
  will be used for decisions.

## Roadmap context

- **Project A — Insurance charges (regression):** done. naive benchmark →
  linear baseline → nonlinear model → overfitting check → summary.
- **Project B — Breast cancer (classification):** this document.
- **Project C — Mall Customers (unsupervised):** K-means clustering, PCA
  intuition.
- **Then:** portfolio-style consolidation of all three projects.

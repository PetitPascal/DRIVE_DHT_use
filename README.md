# DRIVE study: predicting digital health technology use in rheumatology with XGBoost and SHAP

R code for the article **"Leveraging machine learning to identify predictors of digital health technology use among rheumatic patients using German survey data"**

Archived version: https://doi.org/10.5281/zenodo.24538300 · License: MIT

## What this repository does

Secondary analysis of a cross-sectional survey of 337 patients from three German rheumatology outpatient clinics. The pipeline:

1. recodes the survey data (one-hot encoding of categorical predictors),
2. learns all pre-processing on training data only (missRanger imputation, near-zero-variance removal, VIF screening <= 2.5), also inside every cross-validation fold,
3. tunes XGBoost by random search (stratified 5-fold CV), selects predictors by mean |SHAP| in a nested 10x5-fold CV, and sets the classification threshold (Youden) on out-of-fold training predictions,
4. evaluates on a held-out 30% test set, with 95% confidence intervals for AUROC, confusion-matrix metrics and calibration,
5. explains the model with SHAP values (overall and by observed user / non-user),
6. repeats everything for ten seeds and for two sensitivity analyses (without domain D; without domains B and D),
7. combines the seeds: pooled out-of-sample performance, across-seed SHAP with two-stage bootstrap CIs, ROC curves, and a final model fitted on all participants.

## Data availability

The anonymized survey data are **not included** in this repository. They are available from the corresponding author upon reasonable request. The code expects two files in `data_dir`:

| File | Content |
|---|---|
| `DRIVE_Data_engl.csv` | survey data (`;`-separated, decimal comma), outcome column `DHS_use` |
| `nomenclaturef.csv` | codebook with columns `Parameter` (variable name), `Def` (readable label), `categ` (domain A–D) |

[The codebook is included in `data/` / not included.]

## Repository structure

```
DRIVE_DHT_use/
├── R/DRIVE_ML_analysis.R     # complete analysis script
├── data/                     # place the input files here (not tracked)
├── CITATION.cff
├── LICENSE                   # MIT
└── README.md
```

## Requirements

- R 4.6.0
- Packages (versions used): xgboost 3.2.1.1, missRanger 2.6.1, caret 7.0-1, pROC 1.19.1, splitTools 1.0.1, tidyverse 2.0.0. Missing packages are installed automatically by the script.

## How to run

1. Clone the repository and put the two input files in `data/`.
2. Set `data_dir` at the top of the script if needed.
3. Run `R/DRIVE_ML_analysis.R` (e.g., `source("R/DRIVE_ML_analysis.R")`). Results are written to `Drive - ML analysis/`.

## Outputs

| Location | Content |
|---|---|
| `<analysis>/seed_<seed>/` | model (`.ubj`), hyperparameters, performance with 95% CIs, pre-processing logs, SHAP tables and figures, `sessionInfo` |
| `Combined_across_seeds/` | AUROC by seed; hyperparameters (by seed, summary, final); performance (per seed, median [range], pooled out-of-sample); SHAP (per seed, global across seeds); ROC curves; combined SHAP and SHAP comparison figures; final model |

`<analysis>` is `main_analysis`, `sensitivity_1` or `sensitivity_2`.

## Reproducibility

- All random steps (split, folds, imputation, search, XGBoost, bootstrap) are controlled by the seed (ten seeds: 123, 234, 345, 456, 567, 678, 789, 890, 901, 1012).
- Results may differ slightly across package versions, operating systems and thread counts.

## Citation

Please cite the article and the archived code: Pascal Petit (2026). *DRIVE: code for Leveraging machine learning to identify predictors of digital health technology use among rheumatic patients using German survey data* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24538300

## License and contact

MIT License (see `LICENSE`). Questions: [name, e-mail].

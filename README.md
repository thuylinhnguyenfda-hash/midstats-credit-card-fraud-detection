# Banking Transaction Fraud Detection — Group Review

A complete notebook with train-only data exploration, shared preprocessing, SGD logistic/LightGBM building, temporal tree comparisons and optional final test evaluation.

**Status:** prepared for the group's review; **not pushed to GitHub**. The dataset source and license must still be filled by the team.

## Main notebook

`Banking_Fraud_Project.ipynb`

The layout follows the provided Stock_META_Thien sample:
1. Explore and prepare the dataset
2. Build the models
3. Compare the models
4. Conclusions

[Open the current review notebook in Colab]([(https://colab.research.google.com/drive/1V-CkMfJm2Il9-eJhni-n1nrAPncW5XWe)])


## Reproduce locally

1. Use Python 3.11 or later and open a terminal in this project directory.
2. Create a virtual environment and install `requirements.txt`.
3. Put the original `data.csv` in `data/raw/` or set `FRAUD_DATA_CSV` to its location.
4. Open `Banking_Fraud_Project.ipynb` in JupyterLab/VS Code. On a local runtime, `USE_DRIVE` defaults to False.
5. Run from top to bottom. Keep `RUN_FINAL_TEST = False` while reviewing model decisions.

```bash
python -m venv .venv
# Activate the environment with the command for your operating system.
python -m pip install -r requirements.txt
jupyter lab
```

The raw CSV is about 3.35 GB. It is not included in the repository/package. See `data/raw/README.md` for the expected file and source placeholders. The notebook reads the CSV in chunks, keeps only training rows for EDA and uses Arrow-backed data/categorical columns to reduce memory.

## Run in Colab

Colab automatically sets `USE_DRIVE = True`. The default review root is:

```text
/content/drive/MyDrive/Banking_Fraud_Midterm_Member1/GitHub_Review
```

The folder must contain `data/data_dictionary.csv`; its source data can be the existing CSV in the parent project's `Handoff_Outputs/raw_input_data/`. Grant each collaborator access to the project folder and add a shortcut in My Drive, or change `root` and `csv_path` to their actual locations.

Drive is **optional storage**, not a required part of the statistical method. Local runs use a project-relative path. On Colab the CSV is copied once into temporary runtime storage before repeated scans; this copy disappears when the runtime is reset.

## Evaluation protocol

- The first pass reads only transaction date/time to establish chronological membership. Timestamp parsing is required to define the split, not exploratory analysis of target distributions.
- The first ~80% of rows form development, with the last 20% reserved for test. The last 20% of development form validation: approximately **64/16/20 overall**.
- All transactions at the same timestamp stay together. Original row position breaks ordering ties reproducibly.
- Before EDA, the split table shows **only counts and date ranges** for all three partitions.
- Full-training statistics review all 44 original columns. Charts needing many records use `train.sample(n=min(200_000, len(train)), random_state=42)`.
- The static dictionary contains only manually documented meanings/groups/units. Missingness, distinct counts, means, medians, time checks, class/category rates and tests are recalculated on train; there are **no precomputed all-dataset EDA CSV dependencies**.
- No validation/test category-label table, monthly holdout audit or test class prevalence is used in EDA.
- Median/scaler/state vocabulary are fitted on train. Frozen deterministic transforms are applied to held-out inputs and written to prepared files; this is not held-out EDA or parameter fitting.
- Models train on train, early stopping and thresholds use validation, and final test scoring is disabled during group review.

## Models retained

- SGD logistic baseline: the group's original weighted log-loss configuration.
- LightGBM: the group's approved newer Drive configuration (31 leaves, minimum 300 rows per leaf, negative bagging 0.5, L2=1, 800 maximum rounds and 50-round AP early stopping). Its training code is retained; validation evaluation and RAM cleanup are appended.
- Dummy, SGD CV baseline, Decision Tree and Random Forest candidates: original configurations and 3 timestamp-aware folds over the latest 300,000 training rows.
- The selected tree is refitted on full train. Model comparison reads current results rather than copied constants.
- As in the team's submitted scope, the final test section evaluates only logistic and LightGBM; it does not invent test results for tree/forest candidates.

See `docs/CHANGELOG_VI.md` and `docs/model_code_preservation.json` for the exact changes to the latest Drive snapshot.

## Quick review check

After dependencies are installed, run `python tools/check_project.py`. This checks notebook structure, train-only EDA dependencies and the disabled test gate without reading raw data or training models. It complements, rather than replaces, a full run.

## Final test

Leave `RUN_FINAL_TEST = False` while reviewing. Once model settings and the threshold rule are frozen, set it to True and run the final evaluation section. Do not use test scores to continue tuning.

**Historical limitation:** Earlier versions of this project inspected test labels and used those findings in feature decisions. This revision no longer performs that audit, but the old holdout has already been reviewed. It cannot be described as historically untouched. A separate future holdout is preferable for independent confirmation.

## Other limitations

- Source/currency/timezone and history construction remain unconfirmed.
- Chi-square tests treat repeated transactions as independent; p-values are exploratory.
- Internal temporal CV retains preprocessing fitted on final train before folds, so it is not strict fold-isolated preprocessing. This is distinct from test-label EDA and is explicitly retained to avoid silently rewriting Member 3's experiment.
- Weighted/bagged model scores are not calibrated fraud probabilities. Validation F1 uses a threshold optimized on that validation set.
- Different library versions or hardware can give small score differences. `requirements-tested.txt` records the verification environment.

## Outputs and version control

- `data/processed/`: prepared tables, row indices, preprocessing settings and feature schema. Generated locally/Drive, ignored by Git.
- `results/`: small aggregate CSVs, run information and exported figures for review.
- Raw data, prepared Parquet files, environments and model binaries are excluded by `.gitignore`.
- No raw transaction rows, names, card/account/SSN values or credentials are included in the review ZIP.

## Team information to fill

| Member | Name / ID | Contribution |
|---|---|---|
| 1 | TO BE FILLED | Data preparation, train-only sanity checks and EDA |
| 2 | TO BE FILLED | SGD logistic and LightGBM |
| 3 | TO BE FILLED | Temporal CV and tree comparisons |

Before pushing, fill the dataset source/license and member information, review the notebook and confirm the intended repository visibility.

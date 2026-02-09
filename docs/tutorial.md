# MLOps Examples Tutorial: Reproducible Training + Registry

This tutorial walks through a repeatable workflow:
1. Create a new branch
2. Create a new DVC data version
3. Train and register a model
4. Commit the change
5. Repeat to show reproducibility and change tracking

It assumes you are using `mlops-services` locally (MLflow + RustFS).

---

## Prerequisites
- Python 3.11+
- `uv` installed
- Git
- `mlops-services` running locally

If you need to set credentials from `mlops-services`:
```bash
set -a
source ../mlops-services/env/config.env
source ../mlops-services/env/secrets.env
set +a
export AWS_ACCESS_KEY_ID="$RUSTFS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$RUSTFS_SECRET_KEY"
```

If your repo layout differs, set the path for Make:
```bash
make pull MLOPS_SERVICES_DIR=/path/to/mlops-services
```

---

## One-Time Setup (Per Machine)
```bash
uv venv
uv sync
```

If you already did this, you can skip it. Use `uv run ...` for all commands.

---

## Step 0: Start Services
In another terminal:
```bash
cd ../mlops-services
make up
```

Verify:
- MLflow UI: `http://localhost:5000`
- RustFS S3: `http://localhost:9000`

---

## Step 1: Create a New Branch
```bash
git checkout -b demo/repro-1
```

---

## Step 2: Create a New Data Version (DVC)
This generates the Breast Cancer dataset and tracks it with DVC.

```bash
make data
make push
```

---

## Step 3: Train + Register a Model
```bash
make train
```

Verify in MLflow UI:
- Experiment: `mlops-examples/dev`
- A new run exists with metrics and artifacts
- A model is registered: `MLOpsExamples_BreastCancer_RF`

Artifacts in `eval/`:
- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `feature_importance.png`

---

## Step 4: Try a Few Hyperparameter Variations
To see performance shifts, change a few Random Forest hyperparameters and re-run training.

Edit `configs/dev.yaml` and try combinations like:
- `n_estimators`: 50, 200, 500
- `max_depth`: 3, 6, 12
- `min_samples_leaf`: 1, 2, 5
- `max_features`: "sqrt" or "log2"

Run training each time:
```bash
make train
```

In MLflow, compare the runs:
- `val_accuracy`, `val_f1_macro`, `val_precision`, `val_recall`
- `val_roc_auc`, `val_pr_auc`
- ROC/PR curves and feature importance

Repeat a couple of times to get a feel for how model capacity affects performance.

---

## Step 5: Create + Train on a New Data Version
This appends one synthetic row based on per‑class mean/stddev. The random seed is derived from the current dataset hash, so the new row is deterministic for the current dataset.
`data/breast_cancer.appended` is a local guard file used to prevent multiple appends per commit; it is intentionally not tracked.

```bash
make data-append
make push
make train
```

Then commit to lock the data version used in the run:
```bash
git status --short
git add data/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Add breast_cancer dataset version"
```

---

## Step 6: Repeat the Data-Version Process
Run the same steps to generate a new data version and compare results.

```bash
make data-append
make push
make train
git add data/breast_cancer.csv.dvc .dvc/.gitignore
git commit -m "Update dataset version"
```

In MLflow, compare the two runs:
- `val_accuracy`, `val_f1_macro`, `val_precision`, `val_recall`
- `val_roc_auc`, `val_pr_auc`
- Compare confusion/ROC/PR/feature importance plots

---

## Step 7: Reproduce a Registry Model Version
To reproduce a registered model:

1) In MLflow Registry, open the model version and click the run.

2) Note the run tags:
   - `git_sha`
   - `data_sha256`
   
3) Reproduce locally:
```bash
git checkout <git_sha>
make pull
make train
```

You should get matching metrics and the same artifacts for that run.

**Note:** `data_sha256` is a verification tag (not a command input). The `git_sha` points to the commit whose `.dvc` file references the exact data version. You can use `data_sha256` to cross‑check that the pulled dataset matches the run.

---

## Step 8: MLflow Features to Explore
Use the MLflow UI to practice core workflows:

1) Compare runs
   - Select two runs from the experiment list.
   - Click “Compare” to view params/metrics side‑by‑side.
   - Observe how metrics shift across data versions.

2) Filter and search
   - Use the filter bar to find runs with `val_roc_auc > 0.97`.
   - Filter by tag `pipeline = mlops-examples`.

3) Inspect artifacts
   - Open `eval/` and compare `roc_curve.png` or `feature_importance.png`.

4) Model Registry
   - Open the registered model `MLOpsExamples_BreastCancer_RF`.
   - Inspect versions and their source runs.
   - If stages are enabled, promote a version to `Staging`.

---

## Step 9: Cleanup (Optional)
```bash
git checkout main
git branch -D demo/repro-1
```

# MLOps Examples: DVC + MLflow + Model Registry (On-Prem)

This repo is a teaching + demo project for our MLOps stack:

- DVC: data versioning (Git tracks metadata; data lives in object storage)
- MLflow: experiment tracking + artifacts + Model Registry
- RustFS (S3-compatible): artifact storage for DVC (and MLflow server artifacts)
- Postgres: MLflow backend store (runs/registry metadata)

## Prereqs
- Python 3.11+
- Git
- DVC remote and MLflow tracking server reachable from your machine/network

## Local endpoints (mlops-services default)
- MLflow: http://localhost:5000
- RustFS S3: http://localhost:9000
- RustFS Console: http://localhost:9001

---

## One-time setup (per developer machine)

### 1) Install dependencies
This repo uses `uv` for reproducible environments. Python 3.12 is recommended.

```bash
uv venv
uv sync
```

If you prefer, use the Makefile:

```bash
make setup
```

### 2) DVC remote (already configured)

This repo includes a committed DVC remote pointing at RustFS:

- bucket: `dvc-remote`
- prefix: `mlops-examples`
- endpoint: `http://localhost:9000`

If you need to change these, update `.dvc/config`.

Credentials are NOT committed. Set these in your shell (or use your secrets manager):

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

---

## Create / update the dataset (maintainers)

This project uses a tiny deterministic Iris CSV to demonstrate DVC.

```bash
python scripts/make_data.py --out data/iris.csv
dvc add data/iris.csv
git add data/iris.csv.dvc .gitignore
git commit -m "Track iris.csv with DVC"
dvc push
```

Makefile equivalent:

```bash
make data
```

After this, others can run `dvc pull` to fetch the dataset from RustFS.

---

## Run locally (developer workflow)

### 1) Get the dataset

```bash
dvc pull
```

Makefile equivalent:

```bash
make pull
```

### 2) Train + log to MLflow + register a model

```bash
python src/train.py --config configs/dev.yaml
```

Makefile equivalent:

```bash
make train
```

Open MLflow UI and verify:

- Experiment exists (`mlops-examples/dev`)
- A run has params + metrics + eval artifacts
- A model is registered (`MLOpsExamples_Iris_LogReg`)
- Stage promotion (if enabled) is set (e.g. `Staging` -> latest version)

---

## GitLab CI

CI runs:

- `dvc pull`
- `python src/train.py --config configs/ci.yaml`

GitLab CI/CD variables required (masked/protected):

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- optionally `AWS_DEFAULT_REGION`
- optional `MLFLOW_TRACKING_URI` (overrides config)
- optional `MLFLOW_EXPERIMENT_NAME` (overrides config)
- optional `MLFLOW_REGISTERED_MODEL_NAME` (overrides config)
- optional `MLFLOW_MODEL_STAGE` (e.g., `Staging`)

---

## Teaching checklist

A) First successful run
- [ ] Clone repo, install deps
- [ ] Set AWS creds
- [ ] `dvc pull`
- [ ] Run training (`configs/dev.yaml`)
- [ ] Find your run in MLflow UI and inspect:
  - params (C, max_iter, seed)
  - metrics (val_accuracy, val_f1_macro)
  - artifacts (metrics.json, confusion_matrix.png)

B) Prove reproducibility
- [ ] Note the run's `git_sha` and `data_sha256` tags in MLflow
- [ ] Check out that exact git commit
- [ ] `dvc pull`
- [ ] Re-run training and compare metrics (should match or be extremely close)

C) Make a controlled change
- [ ] Change `C` in `configs/dev.yaml` (e.g., 1.0 -> 0.2)
- [ ] Re-run training
- [ ] Compare runs in MLflow (metrics shift, params differ)

D) Data versioning exercise
- [ ] Regenerate data (or add a tiny perturbation in `make_data.py` like shuffling rows)
- [ ] `dvc add data/iris.csv`, commit, `dvc push`
- [ ] Re-run training and observe:
  - new `data_sha256` tag
  - potential metric differences
- [ ] Check out the previous commit, `dvc pull`, rerun and confirm you can reproduce the old run.

E) Registry exercise
- [ ] Identify the latest registered model version
- [ ] If stages are enabled in dev config or env:
  - confirm the model version moved to the configured stage (e.g., `Staging`)
- [ ] Discuss: what would be our policy for promoting `Staging -> Production`?

---

## Notes / conventions

- MLflow tags:
  - `git_sha`: source code version
  - `data_sha256`: dataset content hash
- Buckets:
  - `dvc-remote` for DVC (created by `mlops-services` RustFS init)
  - `mlflow-artifacts` for MLflow server

Note: `uv.lock` is committed to pin exact versions for reproducibility.

If you run into issues, check:

- network access to MLflow/RustFS
- AWS creds for DVC
- that the DVC remote endpointurl is correct

# Reproducibility Checklist

To reproduce a model run, you need:

- **Code**: the exact Git commit (`git_sha`)
- **Data**: the exact raw data version plus the rebuilt processed dataset (`raw_data_sha256`, `processed_data_sha256`, plus the DVC remote)
- **Config**: the training config used for the run
- **Environment**: Python + dependency versions (`uv.lock`)
- **MLflow run**: the run ID to inspect artifacts and metrics
- **Feature snapshot**: the DVC-tracked Parquet snapshot used for training (`feature_snapshot_id`, `feature_snapshot_dvc_hash`)

## Quick Repro Steps
1) In MLflow, open the run and note `git_sha`, `raw_data_sha256`, `processed_data_sha256`, and `feature_snapshot_id`.
2) Checkout that code:
```bash
git checkout <git_sha>
```
3) Pull the exact data:
```bash
make pull
```
4) Rebuild the processed dataset:
```bash
make transform
```
5) Rebuild the feature snapshot:
```bash
make snapshot-docker
```
6) Recreate the train/val/test splits:
```bash
make split-docker
```
7) Re-run training, evaluation, and logging:
```bash
make train
make eval
make log
```
8) Compare metrics/artifacts in MLflow and confirm the rebuilt run matches the original data and snapshot hashes.

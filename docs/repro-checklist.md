# Reproducibility Checklist

To reproduce a model run, you need:

- **Code**: the exact Git commit (`git_sha`)
- **Data**: the exact data version (`data_sha256` + DVC remote)
- **Config**: the training config used for the run
- **Environment**: Python + dependency versions (`uv.lock`)
- **MLflow run**: the run ID to inspect artifacts and metrics

## Quick Repro Steps
1) In MLflow, open the run and note `git_sha` + `data_sha256`.
2) Checkout that code:
```bash
git checkout <git_sha>
```
3) Pull the exact data:
```bash
make pull
```
4) Re-run training:
```bash
make train
```
5) Compare metrics/artifacts in MLflow.

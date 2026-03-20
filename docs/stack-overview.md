# Stack Overview: DVC + MLflow + RustFS

This repo demonstrates a simple, reproducible MLOps stack:

- **DVC**: tracks dataset versions in Git via `.dvc` files; data lives in object storage.
- **MLflow**: tracks experiments, logs metrics/artifacts, and registers models.
- **RustFS (S3-compatible)**: object store for DVC and MLflow artifacts.
- **Feast**: Feature store for training data.
- **Postgres**: MLflow backend store for runs/registry metadata and Feast registry metadata.

### How the pieces connect
- DVC stores content hashes and remote locations for the raw dataset and feature snapshot in `data/*.dvc`.
- MLflow logs metrics and artifacts for each run, and records the code/data/snapshot lineage tags.
- RustFS stores the actual data and artifacts, referenced by both DVC and MLflow.
- Postgres stores MLflow metadata and the Feast registry. Feast reads local Parquet feature snapshots during training.

### Why this matters
This setup ties **code + data + model** to a specific run. You can always:
1) Checkout the exact Git SHA  
2) Pull the exact data version  
3) Re-run training to reproduce results

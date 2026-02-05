import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    for key in ("CI_COMMIT_SHA", "GIT_COMMIT", "GITHUB_SHA"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def plot_confusion(cm, out_path: Path) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", cfg["mlflow"]["experiment_name"])
    registered_model_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL_NAME", cfg["mlflow"]["registered_model_name"]
    )
    model_stage = os.environ.get("MLFLOW_MODEL_STAGE", cfg["mlflow"].get("model_stage", "")).strip()

    data_path = Path(cfg["data"]["path"])
    out_dir = Path(cfg["artifacts"]["out_dir"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Did you run `dvc pull` (or generate + dvc add)?"
        )

    data_hash = sha256_file(data_path)
    git_sha = get_git_sha()

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=float(cfg["train"]["test_size"]),
        random_state=int(cfg["train"]["seed"]),
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=int(cfg["train"]["max_iter"]),
        C=float(cfg["train"]["C"]),
        n_jobs=None,
    )
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="macro")
    cm = confusion_matrix(yte, preds)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    cm_path = out_dir / "confusion_matrix.png"

    metrics = {"val_accuracy": acc, "val_f1_macro": f1}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, cm_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tag("git_sha", git_sha)
        mlflow.set_tag("data_sha256", data_hash)
        mlflow.set_tag("pipeline", "mlops-examples")
        mlflow.set_tag("dataset_path", str(data_path))

        mlflow.log_param("model_type", cfg["train"]["model_type"])
        mlflow.log_param("seed", cfg["train"]["seed"])
        mlflow.log_param("test_size", cfg["train"]["test_size"])
        mlflow.log_param("max_iter", cfg["train"]["max_iter"])
        mlflow.log_param("C", cfg["train"]["C"])

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="eval")
        mlflow.log_artifact(str(cm_path), artifact_path="eval")

        input_example = Xtr.head(5)
        signature = mlflow.models.infer_signature(Xtr, model.predict(Xtr))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        model_version = None
        if model_info.model_uri and registered_model_name:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            run_versions = [v for v in versions if getattr(v, "run_id", None) == run_id]
            if run_versions:
                run_versions.sort(key=lambda v: int(v.version))
                model_version = run_versions[-1].version

            if model_stage and model_version is not None:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=model_version,
                    stage=model_stage,
                )

        print("MLflow run complete")
        print(f"Run ID: {run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Registered model: {registered_model_name}")
        if model_version is not None:
            print(f"Model version: {model_version}")
        if model_stage and model_version is not None:
            print(f"Stage '{model_stage}' -> v{model_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dev.yaml")
    args = parser.parse_args()
    main(args.config)

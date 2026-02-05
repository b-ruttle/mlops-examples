import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def main(out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_iris(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y
    df.to_csv(out, index=False)
    print(f"Wrote dataset to: {out.resolve()}  (rows={len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/iris.csv")
    args = parser.parse_args()
    main(args.out)

.PHONY: setup data pull train

setup:
	uv venv
	uv sync

lock:
	uv lock

data:
	uv run python scripts/make_data.py --out data/iris.csv
	uv run dvc add data/iris.csv

pull:
	uv run dvc pull

train:
	uv run python src/train.py --config configs/dev.yaml

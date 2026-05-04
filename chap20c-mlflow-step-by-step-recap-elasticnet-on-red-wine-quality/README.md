# chap20c - Step-by-step recap: first ElasticNet pipeline on red-wine-quality

The full lesson lives at [`../20c-practical-work-15c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality.md`](../20c-practical-work-15c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality.md).

## What's new vs chap20b

- A real ML pipeline: `pd.read_csv` -> `train_test_split` -> `ElasticNet.fit` -> compute `rmse / mae / r2`
- `mlflow.sklearn.log_model(lr, "mymodel")` to persist the trained model

## Quick run

```bash
docker compose up --build
```

Then in another terminal (host venv activated, `pip install -r requirements.txt` once):

```bash
python train.py --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> experiment `experiment_1` -> the run with params, metrics and the `mymodel/` artifact.

## A tiny grid search

```bash
python train.py --alpha 0.1 --l1_ratio 0.1
python train.py --alpha 0.5 --l1_ratio 0.5
python train.py --alpha 0.9 --l1_ratio 0.9
```

Then in the UI, sort the table by `metrics.rmse` ascending.

## Dataset

`data/red-wine-quality.csv` - UCI Red Wine Quality (1 599 rows, 11 features, integer target `quality`). Comma-separated; converted from the UCI semicolon version for `pd.read_csv` simplicity.

## Tear down

```bash
docker compose down       # keep all runs
docker compose down -v    # wipe everything
```

# chap20p - Step-by-step recap: pyfunc log_model with SklearnWrapper + joblib + conda_env

Lesson: [`../20p-practical-work-15p-mlflow-step-by-step-recap-pyfunc-sklearn-wrapper-with-joblib-and-conda-env.md`](../20p-practical-work-15p-mlflow-step-by-step-recap-pyfunc-sklearn-wrapper-with-joblib-and-conda-env.md).

## What's new vs chap20o

- `joblib.dump(lr, "sklearn_model.pkl")` serializes the estimator
- `class SklearnWrapper(mlflow.pyfunc.PythonModel)`: custom `load_context` + `predict`
- `artifacts` dict bundles the .pkl and the data folder into the package
- `conda_env` dict pins Python 3.10 + mlflow + sklearn + cloudpickle
- `mlflow.pyfunc.log_model(artifact_path=..., python_model=..., artifacts=..., code_path=..., conda_env=...)` logs the full portable package

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Open http://localhost:5000 -> experiment `experiment_custom_sklearn`:
- Artifacts tab: `sklearn_mlflow_pyfunc/`
  - `MLmodel` -> loader: mlflow.pyfunc
  - `conda.yaml` -> your environment
  - `python_model.pkl` -> the SklearnWrapper (cloudpickle)
  - `artifacts/sklearn_model.pkl` -> the ElasticNet (joblib)
  - `artifacts/data/` -> train.csv, test.csv, data.csv
  - `train.py` -> your code

## Load the model back

```python
import mlflow.pyfunc, pandas as pd, os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.pyfunc.load_model("runs:/<RUN_ID>/sklearn_mlflow_pyfunc")
predictions = model.predict(pd.DataFrame([{...}]))
```

## Tear down

```bash
docker compose down
docker compose down -v
```

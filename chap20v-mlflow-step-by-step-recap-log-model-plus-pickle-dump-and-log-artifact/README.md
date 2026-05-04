# chap20v - Step-by-step recap: log_model + pickle.dump + log_artifact

Lesson: [`../20v-practical-work-15v-mlflow-step-by-step-recap-log-model-plus-pickle-dump-and-log-artifact.md`](../20v-practical-work-15v-mlflow-step-by-step-recap-log-model-plus-pickle-dump-and-log-artifact.md).

## What's new vs chap20u

- Drop `registered_model_name` (chap20u's lesson)
- Log the SAME model in TWO formats inside the same run:
  ```python
  mlflow.sklearn.log_model(lr, artifact_path="model")          # canonical flavour
  pickle.dump(lr, open("elastic-net-regression.pkl", "wb"))
  mlflow.log_artifact("elastic-net-regression.pkl")             # side-car .pkl
  ```
- Add a small `get_path_type(path)` helper -> classifies a path as `"file"`, `"directory"`, or `"not a valid path"`
- New experiment name: `experiment_elastic_net_mlflow`

## Why both formats?

- `log_model` -> canonical, future-proof, loadable with `mlflow.pyfunc.load_model`, eligible for the registry
- `log_artifact("...pkl")` -> a single bare pickle, loadable with plain `pickle.load`, useful when a downstream consumer cannot install MLflow

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Open http://localhost:5000 -> `experiment_elastic_net_mlflow` -> latest run -> Artifacts tab:
```
elastic-net-regression.pkl
model/
  MLmodel
  conda.yaml
  python_env.yaml
  requirements.txt
  model.pkl
```

## Loading the side-car from another consumer

```python
import pickle, mlflow.artifacts
local = mlflow.artifacts.download_artifacts("runs:/<RUN_ID>/elastic-net-regression.pkl")
with open(local, "rb") as f:
    lr = pickle.load(f)
```

## Tear down

```bash
docker compose down
docker compose down -v
```

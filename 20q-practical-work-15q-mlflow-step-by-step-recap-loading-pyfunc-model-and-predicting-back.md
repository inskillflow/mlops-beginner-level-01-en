<a id="top"></a>

# Chapter 20q — Step-by-step recap: loading a logged pyfunc model with `mlflow.pyfunc.load_model` and predicting again

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20p](#section-2) |
| 3 | [Where does the `model_uri` come from?](#section-3) |
| 4 | [Why re-load and re-predict in the same script?](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [The code](#section-6) |
| 7 | [Run it, compare the two metric sets](#section-7) |
| 8 | [Loading from outside the run (decoupled scoring script)](#section-8) |
| 9 | [Tear down](#section-9) |
| 10 | [Recap and next chapter](#section-10) |

---

<a id="section-1"></a>

## 1. Objective

Chapter 20p packaged an ElasticNet inside a `pyfunc` artifact. Today we close the loop: we **load** that artifact back with **`mlflow.pyfunc.load_model`** and call its `.predict()` to make sure round-tripping the model preserves its behaviour.

Concretely:

1. After `log_model(...)`, capture the **`ModelInfo`** object it returns.
2. Use `model_info.model_uri` to load the model back with `mlflow.pyfunc.load_model`.
3. Predict on `test_x` and compute RMSE / MAE / R².
4. Compare with the metrics computed before logging — they must be **identical** if serialization worked.
5. Log the round-trip metrics under `loaded_test_*` so they're visible in the UI.

This pattern is the foundation of every "load model from registry → batch score" job.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20p

| Diff | What |
|---|---|
| Capture `model_info = mlflow.pyfunc.log_model(...)` | Hold on to the URI. |
| Add `loaded = mlflow.pyfunc.load_model(model_info.model_uri)` | Read back from MLflow. |
| Re-predict + recompute metrics → log as `loaded_test_rmse` etc. | Sanity check + leaves a trail in the UI. |
| Print both metric sets side-by-side | Easy visual comparison. |
| Fix typos from the original snippet | `sklear_mlflow_pyfunc` → `sklearn_mlflow_pyfunc`, `code_path=["main.py"]` → `["train.py"]`, `python={}.format(3.10)` → `python=3.10`. |
| **Don't hardcode** any `runs:/<id>/...` | The model URI comes from `ModelInfo`. |

Everything else (`SklearnWrapper`, `joblib`, `conda_env`, `set_tags`, `autolog(...,log_models=False)`) is identical to chap 20p.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Where does the `model_uri` come from?

`mlflow.pyfunc.log_model(...)` returns a `ModelInfo` object that exposes the URI:

```python
model_info = mlflow.pyfunc.log_model(
    artifact_path="sklearn_mlflow_pyfunc",
    python_model=SklearnWrapper(),
    artifacts=artifacts,
    code_path=["train.py"],
    conda_env=conda_env,
)

print(model_info.model_uri)
# → 'runs:/8a4f...d1/sklearn_mlflow_pyfunc'
```

Three URI flavours work with `load_model`:

| URI | Reads from |
|---|---|
| `runs:/<run_id>/<artifact_path>` | A specific run's artifact |
| `models:/<name>/<version_or_alias>` | A registered model in the Model Registry (chap 17) |
| `s3://<bucket>/<path>` or `file:///<path>` | Direct artifact storage URL |

> [!IMPORTANT]
> Never hardcode a `run_id` in your training script (the snippet in chap 20p had `runs:/ee0c9144.../...` hardcoded — that's a bug waiting to happen as soon as you re-run the script). Always use `model_info.model_uri` or look up the run dynamically with `MlflowClient`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Why re-load and re-predict in the same script?

Three good reasons:

1. **Validation at the source** — if the round-trip changes a single prediction, you catch it immediately, not in production.
2. **Documentation** — the `loaded_test_*` metrics are stored on the run forever. Anyone who opens it in 6 months can see the model behaved exactly as recorded.
3. **CI/CD gate** — automated pipelines often demand `assert math.isclose(rmse, loaded_rmse, rel_tol=1e-9)` before promoting a model. Doing it inside the training script makes that assertion trivial.

It's only ~5 extra lines and it adds a real safety net.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap20q-mlflow-step-by-step-recap-loading-pyfunc-model-and-predicting-back/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt        ← same as 20p (joblib + cloudpickle)
    └── train.py               ← log_model + load_model + re-predict
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The code

### 6.1 `trainer/train.py`

```python
import argparse
import logging
import os
import warnings

import cloudpickle
import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_custom_sklearn")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    data_dir = "data/red-wine-data"
    os.makedirs(data_dir, exist_ok=True)
    data.to_csv(f"{data_dir}/data.csv",  index=False)
    train.to_csv(f"{data_dir}/train.csv", index=False)
    test.to_csv(f"{data_dir}/test.csv",  index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    mlflow.set_tags({
        "engineering":       "ML platform",
        "release.candidate": "RC1",
        "release.version":   "2.0",
    })

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False,
    )

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print(f"  Elasticnet (alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"  Before log -> RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)

    artifacts = {
        "sklearn_model": sklearn_model_path,
        "data":          data_dir,
    }

    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                ],
            },
        ],
        "name": "sklearn_env",
    }

    # ===== LOG the pyfunc model and KEEP the returned ModelInfo =====
    model_info = mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_path=["train.py"],
        conda_env=conda_env,
    )
    print("Logged model URI:", model_info.model_uri)

    # ===== LOAD it back via the URI we just got =====
    loaded = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

    # ===== RE-PREDICT on test_x and recompute metrics =====
    loaded_predictions = loaded.predict(test_x)
    loaded_rmse, loaded_mae, loaded_r2 = eval_metrics(test_y, loaded_predictions)

    print(f"  After load -> RMSE={loaded_rmse:.6f}  MAE={loaded_mae:.6f}  R2={loaded_r2:.6f}")

    # Log the round-trip metrics under DIFFERENT names so we can compare them
    mlflow.log_metrics({
        "loaded_test_rmse": loaded_rmse,
        "loaded_test_mae":  loaded_mae,
        "loaded_test_r2":   loaded_r2,
    })

    # Sanity check: predictions must be identical (round-trip is lossless)
    assert np.allclose(predicted_qualities, loaded_predictions), \
        "Loaded model produces DIFFERENT predictions! Serialization is broken."

    print("Sanity check OK: original and loaded predictions match exactly.")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 6.2 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt`

Identical to chap 20p (standard SQLite + joblib + cloudpickle in trainer).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run it, compare the two metric sets

```bash
cd chap20q-mlflow-step-by-step-recap-loading-pyfunc-model-and-predicting-back
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

End of trainer's stdout:

```text
  Before log -> RMSE=0.778500  MAE=0.622300  R2=0.105400
Logged model URI: runs:/8a4f...d1/sklearn_mlflow_pyfunc
  After load -> RMSE=0.778500  MAE=0.622300  R2=0.105400
Sanity check OK: original and loaded predictions match exactly.
```

In the UI ([http://localhost:5000](http://localhost:5000)) → **`experiment_custom_sklearn`** → Metrics tab:

```text
rmse              = 0.778500    ← logged before log_model
loaded_test_rmse  = 0.778500    ← logged after load_model
mae               = 0.622300
loaded_test_mae   = 0.622300
r2                = 0.105400
loaded_test_r2    = 0.105400
```

Identical, as expected. If they ever differ, the assertion would fail — that's the point.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Loading from outside the run (decoupled scoring script)

In production you load a model in a **separate** script (a batch scorer, an API container, a Spark UDF…). Same call, different `model_uri`:

```python
import mlflow.pyfunc, os, pandas as pd

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Three equivalent ways to point at the same artifact:

# 1. Specific run
m = mlflow.pyfunc.load_model("runs:/8a4fd1.../sklearn_mlflow_pyfunc")

# 2. Registered model (after chap 17 registers it)
m = mlflow.pyfunc.load_model("models:/wine-quality-elasticnet/Production")

# 3. Direct artifact URI (e.g. inside the docker volume)
m = mlflow.pyfunc.load_model(
    "/mlflow/mlruns/1/8a4fd1.../artifacts/sklearn_mlflow_pyfunc"
)

predictions = m.predict(pd.DataFrame([{...}]))
```

Easy way to test it via Docker:

```bash
docker compose run --rm --entrypoint python trainer -c "
import os, mlflow.pyfunc, pandas as pd
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
m = mlflow.pyfunc.load_model('runs:/<RUN_ID>/sklearn_mlflow_pyfunc')
sample = pd.DataFrame([{'fixed acidity':7.2,'volatile acidity':0.35,'citric acid':0.45,'residual sugar':8.5,'chlorides':0.045,'free sulfur dioxide':30.,'total sulfur dioxide':120.,'density':0.997,'pH':3.2,'sulphates':0.65,'alcohol':9.2}])
print(m.predict(sample))
"
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Tear down

```bash
docker compose down
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Recap and next chapter

You can now **save** a pyfunc model and **load it back** without ever hardcoding a run_id:

```python
model_info = mlflow.pyfunc.log_model(...)
loaded     = mlflow.pyfunc.load_model(model_info.model_uri)
```

Plus a tiny `np.allclose(...)` sanity check that catches serialization bugs immediately.

Next: **[chapter 20r](./20r-practical-work-15r-mlflow-step-by-step-recap-mlflow-evaluate-default-regressor.md)** — instead of computing RMSE/MAE/R² by hand after `load_model`, let MLflow do it (and a lot more) with **`mlflow.evaluate(model_uri, eval_df, targets=..., model_type="regressor")`**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20q — loading a pyfunc model and predicting back</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

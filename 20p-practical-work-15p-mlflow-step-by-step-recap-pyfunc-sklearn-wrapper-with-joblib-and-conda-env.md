<a id="top"></a>

# Chapter 20p — Step-by-step recap: `mlflow.pyfunc.log_model` with a custom `SklearnWrapper` (`PythonModel`), `joblib`, and a hand-crafted `conda_env`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20o](#section-2) |
| 3 | [Why `pyfunc` instead of `mlflow.sklearn.log_model`?](#section-3) |
| 4 | [The three building blocks](#section-4) |
| 5 | [`mlflow.pyfunc.log_model` — all arguments explained](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, inspect the pyfunc artifact tree](#section-8) |
| 9 | [How to load the model back](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

`mlflow.sklearn.log_model` is perfect when you want to log an estimator as-is. But what if you need **custom preprocessing**, **post-processing**, or you want to bundle **a different serializer** (`joblib` instead of `pickle`)? That's what **`mlflow.pyfunc`** is for.

Today we:

1. Serialize the fitted `ElasticNet` with **`joblib`**.
2. Write a **`SklearnWrapper`** class (inherits `mlflow.pyfunc.PythonModel`) that handles load + predict.
3. Bundle a list of **artifacts** (the `.pkl` + the data folder).
4. Declare a **`conda_env`** dict so the model can be reproduced on any machine.
5. Log everything with **`mlflow.pyfunc.log_model`**.

The resulting artifact is a **portable, self-contained model package** that can be loaded on any host — even one that has never seen sklearn or MLflow before — as long as Conda is available.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20o

| Diff | What |
|---|---|
| Serialize with `joblib.dump(lr, "sklearn_model.pkl")` | More efficient than Python's default pickle for sklearn. |
| Write `SklearnWrapper(mlflow.pyfunc.PythonModel)` | Custom `load_context` + `predict` methods. |
| Build an `artifacts` dict | Tells pyfunc which files to bundle (`sklearn_model.pkl` + data dir). |
| Build a `conda_env` dict | Pin Python, mlflow, sklearn, cloudpickle versions. |
| Call `mlflow.pyfunc.log_model(...)` | Log the full pyfunc package (not just the estimator). |
| Add `joblib` and `cloudpickle` to `requirements.txt` | Needed by the trainer container. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Why `pyfunc` instead of `mlflow.sklearn.log_model`?

| | `mlflow.sklearn.log_model` | `mlflow.pyfunc.log_model` + wrapper |
|---|---|---|
| Serializer | Python `pickle` (mlflow default) | **`joblib`** (you choose) |
| Preprocessing | None | You code it in `predict()` |
| Post-processing | None | You code it in `predict()` |
| Dependency bundling | auto `requirements.txt` from environment | **explicit `conda_env` or `pip_requirements`** |
| Portability | Needs exact same mlflow+sklearn versions | Self-contained Conda env |
| Complexity | 1 line | ~30 lines |
| Use when | Vanilla estimator, default env | Custom inference logic, strict env control |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The three building blocks

### 4.1 `mlflow.pyfunc.PythonModel`

The abstract base class you inherit to build a custom wrapper. You must implement two methods:

```python
class SklearnWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """Called ONCE when the model is loaded.
        `context.artifacts` maps the keys from the `artifacts` dict
        to the actual file paths inside the artifact package."""
        self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        """Called for every inference request.
        `model_input` is a pandas DataFrame.
        Must return a numpy array or pandas Series/DataFrame."""
        return self.sklearn_model.predict(model_input.values)
```

### 4.2 The `artifacts` dict

A mapping of `name → local_path`. MLflow copies these files into the artifact package:

```python
artifacts = {
    "sklearn_model": "sklearn_model.pkl",   # the serialized estimator
    "data":          "data/",               # the data folder (optional, for documentation)
}
```

Inside `load_context`, `context.artifacts["sklearn_model"]` is the **absolute path** to the `.pkl` file inside the package on the loading machine.

### 4.3 The `conda_env` dict

A YAML-compatible dict that pins the exact Python version and packages:

```python
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
```

MLflow serializes this as `conda.yaml` inside the artifact tree. When you run `mlflow models serve` or `mlflow models predict`, it creates a fresh Conda environment from this file.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. `mlflow.pyfunc.log_model` — all arguments explained

```python
mlflow.pyfunc.log_model(
    artifact_path="sklearn_mlflow_pyfunc",  # sub-folder inside the run's artifact tree
    python_model=SklearnWrapper(),           # your PythonModel instance
    artifacts=artifacts,                     # files to bundle
    code_path=["train.py"],                 # Python files to include (for reproducibility)
    conda_env=conda_env,                    # environment spec
)
```

| Argument | Purpose |
|---|---|
| `artifact_path` | The folder name in the run's artifact tree (e.g. `sklearn_mlflow_pyfunc/`). |
| `python_model` | An instance of your `PythonModel` subclass. |
| `artifacts` | Dict `name → local_path` — files MLflow will bundle into the package. |
| `code_path` | List of Python files to copy verbatim into the package (included in `sys.path` at load time). |
| `conda_env` | A dict or a path to a `conda.yaml` — the environment pinned for deployment. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20p-mlflow-step-by-step-recap-pyfunc-sklearn-wrapper-with-joblib-and-conda-env/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt        ← adds joblib + cloudpickle
    └── train.py               ← SklearnWrapper + pyfunc.log_model
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. The code

### 7.1 `trainer/requirements.txt`

```
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
joblib==1.4.2
cloudpickle==3.0.0
```

### 7.2 `trainer/train.py`

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
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
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


# ===== Custom PythonModel wrapper =====
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
    print(f"Artifact Location : {exp.artifact_location}")
    print(f"Tags              : {exp.tags}")
    print(f"Lifecycle_stage   : {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")

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
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    # ===== Serialize the model with joblib =====
    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)

    # ===== Artifacts to bundle =====
    artifacts = {
        "sklearn_model": sklearn_model_path,
        "data":          data_dir,
    }

    # ===== Conda environment =====
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

    # ===== Log the pyfunc model =====
    mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_path=["train.py"],
        conda_env=conda_env,
    )

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 7.3 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`

Same structure as 20o. Only the trainer `requirements.txt` changes (adds `joblib` and `cloudpickle`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, inspect the pyfunc artifact tree

```bash
cd chap20p-mlflow-step-by-step-recap-pyfunc-sklearn-wrapper-with-joblib-and-conda-env
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

In the UI ([http://localhost:5000](http://localhost:5000)) → **`experiment_custom_sklearn`** → latest run → **Artifacts**:

```text
sklearn_mlflow_pyfunc/
├── MLmodel                     ← loader: mlflow.pyfunc (NOT mlflow.sklearn)
├── conda.yaml                  ← your conda_env serialized as YAML
├── python_model.pkl            ← the SklearnWrapper instance (cloudpickle)
├── train.py                    ← your code_path file
└── artifacts/
    ├── sklearn_model.pkl       ← the joblib-serialized ElasticNet
    └── data/
        ├── data.csv
        ├── train.csv
        └── test.csv
```

Key differences from a `sklearn.log_model` artifact:
- `MLmodel` says `loader_module: mlflow.pyfunc` (not `mlflow.sklearn`).
- `python_model.pkl` is the *wrapper class*, cloudpickled.
- `artifacts/sklearn_model.pkl` is the *fitted estimator*, joblibbed.
- `conda.yaml` shows your hand-crafted environment.
- `train.py` is bundled verbatim (from `code_path`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. How to load the model back

From **inside the trainer container** (or any Python session with mlflow installed):

```python
import mlflow.pyfunc, pandas as pd, os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

run_id = "<paste the run_id from the UI>"
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/sklearn_mlflow_pyfunc")

# Feed it a pandas DataFrame — SklearnWrapper.predict converts .values internally
sample = pd.DataFrame({
    "fixed acidity": [7.2], "volatile acidity": [0.35],
    "citric acid": [0.45], "residual sugar": [8.5],
    "chlorides": [0.045], "free sulfur dioxide": [30.],
    "total sulfur dioxide": [120.], "density": [0.997],
    "pH": [3.2], "sulphates": [0.65], "alcohol": [9.2],
})
print(model.predict(sample))
```

Run it via the trainer container:

```bash
docker compose run --rm --entrypoint python trainer -c "
import mlflow.pyfunc, pandas as pd, os
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
model = mlflow.pyfunc.load_model('runs:/<RUN_ID>/sklearn_mlflow_pyfunc')
print(model.predict(pd.DataFrame([{'fixed acidity':7.2,'volatile acidity':0.35,'citric acid':0.45,'residual sugar':8.5,'chlorides':0.045,'free sulfur dioxide':30.,'total sulfur dioxide':120.,'density':0.997,'pH':3.2,'sulphates':0.65,'alcohol':9.2}])))
"
```

> [!NOTE]
> On an external machine (not Docker), you'd do `mlflow models serve --model-uri runs:/<RUN_ID>/sklearn_mlflow_pyfunc` and Conda would recreate `sklearn_env` from `conda.yaml` automatically. This is the "self-contained" promise of pyfunc.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tear down

```bash
docker compose down
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Recap and next chapter

Today's additions at a glance:

| Concept | Code |
|---|---|
| Custom serializer | `joblib.dump(lr, "sklearn_model.pkl")` |
| Wrapper class | `class SklearnWrapper(mlflow.pyfunc.PythonModel)` |
| `load_context` | `self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])` |
| `predict` | `return self.sklearn_model.predict(model_input.values)` |
| Artifact bundle | `artifacts = {"sklearn_model": ..., "data": ...}` |
| Environment pin | `conda_env = {"name": "sklearn_env", "dependencies": [...]}` |
| Log | `mlflow.pyfunc.log_model(artifact_path=..., python_model=..., ...)` |
| Load back | `mlflow.pyfunc.load_model(f"runs:/{run_id}/sklearn_mlflow_pyfunc")` |

The full **step-by-step recap** now covers:

| Chapter | Concept |
|---|---|
| 20a → 20c | Basics, `get_tracking_uri`, full ElasticNet |
| 20d → 20f | Docker trainer, env-var URI, `create_experiment` |
| 20g → 20i | `active_run`, `last_active_run`, `log_artifacts`, `set_tags` |
| 20j → 20l | Multiple runs, multiple experiments, `autolog` |
| 20m | PostgreSQL backend + S3 overview |
| 20n | `infer_signature` (automatic, 1 line) |
| 20o | `ModelSignature` + `Schema` + `ColSpec` (manual) |
| **20p** | **`pyfunc.log_model` + `SklearnWrapper` + `joblib` + `conda_env`** |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20p — pyfunc log_model with SklearnWrapper, joblib and conda_env</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

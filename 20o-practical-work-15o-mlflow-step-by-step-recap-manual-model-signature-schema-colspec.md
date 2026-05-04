<a id="top"></a>

# Chapter 20o — Step-by-step recap: writing a model signature **by hand** with `Schema`, `ColSpec` and `ModelSignature`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20n](#section-2) |
| 3 | [The three classes in detail](#section-3) |
| 4 | [Supported MLflow types](#section-4) |
| 5 | [Why include `quality` in the input schema or not?](#section-5) |
| 6 | [Hardcoded `input_example` — the dict-of-arrays pattern](#section-6) |
| 7 | [Project structure](#section-7) |
| 8 | [The code](#section-8) |
| 9 | [Run it, inspect the artifacts](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Chapter 20n introduced `infer_signature` as the easy path (one line, reads the real data). Today we drill into the **manual path** — writing the schema from scratch using `Schema`, `ColSpec` and `ModelSignature`. This matters when:

- The training data isn't available at the moment you log the model (e.g. the model is retrained offline and re-packaged later).
- You want to enforce a stricter contract than the actual data types would produce (e.g. forcing `integer` even though sklearn returns `float64`).
- You're documenting a model for a pre-agreed API contract.

Same Docker stack as always. Only `train.py` evolves.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20n

| Diff | What |
|---|---|
| Replace `infer_signature(test_x, predicted_qualities)` with manual schema | Full `Schema` / `ColSpec` / `ModelSignature` block. |
| Replace `test_x.head(5)` input example with a **hardcoded dict of arrays** | Shows the dict-of-arrays pattern (alternative to a DataFrame). |
| Use `"model"` as the artifact path (instead of `"elasticnet_model"`) | Matching the original tutorial's naming. |
| Experiment name: `"experiment_signature"` | Same experiment name as chap 20n — the two runs appear side-by-side for comparison. |

Everything else (env-var URI, `autolog(log_models=False,...)`, `set_tags`, `log_artifact`) is identical to chap 20n.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. The three classes in detail

### `ColSpec(type, name=None)`

One column in a schema. `type` is a string (see next section). `name` is optional for the output schema but mandatory for inputs (since column order is meaningful for sklearn).

```python
ColSpec("double", "fixed acidity")   # named input column
ColSpec("double")                     # unnamed output column (just a scalar)
```

### `Schema(list_of_ColSpec)`

An ordered list of columns, forming either the input schema or the output schema.

```python
input_schema = Schema([
    ColSpec("double", "fixed acidity"),
    ColSpec("double", "volatile acidity"),
    # ... all 11 feature columns
])
output_schema = Schema([ColSpec("long")])   # ElasticNet output ≈ quality (int-ish)
```

### `ModelSignature(inputs, outputs)`

The full contract: input schema + output schema. What you pass to `log_model`.

```python
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Supported MLflow types

| MLflow type string | Numpy / Pandas equivalent |
|---|---|
| `"double"` | `float64` |
| `"float"` | `float32` |
| `"long"` | `int64` |
| `"integer"` | `int32` |
| `"string"` | `object` (str) |
| `"boolean"` | `bool` |
| `"binary"` | `bytes` |
| `"datetime"` | `datetime64` |

For wine quality features (measured continuously) → `"double"`.  
For the predicted quality score (integer range 3–9) → `"long"` (as in the original tutorial).

> [!TIP]
> `infer_signature` would have chosen `"double"` for the output since `lr.predict()` returns a `float64` array. Using `"long"` manually signals that your *actual* production output is an integer, which the deployment tooling can use to add rounding. This is a case where the manual approach adds real value.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Why **not** include `quality` in the input schema?

The original tutorial snippet listed `quality` as the 12th input column. That's incorrect: `quality` is the **target variable**, not a feature. The model's `predict()` method takes 11 columns and returns the predicted quality.

Correct input schema = 11 feature columns (everything except `quality`).

Including `quality` in the input schema would:
- Confuse `mlflow models serve` (it would expect 12 columns).
- Break any downstream validation.
- Make the model card misleading.

Today's `train.py` uses the correct 11-column schema.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Hardcoded `input_example` — the dict-of-arrays pattern

MLflow's `log_model` accepts three formats for `input_example`:

| Format | Example |
|---|---|
| **Pandas DataFrame** | `test_x.head(5)` (what chap 20n used) |
| **Dict of arrays** | `{"col1": np.array([v1, v2, …]), …}` |
| **List of dicts** | `[{"col1": v1, "col2": v2}, …]` |

Today we use the dict-of-arrays format, which lets you hardcode representative values without needing access to `test_x` at logging time. The 5-row example has realistic wine chemistry values, making the model card immediately useful for documentation.

```python
input_example = {
    "fixed acidity":        np.array([7.2, 7.5, 7.0, 6.8, 6.9]),
    "volatile acidity":     np.array([0.35, 0.3, 0.28, 0.38, 0.25]),
    # … all 11 feature columns
}
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Project structure

```text
chap20o-mlflow-step-by-step-recap-manual-model-signature-schema-colspec/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py            ← Schema + ColSpec + ModelSignature + dict input_example
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. The code

### 8.1 `trainer/train.py`

```python
import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_signature")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")
    print(f"Artifact Location : {exp.artifact_location}")
    print(f"Tags              : {exp.tags}")
    print(f"Lifecycle_stage   : {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

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

    # ===== MANUAL SIGNATURE =====
    input_schema = Schema([
        ColSpec("double", "fixed acidity"),
        ColSpec("double", "volatile acidity"),
        ColSpec("double", "citric acid"),
        ColSpec("double", "residual sugar"),
        ColSpec("double", "chlorides"),
        ColSpec("double", "free sulfur dioxide"),
        ColSpec("double", "total sulfur dioxide"),
        ColSpec("double", "density"),
        ColSpec("double", "pH"),
        ColSpec("double", "sulphates"),
        ColSpec("double", "alcohol"),
    ])
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # ===== HARDCODED INPUT EXAMPLE (dict of arrays) =====
    input_example = {
        "fixed acidity":        np.array([7.2, 7.5, 7.0, 6.8, 6.9]),
        "volatile acidity":     np.array([0.35, 0.3, 0.28, 0.38, 0.25]),
        "citric acid":          np.array([0.45, 0.5, 0.55, 0.4, 0.42]),
        "residual sugar":       np.array([8.5, 9.0, 8.2, 7.8, 8.1]),
        "chlorides":            np.array([0.045, 0.04, 0.035, 0.05, 0.042]),
        "free sulfur dioxide":  np.array([30., 35., 40., 28., 32.]),
        "total sulfur dioxide": np.array([120., 125., 130., 115., 110.]),
        "density":              np.array([0.997, 0.996, 0.995, 0.998, 0.994]),
        "pH":                   np.array([3.2, 3.1, 3.0, 3.3, 3.2]),
        "sulphates":            np.array([0.65, 0.7, 0.68, 0.72, 0.62]),
        "alcohol":              np.array([9.2, 9.5, 9.0, 9.8, 9.4]),
    }

    mlflow.log_artifact("data/red-wine-quality.csv")
    mlflow.sklearn.log_model(
        lr,
        "model",
        signature=signature,
        input_example=input_example,
    )

    print("Artifact path:", mlflow.get_artifact_uri())
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 8.2 `docker-compose.yml` / `mlflow/Dockerfile` / `trainer/Dockerfile` / `trainer/requirements.txt`

All identical to chap 20n (standard SQLite setup).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run it, inspect the artifacts

```bash
cd chap20o-mlflow-step-by-step-recap-manual-model-signature-schema-colspec
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

In the UI ([http://localhost:5000](http://localhost:5000)) → **`experiment_signature`** → you now have **two runs**: one from chap 20n (`infer_signature`) and one from today (manual schema).

Click the new run → **Artifacts** → `model/`:

- `signature.json`:
  ```json
  {
    "inputs": [
      {"name": "fixed acidity", "type": "double"},
      {"name": "volatile acidity", "type": "double"},
      ...
      {"name": "alcohol", "type": "double"}
    ],
    "outputs": [{"type": "long"}]
  }
  ```
  Note **`"long"`** for the output — the chap 20n run had `"double"` (inferred from the float array). Same model, different declared output type.

- `input_example.json` → the 5 representative rows you hardcoded.

> [!TIP]
> Compare the two runs side-by-side: tick both in the runs table → **Compare** → open the **Model** tab. You'll see the output type difference immediately: `long` vs `double`. This illustrates why the manual approach matters for strict typing.

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

You can now build a signature **without touching the real data**:

| | chap 20n (`infer_signature`) | chap 20o (manual) |
|---|---|---|
| Code | 1 line | ~15 lines |
| Needs actual data | Yes | No |
| Output type | Inferred from numpy dtype (`double`) | You choose (`long`) |
| Maintenance | Automatic | Manual |

Next: **[chapter 20p](./20p-practical-work-15p-mlflow-step-by-step-recap-pyfunc-sklearn-wrapper-with-joblib-and-conda-env.md)** — wrap the ElasticNet in a **`PythonModel`** (`SklearnWrapper`) and log it with `mlflow.pyfunc.log_model`, bundling `joblib`, a `conda_env` dict, and a `code_path` into a single portable artifact.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20o — manual model signature with Schema + ColSpec + ModelSignature</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

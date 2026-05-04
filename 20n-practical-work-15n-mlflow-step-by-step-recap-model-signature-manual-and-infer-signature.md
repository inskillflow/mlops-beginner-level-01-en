<a id="top"></a>

# Chapter 20n — Step-by-step recap: model **signature** — manual (`ModelSignature` / `Schema` / `ColSpec`) vs automatic (`infer_signature`)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20m](#section-2) |
| 3 | [What is a model signature?](#section-3) |
| 4 | [Why disable autolog's signature and log it manually?](#section-4) |
| 5 | [Approach A — Manual signature: `ModelSignature` + `Schema` + `ColSpec`](#section-5) |
| 6 | [Approach B — Automatic signature: `infer_signature`](#section-6) |
| 7 | [Comparison table](#section-7) |
| 8 | [Project structure](#section-8) |
| 9 | [The code (Approach B — recommended)](#section-9) |
| 10 | [Run it, inspect the signature in the UI](#section-10) |
| 11 | [Appendix — Approach A full code](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap and next chapter](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

When you save a model with `mlflow.sklearn.log_model(lr, "model")`, MLflow stores the serialized estimator but has no formal contract for what types of data it accepts or produces.

A **model signature** adds that contract: it records the **column names, data types and shape** for both inputs and outputs. This:

- Powers the MLflow UI's inline model documentation.
- Lets `mlflow models serve` validate requests before forwarding them to the model.
- Enables downstream tools (Databricks, Sagemaker, Azure ML) to generate API stubs automatically.

Two approaches to create one:

- **Manual** — you type out each column name and type using `Schema` and `ColSpec`. Verbose, but you have full control over what the signature says (useful when the training data isn't available at signing time).
- **Automatic** — `infer_signature(inputs, outputs)` reads the actual data arrays and infers everything. One line, zero maintenance.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20m

| Diff | What |
|---|---|
| **Disable** autolog's model-logging step (`log_models=False`) | We want to log the model ourselves with an explicit signature. |
| **Disable** autolog's signature step (`log_model_signatures=False`) | Same reason. |
| **Add** `infer_signature(test_x, predicted_qualities)` | Automatic signature in one line. |
| **Add** `signature=…, input_example=…` to `mlflow.sklearn.log_model` | Attach the signature to the saved model. |
| **Revert** to SQLite backend (simpler, not the focus of this chapter) | Focus: signature, not the DB. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What is a model signature?

A signature is a JSON-encoded description of the model's interface:

```json
{
  "inputs": [
    {"name": "fixed acidity",        "type": "double"},
    {"name": "volatile acidity",     "type": "double"},
    {"name": "citric acid",          "type": "double"},
    {"name": "residual sugar",       "type": "double"},
    {"name": "chlorides",            "type": "double"},
    {"name": "free sulfur dioxide",  "type": "double"},
    {"name": "total sulfur dioxide", "type": "double"},
    {"name": "density",              "type": "double"},
    {"name": "pH",                   "type": "double"},
    {"name": "sulphates",            "type": "double"},
    {"name": "alcohol",              "type": "double"}
  ],
  "outputs": [{"type": "double"}]
}
```

MLflow stores this as `model/signature.json` inside the artifact tree. It shows up in the UI on the model's artifact page.

An **input example** is a small JSON snapshot of real rows (the first 5 rows of `train_x`, for instance). MLflow stores it as `model/input_example.json`. Together, the signature + input example is everything a downstream user needs to call the model correctly.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Why disable autolog's signature and log it manually?

```python
mlflow.sklearn.autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
)
```

Three reasons to turn these off and handle them yourself:

1. **Autolog logs the model twice** if you call both `mlflow.autolog()` and `mlflow.sklearn.log_model(...)`. The second call creates an unnamed duplicate artifact.
2. **You want a hand-crafted signature** — e.g. the production schema has strict column names or type constraints you need to verify.
3. **You want a specific artifact path** name like `"elasticnet_model"` (not autolog's default `"model"`).

When using **`infer_signature`** (Approach B), all three flags can also be left at their defaults because `infer_signature` + `log_model` does everything autolog's signature step would do — just under your control.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Approach A — Manual signature: `ModelSignature` + `Schema` + `ColSpec`

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_data = [
    {"name": "fixed acidity",        "type": "double"},
    {"name": "volatile acidity",     "type": "double"},
    {"name": "citric acid",          "type": "double"},
    {"name": "residual sugar",       "type": "double"},
    {"name": "chlorides",            "type": "double"},
    {"name": "free sulfur dioxide",  "type": "double"},
    {"name": "total sulfur dioxide", "type": "double"},
    {"name": "density",              "type": "double"},
    {"name": "pH",                   "type": "double"},
    {"name": "sulphates",            "type": "double"},
    {"name": "alcohol",              "type": "double"},
]
output_data = [{"type": "double"}]

input_schema  = Schema([ColSpec(col["type"], col["name"]) for col in input_data])
output_schema = Schema([ColSpec(col["type"])              for col in output_data])
signature     = ModelSignature(inputs=input_schema, outputs=output_schema)
```

| Class | What it does |
|---|---|
| `ColSpec(type, name=None)` | One column: its MLflow data type and optional name |
| `Schema([ColSpec, …])` | An ordered list of columns (input OR output) |
| `ModelSignature(inputs, outputs)` | The full contract: input schema + output schema |

Supported types: `"double"`, `"float"`, `"integer"`, `"long"`, `"string"`, `"boolean"`, `"binary"`.

**Pros**: you control every detail; works even if the training data isn't available.  
**Cons**: you have to type every column name and maintain the list when features change.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Approach B — Automatic signature: `infer_signature`

```python
from mlflow.models.signature import infer_signature

signature = infer_signature(test_x, predicted_qualities)
```

`infer_signature` reads the actual numpy/pandas data:

- **`test_x`** (the features DataFrame) → input schema (column names + types auto-detected).
- **`predicted_qualities`** (the numpy array from `.predict()`) → output schema (shape + dtype).

One line replaces the entire block in Approach A. The result is a `ModelSignature` object — identical class, identical JSON output. You can then pass it to `log_model` exactly the same way:

```python
mlflow.sklearn.log_model(
    lr,
    "elasticnet_model",
    signature=signature,
    input_example=test_x.head(5),   # first 5 real rows as an example
)
```

**Pros**: zero maintenance, always in sync with the actual data.  
**Cons**: requires the training/test data to be available at the point of signing.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Comparison table

| | Approach A (manual) | Approach B (`infer_signature`) |
|---|---|---|
| Lines of code | ~15 | 1 |
| Column names | Typed by hand | Read from the DataFrame |
| Column types | Typed by hand | Inferred from the actual dtypes |
| Data needed at signing | No | Yes (need `test_x` and `predicted_qualities`) |
| Sync with real data | Manual | Automatic |
| When to prefer | Strict, pre-defined schemas; offline signing | Typical data science workflow |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Project structure

```text
chap20n-mlflow-step-by-step-recap-model-signature-manual-and-infer-signature/
├── README.md
├── docker-compose.yml          ← back to SQLite (focus on signatures)
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile              ← standard (no psycopg2)
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py                ← uses infer_signature + log_model with signature
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. The code (Approach B — recommended)

### 9.1 `trainer/train.py`

```python
import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
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

    # Disable autolog's model-logging: we log the model ourselves with a signature
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

    # Manual metrics (autolog captured training metrics; we add test metrics)
    mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2, "test_mae": mae})

    # ===== APPROACH B — infer_signature (one line) =====
    signature = infer_signature(test_x, predicted_qualities)

    # A real input example: first 5 rows of test_x
    input_example = test_x.head(5)

    # Log model WITH signature + input example
    mlflow.sklearn.log_model(
        lr,
        "elasticnet_model",
        signature=signature,
        input_example=input_example,
    )

    # Log the raw CSV as an artifact too
    mlflow.log_artifact("data/red-wine-quality.csv")

    print("Artifact path:", mlflow.get_artifact_uri())

    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 9.2 `docker-compose.yml`

Back to SQLite (standard setup from 20g→20l):

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20n
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    networks: [recap-net]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5

  trainer:
    build: { context: ./trainer }
    image: mlops/trainer-recap:latest
    container_name: trainer-recap-20n
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    volumes:
      - ./data:/code/data
    networks: [recap-net]
    depends_on:
      mlflow: { condition: service_healthy }

volumes:
  mlflow-db:
  mlflow-artifacts:

networks:
  recap-net:
    driver: bridge
```

### 9.3 `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt`

Identical to chap 20g→20l (standard SQLite image, no psycopg2).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Run it, inspect the signature in the UI

```bash
cd chap20n-mlflow-step-by-step-recap-model-signature-manual-and-infer-signature
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

In the UI ([http://localhost:5000](http://localhost:5000)):

1. Open **`experiment_signature`** → latest run.
2. **Artifacts** tab → expand `elasticnet_model/`:
   ```text
   elasticnet_model/
   ├── MLmodel               ← YAML descriptor (contains the signature)
   ├── conda.yaml
   ├── model.pkl
   ├── requirements.txt
   ├── signature.json        ← the inferred signature
   └── input_example.json   ← the 5 real rows from test_x
   ```
3. Click **`signature.json`**:
   ```json
   {
     "inputs": [
       {"name": "fixed acidity",        "type": "double"},
       {"name": "volatile acidity",     "type": "double"},
       ...
       {"name": "alcohol",              "type": "double"}
     ],
     "outputs": [{"type": "double"}]
   }
   ```
4. Click **`input_example.json`** → 5 rows of actual wine data.

The **model card** at the top of the artifact page shows a summary table of the signature directly in the browser — no need to open the JSON manually.

> [!TIP]
> The signature is also checked when you call `mlflow models serve --model-uri runs:/<run_id>/elasticnet_model`. If you POST a JSON body with a wrong column name or incompatible type, the server rejects it **before** even calling the model. Huge win for catching data drift early.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Appendix — Approach A full code

Drop-in replacement for the signature block in `train.py` if you prefer hand-crafted schemas:

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

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
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Hardcoded input example (not pulled from test_x)
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

mlflow.sklearn.log_model(lr, "elasticnet_model",
                         signature=signature, input_example=input_example)
```

The rest of the script is identical. The output in the UI is the same JSON structure.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Tear down

```bash
docker compose down
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. Recap and next chapter

You now know two ways to attach a contract to your model:

| | Approach A (manual) | Approach B (infer) |
|---|---|---|
| Code | `ModelSignature` + `Schema` + `ColSpec` | `infer_signature(test_x, preds)` |
| Maintenance | Manual | Automatic |
| Best for | Pre-defined / strict schemas | Typical DS workflows |

Both produce identical `signature.json` + `input_example.json` files. Both are read by `mlflow models serve` for request validation.

The complete **step-by-step recap** now covers:

| Chapter | Concept |
|---|---|
| 20a → 20c | Basics, `get_tracking_uri`, full ElasticNet |
| 20d → 20f | Dockerised trainer, env-var URI, `create_experiment` |
| 20g → 20i | `active_run`, `last_active_run`, `log_artifacts`, `set_tags` |
| 20j → 20l | Multiple runs, multiple experiments, `autolog` |
| 20m | PostgreSQL backend + S3 overview |
| **20n** | **Model signature (manual + infer)** |

From here, the natural next chapters are **17** (Model Registry + `MlflowClient`) and **15** (`mlflow.pyfunc.PythonModel` for custom inference).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20n — model signature (manual + infer_signature)</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

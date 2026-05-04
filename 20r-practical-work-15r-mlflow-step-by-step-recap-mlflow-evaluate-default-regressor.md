<a id="top"></a>

# Chapter 20r — Step-by-step recap: automated evaluation with `mlflow.evaluate(model_uri, eval_df, targets=..., model_type="regressor")`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20q](#section-2) |
| 3 | [What `mlflow.evaluate` actually does](#section-3) |
| 4 | [The `eval_df` shape: features + target in one DataFrame](#section-4) |
| 5 | [The 4 main arguments](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, browse the auto-generated metrics + artifacts](#section-8) |
| 9 | [Bonus — the `EvaluationResult` object](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Chapter 20q computed RMSE / MAE / R² by hand after re-loading the model. That's fine, but for a regressor the standard set of metrics (and several diagnostic plots) is **always the same** — so let MLflow generate them for you.

That's what **`mlflow.evaluate(...)`** does. One call, and the run gets:

- A full set of regression metrics: `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`, `max_error`, `mean_absolute_percentage_error`, `sum_on_target`, `mean_on_target`…
- Diagnostic artifacts: `eval_results_table.json`, `shap_summary_plot.png` (if `shap` is installed), feature attribution plots…
- Everything stored on the **current run**, browsable in the UI.

We hand `mlflow.evaluate` the **logged model URI** (from chap 20q) and a **single DataFrame** that contains features + target. It does the rest.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20q

| Diff | What |
|---|---|
| Replace the manual re-predict + log_metrics block | Single `mlflow.evaluate(...)` call. |
| Use `model_info.model_uri` (or `mlflow.get_artifact_uri("sklearn_mlflow_pyfunc")`) | Same model URI as before. |
| Pass `test` (the full DataFrame, with `quality` column) | `evaluate` reads features and target from the same frame. |
| Set `targets="quality"` | Tells MLflow which column is the label. |
| Set `model_type="regressor"` | Selects the regression metric set. |
| Set `evaluators=["default"]` | Use MLflow's built-in regressor evaluator. |

Everything else (`SklearnWrapper`, `joblib`, `conda_env`, `set_tags`, `autolog(...,log_models=False)`) is identical to chap 20q.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What `mlflow.evaluate` actually does

Pseudo-code of what happens internally:

```python
def mlflow_evaluate(model_uri, eval_df, targets, model_type, evaluators):
    model = mlflow.pyfunc.load_model(model_uri)
    X = eval_df.drop(columns=[targets])
    y = eval_df[targets]

    predictions = model.predict(X)

    metrics = {}
    if model_type == "regressor":
        metrics.update({
            "mean_squared_error":      mean_squared_error(y, predictions),
            "root_mean_squared_error": np.sqrt(mean_squared_error(y, predictions)),
            "mean_absolute_error":     mean_absolute_error(y, predictions),
            "r2_score":                r2_score(y, predictions),
            "max_error":               max_error(y, predictions),
            "mean_absolute_percentage_error": ...,
            "sum_on_target":           y.sum(),
            "mean_on_target":          y.mean(),
        })
    elif model_type == "classifier":
        metrics.update({...})  # accuracy, F1, ROC-AUC, etc.

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # Plus a full evaluation table, SHAP plots if available, etc.
    return EvaluationResult(metrics=metrics, artifacts=...)
```

You write none of that. One line gives you the lot.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The `eval_df` shape: features + target in one DataFrame

`mlflow.evaluate` expects a single DataFrame containing both inputs and the label, and you tell it which column is the label via `targets="<col_name>"`:

```text
eval_df  (the full `test` DataFrame in our case)
┌──────────────┬───────────────────┬───┬──────────┐
│ fixed acidity│ volatile acidity  │ ..│  quality │   ← targets="quality"
├──────────────┼───────────────────┼───┼──────────┤
│         7.2  │            0.35   │ ..│       6  │
│         7.5  │            0.30   │ ..│       7  │
│         ...  │            ...    │ ..│      ..  │
└──────────────┴───────────────────┴───┴──────────┘
        FEATURES (auto-detected)            LABEL
```

Concretely we just pass `test`, the DataFrame returned by `train_test_split` — it already has all 12 columns.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. The 4 main arguments

```python
mlflow.evaluate(
    model_info.model_uri,         # the model to score (any pyfunc URI)
    test,                         # the eval DataFrame (features + label)
    targets="quality",            # the label column name
    model_type="regressor",       # OR "classifier" / "question-answering" / ...
    evaluators=["default"],       # built-in evaluator
)
```

| Arg | Purpose |
|---|---|
| `model_uri` (positional) | Anything `mlflow.pyfunc.load_model` understands: `runs:/...`, `models:/...`, file path. |
| `data` (positional, here `test`) | A pandas DataFrame OR a path/URI to a `Dataset`. |
| `targets` | Column name in `data` that holds the ground-truth labels. |
| `model_type` | Picks the metric set. Common values: `"regressor"`, `"classifier"`, `"question-answering"`, `"text-summarization"`. |
| `evaluators` | List of built-in or registered evaluators. `"default"` is the standard one. |

> [!TIP]
> Pass a **registered model URI** (`models:/wine-quality/Production`) instead of a `runs:/...` URI to evaluate the currently-promoted model whenever new data arrives — the foundation of model-monitoring pipelines.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20r-mlflow-step-by-step-recap-mlflow-evaluate-default-regressor/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt        ← same as 20q (joblib + cloudpickle)
    └── train.py               ← log_model + mlflow.evaluate
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. The code

### 7.1 `trainer/train.py`

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
    print(f"  Manual -> RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)

    artifacts = {"sklearn_model": sklearn_model_path, "data": data_dir}

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

    model_info = mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_path=["train.py"],
        conda_env=conda_env,
    )
    print("Logged model URI:", model_info.model_uri)

    # ===== AUTOMATED EVALUATION =====
    result = mlflow.evaluate(
        model_info.model_uri,         # the freshly-logged pyfunc model
        test,                         # full DataFrame (features + label)
        targets="quality",            # label column
        model_type="regressor",       # picks the regression metric set
        evaluators=["default"],       # built-in evaluator
    )

    print("\n--- mlflow.evaluate auto-logged metrics ---")
    for k, v in sorted(result.metrics.items()):
        print(f"  {k} = {v:.6f}")

    print("\n--- mlflow.evaluate generated artifacts ---")
    for name, info in result.artifacts.items():
        print(f"  {name} -> {info.uri}")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 7.2 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt`

Identical to chap 20q.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, browse the auto-generated metrics + artifacts

```bash
cd chap20r-mlflow-step-by-step-recap-mlflow-evaluate-default-regressor
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Trainer's stdout (excerpt):

```text
Manual -> RMSE=0.7785  MAE=0.6223  R2=0.1054

--- mlflow.evaluate auto-logged metrics ---
  example_count             = 400.000000
  max_error                 = 2.184400
  mean_absolute_error       = 0.622300
  mean_absolute_percentage_error = 0.108700
  mean_on_target            = 5.745000
  mean_squared_error        = 0.605900
  r2_score                  = 0.105400
  root_mean_squared_error   = 0.778500
  sum_on_target             = 2298.000000

--- mlflow.evaluate generated artifacts ---
  eval_results_table.json -> mlflow-artifacts:/.../eval_results_table.json
  per_class_metrics.csv    -> ...    (only for classification)
```

Notice `root_mean_squared_error` matches our manual `rmse`. Same for `mean_absolute_error` vs `mae`, etc. — `evaluate` is doing exactly what we did by hand, plus more.

In the UI ([http://localhost:5000](http://localhost:5000)) → **`experiment_custom_sklearn`** → latest run:

- **Metrics tab**: 9+ new metrics from `evaluate` + your 3 manual `rmse`/`mae`/`r2`.
- **Artifacts tab**:
  ```text
  eval_results_table.json    ← row-by-row predictions vs targets
  sklearn_mlflow_pyfunc/     ← the pyfunc model
  ```
  Click `eval_results_table.json` → see every test-set row with its prediction, ready for an analyst.

> [!NOTE]
> Install `shap` in the trainer image and re-run to also get **`shap_summary_plot.png`**, **`shap_beeswarm_plot.png`**, etc. — feature-attribution plots are auto-generated when SHAP is available. Adds ~150 MB to the image, so it's an opt-in.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Bonus — the `EvaluationResult` object

`mlflow.evaluate` returns an `EvaluationResult` you can inspect programmatically:

```python
result = mlflow.evaluate(...)

# Dict of {metric_name: float}
result.metrics             # {'r2_score': 0.1054, 'mean_squared_error': 0.6059, ...}

# Dict of {artifact_name: EvaluationArtifact} — each has .uri and .content()
result.artifacts           # {'eval_results_table.json': EvaluationArtifact(...)}

# The eval table as a DataFrame, ready for further analysis
result.tables["eval_results_table"]    # DataFrame with prediction + target + features
```

This makes `mlflow.evaluate` perfect for **CI/CD model promotion gates**:

```python
result = mlflow.evaluate(model_uri, test, targets="quality", model_type="regressor")
assert result.metrics["root_mean_squared_error"] < 0.85, "Model regressed!"
```

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

You replaced 5 lines of manual metric computation with **one** `mlflow.evaluate(...)` call and got **9+ metrics + a row-level eval table for free**.

Next: **[chapter 20s](./20s-practical-work-15s-mlflow-step-by-step-recap-mlflow-evaluate-custom-metrics-and-scatter-artifact.md)** — extend `mlflow.evaluate` with **custom metrics** (`make_metric`) and **custom visual artifacts** (a matplotlib scatter plot of predictions vs targets).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20r — mlflow.evaluate with default regressor evaluator</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

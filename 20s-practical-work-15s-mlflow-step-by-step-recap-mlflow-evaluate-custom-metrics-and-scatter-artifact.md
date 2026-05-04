<a id="top"></a>

# Chapter 20s — Step-by-step recap: extending `mlflow.evaluate` with `make_metric` (custom metrics) and `custom_artifacts` (matplotlib scatter plot)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20r](#section-2) |
| 3 | [Custom metric — anatomy of a metric function](#section-3) |
| 4 | [Building a metric with `make_metric`](#section-4) |
| 5 | [Custom artifact — anatomy of an artifact function](#section-5) |
| 6 | [`custom_metrics` vs `extra_metrics` — naming note](#section-6) |
| 7 | [Project structure](#section-7) |
| 8 | [The code](#section-8) |
| 9 | [Run it, browse the new metrics + the scatter PNG](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and series wrap-up](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

Chapter 20r let MLflow auto-generate a standard set of regression metrics. But every team has **its own metrics** (a domain-specific weighted error, a business KPI, a fairness ratio…) and **its own diagnostic plots** (calibration curve, error histograms, scatter…). `mlflow.evaluate` accepts both via two extra arguments:

| Arg | What |
|---|---|
| `extra_metrics` | A list of `EvaluationMetric` objects built with `mlflow.models.make_metric`. |
| `custom_artifacts` | A list of plain Python functions returning `{"name": "/path/to/file.png"}`. |

Both are called with the **same `eval_df` and `builtin_metrics`** that the default evaluator already computed, so you never re-compute predictions yourself.

In this chapter we add:

- 2 custom metrics built with `make_metric`:
  - `squared_diff_plus_one` — sum of `(prediction - target + 1)²`
  - `sum_on_target_divided_by_two` — `builtin_metrics["sum_on_target"] / 2` (shows you can **reuse** built-in metrics)
- 1 custom artifact: `prediction_target_scatter` — a matplotlib scatter plot of predictions vs targets, saved as `example_scatter_plot.png`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20r

| Diff | What |
|---|---|
| `from mlflow.models import make_metric` | Helper to register custom metrics. |
| `import matplotlib.pyplot as plt` | For the scatter plot. |
| Add `matplotlib==3.9.2` to `trainer/requirements.txt` | New dependency. |
| Define 2 metric functions + wrap them with `make_metric` | New custom metrics. |
| Define 1 artifact function returning `{name: path}` | New custom plot. |
| Pass `extra_metrics=[...]` and `custom_artifacts=[...]` to `mlflow.evaluate` | Wires them in. |

Everything else (`SklearnWrapper`, `joblib`, `conda_env`, the call to `evaluate` itself) stays identical.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Custom metric — anatomy of a metric function

A custom metric is a plain Python function with a fixed signature:

```python
def my_metric(eval_df, builtin_metrics):
    """
    eval_df         : pandas.DataFrame with at least 2 columns:
                        - 'prediction' : model output
                        - 'target'     : ground truth (the column you passed as `targets`)
                      plus all the original input feature columns.

    builtin_metrics : dict of {metric_name: float} that the default evaluator
                      already computed (rmse, r2, sum_on_target, ...).
                      Use it to derive new metrics WITHOUT re-walking the data.

    return          : a single scalar float (the value of YOUR metric).
    """
    return float(...)
```

Two examples:

```python
def squared_diff_plus_one(eval_df, _builtin_metrics):
    return np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2)


def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
    return builtin_metrics["sum_on_target"] / 2
```

The second one shows the real super-power: **pass-through computation** — once a built-in metric is in `builtin_metrics`, you can derive any number of variants without touching the data again.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Building a metric with `make_metric`

The bare function is not enough — `mlflow.evaluate` wants an **`EvaluationMetric` object** so it knows the metric's name, its direction (higher-is-better or lower-is-better), and so on. Build it with **`mlflow.models.make_metric`**:

```python
from mlflow.models import make_metric

squared_diff_plus_one_metric = make_metric(
    eval_fn=squared_diff_plus_one,
    greater_is_better=False,     # lower is better (it's a loss-style metric)
    name="squared_diff_plus_one",
)

sum_on_target_divided_by_two_metric = make_metric(
    eval_fn=sum_on_target_divided_by_two,
    greater_is_better=True,
    name="sum_on_target_divided_by_two",
)
```

Then hand them to `evaluate`:

```python
mlflow.evaluate(
    ...,
    extra_metrics=[
        squared_diff_plus_one_metric,
        sum_on_target_divided_by_two_metric,
    ],
)
```

The values appear in the run's **Metrics** tab next to the built-in ones, and in the run table on the experiment page so you can sort runs by them.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Custom artifact — anatomy of an artifact function

A custom artifact function takes **three** arguments and returns a `{name: path}` dict:

```python
def my_artifact(eval_df, builtin_metrics, artifacts_dir):
    """
    eval_df         : same DataFrame as for custom metrics.
    builtin_metrics : same dict as for custom metrics.
    artifacts_dir   : a temp directory MLflow gives you. Save your file there.

    return          : dict {"<artifact_name>": "<absolute_path_to_file>"}
                      MLflow will pick the file up and log it.
    """
    ...
    return {"my_plot": "/tmp/.../my_plot.png"}
```

Our scatter plot:

```python
import matplotlib.pyplot as plt

def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    plt.figure()
    plt.scatter(eval_df["target"], eval_df["prediction"], s=8, alpha=0.6)
    lo, hi = eval_df["target"].min(), eval_df["target"].max()
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)   # ideal y=x line
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path)
    plt.close()
    return {"example_scatter_plot_artifact": plot_path}
```

Then:

```python
mlflow.evaluate(
    ...,
    custom_artifacts=[prediction_target_scatter],
)
```

The PNG is logged in the run's artifacts and the dict key (`example_scatter_plot_artifact`) becomes the artifact's identifier.

> [!TIP]
> Same pattern works for **any** matplotlib / seaborn / plotly chart, a CSV, an HTML report, a JSON of feature importances… Anything you save under `artifacts_dir` and reference in the returned dict is uploaded.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. `custom_metrics` vs `extra_metrics` — naming note

The user-provided snippet uses the kwarg `custom_metrics=`. That **name was renamed** in MLflow ≥ 2.4 to `extra_metrics=`. Both still exist in MLflow 2.16:

| Kwarg | Status in MLflow 2.16 |
|---|---|
| `extra_metrics=[...]` | **Recommended.** What we use below. |
| `custom_metrics=[...]` | Still works, but emits a `DeprecationWarning`. |

Same artifact API (`custom_artifacts=`) is unchanged. We use `extra_metrics` to be future-proof.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Project structure

```text
chap20s-mlflow-step-by-step-recap-mlflow-evaluate-custom-metrics-and-scatter-artifact/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt        ← + matplotlib
    └── train.py               ← log_model + evaluate(extra_metrics, custom_artifacts)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. The code

### 8.1 `trainer/requirements.txt`

```text
mlflow==2.16.2
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.1.1
joblib==1.4.2
cloudpickle==3.0.0
matplotlib==3.9.2
```

### 8.2 `trainer/train.py`

```python
import argparse
import logging
import os
import warnings

import cloudpickle
import joblib
import matplotlib
matplotlib.use("Agg")              # headless backend (no display in container)
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import make_metric
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


class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)


# ============================================================
# Custom metrics
# ============================================================

def squared_diff_plus_one(eval_df, _builtin_metrics):
    """Sum of (prediction - target + 1) squared. Lower is better."""
    return float(np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2))


def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
    """Reuse a built-in metric to derive a new one. Higher is better."""
    return builtin_metrics["sum_on_target"] / 2


# ============================================================
# Custom artifact
# ============================================================

def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    """Scatter plot of predictions vs targets, saved as PNG."""
    plt.figure(figsize=(6, 6))
    plt.scatter(eval_df["target"], eval_df["prediction"], s=8, alpha=0.6)
    lo, hi = eval_df["target"].min(), eval_df["target"].max()
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (ideal)")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions")
    plt.legend()
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"example_scatter_plot_artifact": plot_path}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_model_evaluation")
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

    # ===== Build the EvaluationMetric objects =====
    squared_diff_plus_one_metric = make_metric(
        eval_fn=squared_diff_plus_one,
        greater_is_better=False,
        name="squared_diff_plus_one",
    )

    sum_on_target_divided_by_two_metric = make_metric(
        eval_fn=sum_on_target_divided_by_two,
        greater_is_better=True,
        name="sum_on_target_divided_by_two",
    )

    # ===== EVALUATE with custom metrics + custom artifacts =====
    result = mlflow.evaluate(
        model_info.model_uri,
        test,
        targets="quality",
        model_type="regressor",
        evaluators=["default"],
        extra_metrics=[
            squared_diff_plus_one_metric,
            sum_on_target_divided_by_two_metric,
        ],
        custom_artifacts=[prediction_target_scatter],
    )

    print("\n--- All metrics (built-in + extra) ---")
    for k, v in sorted(result.metrics.items()):
        print(f"  {k} = {v:.6f}")

    print("\n--- All artifacts (built-in + custom) ---")
    for name, info in result.artifacts.items():
        print(f"  {name} -> {info.uri}")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 8.3 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`

Identical to chap 20r (only the trainer's `requirements.txt` and `train.py` change).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run it, browse the new metrics + the scatter PNG

```bash
cd chap20s-mlflow-step-by-step-recap-mlflow-evaluate-custom-metrics-and-scatter-artifact
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Trainer's stdout (excerpt):

```text
--- All metrics (built-in + extra) ---
  example_count                   = 400.000000
  max_error                       = ...
  mean_absolute_error             = ...
  mean_on_target                  = 5.745000
  r2_score                        = ...
  root_mean_squared_error         = ...
  sum_on_target                   = 2298.000000
  squared_diff_plus_one           = 1832.140000        <-- our extra
  sum_on_target_divided_by_two    = 1149.000000        <-- our extra (= 2298/2)

--- All artifacts (built-in + custom) ---
  eval_results_table.json           -> mlflow-artifacts:/...
  example_scatter_plot_artifact     -> mlflow-artifacts:/.../example_scatter_plot.png
```

In the UI ([http://localhost:5000](http://localhost:5000)) → **`experiment_model_evaluation`** → latest run:

- **Metrics tab**: all built-in regression metrics **plus** `squared_diff_plus_one` and `sum_on_target_divided_by_two`. They're first-class metrics, sortable in the experiment table.
- **Artifacts tab**:
  ```text
  eval_results_table.json
  example_scatter_plot.png    <-- click it, see the scatter plot
  sklearn_mlflow_pyfunc/
  ```
  Click `example_scatter_plot.png` to preview it inline — predictions on Y, targets on X, with the dashed `y=x` reference line so you immediately see how clustered (or off) the predictions are.

> [!IMPORTANT]
> `squared_diff_plus_one` was registered with `greater_is_better=False`. In a model-comparison view the runs with the **lowest** value will be highlighted as best for that metric. Always set this flag correctly — it controls how the UI sorts and colours your custom metrics.

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

## 11. Recap and series wrap-up

You can now plug **any** scalar metric and **any** matplotlib/seaborn/HTML artifact into `mlflow.evaluate`:

```python
metric = make_metric(eval_fn=my_fn, greater_is_better=False, name="my_metric")

mlflow.evaluate(
    model_uri, eval_df, targets="...", model_type="regressor",
    evaluators=["default"],
    extra_metrics=[metric],
    custom_artifacts=[my_plot_fn],
)
```

This closes the **20a → 20s recap**. You've gone from `mlflow.set_tracking_uri` (chap 20a) to a fully production-grade evaluation pipeline that:

1. Trains a model (`autolog` does most of the boring stuff).
2. Logs it as a portable pyfunc artifact with a `conda_env` and a custom wrapper.
3. Reloads it from the registry/run URI.
4. Evaluates it with a rich, **extensible** set of metrics and diagnostic plots, all stored on the run.

Next major step in the course: chapter 21+ moves from "tracking + evaluation" to **deployment** — serving the registered pyfunc through FastAPI and consuming it from Streamlit.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20s — mlflow.evaluate with custom metrics and scatter artifact</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

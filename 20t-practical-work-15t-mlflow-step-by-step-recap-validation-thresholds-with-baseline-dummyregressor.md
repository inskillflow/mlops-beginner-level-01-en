<a id="top"></a>

# Chapter 20t — Step-by-step recap: `mlflow.evaluate` with `validation_thresholds` (`MetricThreshold`) and a `baseline_model` (`DummyRegressor`)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20s](#section-2) |
| 3 | [Why a baseline model?](#section-3) |
| 4 | [Anatomy of a `MetricThreshold`](#section-4) |
| 5 | [How `mlflow.evaluate` enforces the thresholds](#section-5) |
| 6 | [The `SklearnWrapper` becomes parameterised](#section-6) |
| 7 | [Project structure](#section-7) |
| 8 | [The code](#section-8) |
| 9 | [Run it — pass case and failure case](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Recap and next chapter](#section-11) |

---

<a id="section-1"></a>

## 1. Objective

In chap 20s we extended `mlflow.evaluate` with **custom metrics** and a **scatter artifact**. Today we add the **decision-gate** features:

- A **baseline model** (`sklearn.dummy.DummyRegressor`) — a "no-brain" model whose only job is to predict the mean (or median) of the target. Any real model that doesn't beat it is broken.
- **`MetricThreshold`** rules describing **how much better than the baseline** the candidate model must be (in absolute and relative terms) and an **absolute upper limit** on the error.

When fed both, `mlflow.evaluate` either:

1. **Passes silently** — all thresholds satisfied → training run succeeds.
2. **Raises `ModelValidationFailedException`** — at least one threshold is violated → the run is marked FAILED in the UI and your CI/CD pipeline halts.

This is the foundation of automated model-promotion gates: "only register the model in `Production` if it beats the current baseline by ≥10% RMSE".

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20s

| Diff | What |
|---|---|
| `from sklearn.dummy import DummyRegressor` | The "no-brain" baseline model. |
| `from mlflow.models import MetricThreshold` | The threshold descriptor. |
| Train + log a **second** pyfunc model (`baseline_sklearn_mlflow_pyfunc`) | The reference for `evaluate` to beat. |
| `SklearnWrapper.__init__(self, artifacts_name)` becomes parameterised | Same wrapper class is reused for both candidate and baseline (each with a different artifact key). |
| Build a `thresholds = {"<metric>": MetricThreshold(...)}` dict | One threshold per metric to enforce. |
| Pass `validation_thresholds=thresholds` and `baseline_model=baseline_uri` to `evaluate` | Activates the gate. |

Everything else (`autolog`, `extra_metrics`, `custom_artifacts`, `set_tags`) is identical to chap 20s.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Why a baseline model?

Raw thresholds (e.g. "RMSE < 0.85") are dangerous: maybe the dataset is so easy that a constant predictor reaches 0.84. Your model would technically pass, but it learned nothing.

A **baseline** sets a floor that is automatically calibrated to the dataset:

```python
from sklearn.dummy import DummyRegressor

baseline_model = DummyRegressor()        # default: predicts the mean of y_train
baseline_model.fit(train_x, train_y)
```

Then you ask: "is my ElasticNet meaningfully better than 'always predict the mean'?" That's a much stronger guarantee.

> [!TIP]
> Variants useful in practice:
> - `DummyRegressor(strategy="median")` — predicts the median (robust to outliers).
> - `DummyRegressor(strategy="quantile", quantile=0.9)` — predicts the 90th percentile (e.g. for a "safe" delivery-time forecast).
> - `DummyRegressor(strategy="constant", constant=5.0)` — predicts a fixed value (your business rule).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Anatomy of a `MetricThreshold`

```python
from mlflow.models import MetricThreshold

MetricThreshold(
    threshold=0.6,              # absolute upper limit (or lower, if greater_is_better=True)
    min_absolute_change=0.1,    # candidate must improve by AT LEAST 0.1 in absolute terms
    min_relative_change=0.05,   # ... AND by at least 5% relative to the baseline
    greater_is_better=False,    # for MSE, RMSE, MAE: lower is better
)
```

What each field controls:

| Field | What it asserts |
|---|---|
| `threshold` | Hard absolute limit. For an "lower-is-better" metric like MSE: candidate's MSE ≤ `threshold`. |
| `min_absolute_change` | `\|candidate_metric - baseline_metric\|` must be ≥ this value, in the right direction. |
| `min_relative_change` | The same comparison but as a fraction (`0.05` = 5%). |
| `greater_is_better` | Tells MLflow which direction "better" is. Wrong value → wrong validation. |

All three checks run; **all must pass** for the metric to be considered satisfied. You can supply only some — e.g. `threshold=0.6` alone, or `min_relative_change=0.05` alone — by leaving the others as `None`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. How `mlflow.evaluate` enforces the thresholds

```python
mlflow.evaluate(
    model_info.model_uri,            # candidate model
    test,
    targets="quality",
    model_type="regressor",
    evaluators=["default"],
    extra_metrics=[...],
    custom_artifacts=[...],
    validation_thresholds={
        "mean_squared_error": MetricThreshold(
            threshold=0.6, min_absolute_change=0.1,
            min_relative_change=0.05, greater_is_better=False
        ),
    },
    baseline_model=baseline_model_uri,   # the model to beat
)
```

Internally:

1. MLflow scores **both** the candidate and the baseline on `test`.
2. For every metric in `validation_thresholds`, it checks:
   - candidate value vs `threshold`
   - candidate vs baseline → satisfies `min_absolute_change`?
   - candidate vs baseline → satisfies `min_relative_change`?
3. If everything passes → the function returns normally.
4. If anything fails → it raises **`mlflow.models.evaluation.validation.ModelValidationFailedException`** with a message describing exactly which threshold(s) failed and by how much.

The two model artifacts (candidate + baseline) are both logged inside the run, so the UI shows two columns of metrics side-by-side for visual comparison.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The `SklearnWrapper` becomes parameterised

In chap 20p the wrapper hardcoded `context.artifacts["sklearn_model"]`. Today we log **two** models (candidate + baseline), so each needs its own artifact key. We make the artifact key a constructor argument:

```python
class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, artifact_name):
        self.artifact_name = artifact_name

    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts[self.artifact_name])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)
```

Then we instantiate it twice with different keys:

```python
mlflow.pyfunc.log_model(
    artifact_path="sklearn_mlflow_pyfunc",
    python_model=SklearnWrapper("sklearn_model"),
    artifacts={"sklearn_model": "sklearn_model.pkl", "data": data_dir},
    ...
)

mlflow.pyfunc.log_model(
    artifact_path="baseline_sklearn_mlflow_pyfunc",
    python_model=SklearnWrapper("baseline_sklearn_model"),
    artifacts={"baseline_sklearn_model": "baseline_sklearn_model.pkl"},
    ...
)
```

> [!IMPORTANT]
> When `__init__` takes arguments, MLflow needs to be able to pickle the wrapper instance with `cloudpickle`. Plain primitives like strings work out of the box. Avoid storing non-pickleable objects (open file handles, sockets, Spark sessions…) in `__init__`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Project structure

```text
chap20t-mlflow-step-by-step-recap-validation-thresholds-with-baseline-dummyregressor/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py
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

import cloudpickle
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import MetricThreshold, make_metric
from sklearn.dummy import DummyRegressor
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
    """Reusable wrapper. The artifact_name lets the same class wrap
    multiple models in the same run (candidate + baseline)."""

    def __init__(self, artifact_name):
        self.artifact_name = artifact_name

    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts[self.artifact_name])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)


# ===== custom metrics + artifact (same as chap 20s) =====

def squared_diff_plus_one(eval_df, _builtin_metrics):
    return float(np.sum(np.abs(eval_df["prediction"] - eval_df["target"] + 1) ** 2))


def sum_on_target_divided_by_two(_eval_df, builtin_metrics):
    return builtin_metrics["sum_on_target"] / 2


def prediction_target_scatter(eval_df, _builtin_metrics, artifacts_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(eval_df["target"], eval_df["prediction"], s=8, alpha=0.6)
    lo, hi = eval_df["target"].min(), eval_df["target"].max()
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (ideal)")
    plt.xlabel("Targets"); plt.ylabel("Predictions")
    plt.title("Targets vs. Predictions"); plt.legend()
    plot_path = os.path.join(artifacts_dir, "example_scatter_plot.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"example_scatter_plot_artifact": plot_path}


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

    # ===== Candidate: ElasticNet =====
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    candidate_pred = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, candidate_pred)

    print(f"Candidate ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    # ===== Baseline: DummyRegressor (predicts the mean of y_train) =====
    baseline_model = DummyRegressor()
    baseline_model.fit(train_x, train_y)
    baseline_pred = baseline_model.predict(test_x)
    bl_rmse, bl_mae, bl_r2 = eval_metrics(test_y, baseline_pred)

    print("Baseline DummyRegressor (predicts mean)")
    print(f"  RMSE={bl_rmse:.4f}  MAE={bl_mae:.4f}  R2={bl_r2:.4f}")

    mlflow.log_metrics({
        "baseline_rmse": bl_rmse,
        "baseline_mae":  bl_mae,
        "baseline_r2":   bl_r2,
    })

    # ===== Serialize both models to disk for the wrappers =====
    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)

    baseline_sklearn_model_path = "baseline_sklearn_model.pkl"
    joblib.dump(baseline_model, baseline_sklearn_model_path)

    candidate_artifacts = {"sklearn_model":          sklearn_model_path,
                           "data":                   data_dir}
    baseline_artifacts  = {"baseline_sklearn_model": baseline_sklearn_model_path}

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

    candidate_info = mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper("sklearn_model"),
        artifacts=candidate_artifacts,
        code_path=["train.py"],
        conda_env=conda_env,
    )
    print("Candidate URI:", candidate_info.model_uri)

    baseline_info = mlflow.pyfunc.log_model(
        artifact_path="baseline_sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper("baseline_sklearn_model"),
        artifacts=baseline_artifacts,
        code_path=["train.py"],
        conda_env=conda_env,
    )
    print("Baseline URI :", baseline_info.model_uri)

    # ===== Custom metrics built with make_metric =====
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

    # ===== Validation thresholds (the gate) =====
    thresholds = {
        "mean_squared_error": MetricThreshold(
            threshold=0.6,                # absolute upper limit on candidate's MSE
            min_absolute_change=0.1,      # must beat baseline by at least 0.1
            min_relative_change=0.05,     # ... and by at least 5%
            greater_is_better=False,      # lower is better
        ),
    }

    # ===== EVALUATE with thresholds + baseline =====
    try:
        result = mlflow.evaluate(
            candidate_info.model_uri,
            test,
            targets="quality",
            model_type="regressor",
            evaluators=["default"],
            extra_metrics=[
                squared_diff_plus_one_metric,
                sum_on_target_divided_by_two_metric,
            ],
            custom_artifacts=[prediction_target_scatter],
            validation_thresholds=thresholds,
            baseline_model=baseline_info.model_uri,
        )
        print("\nVALIDATION PASSED. Candidate cleared all thresholds vs baseline.")
        for k, v in sorted(result.metrics.items()):
            print(f"  {k} = {v:.6f}")

    except Exception as e:
        # mlflow.models.evaluation.validation.ModelValidationFailedException
        # at runtime, but we catch broadly to keep the lesson self-contained.
        print("\nVALIDATION FAILED:")
        print(str(e))
        # Re-raise in real CI/CD; here we let the run end so the UI shows the data.
        raise
    finally:
        mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
```

### 8.2 `docker-compose.yml`, `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt`

Identical to chap 20s (matplotlib + joblib + cloudpickle + mlflow).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run it — pass case and failure case

### 9.1 Pass case (default args)

```bash
cd chap20t-mlflow-step-by-step-recap-validation-thresholds-with-baseline-dummyregressor
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
```

Expected (when ElasticNet beats the dummy by ≥ 5% MSE and is below 0.6):

```text
Candidate ElasticNet ... RMSE=0.7785 MAE=0.6223 R2=0.1054   -> MSE ~0.606
Baseline DummyRegressor RMSE=0.8350 MAE=0.6800 R2=-0.0023   -> MSE ~0.697

VALIDATION PASSED. Candidate cleared all thresholds vs baseline.
```

> Note: depending on the random split, the exact numbers vary. If MSE 0.606 is just slightly above 0.6 you'll fail the threshold — that's the point of the gate.

### 9.2 Forced failure (over-regularised candidate)

```bash
docker compose run --rm trainer --alpha 5.0 --l1_ratio 1.0
```

`alpha=5.0` makes ElasticNet predict almost a constant. It barely beats the dummy → at least one threshold fails:

```text
Candidate ElasticNet ... RMSE=0.83 ... MSE ~0.69
Baseline DummyRegressor RMSE=0.83 ... MSE ~0.70

VALIDATION FAILED:
Model validation failed for the following thresholds:
  Metric 'mean_squared_error': absolute change between candidate and baseline
  is 0.01, which is less than the required 0.1.
```

Run is marked as **FAILED** in the UI — exactly what a CI/CD pipeline needs.

### 9.3 In the UI

[http://localhost:5000](http://localhost:5000) → **`experiment_model_evaluation`** → latest run:

- **Metrics tab**: candidate metrics + `baseline_*` metrics + the metrics auto-logged by `evaluate` for the candidate + the baseline (suffixed `_on_baseline_model`).
- **Artifacts tab**: `sklearn_mlflow_pyfunc/`, `baseline_sklearn_mlflow_pyfunc/`, `eval_results_table.json`, `example_scatter_plot.png`.

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

You can now make `mlflow.evaluate` **fail loudly** when a candidate model isn't strictly better than a baseline:

```python
mlflow.evaluate(
    candidate_uri, test, targets="quality", model_type="regressor",
    validation_thresholds={"mean_squared_error": MetricThreshold(...)},
    baseline_model=baseline_uri,
)
# → ModelValidationFailedException if any threshold fails
```

A single `try/except` turns this into a CI/CD-friendly model-promotion gate.

Next: **[chapter 20u](./20u-practical-work-15u-mlflow-step-by-step-recap-registered-model-name-with-mlflow-sklearn-log-model.md)** — once you trust your model, register it in the **MLflow Model Registry** with `mlflow.sklearn.log_model(..., registered_model_name="elasticnet-api")` so it can be loaded by alias (`models:/elasticnet-api/Production`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20t — validation thresholds with a baseline DummyRegressor</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

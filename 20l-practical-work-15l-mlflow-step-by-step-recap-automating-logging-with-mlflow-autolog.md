<a id="top"></a>

# Chapter 20l — Step-by-step recap: automating everything with **`mlflow.autolog()`**

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20k](#section-2) |
| 3 | [What `mlflow.autolog()` actually does](#section-3) |
| 4 | [Manual vs autolog — side by side](#section-4) |
| 5 | [`autolog()` vs `mlflow.sklearn.autolog()`](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, see what landed in the UI for free](#section-8) |
| 9 | [What autolog **doesn't** do (and how to bridge it)](#section-9) |
| 10 | [Mini exercise — autolog a Ridge run](#section-10) |
| 11 | [Tear down](#section-11) |
| 12 | [Recap and what's next](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Until now, every chapter explicitly called `log_param`, `log_metric`, `log_model`, etc. That's good for learning the pieces — and **tedious** in real projects.

**`mlflow.autolog()`** changes that. Add **one line** before your `.fit()` and MLflow will log:

- All the model's **hyperparameters** (every `__init__` arg, even the defaults you didn't touch).
- The **training metrics** sklearn computes (e.g. score on the train set).
- The **fitted model**, with a **signature** and an **input example**.
- A handful of useful **system tags**.

Today we use it to replace ~30 lines of explicit logging with **one line + one `.fit()`**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20k

| Diff | What |
|---|---|
| Add `mlflow.autolog(log_input_examples=True)` | The whole automation switch. |
| Drop `log_params`, `log_metrics`, `log_model` calls | Autolog handles them. |
| Keep `set_tags(...)` and `log_artifacts("data/")` (or `log_artifact`) | Autolog **doesn't** know about your CSVs or your custom tags. |
| Switch to ONE experiment `experiment_autolog` with ONE run | Easier to demonstrate the comparison. |
| Print the run's auto-logged params + metrics from `last_active_run().data` | Show what landed there without us asking. |

Everything else (env-var URI, multi-service Docker, imperative `start_run`/`end_run`, `last_active_run`) is identical to chap 20k.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What `mlflow.autolog()` actually does

When you call `mlflow.autolog()`, MLflow **monkey-patches** the `.fit()` methods of every supported framework it can find (sklearn, XGBoost, LightGBM, PyTorch, Keras, …). The patched `.fit()`:

1. Detects the **active run** (or starts a new one if there isn't one).
2. Reads the model's hyperparameters via `model.get_params()` → `mlflow.log_params(...)`.
3. Records training-time metrics → `mlflow.log_metrics(...)`.
4. Logs the fitted model with `mlflow.<flavor>.log_model(...)`.
5. Optionally, infers a **signature** and an **input example** and attaches them.

You only opt out of pieces you don't want, with kwargs:

```python
mlflow.autolog(
    log_input_examples=True,    # default: False  → grab the first row(s) as example
    log_model_signatures=True,  # default: True   → infer Schema from the fit data
    log_models=True,            # default: True   → include the fitted estimator
    disable=False,              # set True to turn it off
    silent=False,               # set True to mute autolog's own logs
)
```

> [!IMPORTANT]
> Autolog **will not** override anything you log manually. If you call `mlflow.log_metric("rmse", x)` and autolog also logged `training_score`, both end up in the run. Same for params. Use this to add the metrics autolog can't compute (validation/test metrics).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Manual vs autolog — side by side

```python
# ===== Manual (chap 20i style) =====
mlflow.start_run()
mlflow.set_tags(COMMON_TAGS)
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)
preds = lr.predict(test_x)
rmse, mae, r2 = eval_metrics(test_y, preds)
mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
mlflow.sklearn.log_model(lr, "my_new_model_1")
mlflow.log_artifact("data/red-wine-quality.csv")
mlflow.end_run()

# ===== Autolog (today) =====
mlflow.start_run()
mlflow.set_tags(COMMON_TAGS)
mlflow.autolog(log_input_examples=True)
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)
# (preds + manual eval stay if you want test metrics)
mlflow.log_artifact("data/red-wine-quality.csv")  # autolog doesn't know about this
mlflow.end_run()
```

Same UI result for params and model. Plus, autolog gives you **for free**:

- All sklearn defaults (`fit_intercept`, `selection`, `max_iter`, `precompute`, …).
- A `training_score`, `training_mean_squared_error`, etc.
- A model **signature** (input/output schema) and an **input example** (a real row from `train_x`).
- The system tags `mlflow.autologging` and `mlflow.source.type`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. `autolog()` vs `mlflow.sklearn.autolog()`

Two flavours, slightly different scope:

| Function | What it patches |
|---|---|
| **`mlflow.autolog()`** | A best-effort superset: scans your environment and patches **every supported framework it finds** (sklearn, XGBoost, LightGBM, PyTorch, TF/Keras, statsmodels, FastAI, Spark MLlib…). |
| **`mlflow.sklearn.autolog()`** | Only sklearn. Lighter import. Same kwargs as the generic one. |

Both accept the same `log_input_examples=`, `log_model_signatures=`, `log_models=` kwargs.

> [!TIP]
> Use **framework-specific** autolog (`mlflow.sklearn.autolog()`) in production scripts — it's faster to import and you avoid surprises if another framework you happen to import also has its `.fit()` patched. Use the generic `mlflow.autolog()` for quick exploration.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20l-mlflow-step-by-step-recap-automating-logging-with-mlflow-autolog/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py            ← + mlflow.autolog(log_input_examples=True)
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

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
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

    exp = mlflow.set_experiment(experiment_name="experiment_autolog")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")
    print(f"Artifact Location : {exp.artifact_location}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    mlflow.set_tags({
        "engineering":       "ML platform",
        "release.candidate": "RC1",
        "release.version":   "2.0",
    })

    # ===== THE WHOLE POINT OF THIS CHAPTER =====
    mlflow.autolog(log_input_examples=True)

    # No more log_param / log_metric / log_model in this section!
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)            # autolog records params + model + signature

    # Test-set metrics: autolog doesn't compute these, so do them manually
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2, "test_mae": mae})

    # Autolog ignores arbitrary files; log the input CSV by hand
    mlflow.log_artifact("data/red-wine-quality.csv")

    print("Artifact path:", mlflow.get_artifact_uri())
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"\nActive run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")

    print("\n--- Auto-logged PARAMS (subset) ---")
    for k in sorted(run.data.params)[:8]:        # show first 8 to keep it short
        print(f"  {k} = {run.data.params[k]}")
    print(f"  ... ({len(run.data.params)} params in total)")

    print("\n--- Auto-logged METRICS ---")
    for k, v in run.data.metrics.items():
        print(f"  {k} = {v}")
```

### 7.2 `docker-compose.yml`

Same skeleton as 20j/k, only `container_name`s change:

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20l
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
    container_name: trainer-recap-20l
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

### 7.3 `mlflow/Dockerfile`, `trainer/Dockerfile`, `trainer/requirements.txt`

Identical to chap 20j/k.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, see what landed in the UI for free

```bash
cd chap20l-mlflow-step-by-step-recap-automating-logging-with-mlflow-autolog
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Trainer's stdout (truncated):

```text
The set tracking URI is http://mlflow:5000
Name              : experiment_autolog
Experiment_id     : 1
2026/05/04 20:44:01 INFO mlflow.tracking.fluent: Autologging successfully enabled for sklearn.
  RMSE=0.7800  MAE=0.6200  R2=0.1000
Artifact path: mlflow-artifacts:/1/8a4f...d1/artifacts

Active run id   : 8a4f...d1
Active run name : agreeable-eel-19

--- Auto-logged PARAMS (subset) ---
  alpha = 0.7
  copy_X = True
  fit_intercept = True
  l1_ratio = 0.7
  max_iter = 1000
  positive = False
  precompute = False
  random_state = 42
  ... (12 params in total)

--- Auto-logged METRICS ---
  training_score = 0.27...
  training_mean_squared_error = 0.55...
  training_mean_absolute_error = 0.59...
  training_r2_score = 0.27...
  training_root_mean_squared_error = 0.74...
  test_rmse = 0.78
  test_r2 = 0.10
  test_mae = 0.62
```

Notice **two batches of metrics**: the `training_*` ones came from autolog; the `test_*` ones from our manual `log_metrics`.

In the UI ([http://localhost:5000](http://localhost:5000)), open **`experiment_autolog`** → the run:

- **Parameters** tab → all **12** ElasticNet hyperparameters (not just the 2 we cared about).
- **Metrics** tab → the `training_*` set + our `test_*` set.
- **Artifacts** tab:
  - `model/` (the auto-logged model with `MLmodel`, `conda.yaml`, `model.pkl`, **`signature.json`**, **`input_example.json`**)
  - `red-wine-quality.csv` (our manual `log_artifact`)
- **Tags** tab → our 3 custom tags + `mlflow.autologging = sklearn` + the usual `mlflow.user`, `mlflow.source.*`.

Click on **`model/input_example.json`** in the UI → MLflow shows the **first row** of `train_x` it captured. That single row is enough for downstream tooling to know the model's input schema.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. What autolog **doesn't** do (and how to bridge it)

| Limitation | Workaround |
|---|---|
| **Test/validation metrics** are not computed | Compute them yourself + `mlflow.log_metrics({"test_rmse": ...})` (we do exactly this in the chapter). |
| **Custom artifacts** (CSVs, plots, configs) are ignored | Call `mlflow.log_artifact(path)` or `log_artifacts(folder)` explicitly. |
| **Custom tags** must still be set manually | `mlflow.set_tags({...})` (we keep them). |
| **Multiple `.fit()` calls in the same run** stack their logs | If you cross-validate or fit twice, you'll get duplicate metric histories. Wrap each in its own `start_run()` (or use `nested=True`). |
| **Some frameworks log gigabytes** (e.g. autolog for Keras records every batch) | Tune the `every_n_iter=` kwarg on the framework-specific autolog, or `disable=True` for that framework. |

> [!NOTE]
> A common confusion: **autolog must be enabled BEFORE `.fit()`**. If you call `mlflow.autolog()` *after* the model fit, nothing happens. Put it as early as possible in the script (top of `main()` is fine).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Mini exercise — autolog a Ridge run

Replace the model in `train.py`:

```python
from sklearn.linear_model import Ridge

# ...
mlflow.autolog(log_input_examples=True)

lr = Ridge(alpha=alpha, random_state=42)
lr.fit(train_x, train_y)
```

Re-run:

```bash
docker compose run --rm trainer --alpha 0.5
```

In the UI:

- **Parameters** → all of `Ridge`'s defaults (`solver`, `tol`, `fit_intercept`, …) — you didn't write a single `log_param` for them.
- **Artifacts → model/MLmodel** → it now says `loader_module: mlflow.sklearn` with the right Python class name.

Same train script, totally different model — autolog adapts.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Tear down

```bash
docker compose down       # keep volumes
docker compose down -v    # wipe everything
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap and what's next

You now have, in your toolbox, **every essential MLflow building block** + the automation shortcut:

| Concept | Chapter |
|---|---|
| `set_tracking_uri` + first run | 20a |
| `get_tracking_uri` | 20b |
| Full ElasticNet pipeline | 20c |
| Containerized trainer (with bug) | 20d |
| Fix bug via `MLFLOW_TRACKING_URI` env var | 20e |
| `create_experiment` + custom artifact location + shared volume | 20f |
| `active_run` / `last_active_run` + imperative `start_run`/`end_run` | 20g |
| `log_artifacts` + bulk `log_params` / `log_metrics` + `get_artifact_uri` | 20h |
| `set_tags` for searchable metadata | 20i |
| Multiple **runs** in one experiment (helper + `for` loop) | 20j |
| Multiple **experiments** (model factories) | 20k |
| **`mlflow.autolog()`** to automate logging | **20l** ← you are here |

What's left on the main course track:

- **Chapter 14** — model **signature + input example** (manual control over what autolog generates here).
- **Chapter 15** — `mlflow.pyfunc.PythonModel` to ship custom inference logic alongside an sklearn model.
- **Chapter 17** — the **Model Registry** + `MlflowClient` to version, stage and promote models.
- **Chapter 18** — the MLflow **CLI** as a separate Docker service for cleanup, exports and audits.

You have all the prerequisites for any of those — pick the one that solves the next real problem in your project.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20l — automating logging with mlflow.autolog</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

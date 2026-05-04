<a id="top"></a>

# Chapter 20h — Step-by-step recap: `log_artifacts(folder)` + bulk `log_params` / `log_metrics` + `get_artifact_uri`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20g](#section-2) |
| 3 | [What is an artifact?](#section-3) |
| 4 | [`log_artifact` vs `log_artifacts` vs `log_model`](#section-4) |
| 5 | [Bulk versions: `log_params` and `log_metrics`](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, browse the artifacts](#section-8) |
| 9 | [`get_artifact_uri()` — where do my files actually live?](#section-9) |
| 10 | [Bonus — quick CLI commands](#section-10) |
| 11 | [Tear down](#section-11) |
| 12 | [Recap and next chapter](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Three small but very useful additions today:

- **`mlflow.log_artifacts(folder)`** — log **every file inside a folder** in one call (not just one file).
- **`mlflow.log_params(dict)`** and **`mlflow.log_metrics(dict)`** — the **bulk** versions of `log_param` / `log_metric`. One call, many entries.
- **`mlflow.get_artifact_uri()`** — print the storage URI used by the current run, so you can **see where MLflow puts your files**.

The training script also writes the train/test splits to `data/` *before* logging them, demonstrating a realistic workflow: produce → save → log.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20g

| Diff | What |
|---|---|
| Save `train.csv` + `test.csv` to `data/` after `train_test_split` | Realistic intermediate outputs. |
| Replace `log_param("alpha", ...)` / `log_param("l1_ratio", ...)` with `log_params({...})` | One call, many params. |
| Replace `log_metric("rmse", ...)` ×3 with `log_metrics({...})` | One call, many metrics. |
| Add `mlflow.log_artifacts("data/")` | Log the **whole** `data/` folder. |
| Add `mlflow.get_artifact_uri()` and print it | See **exactly** where the files are stored. |
| Switch experiment to `experiment_4` | Just to keep this chapter's runs visually separate. |

Everything else (multi-service Docker, env-var URI, imperative `start_run`/`end_run`, `last_active_run` summary) is **unchanged from 20g**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What is an artifact?

An **artifact** is **any file** produced by your script that you want to **save and find again** in MLflow.

| Example | Type |
|---|---|
| `.csv` | Data table |
| `.pkl`, `.joblib`, `.onnx` | Serialized ML model |
| `.png`, `.jpg`, `.svg` | Plot or visualization |
| `.txt`, `.log`, `.json`, `.html` | Logs / report / dashboard |
| A whole folder | Multi-file output |

If you can write it to disk, you can log it as an artifact. MLflow groups artifacts under each run, so you always know **which run produced which file**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. `log_artifact` vs `log_artifacts` vs `log_model`

| Function | Use it when… | Logs |
|---|---|---|
| `mlflow.log_artifact(path)` | You have **one file** to log | That single file |
| `mlflow.log_artifacts(folder)` | You have **many files in one folder** | Every file inside `folder` (recursive) |
| `mlflow.<flavor>.log_model(model, path)` | You're saving a **model** (sklearn, pyfunc, …) | The model + a `MLmodel` descriptor + (optionally) a signature & input example |

```python
mlflow.log_artifact("courbe_apprentissage.png")      # 1 file
mlflow.log_artifacts("data/")                        # everything in data/
mlflow.sklearn.log_model(lr, "my_new_model_1")       # the trained model
```

All three live under the same run's artifact tree in the UI.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Bulk versions: `log_params` and `log_metrics`

Same data, half the noise:

```python
# Before (one-by-one):
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)

# After (bulk):
mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
```

Functionally identical, but:

- One round-trip to the server instead of N (faster).
- The dict makes the **set of names** visible at a glance — easier code review.
- No risk of mixing the wrong value with the wrong name.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20h-mlflow-step-by-step-recap-log-artifacts-and-bulk-log-params-metrics/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv          ← input
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py                      ← log_artifacts + bulk versions
```

After running, `data/` will also contain `train.csv` and `test.csv` (written by `train.py`).

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

    exp = mlflow.set_experiment(experiment_name="experiment_4")
    print("Name              :", exp.name)
    print("Experiment_id     :", exp.experiment_id)
    print("Artifact Location :", exp.artifact_location)
    print("Tags              :", exp.tags)
    print("Lifecycle_stage   :", exp.lifecycle_stage)
    print("Creation timestamp:", exp.creation_time)

    # ===== Load + split + WRITE intermediate files =====
    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)                    # idempotent
    train.to_csv("data/train.csv", index=False)            # NEW
    test.to_csv("data/test.csv", index=False)              # NEW

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print("Elasticnet (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE:  %s" % mae)
    print("  R2:   %s" % r2)

    # ===== Bulk params & metrics (NEW) =====
    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    # ===== Model (same as before) =====
    mlflow.sklearn.log_model(lr, "my_new_model_1")

    # ===== Folder of artifacts (NEW) =====
    mlflow.log_artifacts("data/")

    # ===== Where did MLflow put them? (NEW) =====
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Active run id   :", run.info.run_id)
    print("Active run name :", run.info.run_name)
```

### 7.2 `docker-compose.yml`

Same as 20g, only `container_name`s change:

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20h
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
    container_name: trainer-recap-20h
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

Identical to chap 20g.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, browse the artifacts

```bash
cd chap20h-mlflow-step-by-step-recap-log-artifacts-and-bulk-log-params-metrics
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.3
```

Verify the files were written on your **host** (thanks to the bind mount):

```bash
ls data/
# → red-wine-quality.csv  train.csv  test.csv
```

In the MLflow UI ([http://localhost:5000](http://localhost:5000)):

1. Open experiment **`experiment_4`** → click the latest run.
2. Tab **Parameters** → `alpha`, `l1_ratio`.
3. Tab **Metrics** → `rmse`, `r2`, `mae`.
4. Tab **Artifacts** → tree:

```text
my_new_model_1/         ← the model (logged by sklearn.log_model)
  MLmodel
  conda.yaml
  model.pkl
  ...
red-wine-quality.csv    ← logged by log_artifacts("data/")
train.csv               ← logged by log_artifacts("data/")
test.csv                ← logged by log_artifacts("data/")
```

> [!TIP]
> Click **`train.csv`** in the artifact tree → MLflow shows a small CSV preview right in the browser. Useful for sanity-checking what you actually shipped.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. `get_artifact_uri()` — where do my files actually live?

In the trainer's stdout you'll see something like:

```text
The artifact path is mlflow-artifacts:/2/8a4f...d1/artifacts
```

Decoded:

- **`mlflow-artifacts:`** — the proxy scheme used by MLflow ≥ 2 when the server stores artifacts itself.
- **`/2`** — `experiment_id`.
- **`/8a4f...d1`** — the `run_id`.
- **`/artifacts`** — the per-run artifact root.

Inside the **`mlflow`** container, those files actually live at `/mlflow/mlruns/2/8a4f...d1/artifacts/`, which is the named volume **`mlflow-artifacts`** declared in the compose file. Want to confirm? Peek inside:

```bash
docker compose exec mlflow ls /mlflow/mlruns
docker compose exec mlflow find /mlflow/mlruns -name "*.csv"
```

> [!NOTE]
> When MLflow returns `mlflow-artifacts:/...`, **don't** treat it as a path on your host. It's a server-side URI. Use the UI, the CLI (`mlflow artifacts download`), or `MlflowClient` to retrieve files.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Bonus — quick CLI commands

The MLflow CLI is just `mlflow` inside a container that has `mlflow` installed. Easiest way: pop a shell in the **trainer** image (no need for a third service):

```bash
# List experiments
docker compose run --rm --entrypoint mlflow trainer experiments search

# List runs of an experiment (use the id you saw in the UI)
docker compose run --rm --entrypoint mlflow trainer runs list --experiment-id 2

# Download just the model from a specific run, into the host's ./data/downloaded/
docker compose run --rm --entrypoint mlflow trainer artifacts download \
    --run-id 8a4f...d1 \
    --artifact-path my_new_model_1 \
    --dst-path /code/data/downloaded
```

> [!IMPORTANT]
> The trainer container already has `MLFLOW_TRACKING_URI=http://mlflow:5000` baked in via the compose `environment:` block, so the CLI talks to the right server without any extra flags.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Tear down

```bash
docker compose down       # keep volumes (and your runs in the UI)
docker compose down -v    # nuke volumes + runs + db
```

To also delete the host-side outputs:

```bash
rm data/train.csv data/test.csv     # PowerShell: Remove-Item data/train.csv,data/test.csv
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap and next chapter

You learned three useful additions:

| API | Replaces | When to use |
|---|---|---|
| `mlflow.log_artifacts(folder)` | Many `log_artifact` calls | A whole folder of outputs |
| `mlflow.log_params(dict)` / `log_metrics(dict)` | Many `log_param` / `log_metric` calls | When you have several at once |
| `mlflow.get_artifact_uri()` | (no equivalent) | To **see** where your files were stored |

Next: **[chapter 20i](./20i-practical-work-15i-mlflow-step-by-step-recap-attaching-metadata-to-runs-with-set-tags.md)** — same setup, but we attach **searchable metadata** to the run with `mlflow.set_tags({...})`, so we can later filter runs by `release.version`, `engineering`, `dataset`, etc.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20h — log_artifacts + bulk log_params/log_metrics</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

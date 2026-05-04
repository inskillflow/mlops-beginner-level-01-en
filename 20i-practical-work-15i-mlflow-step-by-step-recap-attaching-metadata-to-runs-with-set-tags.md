<a id="top"></a>

# Chapter 20i — Step-by-step recap: attaching metadata to runs with `mlflow.set_tags({...})`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What we add today vs chap 20h](#section-2) |
| 3 | [`set_tag` vs `set_tags` vs `delete_tag`](#section-3) |
| 4 | [Run tags vs experiment tags vs MLflow system tags](#section-4) |
| 5 | [Useful tag conventions](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [The code](#section-7) |
| 8 | [Run it, see the tags in the UI](#section-8) |
| 9 | [Filtering runs by tag](#section-9) |
| 10 | [Mini exercise — version your runs](#section-10) |
| 11 | [Tear down](#section-11) |
| 12 | [Recap and what's next](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Today we attach **searchable metadata** to each run with **`mlflow.set_tags({...})`**.

Tags are arbitrary `key → value` strings — perfect for things like:

- `engineering = "ML platform"`
- `release.candidate = "RC1"`
- `release.version = "2.0"`
- `dataset = "red-wine-quality"`
- `git.branch = "feature/elasticnet-sweep"`

Unlike **parameters** (which describe *the algorithm's inputs*) and **metrics** (which describe *the algorithm's outputs*), **tags** describe *the run itself*: who, what version, which environment, which release candidate.

Best of all: you can **filter and group** runs by tag in the MLflow UI and via the search API.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What we add today vs chap 20h

| Diff | What |
|---|---|
| `mlflow.set_tags({...})` right after `start_run()` | Attach 3 tags to the run. |
| Print the active run's tags from `last_active_run().data.tags` | Confirm they were saved. |
| Tiny CLI hint: how to filter by tag in the UI | One sentence at the end. |

That's the whole change. Everything else (env-var URI, multi-service Docker, bulk `log_params`/`log_metrics`, `log_artifacts("data/")`, `get_artifact_uri`) is **identical to chap 20h**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. `set_tag` vs `set_tags` vs `delete_tag`

```python
# Single tag
mlflow.set_tag("release.version", "2.0")

# Many tags at once (NEW today)
mlflow.set_tags({
    "engineering":       "ML platform",
    "release.candidate": "RC1",
    "release.version":   "2.0",
})

# Remove a tag
mlflow.delete_tag("release.candidate")
```

| Function | Use it for |
|---|---|
| `mlflow.set_tag(key, value)` | One tag |
| `mlflow.set_tags(dict)` | Several tags in one server round-trip |
| `mlflow.delete_tag(key)` | Drop a tag from the **current** run |

> [!TIP]
> Both `set_tag` and `set_tags` **upsert**: if the tag already exists on the run, its value is overwritten. No need to delete-then-set.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Run tags vs experiment tags vs MLflow system tags

Three flavours of tags coexist — easy to confuse:

| Scope | API | Lives on |
|---|---|---|
| **Run** (what we do today) | `mlflow.set_tag` / `set_tags` | The current run |
| **Experiment** | `MlflowClient().set_experiment_tag(exp_id, key, value)` *or* the `tags=` argument of `create_experiment` (see chap 20f) | The whole experiment |
| **System** (auto-set by MLflow) | n/a — MLflow sets them for you | The run, prefixed with `mlflow.` |

Common system tags MLflow sets automatically:

- `mlflow.user` — OS user that started the run
- `mlflow.source.name` — script path
- `mlflow.source.type` — `LOCAL`, `JOB`, `NOTEBOOK`…
- `mlflow.source.git.commit` — current Git SHA (if any)
- `mlflow.runName` — display name of the run

> [!IMPORTANT]
> **Don't** set tags whose key starts with `mlflow.` yourself. That namespace is reserved for system tags. Use your own prefix (e.g. `team.`, `release.`, `dataset.`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Useful tag conventions

A few patterns that pay off in real projects:

| Tag key | Why it's useful |
|---|---|
| `release.version` / `release.candidate` | Quickly find every run associated with a release. |
| `engineering` (team or component) | Group runs by who owns them. |
| `dataset` and `dataset.version` | Trace runs back to the exact data. |
| `model.family` (`elasticnet`, `xgboost`, …) | Compare algorithms within an experiment. |
| `purpose` (`baseline`, `hpo`, `prod-candidate`, `debug`) | Filter out noise when reviewing results. |
| `commit.sha` | Reproducibility (if Git auto-tagging isn't on). |

Use `.` as a separator (e.g. `release.version`) — the UI groups them visually.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap20i-mlflow-step-by-step-recap-attaching-metadata-with-set-tags/
├── README.md
├── docker-compose.yml
├── data/
│   └── red-wine-quality.csv
├── mlflow/
│   └── Dockerfile
└── trainer/
    ├── Dockerfile
    ├── requirements.txt
    └── train.py            ← + mlflow.set_tags({...})
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

    exp = mlflow.set_experiment(experiment_name="experiment_4")
    print("Name              :", exp.name)
    print("Experiment_id     :", exp.experiment_id)
    print("Artifact Location :", exp.artifact_location)
    print("Tags              :", exp.tags)
    print("Lifecycle_stage   :", exp.lifecycle_stage)
    print("Creation timestamp:", exp.creation_time)

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

    # ===== NEW: tags describing this run =====
    tags = {
        "engineering":       "ML platform",
        "release.candidate": "RC1",
        "release.version":   "2.0",
    }
    mlflow.set_tags(tags)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print("Elasticnet (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE:  %s" % mae)
    print("  R2:   %s" % r2)

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Active run id   :", run.info.run_id)
    print("Active run name :", run.info.run_name)

    # NEW: confirm the tags are saved on the run
    print("Tags on this run:")
    for k, v in run.data.tags.items():
        if not k.startswith("mlflow."):       # skip system tags for clarity
            print(f"  {k} = {v}")
```

### 7.2 `docker-compose.yml`

Same as 20h, just a unique `container_name`:

```yaml
services:
  mlflow:
    build: { context: ./mlflow }
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20i
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
    container_name: trainer-recap-20i
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

Identical to chap 20g/h.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run it, see the tags in the UI

```bash
cd chap20i-mlflow-step-by-step-recap-attaching-metadata-with-set-tags
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.5 --l1_ratio 0.3
```

End of the trainer's stdout:

```text
Active run id   : 5a2c...91
Active run name : adventurous-dove-37
Tags on this run:
  engineering = ML platform
  release.candidate = RC1
  release.version = 2.0
```

In [http://localhost:5000](http://localhost:5000):

1. Open experiment **`experiment_4`** → click the latest run.
2. Tab **Tags** → your three custom tags appear right above the system tags (`mlflow.user`, `mlflow.source.name`, …).
3. Tab **Artifacts** → unchanged from chap 20h (model + the 3 CSVs).

> [!NOTE]
> Tags appear in **two** places: the run's **Tags** panel **and** as columns you can add to the runs table (click the gear icon → "Show columns" → check `tags.release.version`, etc.).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Filtering runs by tag

In the experiment view, the search bar accepts SQL-like filters:

```text
tags.`release.version` = "2.0"
tags.`engineering` = "ML platform" and metrics.rmse < 0.8
tags.`release.candidate` LIKE "RC%"
```

> [!IMPORTANT]
> **Backticks** are required around tag keys that contain a `.` (which is most of yours). The bar will silently ignore an unquoted dotted key.

The same filters work via `MlflowClient().search_runs(..., filter_string="...")` — perfect for scripts that promote "the best run with `release.candidate = RC1`" to the model registry (chap 17 covered the registry).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Mini exercise — version your runs

Run the trainer **twice** with different hyperparams **and a different release tag**:

```bash
docker compose run --rm \
  -e RELEASE_VERSION=2.0 \
  trainer --alpha 0.3 --l1_ratio 0.3

docker compose run --rm \
  -e RELEASE_VERSION=2.1 \
  trainer --alpha 0.5 --l1_ratio 0.5
```

Modify `train.py` so the `release.version` tag reads `os.getenv("RELEASE_VERSION", "dev")`:

```python
tags = {
    "engineering":     "ML platform",
    "release.version": os.getenv("RELEASE_VERSION", "dev"),
}
mlflow.set_tags(tags)
```

Then in the UI search bar:

```text
tags.`release.version` = "2.1"
```

…and only the second run shows up. Now imagine doing this for **every** training in CI: instant traceability between code releases and trained models.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Tear down

```bash
docker compose down
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap and what's next

You now have, in your toolbox, every essential MLflow building block:

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
| **`set_tags` for searchable metadata** | **20i** ← you are here |

You can stop here and have a perfectly serviceable MLflow workflow. From here you can branch into:

- **Chapter 13** — autologging (`mlflow.sklearn.autolog`) — let MLflow log everything by itself.
- **Chapter 14** — model **signature + input example** — describe what your model expects.
- **Chapter 17** — the **Model Registry** with `MlflowClient` — version, stage, promote.
- **Chapter 18** — the MLflow **CLI** as a separate Docker service — automate cleanup, exports, and audits.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20i — set_tags for run metadata</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

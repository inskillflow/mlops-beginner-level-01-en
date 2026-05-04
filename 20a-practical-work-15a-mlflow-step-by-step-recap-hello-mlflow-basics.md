<a id="top"></a>

# Chapter 20a — Step-by-step recap: Hello MLflow (the absolute basics)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Prerequisite](#section-2) |
| 3 | [Project structure](#section-3) |
| 4 | [The code](#section-4) |
| 5 | [Run it, log a first run, tear down](#section-5) |
| 6 | [Recap and next chapter](#section-6) |

---

<a id="section-1"></a>

## 1. Objective

Chapter 20 starts a **step-by-step recap** of the MLflow API you've already used through chapters 06 → 18. We rebuild it from scratch one tiny line at a time, so the muscle memory really sticks. Same MLflow, same Docker, but **stripped down to the bare minimum** at each step.

Today (**20a**) is the smallest possible MLflow program:

- Point at a tracking server (`mlflow.set_tracking_uri`)
- Pick an experiment (`mlflow.set_experiment`)
- Open a run (`mlflow.start_run(run_name=...)`)
- Log two hyperparameters and two metrics

That's it. No scikit-learn, no FastAPI, no Streamlit. Just the 5 MLflow calls you'll re-use forever.

> [!NOTE]
> If you've completed chapter 05c, you already saw this exact script. Today is the formal "**chapter 1 of the recap**". Chapters 20b and 20c will add one tiny thing each.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisite

You need:

1. **Docker Desktop** (or Docker Engine + Compose v2). If not yet installed, see [chapter 06, sections 1 to 3](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md).
2. **A host-side Python venv** with `mlflow==2.16.2` installed (used to run the script that talks to the dockerized server).

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install mlflow==2.16.2
```

```powershell
# Windows PowerShell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install mlflow==2.16.2
```

> [!IMPORTANT]
> **Don't forget `set_tracking_uri`.** Without it, MLflow logs into a local `./mlruns` folder and you'll wonder why nothing shows up in the dockerized UI. Always point at the running server explicitly.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap20a-mlflow-step-by-step-recap-hello-mlflow-basics/
├── README.md
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
└── hello_mlflow.py
```

Same single-service Docker skeleton as chap 05c.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The code

### 4.1 `mlflow/Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /mlflow
RUN pip install --no-cache-dir mlflow==2.16.2
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///database/mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", "--port", "5000"]
```

### 4.2 `docker-compose.yml`

```yaml
services:
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-recap:latest
    container_name: mlflow-recap-20a
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    restart: unless-stopped

volumes:
  mlflow-db:
  mlflow-artifacts:
```

### 4.3 `hello_mlflow.py` — runs from the host venv

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hello_mlflow")

with mlflow.start_run(run_name="my_first_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.18)

print("Done. Open http://localhost:5000 to see your run.")
```

Five MLflow lines. Read them top-to-bottom:

| Line | What it does |
|---|---|
| `mlflow.set_tracking_uri("http://localhost:5000")` | Tell MLflow **where** the tracking server lives. From the host, that's `localhost`. |
| `mlflow.set_experiment("hello_mlflow")` | Group runs under a named experiment. Created on first call, reused after. |
| `mlflow.start_run(run_name="my_first_run")` | Open a run. Everything inside the `with` block is attached to this run. |
| `mlflow.log_param("learning_rate", 0.01)` | Persist an **input** (immutable, set once per run). |
| `mlflow.log_metric("accuracy", 0.92)` | Persist an **output** (numeric, can have a history). |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Run it, log a first run, tear down

### 5.1 Start the MLflow server

```bash
cd chap20a-mlflow-step-by-step-recap-hello-mlflow-basics
docker compose up --build
```

When you see `Listening at: http://0.0.0.0:5000`, open [http://localhost:5000](http://localhost:5000). Empty UI for now.

### 5.2 Log your first run from the host

In **another terminal** (keep the server running):

```bash
# activate your host venv first
python hello_mlflow.py
```

Output:

```text
Done. Open http://localhost:5000 to see your run.
```

Refresh the UI. The experiment **`hello_mlflow`** appears with one run **`my_first_run`** containing 2 params + 2 metrics.

### 5.3 Mini exercise

Run `python hello_mlflow.py` two more times. Each time, edit one of the `log_param` or `log_metric` values:

```python
mlflow.log_param("learning_rate", 0.001)
mlflow.log_metric("accuracy", 0.95)
```

You'll have **three runs** in the same experiment. Click the column header `metrics.accuracy` in the UI to sort and find the best.

### 5.4 Tear down

```bash
docker compose down       # keep the runs (volumes survive)
docker compose down -v    # wipe everything
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Recap and next chapter

You've now (re)established:

- **5 essential MLflow calls** — `set_tracking_uri`, `set_experiment`, `start_run`, `log_param`, `log_metric`.
- **The single-service Docker pattern** — server in container, script on host.
- **The "always set the tracking URI" reflex** — without it, MLflow silently logs nowhere useful.

Next: **[chapter 20b](./20b-practical-work-15b-mlflow-step-by-step-recap-printing-the-tracking-uri.md)** — the same script with **one new line**: `print(mlflow.get_tracking_uri())`. A tiny diff that teaches a great defensive habit.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20a — Hello MLflow basics</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

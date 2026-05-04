<a id="top"></a>

# Chapter 20b — Step-by-step recap: confirming the tracking URI with `get_tracking_uri()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [The single line we add today](#section-2) |
| 3 | [Project structure](#section-3) |
| 4 | [The code](#section-4) |
| 5 | [Run it, observe the output, tear down](#section-5) |
| 6 | [Recap and next chapter](#section-6) |

---

<a id="section-1"></a>

## 1. Objective

Tiniest possible diff from chapter 20a: we **print** the URI MLflow is currently pointing at. This is the most useful one-liner you can sprinkle in any MLflow script — it instantly answers the question "**am I logging to the server I think I'm logging to?**".

> [!IMPORTANT]
> Most "my run isn't showing up in the UI!" problems come down to a wrong tracking URI. Print it once and the mystery dies.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. The single line we add today

Right after `mlflow.set_tracking_uri(...)`:

```python
print("Tracking URI:", mlflow.get_tracking_uri())     # NEW
```

That's it. The whole chapter is built around **one extra line**.

`mlflow.get_tracking_uri()` returns whatever was set most recently — either via `set_tracking_uri(...)` in code, or via the env var `MLFLOW_TRACKING_URI`, or the default (`file:./mlruns`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri/
├── README.md
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
└── hello_mlflow.py     ← +1 line vs chap 20a
```

Identical to chap 20a apart from `hello_mlflow.py`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The code

### 4.1 `hello_mlflow.py` — diff vs chap 20a

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
print("Tracking URI:", mlflow.get_tracking_uri())     # NEW

mlflow.set_experiment("hello_mlflow")

with mlflow.start_run(run_name="my_first_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.18)

print("Done. Open http://localhost:5000 to see your run.")
```

### 4.2 `docker-compose.yml` and `mlflow/Dockerfile`

Identical to [chapter 20a](./20a-practical-work-15a-mlflow-step-by-step-recap-hello-mlflow-basics.md). No change.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Run it, observe the output, tear down

### 5.1 Start the server

```bash
cd chap20b-mlflow-step-by-step-recap-printing-the-tracking-uri
docker compose up --build
```

### 5.2 Run the script from the host venv

```bash
python hello_mlflow.py
```

Output:

```text
Tracking URI: http://localhost:5000
Done. Open http://localhost:5000 to see your run.
```

The first line confirms what you intended.

### 5.3 Mini exercise — observe what happens **without** `set_tracking_uri`

Comment out the `set_tracking_uri` line:

```python
# mlflow.set_tracking_uri("http://localhost:5000")
print("Tracking URI:", mlflow.get_tracking_uri())
```

Re-run:

```text
Tracking URI: file:///path/to/your/cwd/mlruns
Done. Open http://localhost:5000 to see your run.
```

The URI now points at a **local folder** (`file:///...mlruns`). Your "run" was logged there, **not** in the dockerized server. This is exactly the silent failure mode you'll catch in two seconds with the `print` line.

Restore the `set_tracking_uri` line and re-run — back to `http://localhost:5000`.

### 5.4 Tear down

```bash
docker compose down
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Recap and next chapter

- `mlflow.get_tracking_uri()` returns the **currently active** tracking URI.
- Always print it once at script start. Three letters of code save hours of confusion.
- Without `set_tracking_uri(...)` and without `MLFLOW_TRACKING_URI`, MLflow falls back to a local `mlruns/` folder — runs go nowhere visible.

Next: **[chapter 20c](./20c-practical-work-15c-mlflow-step-by-step-recap-elasticnet-on-red-wine-quality.md)** — same Docker setup, but the script becomes a **real ML pipeline**: load the red-wine-quality CSV, train an `ElasticNet`, log params + metrics + model with `mlflow.sklearn.log_model`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 20b — Confirming the tracking URI</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>

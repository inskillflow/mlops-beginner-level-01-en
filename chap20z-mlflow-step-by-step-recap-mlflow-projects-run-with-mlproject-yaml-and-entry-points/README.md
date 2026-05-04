# chap20z - Step-by-step recap: mlflow.projects.run + MLproject YAML

Lesson: [`../20z-practical-work-15z-mlflow-step-by-step-recap-mlflow-projects-run-with-mlproject-yaml-and-entry-points.md`](../20z-practical-work-15z-mlflow-step-by-step-recap-mlflow-projects-run-with-mlproject-yaml-and-entry-points.md).

## What's new vs chap20y

- Add `trainer/MLproject` (no extension) -- declarative entry points + typed parameters
- Add `trainer/python_env.yaml` -- env recipe for `env_manager="virtualenv"`
- Add `trainer/run_project.py` -- launcher calling `mlflow.projects.run(...)`
- `train.py` itself is unchanged from chap20y -- packaging is purely additive
- `env_manager="local"` so we don't try to recreate a venv inside an already-complete container

## Three equivalent ways to launch

```bash
# 1. Via our Python launcher (default)
docker compose run --rm trainer

# 2. Via the MLflow CLI
docker compose run --rm --entrypoint sh trainer -c \
  "mlflow run . -e ElasticNet -P alpha=0.3 -P l1_ratio=0.3 \
                --experiment-name 'Project exp 1' --env-manager local"

# 3. Default `main` entry point with default params
docker compose run --rm --entrypoint sh trainer -c \
  "mlflow run . --env-manager local"
```

Open http://localhost:5000 -> `Project exp 1`. Each launch adds tags `mlflow.project.entryPoint`, `mlflow.project.backend`, `mlflow.source.name` automatically.

## Tear down

```bash
docker compose down
docker compose down -v
```

# chap20f - Step-by-step recap: create_experiment with tags + custom artifact_location

The full lesson lives at [`../20f-practical-work-15f-mlflow-step-by-step-recap-create-experiment-with-tags-and-custom-artifact-location.md`](../20f-practical-work-15f-mlflow-step-by-step-recap-create-experiment-with-tags-and-custom-artifact-location.md).

## What's new vs chap20e

- `mlflow.create_experiment(name=, tags=, artifact_location=)` instead of `set_experiment(name)`
- `Path("/mlflow/myartifacts").as_uri()` to build a portable `file:///...` URI
- Shared named volume `myartifacts` mounted on BOTH services so the UI can list artifacts
- `try / except mlflow.exceptions.MlflowException` to make the script idempotent

## Quick run

```bash
docker compose up -d --build mlflow

docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.4
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.3
docker compose run --rm trainer --alpha 0.1 --l1_ratio 0.9
```

Output of the first run:

```
Created experiment 'exp_create_exp_artifact' with id=1
Name              : exp_create_exp_artifact
Tags              : {'version': 'v1', 'priority': 'p1'}
Artifact Location : file:///mlflow/myartifacts
```

Output of subsequent runs:

```
Experiment already exists: id=1 (RESOURCE_ALREADY_EXISTS)
```

Open http://localhost:5000 -> experiment `exp_create_exp_artifact` -> click a run -> Artifacts tab shows `model/`.

## Inspect the shared volume

```bash
docker compose exec mlflow ls -R /mlflow/myartifacts
```

## set_experiment vs create_experiment

| Feature | `set_experiment` | `create_experiment` |
|---|---|---|
| If exists | Reuses silently | Raises `RESOURCE_ALREADY_EXISTS` |
| Tags at creation | No | Yes |
| Custom `artifact_location` | No | Yes |
| Best for | Day-to-day training scripts | One-off bootstrap scripts |

## Tear down

```bash
docker compose down       # keep all volumes
docker compose down -v    # wipe everything (mlflow-db + mlflow-artifacts + myartifacts)
```

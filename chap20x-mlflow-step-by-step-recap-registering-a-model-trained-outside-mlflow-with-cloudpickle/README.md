# chap20x - Step-by-step recap: registering a model trained OUTSIDE MLflow

Lesson: [`../20x-practical-work-15x-mlflow-step-by-step-recap-registering-a-model-trained-outside-mlflow-with-cloudpickle.md`](../20x-practical-work-15x-mlflow-step-by-step-recap-registering-a-model-trained-outside-mlflow-with-cloudpickle.md).

## What's new vs chap20w

- TWO services in the docker-compose:
  - **`pretrainer`** -- pure sklearn, NO mlflow, dumps `elastic-net-regression.pkl` to a shared volume
  - **`registrar`** -- bridge: `pickle.load(...)` then `mlflow.sklearn.log_model(..., serialization_format="cloudpickle", registered_model_name=...)`
- New experiment: `experiment_register_outside`
- Run has tags `imported=true`, `source=external_pickle`, `filename=...` -- and NO params/metrics (those would be lies)
- Mirrors the real-world pattern of importing a vendor / legacy / partner-team model into MLflow

## Quick run

```bash
docker compose up -d --build mlflow

docker compose run --rm pretrainer --alpha 0.4 --l1_ratio 0.4
# -> writes /shared/elastic-net-regression.pkl

docker compose run --rm registrar
# -> Successfully registered model 'elastic-net-regression-outside-mlflow' (Version 1)
```

Open http://localhost:5000:
- **Experiments tab** -> `experiment_register_outside` -> 1 run with import tags
- **Models tab** -> `elastic-net-regression-outside-mlflow` -> Version 1

## Loading the imported model from any consumer

```python
import mlflow.pyfunc
m = mlflow.pyfunc.load_model("models:/elastic-net-regression-outside-mlflow/1")
```

## Tear down

```bash
docker compose down
docker compose down -v   # -v also drops the `shared` volume holding the .pkl
```

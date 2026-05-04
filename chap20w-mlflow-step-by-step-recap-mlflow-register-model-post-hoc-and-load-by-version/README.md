# chap20w - Step-by-step recap: mlflow.register_model post-hoc + load by version

Lesson: [`../20w-practical-work-15w-mlflow-step-by-step-recap-mlflow-register-model-post-hoc-and-load-by-version.md`](../20w-practical-work-15w-mlflow-step-by-step-recap-mlflow-register-model-post-hoc-and-load-by-version.md).

## What's new vs chap20u

- Drop `registered_model_name=` kwarg from `log_model`
- Two-step registry pattern:
  ```python
  mlflow.sklearn.log_model(lr, "model")                                  # log only
  run = mlflow.active_run()
  mv = mlflow.register_model(
      model_uri=f"runs:/{run.info.run_id}/model",
      name="elastic-api-2",
  )                                                                       # then register
  loaded = mlflow.pyfunc.load_model(f"models:/{mv.name}/{mv.version}")    # load by version
  ```
- `mv.version` is the auto-incremented version string -- no hardcoded `/1`
- Round-trip sanity check `assert np.allclose(...)` between in-process and registry-loaded predictions

## Why split log + register?

- Production pipelines often run training and registration as separate jobs (registration may wait for evaluation gates to pass)
- Allows registering an EXISTING run's artifact long after the training finished
- Allows registering the same artifact under multiple names

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.6 --l1_ratio 0.6
docker compose run --rm trainer --alpha 0.4 --l1_ratio 0.2
```

Open http://localhost:5000:
- **Experiments tab** -> `experiment_register_model_api` -> 2 runs
- **Models tab** -> `elastic-api-2` -> Version 1 + Version 2

## Tear down

```bash
docker compose down
docker compose down -v
```

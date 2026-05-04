# chap20k - Step-by-step recap: multiple experiments (ElasticNet vs Ridge vs Lasso)

Lesson: [`../20k-practical-work-15k-mlflow-step-by-step-recap-multiple-experiments-comparing-elasticnet-ridge-lasso.md`](../20k-practical-work-15k-mlflow-step-by-step-recap-multiple-experiments-comparing-elasticnet-ridge-lasso.md).

## What's new vs chap20j

- 3 algorithms -> 3 experiments (`exp_multi_EL`, `exp_multi_Ridge`, `exp_multi_Lasso`)
- Each experiment has 3 runs (alphas = [args.alpha, 0.9, 0.4])
- Model factories pattern: one helper trains any of the 3 estimators
- Total: **9 runs** in **3 experiments**, in one execution

## Quick run

```bash
docker compose up -d --build mlflow
docker compose run --rm trainer --alpha 0.7 --l1_ratio 0.7
```

Open http://localhost:5000 -> 3 experiments in the left sidebar.

## Add a 4th algorithm

```python
from sklearn.linear_model import HuberRegressor

def make_huber(alpha, l1_ratio):
    return HuberRegressor(alpha=alpha), {"alpha": alpha}

EXPERIMENTS.append(("exp_multi_Huber", make_huber))
```

## Tear down

```bash
docker compose down
docker compose down -v
```

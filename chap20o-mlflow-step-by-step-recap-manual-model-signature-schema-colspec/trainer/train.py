import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.7)
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

    exp = mlflow.set_experiment(experiment_name="experiment_signature")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")
    print(f"Artifact Location : {exp.artifact_location}")
    print(f"Tags              : {exp.tags}")
    print(f"Lifecycle_stage   : {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    mlflow.set_tags({
        "engineering":       "ML platform",
        "release.candidate": "RC1",
        "release.version":   "2.0",
    })

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False,
    )

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print(f"  Elasticnet (alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    input_schema = Schema([
        ColSpec("double", "fixed acidity"),
        ColSpec("double", "volatile acidity"),
        ColSpec("double", "citric acid"),
        ColSpec("double", "residual sugar"),
        ColSpec("double", "chlorides"),
        ColSpec("double", "free sulfur dioxide"),
        ColSpec("double", "total sulfur dioxide"),
        ColSpec("double", "density"),
        ColSpec("double", "pH"),
        ColSpec("double", "sulphates"),
        ColSpec("double", "alcohol"),
    ])
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    input_example = {
        "fixed acidity":        np.array([7.2, 7.5, 7.0, 6.8, 6.9]),
        "volatile acidity":     np.array([0.35, 0.3, 0.28, 0.38, 0.25]),
        "citric acid":          np.array([0.45, 0.5, 0.55, 0.4, 0.42]),
        "residual sugar":       np.array([8.5, 9.0, 8.2, 7.8, 8.1]),
        "chlorides":            np.array([0.045, 0.04, 0.035, 0.05, 0.042]),
        "free sulfur dioxide":  np.array([30., 35., 40., 28., 32.]),
        "total sulfur dioxide": np.array([120., 125., 130., 115., 110.]),
        "density":              np.array([0.997, 0.996, 0.995, 0.998, 0.994]),
        "pH":                   np.array([3.2, 3.1, 3.0, 3.3, 3.2]),
        "sulphates":            np.array([0.65, 0.7, 0.68, 0.72, 0.62]),
        "alcohol":              np.array([9.2, 9.5, 9.0, 9.8, 9.4]),
    }

    mlflow.log_artifact("data/red-wine-quality.csv")
    mlflow.sklearn.log_model(
        lr,
        "model",
        signature=signature,
        input_example=input_example,
    )

    print("Artifact path:", mlflow.get_artifact_uri())
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")

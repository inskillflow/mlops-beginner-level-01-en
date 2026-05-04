import argparse
import logging
import os
import warnings

import cloudpickle
import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.sklearn_model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        return self.sklearn_model.predict(model_input.values)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_custom_sklearn")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    data_dir = "data/red-wine-data"
    os.makedirs(data_dir, exist_ok=True)
    data.to_csv(f"{data_dir}/data.csv",  index=False)
    train.to_csv(f"{data_dir}/train.csv", index=False)
    test.to_csv(f"{data_dir}/test.csv",  index=False)

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
    print(f"  Before log -> RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    sklearn_model_path = "sklearn_model.pkl"
    joblib.dump(lr, sklearn_model_path)

    artifacts = {
        "sklearn_model": sklearn_model_path,
        "data":          data_dir,
    }

    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                ],
            },
        ],
        "name": "sklearn_env",
    }

    model_info = mlflow.pyfunc.log_model(
        artifact_path="sklearn_mlflow_pyfunc",
        python_model=SklearnWrapper(),
        artifacts=artifacts,
        code_path=["train.py"],
        conda_env=conda_env,
    )
    print("Logged model URI:", model_info.model_uri)

    loaded = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

    loaded_predictions = loaded.predict(test_x)
    loaded_rmse, loaded_mae, loaded_r2 = eval_metrics(test_y, loaded_predictions)

    print(f"  After load -> RMSE={loaded_rmse:.6f}  MAE={loaded_mae:.6f}  R2={loaded_r2:.6f}")

    mlflow.log_metrics({
        "loaded_test_rmse": loaded_rmse,
        "loaded_test_mae":  loaded_mae,
        "loaded_test_r2":   loaded_r2,
    })

    assert np.allclose(predicted_qualities, loaded_predictions), \
        "Loaded model produces DIFFERENT predictions! Serialization is broken."

    print("Sanity check OK: original and loaded predictions match exactly.")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hello_mlflow")

with mlflow.start_run(run_name="my_first_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.18)

print("Done. Open http://localhost:5000 to see your run.")

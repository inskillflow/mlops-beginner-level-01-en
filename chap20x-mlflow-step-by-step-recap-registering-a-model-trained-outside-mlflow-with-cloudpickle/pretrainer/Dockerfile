FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_outside_mlflow.py .
ENTRYPOINT ["python", "train_outside_mlflow.py"]

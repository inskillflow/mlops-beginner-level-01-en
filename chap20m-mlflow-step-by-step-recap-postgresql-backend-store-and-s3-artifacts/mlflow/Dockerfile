FROM python:3.12-slim
WORKDIR /mlflow
RUN pip install --no-cache-dir \
        mlflow==2.16.2 \
        psycopg2-binary==2.9.10
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflowuser:mlflowpassword@postgres:5432/mlflowdb", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", "--port", "5000"]

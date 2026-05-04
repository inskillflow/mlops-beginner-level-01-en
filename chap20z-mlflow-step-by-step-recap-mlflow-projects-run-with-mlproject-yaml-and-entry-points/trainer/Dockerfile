FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY MLproject python_env.yaml train.py run_project.py ./
ENTRYPOINT ["python", "run_project.py"]

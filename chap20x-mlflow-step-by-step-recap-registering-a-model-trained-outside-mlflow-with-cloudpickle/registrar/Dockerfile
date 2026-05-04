FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY register_external.py .
ENTRYPOINT ["python", "register_external.py"]

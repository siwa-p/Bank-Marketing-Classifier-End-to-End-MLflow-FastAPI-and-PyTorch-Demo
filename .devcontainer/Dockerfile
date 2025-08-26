FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl postgresql-client && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt /tmp/pip-tmp/

RUN pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt

RUN rm -rf /tmp/pip-tmp

USER vscode

WORKDIR /

ENV BACKEND_URI "postgresql://postgres:password@postgres:5432/mlflow_db"

EXPOSE 5000

CMD mlflow server --backend-store-uri $BACKEND_URI --host 0.0.0.0 --serve-artifacts --artifacts-destination s3://mlflow-artifacts

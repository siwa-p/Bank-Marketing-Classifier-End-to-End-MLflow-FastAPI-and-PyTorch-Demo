FROM mcr.microsoft.com/vscode/devcontainers/python:3.12-bullseye

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl postgresql-client && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /app/
WORKDIR /app

ENV BACKEND_URI "postgresql://postgres:password@postgres:5432/mlflow_db"

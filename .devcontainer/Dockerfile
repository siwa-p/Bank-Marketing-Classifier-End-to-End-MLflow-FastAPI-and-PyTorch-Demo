FROM nvidia/cuda:13.0.0-tensorrt-devel-ubuntu24.04

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl postgresql-client && \
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

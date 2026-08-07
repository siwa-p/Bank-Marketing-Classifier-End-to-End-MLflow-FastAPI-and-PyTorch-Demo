FROM mcr.microsoft.com/vscode/devcontainers/python:3.12-bullseye

RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends git curl postgresql-client && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV BACKEND_URI "postgresql://postgres:password@postgres:5432/mlflow_db"

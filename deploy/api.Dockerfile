# API image: main dependencies only (no training libraries), non-root, read-only DB role.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /srv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY backend/app ./app
RUN uv sync --locked --no-default-groups

FROM python:3.12-slim
RUN useradd --system --uid 10001 --home /srv shelfsense
WORKDIR /srv
COPY --from=build --chown=shelfsense /srv /srv
USER shelfsense
ENV PATH="/srv/.venv/bin:$PATH" PYTHONUNBUFFERED=1 LOG_FORMAT=json
EXPOSE 8000
# Hosting platforms inject PORT; 8000 is the container's own default.
CMD ["sh", "-c", "exec uvicorn --factory app.api.main:create_app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]

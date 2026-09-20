FROM ghcr.io/berriai/litellm:v1.101.0
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY jev_router /app/jev_router
COPY deploy/routers /app/deploy/routers
# The base image's venv is uv-built and ships no pip; install into it with uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --python /app/.venv/bin/python --no-cache "typesafe-sdk>=0.7,<1" pyyaml
ENV PYTHONPATH=/app JEV_ROUTER_CONFIG=/app/deploy/routers

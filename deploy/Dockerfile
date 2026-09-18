FROM ghcr.io/berriai/litellm:main-v1.101.0
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY jev_router /app/jev_router
COPY deploy/router.yaml /app/deploy/router.yaml
RUN pip install --no-cache-dir "typesafe-sdk>=0.7,<1" pyyaml
ENV PYTHONPATH=/app JEV_ROUTER_CONFIG=/app/deploy/router.yaml

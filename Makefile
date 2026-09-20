# Common tasks. `make` with no target prints this list.
.DEFAULT_GOAL := help
DATASET ?= evals/datasets/routing-golden.yaml

help:            ## show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-14s %s\n", $$1, $$2}'

setup:           ## create .venv and install the package with dev tools
	uv venv --clear && uv pip install -e ".[dev]"

test:            ## unit + integration tests (boots a mock proxy; no keys needed)
	.venv/bin/pytest -q

lint:            ## ruff + pyright
	.venv/bin/ruff check jev_router tests && .venv/bin/ruff format --check jev_router tests && .venv/bin/pyright jev_router

fmt:             ## apply ruff fixes and formatting
	.venv/bin/ruff check --fix jev_router tests && .venv/bin/ruff format jev_router tests

up:              ## build and start the proxy + redis (reads deploy/.env)
	cd deploy && docker compose up -d --build

down:            ## stop the proxy
	cd deploy && docker compose down

logs:            ## follow proxy logs (routing decisions are logged at INFO)
	cd deploy && docker compose logs -f litellm

preflight:       ## check every tier accepts its effort and caches, against the running proxy
	.venv/bin/jev-router eval preflight

eval:            ## evaluate deploy/router.yaml with real Jev, no model calls (free)
	.venv/bin/jev-router eval run $(DATASET) --mode simulate

eval-live:       ## evaluate against the running proxy with real providers (costs money)
	.venv/bin/jev-router eval run $(DATASET) --mode live

.PHONY: help setup test lint fmt up down logs preflight eval eval-live

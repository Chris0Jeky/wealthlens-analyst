PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)

.PHONY: help install lint format test dev db-up ingest-slice fetch-corpus \
        eval-golden-validate eval-deterministic eval-ragas eval-report

# ── Help ──────────────────────────────────────────────────────────────────
help: ## Show all targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Check targets fail loudly: a non-zero exit from ruff/mypy/pytest aborts the
# recipe. No error swallowing.
install: ## Install the package editable (+dev,evals extras)
	$(PYTHON) -m pip install -e ".[dev,evals]"

lint: ## Ruff check + format check + strict mypy
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format .

test: ## Run the pytest suite
	$(PYTHON) -m pytest -q

db-up: ## Start Postgres+pgvector (localhost:15432)
	docker compose up -d analyst-db

dev: ## Start the dev server (uvicorn, 127.0.0.1:8100)
	$(PYTHON) -m uvicorn --factory wealthlens_analyst.api.app:create_app --reload --host 127.0.0.1 --port 8100

ingest-slice: ## Ingest the frozen corpus slice (chunk -> provenance gate -> write; FTS auto; embed if OPENAI_API_KEY set)
	$(PYTHON) -m wealthlens_analyst.ingest.slice_corpus

fetch-corpus: ## Download corpus documents per data/corpus_manifest.yml into data/raw/ (verifies/records sha256)
	$(PYTHON) scripts/fetch_corpus.py

eval-golden-validate: ## Validate the golden set against its JSON schema (static, CI-safe)
	$(PYTHON) evals/checks/deterministic.py

# Static checks today; gains --live (citation resolvability, refusal set,
# latency/cost bounds against a serving /ask) when backlog task H1-23 lands.
eval-deterministic: ## Run the deterministic eval checks
	$(PYTHON) evals/checks/deterministic.py

eval-ragas: ## Run RAGAS metrics over the reviewed golden subset (pending H1-25)
	$(PYTHON) evals/run_ragas.py

eval-report: eval-deterministic eval-ragas ## Generate the combined committed eval report (assembly pending H1-26)
	@echo "eval-report: deterministic + RAGAS ran; report assembly lands with H1-26"

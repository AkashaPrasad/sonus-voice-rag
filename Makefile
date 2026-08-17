.PHONY: dev index bench test verify deploy

VENV := .venv/bin

dev:
	$(VENV)/uvicorn backend.app.main:app --reload --port 8000

index:
	$(VENV)/python ingest/build_index.py --profile $(or $(CORPUS_PROFILE),dev) \
	  --langs $(or $(CORPUS_LANGS),hi,ta,te,bn) --rows $(or $(ROWS),1000) \
	  --strategy $(or $(CHUNK_STRATEGY),s5_hierarchical) --out index

test:
	$(VENV)/python -m pytest backend/tests -q

bench:
	$(VENV)/python bench/latency_bench.py --n 320 --runs 3
	$(VENV)/python bench/guardrail_eval.py
	$(VENV)/python bench/chunking_eval.py

verify: test bench

deploy:
	railway up --service vaani-api --detach
	cd frontend && npx vite build && \
	  npx wrangler pages deploy dist --project-name=sonus --branch=main --commit-dirty=true

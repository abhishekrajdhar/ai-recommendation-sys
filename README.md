## Recommender Agent

Production-ready FastAPI service for a stateless conversational SHL assessment recommender. It scrapes SHL Individual Test Solutions, indexes the catalog with BM25 plus FAISS semantic retrieval, and uses deterministic agent routing so it clarifies, recommends, refines, compares, or refuses predictably.

## Architecture

- `app/main.py`: FastAPI app with `GET /health` and `POST /chat`.
- `app/routes/chat.py`: API route preserving the exact required response schema.
- `app/models/schemas.py`: Pydantic request, response, catalog, and state models.
- `app/scraper/scrape_shl.py`: robust SHL catalog scraper for Individual Test Solutions.
- `app/services/state_extractor.py`: regex and rule-based stateless context extraction.
- `app/services/guardrails.py`: injection, jailbreak, off-topic, legal, and general hiring rejection.
- `app/services/retrieval.py`: BM25 + FAISS hybrid retrieval with weighted scoring.
- `app/services/reranker.py`: deterministic reranking from extracted hiring requirements.
- `app/services/comparison.py`: grounded catalog comparison tables.
- `app/services/agent.py`: explicit mode orchestration.
- `app/tests/test_agent.py`: evaluation-style pytest coverage.

The agent does not rely on hidden conversation memory. Every request reconstructs state from the supplied `messages` array.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy environment defaults:

```bash
cp .env.example .env
```

Environment variables:

- `SHL_CATALOG_PATH`: catalog JSON path, defaults to `app/data/shl_catalog.json` with fallback to `app/data/sample_catalog.json`.
- `SHL_USE_LLM`: set `true` to let the configured LLM polish recommendation wording. Routing and returned recommendations remain deterministic.
- `LLM_PROVIDER`: `openai` or `gemini`.
- `OPENAI_API_KEY`, `OPENAI_MODEL`: defaults to `gpt-4.1-mini`.
- `GEMINI_API_KEY`, `GEMINI_MODEL`: defaults to `gemini-2.5-flash`.

## Scrape And Index

Scrape SHL Individual Test Solutions:

```bash
python -m app.scraper.scrape_shl --output app/data/shl_catalog.json
```

Build the FAISS index:

```bash
python scripts/build_index.py
```

The scraper handles duplicate URLs, pagination, malformed pages, and missing metadata. It stores:

- remote testing support
- adaptive support
- job levels
```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Chat request:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a java backend developer."
    },
    {
      "role": "assistant",
      "content": "What seniority level are you hiring for this java backend role (Junior, Mid, or Senior)?"
    },
    {
      "role": "user",
      "content": "Mid-level."
    }
  ]
}'
```

Exact response shape:

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "string"
    }
  ],
  "end_of_conversation": false
}
```

## Agent Behavior

Routing is explicit:

1. Prompt injection, jailbreak, unrelated topics, general hiring advice, and legal advice are refused.
2. Comparison queries are answered only from catalog records.
3. Vague queries are clarified before recommendations.
4. Recommendations happen only after role/domain plus seniority, assessment type, or skills are known.
5. Refinements are handled by re-extracting state from the full stateless message history.

Minimum recommendation context:

- role or domain
- at least one of seniority, assessment type, or required skills

The API caps recommendations at 10 and only returns URLs already present in the scraped catalog.

## Retrieval

Hybrid retrieval improves Recall@10 by combining lexical precision with semantic matching:

1. BM25 retrieves keyword matches.
2. FAISS retrieves semantic neighbors using `sentence-transformers/all-MiniLM-L6-v2`.
3. Results are merged and deduplicated.
4. Weighted score is computed:

```text
final_score = 0.65 * semantic_score + 0.35 * keyword_score
```

5. A deterministic reranker boosts matches for extracted skills, seniority, personality, cognitive, technical, and remote requirements.

The bundled sample catalog lets the service and tests run before live scraping. Production should run the scraper and index builder.

## Evaluation

Automated tests

Run the unit and integration tests with pytest:

```bash
pytest
```

The test suite includes automated checks for clarification, refinement handling, grounded comparison behavior, prompt-injection resistance, refusal conditions, schema stability, and a small retrieval recall check.

Evaluation harness (retrieval + grounding)

This repository includes a lightweight evaluation harness used during development to measure retrieval quality against a small seeded ground-truth. The harness lives at `scripts/evaluate_retrieval.py` and the sample ground-truth is `app/tests/ground_truth.json`.

What it computes

- Recall@k (how often a ground-truth URL appears in the top-k results)
- Precision@k (ratio of relevant results in the top-k)
- MRR (Mean Reciprocal Rank) to summarize ranking position
- Per-query printed top-k result lists for manual inspection

How to run the harness

Run the evaluation script from the repository root (the script will import local modules, so set PYTHONPATH to the repo root):

```bash
PYTHONPATH=. python3 scripts/evaluate_retrieval.py
```

The script prints per-query top-k results and a small summary table with aggregated Recall@1/3/5/10, Precision@k, and MRR. Use the printed per-query lists to inspect common failure modes (domain mismatch, underspecified queries, noisy metadata).

Recommended next steps for evaluation

- Expand `app/tests/ground_truth.json` from the current seed (5 queries) to a larger set (50–300 queries) covering core domains (technical, personality, situational, finance, graduate) to get statistically meaningful metrics.
- Add a groundedness check: verify that returned recommendations cite catalog text (e.g., skills or categories appear in the retrieved record) and surface a groundedness score per reply.
- Add semantic-similarity diagnostics: average cosine similarity of returned items to the query embedding to spot weak semantic matches.
- Add a CI job that runs the evaluation harness and fails when Recall@10 or MRR fall below desired thresholds.

Interpretation guidance

- Low Recall@k indicates missing or poorly indexed catalog items or that retrieval signals (BM25/embeddings) need tuning.
- Low MRR shows relevant items are ranked too low — tune exact-skill boosting and reranker penalties/thresholds.
- Discrepancies between precision and recall can indicate noisy catalog metadata; improving `retrieval_text` (skills, job levels, categories) usually helps.

If you'd like, I can add an expanded ground-truth file, a small groundedness checker script, or a GitHub Action that runs the harness on PRs and fails the build when metrics regress.

## Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 --env-file .env shl-recommender
```

Production startup command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker: notes and recommended workflows

There are two common Docker workflows depending on whether you prebuild catalog/index artifacts or build them inside the image.

1) Recommended — prebuild artifacts locally (fast startup)

- Generate catalog and vectorstore locally (or in CI) and place them under the repo before building the image:

```bash
python -m app.scraper.scrape_shl --output app/data/shl_catalog.json --use-json-catalog
python scripts/build_index.py
```

- Build the Docker image (artifacts already present in `app/data` and `app/vectorstore`):

```bash
docker build -t shl-recommender:latest .
```

- Run the container with your environment file:

```bash
docker run -p 8000:8000 --env-file .env shl-recommender:latest
```

This avoids downloading model weights or building FAISS during container start and gives fastest boot times.

2) Build-in-image (convenient but slower and resource heavy)

- If you prefer the image to produce the catalog and index during build, ensure the `Dockerfile` runs the scraper and index builder during the Docker build. This requires longer build times and (optionally) `HF_TOKEN` available as a build-arg or secret.

- Example Build Command (may take several minutes):

```bash
docker build -t shl-recommender:latest .
```

Notes:
- If you build artifacts inside the image, provide `HF_TOKEN` (as a build secret or env var) to speed up model downloads and avoid rate limits.
- For local development, you can mount the `app/data` and `app/vectorstore` directories into the container so you don't need to rebuild the image after updating the catalog:

```bash
docker run -v "$(pwd)/app/data:/app/data" -v "$(pwd)/app/vectorstore:/app/vectorstore" -p 8000:8000 --env-file .env shl-recommender:latest
```

- Health check URL: `http://localhost:8000/health` (same as the Render health check).

Troubleshooting
- If the container fails to start due to FAISS or embedding model errors, verify the vectorstore files exist at `app/vectorstore/` and that the catalog JSON path points to a valid file.
- If model downloads are slow or fail, set `HF_TOKEN` as an env var to enable authenticated downloads with higher rate limits.

## Render Deployment

`render.yaml` defines a Docker web service with `/health` as the health check. In Render, set secret environment variables for `OPENAI_API_KEY` or `GEMINI_API_KEY` if `SHL_USE_LLM=true`.

Recommended deployment flow:

1. Scrape catalog locally or in a build job.
2. Build FAISS index with `python scripts/build_index.py`.
3. Commit or attach generated catalog/index artifacts according to your deployment policy.
4. Deploy with the included `render.yaml`.

## Design Tradeoffs

The agent is intentionally deterministic for routing and validation. LLM usage is optional and limited to final wording, which protects schema stability and prevents fabricated recommendations. The scraper extracts best-effort metadata from heterogeneous SHL pages and validates every item through Pydantic before indexing. Retrieval gracefully falls back to BM25 if semantic dependencies or index artifacts are unavailable, but the intended production path is BM25 plus FAISS.

#Recommender Agent

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
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
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

Run tests:

```bash
pytest
```

The test suite includes at least 15 automated checks covering:

- vague query clarification
- refinement handling
- grounded comparison
- hallucination prevention
- prompt injection resistance
- off-topic refusal
- legal and general hiring refusal
- schema stability
- Recall@10 for a known catalog item

## Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 --env-file .env shl-recommender
```

Production startup command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Render Deployment

`render.yaml` defines a Docker web service with `/health` as the health check. In Render, set secret environment variables for `OPENAI_API_KEY` or `GEMINI_API_KEY` if `SHL_USE_LLM=true`.

Recommended deployment flow:

1. Scrape catalog locally or in a build job.
2. Build FAISS index with `python scripts/build_index.py`.
3. Commit or attach generated catalog/index artifacts according to your deployment policy.
4. Deploy with the included `render.yaml`.

## Design Tradeoffs

The agent is intentionally deterministic for routing and validation. LLM usage is optional and limited to final wording, which protects schema stability and prevents fabricated recommendations. The scraper extracts best-effort metadata from heterogeneous SHL pages and validates every item through Pydantic before indexing. Retrieval gracefully falls back to BM25 if semantic dependencies or index artifacts are unavailable, but the intended production path is BM25 plus FAISS.

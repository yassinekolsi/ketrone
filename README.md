# Legal GraphRAG Pipeline for Oman Legislation

This repository implements a practical GraphRAG pipeline for the public Oman legislation portals:

- Arabic source of truth: `https://qanoon.om/`
- English translations when linked: `https://decree.om/`

The implementation is intentionally deterministic where the site already exposes structure: REST discovery, embedded English links, PDF URL issuer codes, title-adjacent number blocks, and inline legal cross-references. LLMs are used only where they add value: topic extraction and optional final answer synthesis.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker compose up -d
```

Run the required day-1 REST smoke test:

```powershell
python -m src.scraper.rest_probe
```

Expected result: both `qanoon` and `decree` return `REST - ok id=...`. If either endpoint fails in a future run, the crawler supports sitemap fallback.

## Sample Pipeline

A small real sample is included in `data/sample_output/`, generated from live Qanoon pages. To regenerate it:

```powershell
python -m src.scraper.crawl crawl --limit 8 --sample
python -m src.llm_agents.chunking --input-path data\sample_output\documents.jsonl --output-path data\sample_output\chunks.jsonl
python -m src.llm_agents.topic_extractor --input-path data\sample_output\documents.jsonl --output-path data\sample_output\topics.jsonl
python -m src.vector_ops.embed --chunks-path data\sample_output\chunks.jsonl --topics-path data\sample_output\topics.jsonl --output-dir data\sample_output
python src\search_client.py --query "What decrees govern health specialties?" --data-dir data\sample_output --top-k 10 --top-n 3
```

Cross-encoder reranking is part of this default command. Use `--no-rerank` only for an ablation or a constrained demo environment.

The search client supports two backends:

- `--backend local` reads the JSONL artifacts from disk for quick offline demos.
- `--backend neo4j` queries Neo4j full-text/vector indexes and expands context through live graph relationships.

## Evaluation Pipeline

The six-document sample proves the plumbing, not retrieval quality. For a broader retrieval check, run a larger crawl and the labeled evaluation harness:

```powershell
python -m src.scraper.crawl crawl --limit 80 --output-dir data\eval_output --state-path data\eval_state.sqlite3
python -m src.llm_agents.chunking --input-path data\eval_output\documents.jsonl --output-path data\eval_output\chunks.jsonl
python -m src.llm_agents.topic_extractor --input-path data\eval_output\documents.jsonl --output-path data\eval_output\topics.jsonl
python -m src.vector_ops.embed --chunks-path data\eval_output\chunks.jsonl --topics-path data\eval_output\topics.jsonl --output-dir data\eval_output
python -m src.evaluation.evaluate_retrieval --data-dir data\eval_output --output-path data\eval_output\retrieval_eval.json --top-k 20 --metric-k 3 --dense-weight 0.35
```

Latest local evaluation: 80 posts inspected, 67 full documents written, 247 chunks generated, and six known-answer positive queries. The committed evaluator measures every stage over the same candidate depth and records whether the cross-encoder actually ran:

| Stage | P@3 | R@3 | H@1 | H@3 | MRR | Reranker executed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense only | 0.222 | 0.667 | 0.667 | 0.667 | 0.699 | 0/6 |
| Hybrid + graph context | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 | 0/6 |
| Hybrid + graph context + rerank | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 | 6/6 |

The hybrid stage lifts recall and MRR substantially. Reranking preserves the ceiling on this small/easy label set rather than producing a fabricated extra gain; a larger hard-query set is needed to measure its incremental benefit. The negative-control query also shows that production use still needs a calibrated rejection threshold.

Full crawl:

```powershell
python -m src.scraper.crawl crawl --limit 100 --sample
python -m src.scraper.crawl crawl
```

The crawler uses SQLite checkpointing at `data/crawl_state.sqlite3`, randomized pacing, retries with exponential backoff, resumable WordPress REST pagination, and replay of pending/failed slugs before new discovery work.

Optional crawler controls are available through `.env`: `CRAWLER_PROXY_URL` for proxy routing and `CRAWLER_COOKIE_JAR` for cookie persistence. The `--browser-fallback` flag can use Playwright for detail-page HTML fallback if Playwright is installed locally.

## Graph Loading

Start Neo4j:

```powershell
docker compose up -d
```

Generate pipeline artifacts, then load them:

```powershell
python -m src.llm_agents.chunking
python -m src.llm_agents.topic_extractor
python -m src.vector_ops.embed
python -m src.vector_ops.community --output-path data\raw\communities.jsonl --summaries-path data\raw\community_summaries.jsonl
python -m src.ingestion.load_graph --communities-path data\raw\communities.jsonl
```

Run a live Neo4j-backed search:

```powershell
python src\search_client.py --backend neo4j --query "Which decree abolished the Higher Institute of Health Specialties?" --top-k 10 --top-n 3
```

Neo4j defaults:

- URI: `bolt://localhost:7687`
- User: `neo4j`
- Password: `password123`

Override with `.env` values from `.env.example`.

For Neo4j Aura, use the Aura-provided `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE`. If local Windows/Python certificate validation fails while the host is otherwise reachable, `neo4j+ssc://...` can be used as a local diagnostic workaround; keep `neo4j+s://...` for normal trusted environments.

## FastAPI And Render

Run the same search pipeline over HTTP:

```powershell
uvicorn src.api:app --reload
Invoke-RestMethod http://127.0.0.1:8000/search -Method Post -ContentType application/json -Body '{"query":"Which decree abolished the Higher Institute of Health Specialties?","top_n":3}'
```

Endpoints:

- `POST /search` runs hybrid candidate generation, graph expansion, cross-encoder reranking by default, and final synthesis.
- `GET /health` is the cheap liveness URL for Render and UptimeRobot.
- `GET /ready` checks the configured local or Aura search backend.
- `GET /docs` exposes Swagger UI.

`render.yaml` contains the build/start commands and non-secret defaults. Create a Render Blueprint from the repository, then provide `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `GOOGLE_API_KEY` in Render. Point UptimeRobot at `https://<service>.onrender.com/health`; use `/ready` for backend-aware monitoring.

The configured E5 model plus BGE cross-encoder is not a 512 MB workload. A free instance is useful for testing deployment wiring but can run out of memory on its first real search; choose an instance with roughly 2 GB or more for the default three-stage pipeline. UptimeRobot can detect outages and generate traffic, but it cannot guarantee zero downtime or add memory to the service.

## Gemini

The LLM client supports Google Gemini directly through the Generative Language REST API:

```powershell
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-3.1-flash-lite
```

When these variables are set, topic extraction and final synthesis use Gemini. If no hosted LLM key is configured, topic extraction falls back to deterministic taxonomy matching and search returns extractive snippets.

## Design Highlights

- REST first: `qanoon.om/wp-json/wp/v2/posts` and `decree.om/wp-json/wp/v2/posts` are checked before crawling.
- No slug guessing: English pages are followed from the embedded `decree.om` link in each Arabic post.
- Gazette handling: `og####` posts are index pages, not `Document` nodes.
- Exact number parsing: primary document numbers are parsed only from the paragraph directly below `h1.entry-title`.
- Issuer metadata: ministerial issuer codes are parsed from `data.qanoon.om/ar/md/{issuer}/...` PDF paths.
- Cross-reference edges: `REFERENCES`, `REPEALS`, and `AMENDS` come from inline links plus Arabic legal keywords.
- Batched graph writes: documents, chunks, topics, legal references, and optional community assignments are written with `UNWIND` batches.
- Multilingual embeddings: default model is `intfloat/multilingual-e5-small`, with a deterministic hash fallback if model dependencies are unavailable.
- Search backends: local JSONL retrieval for portable demos and live Neo4j retrieval for graph/vector/full-text queries.
- Bonus features: topic merge audit, default cross-encoder reranking, and Louvain community detection with LLM-generated community summaries.

## Key Commands

```powershell
python -m src.scraper.crawl probe
python -m src.scraper.crawl crawl --limit 50 --sample
python -m src.llm_agents.chunking --input-path data\sample_output\documents.jsonl --output-path data\sample_output\chunks.jsonl
python -m src.llm_agents.topic_extractor --input-path data\sample_output\documents.jsonl --output-path data\sample_output\topics.jsonl
python -m src.vector_ops.embed --chunks-path data\sample_output\chunks.jsonl --topics-path data\sample_output\topics.jsonl --output-dir data\sample_output
python -m src.vector_ops.merge_topics --topics-path data\sample_output\topics.embedded.jsonl --threshold 0.88
python -m src.vector_ops.community --documents-path data\sample_output\documents.jsonl --topics-path data\sample_output\topics.jsonl --output-path data\sample_output\communities.jsonl --summaries-path data\sample_output\community_summaries.jsonl
python -m src.ingestion.load_graph --documents-path data\sample_output\documents.jsonl --chunks-path data\sample_output\chunks.embedded.jsonl --topics-path data\sample_output\topics.embedded.jsonl --communities-path data\sample_output\communities.jsonl
python -m src.evaluation.evaluate_retrieval --data-dir data\sample_output --output-path data\sample_output\retrieval_eval.json
python src\search_client.py --query "What laws regulate sports entities?" --data-dir data\sample_output
```

## Tests

```powershell
python -m pytest
```

The tests cover English-link extraction, gazette skipping, exact number-block parsing, PDF issuer parsing, header/footer removal, Markdown table preservation, chunking, crawl-state resume records, UA rotation, exponential-backoff configuration, default reranking, API behavior, and deterministic cross-reference classification.

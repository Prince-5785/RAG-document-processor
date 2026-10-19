# InsureQuery (RAG Document Processor): Review and Roadmap

Review of the RAG project against the resume, plus changes to turn it from a hackathon project into your own product.

---

## Part 1: Resume vs. Code

Resume project: **InsureQuery – LLM Powered Insurance System**

| Resume claim | In the code? |
|---|---|
| FastAPI, ChromaDB, Groq API, HuggingFace, PyPDF2 | **Yes.** `main_api.py`, `vector_store.py`, `llm_service.py` and `embedding_service.py` (sentence-transformers) exist. PyPDF2 is only a fallback after pdfplumber. |
| End-to-end RAG pipeline | **Yes.** Retrieve, then cross-encoder rerank, then LLM. |
| Source citations | **Partial.** The Streamlit decision path returns `relevant_clauses` with file name and score. The `/hackrx/run` API returns plain answer strings with no sources. |
| Similarity-threshold guardrail to avoid hallucinated answers | **No.** `score_threshold: 0.3` is read from config but never applied. `similarity_search_with_threshold()` exists but nothing calls it. The only guardrail is a prompt instruction to say "not found". |
| Evaluated retrieval accuracy on a hand-labeled Q/A set, +35% from tuning chunk size and threshold | **No.** No labeled dataset, eval script or metrics. Tests are unit tests only. The only accuracy note is a commit message, "100 percent accuracy (for sample data)". |
| Token-by-token SSE streaming | **No.** `/hackrx/run` returns one JSON response. No `StreamingResponse`, no `stream=True` on Groq calls. |
| Cached embeddings in Redis | **No.** The cache is pickle files on disk (`data/cache`), query embeddings only. Redis is not in the code, `requirements.txt` or Docker. |
| Input validation and error handling for uploads and API failures | **Partial.** Retries, key rotation, a model fallback chain and try/except exist. No file size or type check on the API, and it assumes every download is a PDF. |
| Containerized with Docker | **Partial.** The Dockerfile runs Streamlit, not the FastAPI service. `docker-compose.yml` references an `nginx.conf` that doesn't exist. |
| GitHub Actions CI | **No.** No `.github/` folder. |

Other mismatches:
- The project is named "CodersHub / RAG document processor", not InsureQuery. Rename the repo and README to match the resume.
- The resume lists PyPDF2 as the parser, but pdfplumber is the primary one.

### Security issue: fix now

`main_api.py` has the API key hardcoded as `COMPETITION_API_KEY = "2535b8fd…"`, and it is in git history (commit `ed3c9aa`). Move it to an environment variable. Rotate the key if it still works for anything, and ideally scrub the history.

---

## Part 2: Changes needed to make the resume true

1. **Threshold guardrail.** In `answer_question` and `query`, drop candidates below the threshold. If the best rerank score is below the cutoff, return "not found" without calling the LLM. Cross-encoder scores are not on the same scale as cosine similarity, so tune the cutoff on your eval set.
2. **Citations.** Change the response model to `{answer, sources: [{file, page, chunk_text, score}]}`. Store the page number during PDF extraction.
3. **Evaluation script.** Create `eval/qa_set.json` with 30–50 hand-written questions, each labeled with the expected chunk or answer. Write `eval/run_eval.py` to report hit@k or recall@k. Run it for the baseline, then for chunk-size and threshold variations, and save the results table in the README. Only claim 35% if your numbers show it. Otherwise change the resume to the real figure.
4. **SSE streaming.** Add `POST /query/stream`. Call Groq with `stream=True` and return `StreamingResponse(..., media_type="text/event-stream")`, sending the sources first and then the tokens.
5. **Redis.** Replace the pickle cache with Redis, keyed by a hash of model name and text, with a TTL. Add Redis to `docker-compose.yml`. Fall back to no cache if Redis is down.
6. **Validation.** Check file type, size limit and content type on download and upload. Return proper 4xx errors and add a request timeout.
7. **Docker.** Make the Dockerfile run `uvicorn main_api:app`. Add `redis` to compose and remove or fix the nginx service.
8. **CI.** Add `.github/workflows/ci.yml` that runs `pytest` and builds the Docker image.

---

## Part 3: Changes that would make the project better

- **Hybrid retrieval.** Add BM25 next to vector search and merge the results. Insurance clauses are full of exact terms like "waiting period" and "sub-limit".
- **Fix the shared index.** `/hackrx/run` calls `reset_index()` on every request, so concurrent users overwrite each other's documents. Use one collection per document (keyed by file hash), or skip re-indexing a document already seen.
- **Parallel LLM calls.** `asyncio.gather` over questions helps, but the embedding and cross-encoder calls block the event loop. Wrap them in `run_in_executor`.
- **Chunking.** Currently 1000/300 character chunks. Add page and section metadata. Try smaller chunks with parent-context expansion and let the eval numbers decide.
- **Tests.** Add API tests with FastAPI's `TestClient` and mocked Groq.
- **README.** Add an architecture diagram, the eval results table and a short demo GIF. Fix `claim_id` generation, because Python's `hash()` changes between runs.

Highest value: threshold guardrail, eval script, SSE streaming and Redis. They make the resume bullets true and give you concrete numbers for interviews.

---

## Part 4: Turning it into your own project (not a hackathon entry)

### 4.1 Remove the hackathon-specific parts

- **Hardcoded contest key.** Replace `COMPETITION_API_KEY` with per-user API keys or JWT auth, stored in a database and read from the environment.
- **Names and branding.** Change "CodersHub RAG API for HackRX", the "CodersHub Insurance Query Solver" UI title and the `/hackrx/run` route. Use one project name (InsureQuery) everywhere: README, config, Docker image, API title.
- **Request shape.** The API takes `{documents: <URL>, questions: [...]}` because the contest required it. Replace it with:
  - `POST /documents` to upload a file and get a `document_id`
  - `GET /documents` and `DELETE /documents/{id}`
  - `POST /query` with `{document_ids, question}`
- **Git history.** Start a fresh repo or clean the history so the committed key and the "100 percent accuracy (for sample data)" commit message don't follow you. Rotate any key that was ever committed.
- **Git hygiene.** Remove `.DS_Store` and `__MACOSX` from the repo, and rename the repo to match the project.

### 4.2 Fix design flaws that only show up with real users

- **Shared global index.** Every request wipes the single ChromaDB collection. Use one collection per document or per user, or a `document_id` metadata filter. Skip re-indexing files whose hash you've seen.
- **Blocking work in async code.** Embedding and cross-encoder calls run synchronously inside `async` handlers and freeze the server under load. Use a thread pool or a worker.
- **Indexing in the request.** Move document processing to a background job (FastAPI `BackgroundTasks`, or Celery/RQ with Redis). The client polls `/documents/{id}/status`.
- **API key rotation as a scaling method.** Rotating several Groq keys works around rate limits for a contest. For a product, use one key with proper retry and rate limiting, and make the provider swappable behind an interface.
- **Local state.** The pickle cache, on-disk ChromaDB and `data/` folder assume one machine. Use Redis for caching and a persistent vector store volume, or Postgres with pgvector.
- **Insecure pickle cache.** `pickle.load` on files in a cache folder is unsafe. Redis fixes this.
- **Unstable claim IDs.** Use a UUID or a content hash instead of Python's `hash()`.

### 4.3 Define the scope

Pick one direction:

- **Insurance-specific.** Matches the InsureQuery name. Make the claim decision (approve or reject with a payout) a proper structured feature: output a Pydantic schema, check it against the retrieved clauses, and return the clause IDs it relied on.
- **General "chat with your documents".** Use domain-neutral prompts, with the insurance prompts as one preset.

Either way, move the prompts out of code and into versioned, testable files.

### 4.4 Add what a real project needs

- **Auth and users.** Signup/login (JWT), per-user documents, per-user rate limits.
- **Real front end.** Streamlit is fine for a demo. A small React UI with upload, chat and a source viewer that highlights the cited passage matches your React/TypeScript skills.
- **Observability.** Structured logging, request IDs, and a `/metrics` endpoint with latency per stage (retrieve, rerank, LLM).
- **Config and secrets.** Use `pydantic-settings` with a `.env.example`, and keep secrets out of YAML.
- **Tests and CI.** API tests with `TestClient` and a mocked LLM, plus GitHub Actions for tests, lint and Docker build.
- **Docs.** Architecture diagram, `docs/` folder, screenshot or GIF, and the eval results table.

---

## Part 5: Suggested order of work

1. Rename, remove the key, and start a clean repo.
2. Replace the single global index with per-document collections, then add upload, list and delete endpoints.
3. Add auth, background indexing, and Redis for the cache and job queue.
4. Add the threshold guardrail, citations with page numbers, the eval set and SSE streaming.
5. Add hybrid retrieval (BM25 plus vectors), then the React UI.
6. Add CI, Docker (API, Redis and vector store in one compose file), and a deployed demo.

Steps 1–3 turn it from a contest script into an application. Steps 4–5 are what you'll talk about in interviews.

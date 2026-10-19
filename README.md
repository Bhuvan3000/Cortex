# Cortex

RAG-based document Q&A system. Upload PDFs, then ask questions and get answers grounded in the documents, with source citations.

Stack: FastAPI, LangChain, ChromaDB, local Sentence Transformer embeddings, Gemini for answer generation.

## Pipeline

1. Upload a PDF to `/documents`
2. Text is extracted per page and split into overlapping chunks
3. Chunks are embedded locally with `all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection with metadata (`doc_id`, `source`, `page`, `chunk_index`, `category`, `tags`, `uploaded_at`)
4. `/query` embeds the question, runs cosine similarity search with optional metadata filters, and returns ranked source chunks
5. If `GOOGLE_API_KEY` is set, the retrieved chunks are passed to Gemini, which writes an answer citing chunks as `[1]`, `[2]` that match the `rank` of each source. Without a key, the API returns retrieval-only results

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

On Windows, activate with `.venv\Scripts\activate` and copy with `copy .env.example .env`.

Interactive docs: http://localhost:8000/docs

The first start downloads the embedding model (about 90 MB).

## Configuration

Set in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | empty | Enables Gemini answer generation |
| `CORTEX_LLM_MODEL` | `gemini-3.8-flash` | Gemini model name |
| `CORTEX_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CORTEX_CHUNK_SIZE` | `1000` | Characters per chunk |
| `CORTEX_CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `CORTEX_TOP_K` | `4` | Default number of chunks retrieved |
| `CORTEX_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `CORTEX_PERSIST_DIR` | `./data/chroma` | ChromaDB storage |
| `CORTEX_UPLOAD_DIR` | `./data/uploads` | Stored PDFs |

For non-English documents, set `CORTEX_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and re-upload with `overwrite=true`.

## API

```bash
curl -F "file=@paper.pdf" -F "category=ai-research" -F "tags=agents,llm" http://localhost:8000/documents

curl http://localhost:8000/documents

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What problem does the paper address?", "top_k": 4}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the method", "filters": {"category": "ai-research", "page": [1, 2]}, "score_threshold": 0.3}'

curl -X DELETE http://localhost:8000/documents/<doc_id>
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service status and configuration |
| `/documents` | POST | Ingest a PDF (`file`, `category`, `tags`, `overwrite`) |
| `/documents` | GET | List indexed documents |
| `/documents/{doc_id}` | DELETE | Remove a document and its chunks |
| `/query` | POST | Similarity search with metadata filters and optional generation |

Query options: `question`, `top_k` (1 to 20), `filters`, `score_threshold`, `generate` (set `false` for retrieval only).

Filter keys: `doc_id`, `source`, `category`, `page`. Each accepts a single value or a list.

Re-uploading an identical PDF is detected by content hash and skipped unless `overwrite=true`.

## Limitations

- Scanned or image-only PDFs are not supported (no OCR)
- Answers are only as good as the retrieved chunks, so specific questions work better than broad ones

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
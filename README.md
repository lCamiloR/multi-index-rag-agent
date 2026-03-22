# Multi-index RAG agent

CLI chat application that ingests documents from disk into **separate FAISS indexes** (one per asset folder), then answers questions with a **LangGraph** workflow: an LLM **routes** the user’s intent to either a general “organization” assistant or a **domain-specific RAG assistant** that can call retrieval against exactly one index.

## Requirements

- **Python 3.13+** (see `pyproject.toml`)
- **LLM Model** for whatever model `LLM_MODEL_VERSION` names: this repo only calls LangChain’s `init_chat_model(model=...)` and does not read an API key from `AgentConfig`.
- Enough disk and RAM for **Sentence Transformers** embeddings (models download on first use)

## Install dependencies

This project uses **[uv](https://docs.astral.sh/uv/)** and a lockfile.

```bash
cd /path/to/multi-index-rag-agent
uv sync
```

That creates a virtual environment and installs everything from `uv.lock` / `pyproject.toml`.

If you prefer pip:

```bash
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Configuration

Create a `.env` file in the **project root** (same directory as `pyproject.toml`). Values declared on `AgentConfig` in `src/config.py` are loaded from the environment and from that file via Pydantic Settings.

| Variable | Purpose |
|----------|---------|
| `LLM_MODEL_VERSION` | Chat model id passed to LangChain’s `init_chat_model` (required). The CLI builds `RagAgent(AGENT_CONFIG.LLM_MODEL_VERSION)` in `src/main.py`. |
| `EMBEDDING_MODEL` | Hugging Face model id for sentence-transformers / `HuggingFaceEmbeddings` (required). |

There is **no** API key field in `AgentConfig` or elsewhere in application code: `RagAgent` only passes the model string to `init_chat_model`. Supply credentials the way LangChain and your chosen provider expect (environment, CLI profile, etc.).

Optional paths (override via env using the same names as in `AgentConfig`):

- `ASSETS_PATH` — defaults to `assets/`
- `FAISS_INDEXING_PATH` — defaults to `faiss_index/`

## How the project works

### 1. Assets and indexes

Place files under `assets/<INDEX_NAME>/`, where `<INDEX_NAME>` is the folder name; that name becomes the **FAISS index name** on disk.

Supported extensions: **pdf, docx, xlsx, csv, txt** (see `FileIngestionHandler`).

Each index folder **must** contain a `prompt.yaml` used when building the graph: **`system`**, **`intent`**, and **`classification_prompt`**. The router prompt is assembled from every index’s intent and classification text; the graph raises if `prompt.yaml` is missing.

### 2. Ingestion (on startup)

When you run the app, `ingest_files_routine()` in `src/main.py`:

1. Fingerprints supported files under `assets/` (mtime + size).
2. Compares them to `faiss_index/.ingestion_manifest.toon` and checks that every asset folder has a matching FAISS index.
3. If nothing changed and indexes exist, ingestion is **skipped**.
4. Otherwise: **Docling** converts each file to markdown (and writes a sibling `.md`), **Chunker** splits text (`MarkdownHeaderTextSplitter` on `##` for non-CSV; **CSV** uses line-oriented chunks with row metadata), **Hugging Face embeddings** are computed, and **FAISS** indexes are written under `faiss_index/`.

### 3. Chat (LangGraph)

`RagAgent` (`src/reasoning/graph.py`) compiles a graph that:

- Runs a **router** node to classify the latest user message into an intent.
- Routes to **`org_assistant`** for organizational / non-domain chat, or to **`<index>_assistant`** for a domain that has a FAISS index and `prompt.yaml`.
- Domain assistants are bound to a **single** retrieval tool per index (`RetrievalHandler.query_vectorstore`), so tool use stays scoped to that index.

When `RagAgent` compiles the graph (on startup), it writes a timestamped **`graph_*.mmd`** Mermaid diagram at the project root (`save_graph_schema` in `src/reasoning/graph.py`).

### 4. Run the CLI

From the project root, use a module run so imports resolve:

```bash
uv run python -m src.main
```

The app ingests if needed, then starts an interactive **Rich** prompt; type `exit` to quit.

---

**Summary:** Documents live in `assets/<domain>/` with `prompt.yaml`; ingestion builds/updates `faiss_index/` with a TOON manifest for change detection; the CLI agent routes questions and retrieves only from the index that matches the classified intent.

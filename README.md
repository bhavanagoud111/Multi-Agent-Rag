# MultiAgentic RAG

A **multi-agent research RAG (retrieval-augmented generation)** demo built with **[LangGraph](https://github.com/langchain-ai/langgraph)**. The system routes user questions, plans multi-step research over an indexed document, retrieves evidence with hybrid search and reranking, synthesizes answers, and runs a **hallucination check** with optional human-in-the-loop retry.

---

## Use case

This project is aimed at teams and developers who want a **concrete pattern** for:

- **Answering questions grounded in long PDFs** (compliance, sustainability reports, policies) instead of trusting the model’s memory alone.
- **Breaking complex questions into steps** (plan → retrieve per step → aggregate → answer) rather than a single retrieval call.
- **Improving reliability** with structured routing, ensemble retrieval, reranking, and an explicit **groundedness / hallucination** gate before accepting the final answer.

Point **`retriever.file`** at **any PDF** you want (annual report, policy, handbook, spec, etc.), and set **`document.title`** / **`document.scope`** in `config.yaml` so the router and answers match your use case.

---

## How it works

### High-level flow

1. **User input** enters the main LangGraph (`main_graph/graph_builder.py`) with conversation state (`AgentState` in `main_graph/graph_states.py`).
2. **Router** (`analyze_and_route_query`) classifies the message into one of three kinds:
   - **`document`** — in scope for the indexed PDF; continue down the RAG path.
   - **`more-info`** — the model needs clarification; **`ask_for_more_info`** responds without retrieval.
   - **`general`** — off-topic or not answerable from the index; **`respond_to_general_query`** declines without using the vector store.
3. **Research path** (for `document`):
   - **`create_research_plan`** produces an ordered list of research steps (structured output).
   - **`conduct_research`** runs **one step at a time**: for each step it invokes the **researcher subgraph** (`subgraph/graph_builder.py`).
4. **Researcher subgraph** (per step):
   - **`generate_queries`** expands the step into multiple search queries (plus the original step text).
   - **`retrieve_and_rerank_documents`** runs **in parallel** for each query (`Send` fan-out): documents come from a **Chroma** vector store loaded from disk, via an **ensemble** (similarity, MMR, BM25) and **Cohere reranking** (`ContextualCompressionRetriever`).
   - Retrieved chunks are merged in state and returned to the main graph.
5. After all plan steps finish, **`respond`** formats retrieved chunks as XML context and generates the **final answer** (stronger model from config).
6. **`check_hallucinations`** scores whether the answer is grounded in the retrieved facts. If not, the graph can **`interrupt`** and the CLI (`app.py`) may ask you to retry generation (`y` resumes with `Command`).

The graph uses **`MemorySaver`** checkpointing so thread state and interrupts integrate with LangGraph’s resume flow.

### Indexing pipeline (`retriever/retriever.py`)

When `load_documents` is **`true`** in `config.yaml`:

1. The PDF is converted to Markdown with **Docling** (`DocumentConverter`).
2. Text is split with **`MarkdownHeaderTextSplitter`** using the headers defined in config (`#` / `##`).
3. Chunks are embedded with **OpenAI embeddings** and stored in **Chroma** under `retriever.directory` (default `vector_db`) and `retriever.collection_name`.
4. **BM25** and vector retrievers are built for later ensemble use during indexing validation; the **running app** loads the persisted store and rebuilds the compression retriever in the subgraph module.

Configuration is read from **`config.yaml`** at the project root (`utils/utils.py`).

---

## Tech stack

| Area | Choices |
|------|---------|
| Orchestration | LangGraph (`StateGraph`, checkpoints, `interrupt`, parallel `Send`) |
| LLMs | OpenAI via `langchain-openai` (`gpt-4o`, `gpt-4o-mini` names from config) |
| Vector store | Chroma (persistent) |
| Retrieval | Ensemble (similarity + MMR + BM25), **Cohere** rerank (`langchain_cohere`) |
| PDF ingestion | Docling → Markdown → header-based chunks |
| Config | YAML (`config.yaml`) |

---

## Prerequisites

- **Python 3.10+** recommended.
- **API keys** (set in environment or a `.env` file in the project root):
  - **`OPENAI_API_KEY`** — embeddings, chat models, query generation.
  - **`COHERE_API_KEY`** — reranking in the researcher subgraph.

Install dependencies:

```bash
pip install -r requirements.txt
```

The repository’s `requirements.txt` may not list every transitive dependency; if imports fail at runtime, install the missing packages (for example `langgraph`, `docling`, `langchain-text-splitters`, `python-dotenv`, `pyyaml`, `pydantic`) until the app starts cleanly.

---

## Configuration (`config.yaml`)

**Document persona (any PDF/report use case)**

| Key | Role |
|-----|------|
| `document.title` | Short label for prompts (e.g. “Q3 2024 earnings”, “Employee handbook”). |
| `document.scope` | What counts as “in scope” for the **`document`** route (answerable from your PDF). |

**Retriever / index**

| Key | Role |
|-----|------|
| `file` | Path to the PDF to index (default placeholder: `retriever/your-document.pdf`). |
| `load_documents` | **`true`** to convert/split/embed the PDF into Chroma; required for first-time indexing. |
| `headers_to_split_on` | Markdown header levels used when splitting (e.g. `#`, `##`). Tune if your PDF structure differs. |
| `collection_name` | Chroma collection name (change if you switch corpora). |
| `directory` | Folder for persisted Chroma data (default `vector_db`). |
| `top_k` / `top_k_compression` | Retrieval and post-rerank depth. |
| `ensemble_weights` | Weights for similarity, MMR, and BM25 in the ensemble. |
| `cohere_rerank_model` | Cohere reranker model id. |

Under `llm`, model names and `temperature` are set for routing, planning, and answering.

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/nicoladisabato/MultiAgenticRAG.git
cd MultiAgenticRAG
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure secrets

Create a `.env` file in the project root (or export variables in your shell):

```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

### 5. Index the document

1. Copy your PDF into the project (e.g. `retriever/my-report.pdf`) and set **`retriever.file`** to that path.
2. Edit **`document.title`** and **`document.scope`** so the assistant matches your content (e.g. finance vs HR vs technical spec).
3. Open **`config.yaml`** and set **`load_documents: true`**.
4. Run:

```bash
python3 -m retriever.retriever
```

Wait until indexing completes and Chroma data appears under `vector_db/` (or your configured `directory`).

5. For normal Q&A runs you can set **`load_documents: false`** to skip re-ingestion on every indexer run (the vector store on disk is still used by the app). **First-time indexing must use `load_documents: true`.**

### 6. Run the chat application

```bash
python3 app.py
```

- Type questions at the `>` prompt.
- Type **`-q`** to quit.
- If the hallucination step triggers and you choose to retry, follow the on-screen prompt (**`y`** to regenerate when asked).

Ask questions that the indexed PDF can support (figures, policies, narrative sections, tables).

---

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | CLI loop, streaming output, interrupt/resume for optional retry. |
| `main_graph/` | Main LangGraph: routing, planning, research loop, response, hallucination gate. |
| `subgraph/` | Researcher LangGraph: query expansion, parallel retrieval + rerank. |
| `retriever/` | PDF → Markdown → chunks → Chroma + BM25 setup for indexing. |
| `utils/` | Shared config load, prompts, document merge helpers. |
| `config.yaml` | Models, paths, retrieval hyperparameters. |

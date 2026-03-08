# Healthcare RAG System

A Retrieval-Augmented Generation (RAG) pipeline for querying healthcare and clinical documentation. Ask natural-language questions against your own PDF documents and receive grounded, cited answers powered by a local embedding model and the Groq LLM API.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that combines two steps:

1. **Retrieve** — find the most relevant passages from a document store based on the user's question
2. **Generate** — pass those passages as context to a large language model (LLM) so it answers using only what was retrieved

This prevents the LLM from hallucinating facts it was never trained on and keeps answers grounded in your specific documents.

---

## Key Concepts

| Concept | What it does in this project |
|---|---|
| **Chunking** | PDFs are split into overlapping token-sized segments (750 tokens, 100 overlap) so long documents fit within LLM context windows |
| **Vector Embeddings** | Each chunk is converted to a dense numerical vector using `all-MiniLM-L6-v2` (runs locally). Similar meaning = similar vectors |
| **ChromaDB** | A local vector database that stores embeddings and enables fast similarity search |
| **BM25** | A classical keyword-based ranking algorithm. Catches exact term matches that vector search can miss |
| **Hybrid Retrieval** | Combines vector search + BM25 results using Reciprocal Rank Fusion (RRF), getting the best of both approaches |
| **Cross-Encoder Reranking** | A second, more accurate model (`ms-marco-MiniLM-L-6-v2`) rescores each candidate chunk against the query to pick the most relevant ones |
| **Citation Guard** | Declines to answer if the reranker confidence is too low — important in healthcare where a wrong answer is worse than no answer |
| **Groq LLM** | A hosted LLM API (Llama 3.3 70B) used only for the final answer generation step |

---

## Pipeline Phases

### Phase 1 — Basic Vector Search
```
Question → Vector Search → Deduplicate → LLM → Answer
```

### Phase 2 — Production (Hybrid + Rerank + Guard)
```
Question → Vector Search + BM25 → RRF Fusion → Cross-Encoder Rerank → Citation Guard → LLM → Answer
```

Phase 2 is more accurate and safer for healthcare use. Phase 1 is useful for debugging retrieval quality.

---

## Project Structure

```
RAG/
├── ingest.py                   # Step 1: load, chunk, embed, store documents
├── query.py                    # Step 2: ask questions against stored documents
├── config/
│   ├── settings.yaml           # chunk size, retrieval k, model names, thresholds
│   └── prompts.yaml            # system and answer prompt templates
├── data/
│   ├── raw/                    # place your PDF/HTML/TXT documents here
│   ├── processed/              # chunked JSON output (used by BM25)
│   ├── chroma_db/              # ChromaDB vector store (auto-created)
│   └── eval/                   # golden dataset and evaluation results
└── src/
    ├── ingestion/              # loader, chunker, embedder
    ├── retrieval/              # vector store, BM25, hybrid, reranker
    ├── generation/             # LLM generator, prompt manager, citation guard
    ├── evaluation/             # golden dataset builder, RAGAS evaluator
    └── pipeline.py             # orchestrates phases 1 and 2
```

---

## Setup

### Prerequisites

- macOS (Intel or Apple Silicon)
- [pyenv](https://github.com/pyenv/pyenv) installed
- A free [Groq API key](https://console.groq.com)

### 1. Install Python 3.11.9

> Python 3.13 is not supported — PyTorch dropped Intel Mac support in 2.3+ and Python 3.13 requires PyTorch 2.6+. Python 3.11 is the stable choice for Intel Macs.

```bash
pyenv install 3.11.9
```

### 2. Configure pyenv for this project

```bash
cd /path/to/RAG
pyenv local 3.11.9
```

### 3. Set up pyenv shell integration (one-time)

This ensures every new terminal automatically uses the correct Python:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Set your Groq API key (one-time)

```bash
echo 'export GROQ_API_KEY="your_key_here"' >> ~/.zshrc
source ~/.zshrc
```

Get a free key at [https://console.groq.com](https://console.groq.com).

### 5. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

> If you see torch-related errors, make sure you are on Python 3.11.9 and not 3.13. Run `python3 --version` to verify.

---

## Usage

### Step 1 — Ingest documents

Place your PDF, HTML, or TXT files in `data/raw/`, then run:

```bash
python3 ingest.py
```

To wipe and re-ingest from scratch:

```bash
python3 ingest.py --reset
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--docs` | `data/raw/` | Directory containing source documents |
| `--collection` | `docs` | ChromaDB collection name |
| `--reset` | off | Wipe existing collection before ingesting |

### Step 2 — Query

```bash
python3 query.py "What are the recommended treatments for type 2 diabetes?"
```

Use `--phase` to choose the pipeline:

```bash
# Phase 1 — basic vector search (faster, good for testing)
python3 query.py "What is insulin resistance?" --phase 1

# Phase 2 — hybrid + rerank + citation guard (default, more accurate)
python3 query.py "What does the WHO recommend for hypertension management?" --phase 2
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--phase` | `2` | `1` = vector only, `2` = hybrid + rerank + guard |
| `--collection` | `docs` | ChromaDB collection to query |

---

## Configuration

All tunable parameters are in `config/settings.yaml`:

```yaml
chunking:
  chunk_size: 750       # tokens per chunk
  chunk_overlap: 100    # overlap between chunks

retrieval:
  vector_top_k: 20      # candidates from vector search
  bm25_top_k: 20        # candidates from BM25 search
  final_top_k: 5        # chunks passed to the LLM after reranking

embedding:
  model: all-MiniLM-L6-v2   # local model, no API key needed

llm:
  model: llama-3.3-70b-versatile
  temperature: 0.0
  max_tokens: 1024

citation:
  decline_threshold: 0.0    # reranker score below this = decline to answer
```

---

## Evaluation

### Build a golden dataset

Run the interactive tool to create verified Q&A pairs:

```bash
python3 -m src.evaluation.build_dataset
```

The system answers each question, you approve or correct it, and verified pairs are saved to `data/eval/golden_dataset.json`.

### Run RAGAS evaluation

```bash
python3 -m src.evaluation.run_eval
```

Measures four metrics using [RAGAS](https://docs.ragas.io):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the claims in the answer supported by the retrieved chunks? |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Did retrieval find all the chunks needed to answer? |

Results are saved to `data/eval/eval_results.json`.

---

## Recommended Documents (Healthcare)

Good sources for healthcare RAG:

| Source | URL |
|---|---|
| WHO Clinical Guidelines | https://www.who.int/publications |
| CDC Treatment Guidelines | https://www.cdc.gov/guidelines |
| FDA Drug Labels | https://labels.fda.gov |
| NIH Health Information | https://www.nih.gov/health-information |
| PubMed Central (free papers) | https://pmc.ncbi.nlm.nih.gov |
# Diabeties_model_RAG

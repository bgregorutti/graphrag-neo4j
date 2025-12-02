# graphrag-neo4j

Lightweight Graph Retrieval-Augmented Generation (RAG) demo using Neo4j for structured context storage, SentenceTransformers for embeddings, and HuggingFace Transformers for text generation.

- Language: Python
- Main files: `store.py` (ingest + store graph), `rag.py` (retrieve + generate), `example.json` (sample data)
- Primary dependencies: neo4j, transformers, sentence-transformers, numpy, loguru, dotenv (see `requirements.txt`)

## What this does (short)
1. Parse structured document content and store it into a Neo4j graph with vector embeddings.
2. Retrieve relevant nodes by embedding similarity and graph expansion.
3. Build a prompt with retrieved context and generate an answer via a text-generation model.

## Quick prerequisites
- Python 3.8+
- Running Neo4j instance (uri, user, password)
- Network access to download models (SentenceTransformers and Transformers)

## Install
1. Create virtual environment
   python -m venv .venv
   source .venv/bin/activate
2. Install deps
   pip install -r requirements.txt

## Environment
Create a `.env` file with:
URI=bolt://localhost:7687
USER=neo4j
PASSWORD=your_password

## Example data
`example.json` contains a small document structure (title, sections, subsections) used by `store.py` to populate the graph.

## Usage

- Store example graph into Neo4j
  1. Ensure `.env` is present and Neo4j is reachable.
  2. Run:
     python store.py
  This will:
   - load `example.json`
   - delete existing nodes (full wipe)
   - create nodes with embeddings (uses `all-MiniLM-L6-v2`)
   - create `contains` / `is_part_of` relationships

- Run a sample Graph-RAG question
  python rag.py
  This will:
   - connect to Neo4j
   - initialize SentenceTransformer (`all-MiniLM-L6-v2`) and a text-generation pipeline (default `openai-community/gpt2`)
   - retrieve top-k relevant nodes and related nodes
   - build a prompt and run text generation

## Key functions (brief)
- store.parse(pdf_text) -> list
  Parse document text into sections/subsections/paragraphs.

- store.add_node_with_embedding(tx, node_type, name, content=None)
  Creates node with an embedding property (embedding computed with SentenceTransformer).

- store.store_graph(example, session)
  Persist example JSON into Neo4j with embeddings and relationships.

- store.query_paragraphs(question, driver) -> list
  Return paragraph contents for a given section substring.

- rag.graph_rag(question, driver, generator, top_k=5) -> str
  High-level RAG: retrieve context, create prompt, generate answer.

- rag.retrieve_relevant_nodes_with_graph(question, driver, top_k=5) -> list
  Retrieves top-k nodes by embedding similarity and expands context via graph traversal.

## Models & Defaults
- Embeddings: sentence-transformers model `all-MiniLM-L6-v2` (used in code)
- Generation: HuggingFace pipeline `text-generation` (example uses `openai-community/gpt2`); change to any compatible generative model as needed.
Note: model names can be replaced in the code to use larger or remote models.

## Notes & limitations
- store.py deletes all nodes on run — use with care.
- Embeddings are stored as a property on nodes; ensure Neo4j driver and database accept list properties as used.
- No batching or rate-limiting: large datasets may require adapting the code (batch writes, streaming).
- The example generation uses a small local GPT-2 model; for high-quality answers use an appropriate large model or hosted inference.

## File overview
- rag.py — retrieval + prompt construction + generation pipeline
- store.py — parsing, creating nodes/embeddings, relationships, and convenience query functions
- example.json — sample document structure
- requirements.txt — Python dependencies

## Contributing & License
- There is no LICENSE file in this repository. Add a license if you intend to make this public/open-source.
- Contributions: open issues / PRs with suggested improvements (ingestion robustness, batching, better parsing, configurable traversal depth, selective deletes).

## Contact
Repository: https://github.com/bgregorutti/graphrag-neo4j

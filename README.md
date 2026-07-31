# Document Q&A Assistant using RAG

A resume-ready Retrieval-Augmented Generation project that lets users upload documents, index them into a FAISS vector database, and ask questions using online LLM providers with answers grounded in retrieved document context.

## Features

- Upload and index PDF, TXT, and Markdown documents.
- Split long documents into overlapping chunks for better retrieval quality.
- Generate embeddings with an online provider such as OpenRouter or OpenAI.
- Store and reuse vectors locally with FAISS.
- Ask questions through a built-in browser UI or FastAPI endpoint.
- Return source snippets with each answer for explainability.
- Optional Streamlit chat UI.
- Dockerfile included for deployment.

## Architecture

```text
Document Upload
    |
    v
Text Extraction -> Chunking -> Embeddings -> FAISS Vector Store
                                                |
User Question -> Query Embedding -> Similarity Search
                                                |
                                                v
                                    Retrieved Context + Question
                                                |
                                                v
                                           Online Chat Model
                                                |
                                                v
                                      Answer with Source Snippets
```

## Project Structure

```text
document-assistant/
├── app/
│   ├── config.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── main.py
│   ├── rag_pipeline.py
│   └── schemas.py
├── data/documents/
├── vectorstore/
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

For the most reliable online demo, use OpenRouter. Create an API key at `https://openrouter.ai`, then set:

```env
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_CHAT_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
USE_LOCAL_FALLBACK=false
MAX_OUTPUT_TOKENS=700
```

You can still use direct OpenAI by setting `AI_PROVIDER=openai` and adding `OPENAI_API_KEY`, but this project now defaults to OpenRouter because it avoids depending on your exhausted OpenAI quota.

If you are only testing without a working provider key, set `USE_LOCAL_FALLBACK=true`. For an online portfolio deployment, keep it `false` so configuration problems fail clearly.

## Run the API

```bash
uvicorn app.main:app --reload
```

Open the web app at:

```text
http://localhost:8000/
```

Upload a document, click **Index Document**, then ask questions directly in the page.

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Upload a document:

```bash
curl -X POST "http://localhost:8000/upload" ^
  -F "file=@sample.pdf"
```

Ask a question:

```bash
curl -X POST "http://localhost:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What are the main points in the document?\"}"
```

## Optional Streamlit UI

Start the FastAPI server first, then run:

```bash
streamlit run streamlit_app.py
```

## Docker

```bash
docker build -t document-assistant .
docker run --env-file .env -p 8000:8000 document-assistant
```

## Deployment

This project includes deployment files for Render and Vercel.

See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for Vercel.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Render.

## Resume Bullets

- Built a Retrieval-Augmented Generation based Document Q&A Assistant using LangChain, FAISS, FastAPI, and OpenAI APIs for semantic search over user-provided documents.
- Designed an end-to-end ingestion pipeline with PDF/text loading, recursive chunking, embedding generation, local vector indexing, top-k retrieval, and context-grounded answer generation.
- Developed production-style FastAPI endpoints for document upload and AI inference, returning source snippets to improve transparency and answer traceability.

## Future Improvements

- Add RAGAS evaluation for faithfulness, answer relevancy, and context precision.
- Add authentication and per-user vector indexes.
- Add cloud deployment with AWS ECS, Azure App Service, or Render.
- Add conversation memory and multi-document filtering.

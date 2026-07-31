# Deployment Guide

This project is ready to deploy as a Render Web Service.

## 1. Prepare Secrets

Create or rotate your OpenRouter key, then keep it out of Git.

Required production environment variable:

```env
OPENROUTER_API_KEY=your_rotated_openrouter_key
```

The repository includes `render.yaml`, which sets all non-secret environment variables and marks `OPENROUTER_API_KEY` as a dashboard-only secret.

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Build document RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/document-assistant.git
git push -u origin main
```

Do not commit `.env`.

## 3. Deploy on Render

1. Go to `https://dashboard.render.com`.
2. Select **New +**.
3. Choose **Blueprint** if you want Render to use `render.yaml`, or choose **Web Service** manually.
4. Connect your GitHub repository.
5. Add the secret environment variable:

```env
OPENROUTER_API_KEY=your_rotated_openrouter_key
```

6. Deploy.

## Manual Render Settings

If you do not use the Blueprint, set these values manually:

```text
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Health Check Path: /health
```

Environment variables:

```env
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_rotated_openrouter_key
OPENROUTER_CHAT_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
USE_LOCAL_FALLBACK=false
VECTORSTORE_DIR=/tmp/vectorstore
DOCUMENTS_DIR=/tmp/documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
MAX_OUTPUT_TOKENS=700
```

## Important Notes

- Render free instances use ephemeral storage. Uploaded documents and FAISS indexes can disappear after redeploys/restarts.
- For a portfolio demo, this is acceptable: upload a document, ask questions, and demo the workflow live.
- For production, add persistent storage, object storage, or a managed vector database such as Pinecone, Qdrant Cloud, Weaviate Cloud, or Supabase pgvector.
- Rotate any API key that appeared in screenshots or files.

# Vercel Deployment Guide

Vercel is the best fit between Vercel and Netlify for this project because Vercel supports FastAPI with the Python runtime. Netlify is excellent for static sites and JavaScript/TypeScript functions, but this app uses Python, LangChain, and FAISS.

## 1. Required Files

This repo includes the Vercel files you need:

```text
api/index.py
vercel.json
.python-version
```

Vercel routes all requests to the FastAPI app through `api/index.py`.

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Deploy document assistant to Vercel"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/document-assistant.git
git push -u origin main
```

Do not commit `.env`.

## 3. Import on Vercel

1. Go to `https://vercel.com/new`.
2. Import your GitHub repository.
3. Keep framework preset as **Other** if Vercel does not auto-detect FastAPI.
4. Do not set a custom build command.
5. Add the environment variables below.
6. Deploy.

## 4. Environment Variables

Set these in **Vercel Project Settings -> Environment Variables**:

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

Use a rotated OpenRouter key. Any key shown in screenshots should be treated as leaked.

## 5. Important Vercel Limitation

Vercel Functions use temporary storage. Uploaded files and FAISS indexes can disappear when the function instance is recycled.

For a portfolio demo, this still works:

1. Open the deployed URL.
2. Upload a document.
3. Ask questions immediately.

For a production-grade app, replace local FAISS persistence with a hosted vector database such as Qdrant Cloud, Pinecone, Weaviate Cloud, or Supabase pgvector.

## 6. Netlify Note

Netlify is not recommended for this exact codebase because the backend is Python FastAPI. To use Netlify cleanly, deploy the UI on Netlify and host the Python API elsewhere, or rewrite the API as JavaScript/TypeScript functions.

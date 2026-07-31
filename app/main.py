from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.document_loader import SUPPORTED_EXTENSIONS, load_document
from app.rag_pipeline import ask_question, index_documents, vectorstore_exists
from app.schemas import AskRequest, AskResponse, HealthResponse, Source, UploadResponse

app = FastAPI(
    title="Document Q&A Assistant",
    description="A Retrieval-Augmented Generation API using LangChain, FAISS, and OpenAI.",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Document Q&A Assistant</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #657487;
      --line: #d9e0e8;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --ok: #0f766e;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1060px, calc(100% - 32px));
      margin: 32px auto;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 18px;
    }
    header {
      grid-column: 1 / -1;
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 30px; line-height: 1.1; }
    p { color: var(--muted); line-height: 1.55; }
    .status {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 14px;
      color: var(--muted);
      white-space: nowrap;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    h2 { margin: 0 0 12px; font-size: 18px; }
    label {
      display: block;
      font-weight: 650;
      margin-bottom: 8px;
    }
    input[type="file"], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    textarea {
      min-height: 118px;
      resize: vertical;
    }
    button {
      width: 100%;
      min-height: 42px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
      margin-top: 12px;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      opacity: .62;
      cursor: wait;
    }
    .message {
      min-height: 24px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .answer {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 170px;
      white-space: pre-wrap;
      line-height: 1.55;
      background: #fbfcfe;
    }
    .sources {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .source {
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      background: #f8fafc;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; margin-top: 18px; }
      header { align-items: start; flex-direction: column; }
      .status { white-space: normal; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Document Q&A Assistant</h1>
        <p>Upload a document, then ask questions grounded in the retrieved content.</p>
      </div>
      <div class="status" id="status">Checking vector store...</div>
    </header>

    <section>
      <h2>Upload Document</h2>
      <label for="file">PDF, TXT, or Markdown</label>
      <input id="file" type="file" accept=".pdf,.txt,.md" />
      <button id="uploadButton">Index Document</button>
      <div class="message" id="uploadMessage"></div>
    </section>

    <section>
      <h2>Ask a Question</h2>
      <label for="question">Question</label>
      <textarea id="question" placeholder="What are the key points in this document?"></textarea>
      <button id="askButton">Ask</button>
      <div class="message" id="askMessage"></div>
      <div class="answer" id="answer">Your answer will appear here.</div>
      <div class="sources" id="sources"></div>
    </section>
  </main>

  <script>
    const statusEl = document.getElementById("status");
    const uploadButton = document.getElementById("uploadButton");
    const askButton = document.getElementById("askButton");
    const uploadMessage = document.getElementById("uploadMessage");
    const askMessage = document.getElementById("askMessage");
    const answerEl = document.getElementById("answer");
    const sourcesEl = document.getElementById("sources");

    async function refreshStatus() {
      const response = await fetch("/health");
      const payload = await response.json();
      statusEl.textContent = payload.vectorstore_ready
        ? "Vector store ready"
        : "Upload a document to create the vector store";
    }

    uploadButton.addEventListener("click", async () => {
      const fileInput = document.getElementById("file");
      if (!fileInput.files.length) {
        uploadMessage.textContent = "Choose a document first.";
        uploadMessage.className = "message bad";
        return;
      }

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      uploadButton.disabled = true;
      uploadMessage.textContent = "Indexing document...";
      uploadMessage.className = "message";

      try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Upload failed.");
        uploadMessage.textContent = `Indexed ${payload.chunks_indexed} chunks from ${payload.filename}.`;
        uploadMessage.className = "message ok";
        await refreshStatus();
      } catch (error) {
        uploadMessage.textContent = error.message;
        uploadMessage.className = "message bad";
      } finally {
        uploadButton.disabled = false;
      }
    });

    askButton.addEventListener("click", async () => {
      const question = document.getElementById("question").value.trim();
      if (!question) {
        askMessage.textContent = "Type a question first.";
        askMessage.className = "message bad";
        return;
      }

      askButton.disabled = true;
      askMessage.textContent = "Retrieving context and generating an answer...";
      askMessage.className = "message";
      sourcesEl.innerHTML = "";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Question answering failed.");
        answerEl.textContent = payload.answer;
        askMessage.textContent = "Answer generated.";
        askMessage.className = "message ok";
        sourcesEl.innerHTML = payload.sources.map(source => {
          const page = source.page ? `, page ${source.page}` : "";
          return `<div class="source"><strong>${source.source}${page}</strong><br>${source.snippet}</div>`;
        }).join("");
      } catch (error) {
        askMessage.textContent = error.message;
        askMessage.className = "message bad";
      } finally {
        askButton.disabled = false;
      }
    });

    refreshStatus().catch(() => {
      statusEl.textContent = "API is running";
    });
  </script>
</body>
</html>
"""


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="running", vectorstore_ready=vectorstore_exists())


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Use one of: {supported}")

    safe_name = f"{uuid4().hex}_{Path(file.filename or 'document').name}"
    destination = settings.documents_dir / safe_name
    content = await file.read()
    destination.write_bytes(content)

    try:
        documents = load_document(destination)
        chunks_indexed = index_documents(documents, replace=True)
        _remove_other_documents(settings.documents_dir, destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(
        filename=file.filename or safe_name,
        chunks_indexed=chunks_indexed,
        message="Document indexed successfully.",
    )


def _remove_other_documents(documents_dir: Path, keep_path: Path) -> None:
    for path in documents_dir.iterdir():
        if path.is_file() and path != keep_path and path.name != ".gitkeep":
            path.unlink(missing_ok=True)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        answer, docs = ask_question(request.question, request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        Source(
            source=Path(str(doc.metadata.get("source", "unknown"))).name,
            page=doc.metadata.get("page") + 1 if isinstance(doc.metadata.get("page"), int) else None,
            snippet=doc.page_content[:350],
        )
        for doc in docs
    ]
    return AskResponse(question=request.question, answer=answer, sources=sources)

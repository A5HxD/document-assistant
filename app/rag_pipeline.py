from pathlib import Path
import re
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.embeddings import get_embedding_model, get_fallback_embedding_model


SYSTEM_PROMPT = """You are a precise document Q&A assistant.
Answer only from the provided context. If the answer is not present, say you do not know.
Keep the answer concise and include useful specifics from the context."""


def split_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def vectorstore_exists() -> bool:
    settings = get_settings()
    return (settings.active_vectorstore_dir / "index.faiss").exists() and (settings.active_vectorstore_dir / "index.pkl").exists()


def _load_vectorstore() -> FAISS:
    settings = get_settings()
    if not vectorstore_exists():
        raise FileNotFoundError("No vector store found. Upload a document before asking questions.")

    embeddings = get_embedding_model()
    try:
        return FAISS.load_local(
            str(settings.active_vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        if not settings.use_local_fallback:
            raise
        return FAISS.load_local(
            str(settings.active_vectorstore_dir),
            get_fallback_embedding_model(),
            allow_dangerous_deserialization=True,
        )


def reset_vectorstore() -> None:
    settings = get_settings()
    if settings.active_vectorstore_dir.exists():
        shutil.rmtree(settings.active_vectorstore_dir)
    settings.active_vectorstore_dir.mkdir(parents=True, exist_ok=True)


def index_documents(documents: list[Document], replace: bool = True) -> int:
    settings = get_settings()
    if replace:
        reset_vectorstore()

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were extracted from the uploaded document.")

    embeddings = get_embedding_model()
    if vectorstore_exists():
        try:
            vectorstore = _load_vectorstore()
            vectorstore.add_documents(chunks)
        except Exception:
            embedding_model = get_fallback_embedding_model() if settings.use_local_fallback else embeddings
            vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception:
            if not settings.use_local_fallback:
                raise
            vectorstore = FAISS.from_documents(chunks, get_fallback_embedding_model())

    vectorstore.save_local(str(settings.active_vectorstore_dir))
    return len(chunks)


def ask_question(question: str, top_k: int | None = None) -> tuple[str, list[Document]]:
    settings = get_settings()
    vectorstore = _load_vectorstore()
    k = top_k or settings.retrieval_k
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception:
        if not settings.use_local_fallback:
            raise
        vectorstore = FAISS.load_local(
            str(settings.active_vectorstore_dir),
            get_fallback_embedding_model(),
            allow_dangerous_deserialization=True,
        )
        docs = vectorstore.similarity_search(question, k=k)
    context = _format_context(docs)

    try:
        if not settings.llm_api_key or settings.llm_api_key.startswith("your_"):
            raise ValueError(f"Missing API key for AI_PROVIDER={settings.ai_provider}. Add it to .env.")
        llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=0,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            max_tokens=settings.max_output_tokens,
        )
        response = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", f"Context:\n{context}\n\nQuestion:\n{question}"),
            ]
        )
        return response.content, docs
    except Exception:
        if not settings.use_local_fallback:
            raise
        return _extractive_answer(question, docs), docs


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page = doc.metadata.get("page")
        location = f"{source}, page {page + 1}" if isinstance(page, int) else source
        blocks.append(f"[Source {index}: {location}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _extractive_answer(question: str, docs: list[Document]) -> str:
    if not docs:
        return "I do not know because no relevant document context was retrieved."

    context = _clean_text(" ".join(doc.page_content for doc in docs))
    question_lower = question.lower().strip()

    direct_answer = _answer_resume_fact(question_lower, context)
    if direct_answer:
        return direct_answer

    question_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9+#.]+", question_lower)
        if len(term) > 2 and term not in {"what", "who", "where", "when", "which", "give", "tell", "about"}
    }
    sentences: list[tuple[int, str]] = []
    for doc in docs:
        for sentence in _split_sentences(doc.page_content):
            cleaned = _clean_text(sentence)
            if cleaned:
                lowered = cleaned.lower()
                score = sum(2 for term in question_terms if term in lowered)
                score += sum(1 for term in question_terms if any(word.startswith(term) for word in lowered.split()))
                sentences.append((score, cleaned))

    ranked = sorted(sentences, key=lambda item: item[0], reverse=True)
    best = [sentence for score, sentence in ranked[:3] if score > 0]
    if not best:
        best = [_clean_text(docs[0].page_content[:500])]

    return " ".join(best)


def _answer_resume_fact(question: str, context: str) -> str | None:
    name = _extract_name(context)
    email = _first_match(r"[\w.+-]+@[\w-]+\.[\w.-]+", context)
    phone = _first_match(r"(?:\+?\d[\d -]{8,}\d)", context)
    github = _first_match(r"(?:https?://)?github\.com/[A-Za-z0-9_.-]+", context)
    linkedin = _first_match(r"(?:https?://)?linkedin\.com/in/[A-Za-z0-9_.-]+", context)

    if any(token in question for token in {"name", "candidate", "person"}):
        return f"The candidate's name is {name}." if name else None

    if any(token in question for token in {"email", "mail"}):
        return f"The email address is {email}." if email else None

    if any(token in question for token in {"phone", "mobile", "contact", "number"}):
        parts = []
        if phone:
            parts.append(f"phone: {phone}")
        if email:
            parts.append(f"email: {email}")
        if parts:
            return "Contact details: " + ", ".join(parts) + "."
        return None

    if "github" in question:
        return f"The GitHub profile is {github}." if github else None

    if "linkedin" in question:
        return f"The LinkedIn profile is {linkedin}." if linkedin else None

    if any(token in question for token in {"skill", "skills", "technology", "technologies", "tools"}):
        skills = _extract_after_heading(context, ["Technical Skills", "Soft Skills", "Skills"], ["Projects", "Experience", "Education", "Certifications"])
        keyword_skills = _extract_skill_keywords(context)
        combined = ", ".join(dict.fromkeys(keyword_skills + ([skills] if skills else [])))
        if combined:
            return "Key skills include " + _trim_answer(combined, 520) + "."

    if any(token in question for token in {"education", "degree", "college", "university"}):
        education = _extract_after_heading(context, ["Education"], ["Experience", "Projects", "Skills", "Certifications"])
        if education:
            return "Education: " + _trim_answer(education, 320) + "."

    if any(token in question for token in {"project", "projects"}):
        projects = _extract_after_heading(context, ["Projects"], ["Skills", "Education", "Experience", "Certifications"])
        if projects:
            return "Projects mentioned include " + _trim_answer(projects, 520) + "."

    if any(token in question for token in {"summary", "profile", "about"}):
        summary = _extract_after_heading(context, ["Summary"], ["Education", "Experience", "Projects", "Skills"])
        if summary:
            return _trim_answer(summary, 520) + "."

    return None


def _extract_name(context: str) -> str | None:
    first_line = context.split(" Student ", 1)[0]
    first_line = re.sub(r"\+?\d[\d -]{8,}\d", "", first_line).strip(" -|")
    match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", first_line)
    return match.group(1) if match else None


def _extract_after_heading(context: str, headings: list[str], stop_headings: list[str]) -> str | None:
    heading_pattern = "|".join(re.escape(heading) for heading in headings)
    stop_pattern = "|".join(re.escape(heading) for heading in stop_headings)
    match = re.search(
        rf"(?:^|\s)({heading_pattern})\s*[:\-]?\s*(.*?)(?=\s(?:{stop_pattern})\s*[:\-]?|$)",
        context,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = _clean_text(match.group(2))
    return value or None


def _split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    return re.split(r"(?<=[.!?])\s+|(?:\s*[•*]\s*)", text)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("• ", "•")
    return text.strip(" -|;,.")


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def _extract_skill_keywords(context: str) -> list[str]:
    known_skills = [
        "Python",
        "Scikit-Learn",
        "Pandas",
        "TensorFlow",
        "LangChain",
        "LLMs",
        "Prompt Engineering",
        "Fine-tuning",
        "Machine Learning",
        "Deep Learning",
        "Neural Networks",
        "Model Evaluation",
        "Matplotlib",
        "Seaborn",
        "Power BI",
        "Tableau",
        "Databases",
        "MySQL",
        "SQL",
        "Git",
        "GitHub",
        "AWS",
        "XGBoost",
        "LightGBM",
        "Isolation Forest",
        "Optuna",
        "Data Analysis",
        "Feature Engineering",
        "Data Structures",
        "Algorithms",
        "OOP",
    ]
    lowered = context.lower()
    return [skill for skill in known_skills if skill.lower() in lowered]


def _trim_answer(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."

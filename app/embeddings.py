import hashlib
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


class HashEmbeddings(Embeddings):
    """Small deterministic embedding model for offline demos and quota-free testing."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector


def get_embedding_model() -> Embeddings:
    settings = get_settings()
    if not settings.embedding_api_key or settings.embedding_api_key.startswith("your_"):
        if not settings.use_local_fallback:
            raise ValueError(f"Missing API key for AI_PROVIDER={settings.ai_provider}. Add it to .env.")
        return HashEmbeddings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )


def get_fallback_embedding_model() -> HashEmbeddings:
    return HashEmbeddings()

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    ai_provider: str = Field(default="openrouter", alias="AI_PROVIDER")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_chat_model: str = Field(default="google/gemini-2.5-flash-lite", alias="OPENROUTER_CHAT_MODEL")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-small", alias="OPENROUTER_EMBEDDING_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_chat_model: str = Field(default="gemini-3.6-flash", alias="GEMINI_CHAT_MODEL")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai/", alias="GEMINI_BASE_URL")

    use_local_fallback: bool = Field(default=True, alias="USE_LOCAL_FALLBACK")
    vectorstore_dir: Path = Field(default=Path("vectorstore"), alias="VECTORSTORE_DIR")
    documents_dir: Path = Field(default=Path("data/documents"), alias="DOCUMENTS_DIR")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")
    max_output_tokens: int = Field(default=700, alias="MAX_OUTPUT_TOKENS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def ensure_directories(self) -> None:
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.active_vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_vectorstore_dir(self) -> Path:
        provider = self.ai_provider.replace("/", "_").replace("\\", "_")
        return self.vectorstore_dir / provider

    @property
    def llm_api_key(self) -> str:
        if self.ai_provider == "openrouter":
            return self.openrouter_api_key
        if self.ai_provider == "gemini":
            return self.gemini_api_key
        return self.openai_api_key

    @property
    def llm_base_url(self) -> str | None:
        if self.ai_provider == "openrouter":
            return self.openrouter_base_url
        if self.ai_provider == "gemini":
            return self.gemini_base_url
        return self.openai_base_url

    @property
    def chat_model(self) -> str:
        if self.ai_provider == "openrouter":
            return self.openrouter_chat_model
        if self.ai_provider == "gemini":
            return self.gemini_chat_model
        return self.openai_chat_model

    @property
    def embedding_api_key(self) -> str:
        if self.ai_provider == "openrouter":
            return self.openrouter_api_key
        return self.openai_api_key

    @property
    def embedding_base_url(self) -> str | None:
        if self.ai_provider == "openrouter":
            return self.openrouter_base_url
        return self.openai_base_url

    @property
    def embedding_model(self) -> str:
        if self.ai_provider == "openrouter":
            return self.openrouter_embedding_model
        return self.openai_embedding_model


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

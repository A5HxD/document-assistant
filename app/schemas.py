from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What are the key findings in this document?"])
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool

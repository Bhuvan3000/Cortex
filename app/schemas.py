from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int | None
    chunks: int
    status: Literal["ingested", "replaced", "already_ingested"]


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    category: str
    tags: list[str]
    uploaded_at: str
    pages: int
    chunks: int


class QueryFilters(BaseModel):
    doc_id: str | list[str] | None = None
    source: str | list[str] | None = None
    category: str | list[str] | None = None
    page: int | list[int] | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    filters: QueryFilters | None = None
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    generate: bool = True


class SourceChunk(BaseModel):
    rank: int
    doc_id: str
    source: str
    page: int
    chunk_index: int
    category: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str | None
    mode: Literal["generative", "retrieval_only"]
    sources: list[SourceChunk]

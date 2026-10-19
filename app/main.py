from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile

from app import ingestion, rag
from app.config import settings
from app.schemas import (
    DocumentInfo,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.vectorstore import get_store


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_store()
    yield


app = FastAPI(
    title="Cortex",
    description="Retrieval-Augmented Generation research assistant over PDF documents",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model,
        "generation_enabled": rag.llm_available(),
        "documents": len(ingestion.list_documents()),
    }


@app.post("/documents", response_model=IngestResponse, status_code=201)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    overwrite: bool = Form(default=False),
):
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="File exceeds upload size limit")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    tag_list = [tag for tag in (tags or "").split(",") if tag.strip()]
    try:
        result = ingestion.ingest_pdf(
            data,
            file.filename or "document.pdf",
            category=category,
            tags=tag_list,
            overwrite=overwrite,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result["status"] == "already_ingested":
        response.status_code = 200
    return result


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    return ingestion.list_documents()


@app.delete("/documents/{doc_id}", status_code=204)
def remove_document(doc_id: str):
    if not ingestion.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    return rag.answer_question(
        question=request.question,
        top_k=request.top_k,
        filters=request.filters,
        score_threshold=request.score_threshold,
        generate=request.generate,
    )

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.vectorstore import get_store

BATCH_SIZE = 256


def compute_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def pdf_path(doc_id: str) -> Path:
    return Path(settings.upload_dir) / f"{doc_id}.pdf"


def chunk_ids_for(doc_id: str) -> list[str]:
    return get_store().get(where={"doc_id": doc_id}, include=[])["ids"]


def ingest_pdf(
    data: bytes,
    filename: str,
    category: str | None = None,
    tags: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    doc_id = compute_doc_id(data)
    store = get_store()
    existing = chunk_ids_for(doc_id)

    if existing and not overwrite:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "pages": None,
            "chunks": len(existing),
            "status": "already_ingested",
        }

    path = pdf_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    try:
        reader = PdfReader(str(path))
        pages = [
            Document(page_content=page.extract_text() or "", metadata={"page": number})
            for number, page in enumerate(reader.pages)
        ]
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ValueError(f"Unable to parse PDF: {exc}") from exc

    pages = [page for page in pages if page.page_content.strip()]
    if not pages:
        path.unlink(missing_ok=True)
        raise ValueError("PDF contains no extractable text")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    tag_value = ",".join(tag.strip() for tag in (tags or []) if tag.strip())
    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            "doc_id": doc_id,
            "source": filename,
            "page": int(chunk.metadata.get("page", 0)) + 1,
            "chunk_index": index,
            "start_index": int(chunk.metadata.get("start_index", 0)),
            "category": category or "general",
            "tags": tag_value,
            "uploaded_at": uploaded_at,
        }

    if existing:
        store.delete(ids=existing)

    ids = [f"{doc_id}-{index}" for index in range(len(chunks))]
    for start in range(0, len(chunks), BATCH_SIZE):
        store.add_documents(
            chunks[start : start + BATCH_SIZE], ids=ids[start : start + BATCH_SIZE]
        )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len({chunk.metadata["page"] for chunk in chunks}),
        "chunks": len(chunks),
        "status": "replaced" if existing else "ingested",
    }


def list_documents() -> list[dict]:
    result = get_store().get(include=["metadatas"])
    documents: dict[str, dict] = {}
    for meta in result["metadatas"]:
        entry = documents.setdefault(
            meta["doc_id"],
            {
                "doc_id": meta["doc_id"],
                "source": meta["source"],
                "category": meta["category"],
                "tags": [tag for tag in meta.get("tags", "").split(",") if tag],
                "uploaded_at": meta["uploaded_at"],
                "pages": set(),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
        entry["pages"].add(meta["page"])
    return sorted(
        (
            {**entry, "pages": len(entry["pages"])}
            for entry in documents.values()
        ),
        key=lambda item: item["uploaded_at"],
        reverse=True,
    )


def delete_document(doc_id: str) -> bool:
    ids = chunk_ids_for(doc_id)
    if not ids:
        return False
    get_store().delete(ids=ids)
    pdf_path(doc_id).unlink(missing_ok=True)
    return True

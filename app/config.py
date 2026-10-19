import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass(frozen=True)
class Settings:
    persist_dir: str = os.getenv("CORTEX_PERSIST_DIR", "./data/chroma")
    upload_dir: str = os.getenv("CORTEX_UPLOAD_DIR", "./data/uploads")
    collection_name: str = os.getenv("CORTEX_COLLECTION", "cortex_documents")
    embedding_model: str = os.getenv(
        "CORTEX_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CORTEX_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CORTEX_CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("CORTEX_TOP_K", "4"))
    max_upload_mb: int = int(os.getenv("CORTEX_MAX_UPLOAD_MB", "25"))
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = os.getenv("CORTEX_LLM_MODEL", "gemini-2.5-flash")


settings = Settings()

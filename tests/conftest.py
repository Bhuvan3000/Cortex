import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["CORTEX_PERSIST_DIR"] = os.path.join(_tmp, "chroma")
os.environ["CORTEX_UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
os.environ["GOOGLE_API_KEY"] = ""

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import vectorstore


@pytest.fixture(scope="session", autouse=True)
def fake_embeddings():
    embeddings = DeterministicFakeEmbedding(size=128)
    original = vectorstore.get_embeddings
    vectorstore.get_embeddings = lambda: embeddings
    vectorstore.get_store.cache_clear()
    yield
    vectorstore.get_embeddings = original
    vectorstore.get_store.cache_clear()


@pytest.fixture(scope="session")
def client(fake_embeddings):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def make_pdf(pages: list[str]) -> bytes:
    count = len(pages)
    font_id = 3 + 2 * count
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids ["
        + " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
        + f"] /Count {count} >>",
    ]
    for index, text in enumerate(pages):
        content_id = 4 + 2 * index
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET"
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        )
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF"
    ).encode()
    return bytes(output)


@pytest.fixture
def pdf_factory():
    return make_pdf

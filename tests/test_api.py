def upload(client, pdf_bytes, name, **form):
    return client.post(
        "/documents",
        files={"file": (name, pdf_bytes, "application/pdf")},
        data=form,
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_non_pdf(client):
    response = upload(client, b"plain text", "notes.txt")
    assert response.status_code == 415


def test_rejects_corrupt_pdf(client):
    response = upload(client, b"%PDF-1.4 garbage", "broken.pdf")
    assert response.status_code == 422


def test_ingest_list_query_delete(client, pdf_factory):
    alpha = pdf_factory(["Transformers use self attention to model sequences"])
    beta = pdf_factory(["Chroma stores embeddings for vector similarity search"])

    first = upload(client, alpha, "alpha.pdf", category="ml", tags="nlp,attention")
    assert first.status_code == 201
    assert first.json()["status"] == "ingested"
    alpha_id = first.json()["doc_id"]

    second = upload(client, beta, "beta.pdf", category="infra")
    assert second.status_code == 201
    beta_id = second.json()["doc_id"]

    duplicate = upload(client, alpha, "alpha.pdf")
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "already_ingested"

    replaced = upload(client, alpha, "alpha.pdf", overwrite="true", category="ml")
    assert replaced.json()["status"] == "replaced"

    listing = client.get("/documents").json()
    assert {doc["doc_id"] for doc in listing} >= {alpha_id, beta_id}
    alpha_doc = next(doc for doc in listing if doc["doc_id"] == alpha_id)
    assert alpha_doc["category"] == "ml"

    result = client.post(
        "/query", json={"question": "What is self attention?", "top_k": 5}
    ).json()
    assert result["mode"] == "retrieval_only"
    assert len(result["sources"]) >= 2

    filtered = client.post(
        "/query",
        json={
            "question": "What is self attention?",
            "top_k": 5,
            "filters": {"category": "infra"},
        },
    ).json()
    assert filtered["sources"]
    assert {hit["doc_id"] for hit in filtered["sources"]} == {beta_id}

    combined = client.post(
        "/query",
        json={
            "question": "vectors",
            "filters": {"doc_id": [alpha_id, beta_id], "page": 1},
            "top_k": 5,
        },
    ).json()
    assert {hit["page"] for hit in combined["sources"]} == {1}

    assert client.delete(f"/documents/{alpha_id}").status_code == 204
    assert client.delete(f"/documents/{alpha_id}").status_code == 404

    after = client.post(
        "/query",
        json={"question": "self attention", "filters": {"doc_id": alpha_id}},
    ).json()
    assert after["sources"] == []

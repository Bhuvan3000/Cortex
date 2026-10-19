from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.schemas import QueryFilters
from app.vectorstore import get_store

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions strictly from the provided context excerpts. "
            "Cite supporting excerpts inline using their bracketed numbers like [1]. "
            "If the context does not contain the answer, say you could not find it in the documents.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

FILTER_KEYS = ("doc_id", "source", "category", "page")


def build_where(filters: QueryFilters | None) -> dict | None:
    if filters is None:
        return None
    conditions = []
    for key in FILTER_KEYS:
        value = getattr(filters, key)
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            conditions.append({key: {"$in": value}})
        else:
            conditions.append({key: value})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(
    question: str,
    top_k: int,
    filters: QueryFilters | None,
    score_threshold: float | None,
) -> list[dict]:
    results = get_store().similarity_search_with_score(
        question, k=top_k, filter=build_where(filters)
    )
    hits = []
    for document, distance in results:
        score = 1.0 - float(distance)
        if score_threshold is not None and score < score_threshold:
            continue
        meta = document.metadata
        hits.append(
            {
                "rank": len(hits) + 1,
                "doc_id": meta["doc_id"],
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "category": meta["category"],
                "score": round(score, 4),
                "text": document.page_content,
            }
        )
    return hits


@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def llm_available() -> bool:
    return bool(settings.google_api_key)


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{hit['rank']}] {hit['source']} (page {hit['page']})\n{hit['text']}"
        for hit in hits
    )


def generate_answer(question: str, hits: list[dict]) -> str:
    chain = PROMPT | get_llm() | StrOutputParser()
    return chain.invoke({"context": format_context(hits), "question": question})


def answer_question(
    question: str,
    top_k: int,
    filters: QueryFilters | None,
    score_threshold: float | None,
    generate: bool,
) -> dict:
    hits = retrieve(question, top_k, filters, score_threshold)

    if not hits:
        return {
            "question": question,
            "answer": "No relevant context was found in the indexed documents.",
            "mode": "retrieval_only",
            "sources": [],
        }

    if generate and llm_available():
        return {
            "question": question,
            "answer": generate_answer(question, hits),
            "mode": "generative",
            "sources": hits,
        }

    return {
        "question": question,
        "answer": None,
        "mode": "retrieval_only",
        "sources": hits,
    }

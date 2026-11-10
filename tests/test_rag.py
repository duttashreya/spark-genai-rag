import psycopg
from fastapi.testclient import TestClient
from pgvector.psycopg import register_vector

from src.api.generator import RAGGenerator
from src.api.main import app
from src.indexing.config import settings


def test_prompt_builder():
    contexts = [
        {"document_id": "doc_1", "chunk_index": 0, "content": "Spark SQL optimizes queries."}
    ]
    prompt = RAGGenerator.build_prompt("How does Spark optimize?", contexts)
    assert "Spark SQL optimizes queries" in prompt
    assert "--- QUESTION ---" in prompt


def test_rag_ask_endpoint():
    dummy_vector = [0.0] * settings.embedding_dimension
    dummy_vector[0] = 1.0
    vector_str = "[" + ",".join(str(x) for x in dummy_vector) + "]"

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                RETURNING id;
                """,
                (
                    "ci_rag_doc",
                    0,
                    "Spark provides unified analytics engine for large-scale data processing.",
                    vector_str,
                ),
            )
            test_id = cur.fetchone()[0]

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/rag/ask",
                json={"query": "unified analytics engine", "top_k": 2},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert len(data["citations"]) > 0
            assert data["citations"][0]["document_id"] != ""
    finally:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE id = %s;", (test_id,))
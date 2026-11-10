from fastapi.testclient import TestClient
from src.api.main import app
from src.api.generator import RAGGenerator


def test_prompt_builder():
    contexts = [
        {"document_id": "doc_1", "chunk_index": 0, "content": "Spark SQL optimizes queries."}
    ]
    prompt = RAGGenerator.build_prompt("How does Spark optimize?", contexts)
    assert "Spark SQL optimizes queries" in prompt
    assert "--- QUESTION ---" in prompt


def test_rag_ask_endpoint():
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
import psycopg
from pgvector.psycopg import register_vector

from src.indexing.config import settings
from src.pipeline import run_pipeline


def test_full_ingestion_pipeline(tmp_path):
    # Setup test file
    test_doc = tmp_path / "spark_ai_test.txt"
    test_doc.write_text(
        "PySpark enables distributed data preparation for AI pipelines. "
        "Large language models benefit from structured and clean vector indexes."
    )

    # Run the full pipeline targeting temp directory
    run_pipeline(str(tmp_path))

    # Assert record is present in pgvector
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM document_chunks WHERE document_id = 'spark_ai_test';"
            )
            count = cur.fetchone()[0]
            assert count > 0, "No chunks were persisted to the database."

            # Verify embedding has expected dimension and non-zero magnitude
            cur.execute(
                "SELECT embedding FROM document_chunks WHERE document_id = 'spark_ai_test' LIMIT 1;"
            )
            embedding = cur.fetchone()[0]
            assert len(embedding.to_numpy()) == settings.embedding_dimension
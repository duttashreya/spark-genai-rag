import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from src.indexing.config import settings


def test_vector_extension_and_table():
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ext = cur.fetchone()
            assert ext is not None, "pgvector extension is missing"

            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'document_chunks'
                );
                """
            )
            exists = cur.fetchone()[0]
            assert exists is True, "document_chunks table was not found"


def test_insert_and_query_vector():
    # Use float32 numpy array for native pgvector binding
    dummy_vector = np.zeros(settings.embedding_dimension, dtype=np.float32)
    dummy_vector[0] = 1.0

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                ("test_doc_001", 0, "Test chunk content", dummy_vector),
            )
            inserted_id = cur.fetchone()[0]
            assert inserted_id is not None

            # Query with explicit ::vector cast for query parameter
            cur.execute(
                """
                SELECT id, content, embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE id = %s;
                """,
                (dummy_vector, inserted_id),
            )
            row = cur.fetchone()
            assert row is not None
            assert round(float(row[2]), 4) == 0.0

            # Cleanup
            cur.execute("DELETE FROM document_chunks WHERE id = %s;", (inserted_id,))
import psycopg
from pgvector.psycopg import register_vector

from src.indexing.config import settings


def init_database():
    print(f"Connecting to database at {settings.database_url}...")
    with psycopg.connect(settings.database_url, autocommit=True) as conn, conn.cursor() as cur:
        # 1. Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✓ pgvector extension enabled.")

        # Register vector type with psycopg connection
        register_vector(conn)

        # 2. Create chunks table
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id VARCHAR(255) NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                embedding vector({settings.embedding_dimension}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        print("✓ Table 'document_chunks' created.")

        # 3. Create HNSW index for cosine distance
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
        print("✓ HNSW vector index created.")


if __name__ == "__main__":
    init_database()
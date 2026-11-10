import psycopg

from src.indexing.config import settings


def enable_full_text_search():
    with psycopg.connect(settings.database_url, autocommit=True) as conn, conn.cursor() as cur:
        # 1. Add generated tsvector column for fast text search
        cur.execute(
            """
            ALTER TABLE document_chunks 
            ADD COLUMN IF NOT EXISTS content_tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
            """
        )
        # 2. Add GIN index over the tsvector
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv 
            ON document_chunks USING gin(content_tsv);
            """
        )
        print("✓ Full-text search (tsvector + GIN index) successfully enabled.")


if __name__ == "__main__":
    enable_full_text_search()
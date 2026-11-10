import io
import psycopg
from pgvector.psycopg import register_vector
from pyspark.sql import DataFrame
from src.indexing.config import settings


def write_partition_to_pg(rows_iter):
    """Worker-level execution: streams a partition of rows directly to PostgreSQL."""
    rows = list(rows_iter)
    if not rows:
        return

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # Batch insert using parameterized SQL
            insert_sql = """
                INSERT INTO document_chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector);
            """
            data_tuples = [
                (
                    row.document_id,
                    row.chunk_index,
                    row.content,
                    row.embedding,
                )
                for row in rows
            ]
            cur.executemany(insert_sql, data_tuples)


def save_chunks_to_vector_db(df: DataFrame) -> int:
    """Executes distributed write across all partitions and returns total rows saved."""
    count = df.count()
    # Execute foreachPartition to perform writes without pulling data to driver
    df.foreachPartition(write_partition_to_pg)
    return count
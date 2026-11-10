from typing import Any, Dict, List
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from src.indexing.config import settings


class HybridRetriever:
    def __init__(self):
        # Query encoder running CPU inference
        self.encoder = SentenceTransformer(settings.embedding_model_name)

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        query_vector = self.encoder.encode(query, normalize_embeddings=True).astype(np.float32)

        hybrid_query = f"""
        WITH dense_search AS (
            SELECT 
                id,
                document_id,
                chunk_index,
                content,
                ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS dense_rank
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        sparse_search AS (
            SELECT 
                id,
                document_id,
                chunk_index,
                content,
                ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC) AS sparse_rank
            FROM document_chunks
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY sparse_rank
            LIMIT %s
        )
        SELECT 
            COALESCE(d.id, s.id) AS id,
            COALESCE(d.document_id, s.document_id) AS document_id,
            COALESCE(d.chunk_index, s.chunk_index) AS chunk_index,
            COALESCE(d.content, s.content) AS content,
            (
                COALESCE(1.0 / (%s + d.dense_rank), 0.0) +
                COALESCE(1.0 / (%s + s.sparse_rank), 0.0)
            ) AS rrf_score
        FROM dense_search d
        FULL OUTER JOIN sparse_search s ON d.id = s.id
        ORDER BY rrf_score DESC
        LIMIT %s;
        """

        with psycopg.connect(settings.database_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    hybrid_query,
                    (
                        query_vector,
                        query_vector,
                        top_k * 2,
                        query,
                        query,
                        top_k * 2,
                        rrf_k,
                        rrf_k,
                        top_k,
                    ),
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": str(row[0]),
                "document_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "rrf_score": float(row[4]),
            })
        return results
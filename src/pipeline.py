import argparse
import time
from src.etl.spark_session import get_spark_session
from src.etl.ingest import process_raw_documents
from src.indexing.embedder import compute_embeddings_udf
from src.indexing.sink import save_chunks_to_vector_db


def run_pipeline(input_dir: str):
    start_time = time.time()
    spark = get_spark_session()
    
    print(f"[*] Starting ingestion pipeline for directory: {input_dir}")
    
    # 1. Document parsing and chunking
    chunks_df = process_raw_documents(spark, input_dir)
    
    # 2. Compute distributed embeddings using Pandas UDF
    embedded_df = chunks_df.withColumn("embedding", compute_embeddings_udf("content"))
    
    # Cache before writing
    embedded_df.cache()
    
    # 3. Bulk sink to pgvector
    print("[*] Writing chunk embeddings to pgvector...")
    records_saved = save_chunks_to_vector_db(embedded_df)
    
    elapsed = time.time() - start_time
    print(f"[✓] Pipeline complete. Saved {records_saved} chunks in {elapsed:.2f} seconds.")
    
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spark GenAI RAG Ingestion")
    parser.add_argument("--input-dir", type=str, default="data/sample", help="Input directory")
    args = parser.parse_args()

    run_pipeline(args.input_dir)
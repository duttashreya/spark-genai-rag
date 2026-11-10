import os
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode, input_file_name, regexp_extract
from src.etl.spark_session import get_spark_session
from src.etl.chunker import chunk_text_udf


def process_raw_documents(spark, input_dir: str) -> DataFrame:
    """Reads raw text documents, extracts document IDs, and explodes into chunks."""
    # Read all text files from the directory
    raw_df = spark.read.text(input_dir, wholetext=True)

    # Extract clean document_id from the file path
    transformed_df = (
        raw_df.withColumn("file_path", input_file_name())
        .withColumn("document_id", regexp_extract(col("file_path"), r"([^/]+)\.txt$", 1))
        .withColumnRenamed("value", "raw_text")
        .withColumn("chunks", chunk_text_udf(col("raw_text")))
        .select(
            col("document_id"),
            explode(col("chunks")).alias("chunk_data"),
        )
        .select(
            col("document_id"),
            col("chunk_data.chunk_index").alias("chunk_index"),
            col("chunk_data.content").alias("content"),
        )
    )

    return transformed_df


if __name__ == "__main__":
    sample_dir = "data/sample"
    os.makedirs(sample_dir, exist_ok=True)

    # Generate sample documents if empty
    sample_file = os.path.join(sample_dir, "apache_spark_overview.txt")
    if not os.path.exists(sample_file):
        with open(sample_file, "w") as f:
            f.write(
                "Apache Spark is a unified analytics engine for large-scale data processing. "
                "It provides high-level APIs in Java, Scala, Python and R, and an optimized "
                "engine that supports general execution graphs. It also supports a rich set of "
                "higher-level tools including Spark SQL for SQL and structured data processing, "
                "pandas API on Spark for pandas workloads, MLlib for machine learning, GraphX "
                "for graph processing, and Structured Streaming for incremental computation and stream processing."
            )

    spark = get_spark_session()
    print("Ingesting sample documents...")
    df = process_raw_documents(spark, sample_dir)
    df.show(truncate=False)
    spark.stop()
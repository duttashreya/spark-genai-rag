from src.etl.chunker import chunk_text
from src.etl.spark_session import get_spark_session
from src.etl.ingest import process_raw_documents


def test_chunk_text_logic():
    text = "word " * 150
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert chunks[0]["chunk_index"] == 0
    assert "content" in chunks[0]


def test_spark_document_processing(tmp_path):
    # Create a temporary text file
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("Data engineering meets artificial intelligence. " * 30)

    spark = get_spark_session()
    df = process_raw_documents(spark, str(tmp_path))
    results = df.collect()
    spark.stop()

    assert len(results) > 0
    assert results[0]["document_id"] == "test_doc"
    assert results[0]["chunk_index"] == 0
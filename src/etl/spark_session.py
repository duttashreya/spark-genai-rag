from pyspark.sql import SparkSession

from src.indexing.config import settings


def get_spark_session() -> SparkSession:
    """Builds and returns a memory-optimized Spark session for local processing."""
    return (
        SparkSession.builder.appName(settings.spark_app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = get_spark_session()
    print(f"✓ Spark Session successfully initialized. Version: {spark.version}")
    spark.stop()
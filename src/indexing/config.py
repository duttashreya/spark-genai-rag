from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384  # bge-small-en-v1.5 outputs 384-dim vectors
    spark_app_name: str = "EnterpriseRAGIngestion"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
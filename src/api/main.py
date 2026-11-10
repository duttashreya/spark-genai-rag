import time
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.indexing.retriever import HybridRetriever

retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    # Startup: Load the encoder once into memory
    retriever = HybridRetriever()
    yield
    # Shutdown logic (if needed)
    retriever = None


app = FastAPI(
    title="Enterprise Spark-pgvector RAG Service",
    description="Production-grade Hybrid Search & Retrieval Engine",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        json_schema_extra={"example": "What execution graphs does Spark support?"},
    )
    top_k: int = Field(default=3, ge=1, le=20)


class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    rrf_score: float


class QueryResponse(BaseModel):
    query: str
    latency_ms: float
    total_results: int
    results: List[DocumentChunkResponse]


@app.post("/v1/search", response_model=QueryResponse)
async def search_endpoint(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever engine not initialized.")

    start_time = time.perf_counter()
    records = retriever.retrieve(query=request.query, top_k=request.top_k)
    latency_ms = (time.perf_counter() - start_time) * 1000

    return QueryResponse(
        query=request.query,
        latency_ms=round(latency_ms, 2),
        total_results=len(records),
        results=records,
    )


@app.get("/healthz")
def healthcheck():
    return {"status": "healthy"}
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.generator import RAGGenerator
from src.indexing.retriever import HybridRetriever

retriever = None
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    retriever = HybridRetriever()
    generator = RAGGenerator()
    yield
    retriever = None
    generator = None


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
    results: list[DocumentChunkResponse]


class RAGResponse(BaseModel):
    query: str
    answer: str
    latency_ms: float
    citations: list[dict[str, Any]]


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


@app.post("/v1/rag/ask", response_model=RAGResponse)
async def ask_endpoint(request: QueryRequest):
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")

    start_time = time.perf_counter()
    contexts = retriever.retrieve(query=request.query, top_k=request.top_k)
    gen_result = generator.generate_answer(query=request.query, contexts=contexts)
    latency_ms = (time.perf_counter() - start_time) * 1000

    return RAGResponse(
        query=request.query,
        answer=gen_result["answer"],
        latency_ms=round(latency_ms, 2),
        citations=gen_result["citations"],
    )


@app.get("/healthz")
def healthcheck():
    return {"status": "healthy"}
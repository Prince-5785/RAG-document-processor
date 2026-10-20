"""
InsureQuery Production FastAPI Application.
Enterprise RAG engine for insurance policy analysis, hybrid retrieval,
threshold guardrails, source citations, token-by-token SSE streaming, and structured claim decisions.
"""

import os
import time
import uuid
import logging
import tempfile
import asyncio
import secrets
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline

# Configure logger
logger = logging.getLogger("insurequery.api")

app = FastAPI(
    title="InsureQuery API",
    description="Enterprise RAG Engine for Insurance Document Intelligence, Hybrid Retrieval, and Claim Verification",
    version="2.0.0"
)

# Lock CORS to production domain (customizable via ALLOWED_ORIGINS env)
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.environ.get("ALLOWED_ORIGINS", "https://prince-gupta.me").split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global metrics tracker
METRICS = {
    'total_requests': 0,
    'total_queries': 0,
    'total_stream_queries': 0,
    'total_claim_evaluations': 0,
    'total_documents_uploaded': 0,
    'start_time': time.time(),
    'latencies': []
}


# --- Request ID and Logging Middleware ---
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.time()
    METRICS['total_requests'] += 1

    logger.info(f"[{request_id}] Incoming {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        elapsed_ms = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        
        # Track latency (bounded list)
        METRICS['latencies'].append(elapsed_ms)
        if len(METRICS['latencies']) > 1000:
            METRICS['latencies'].pop(0)

        logger.info(f"[{request_id}] Completed {response.status_code} in {elapsed_ms:.2f}ms")
        return response
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Unhandled error after {elapsed_ms:.2f}ms: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "message": str(e), "request_id": request_id}
        )


# --- Authentication Dependency ---
async def verify_auth(authorization: Optional[str] = Header(None)):
    """
    Fails closed on authentication. Requires INSUREQUERY_API_KEY environment variable.
    Uses timing-safe comparison with secrets.compare_digest.
    """
    configured_key = os.environ.get("INSUREQUERY_API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth not configured"
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required"
        )
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer' or not secrets.compare_digest(token, configured_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key credentials"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use 'Bearer <api_key>'"
        )
    return True


# --- Initialize Pipeline ---
try:
    pipeline = RAGPipeline()
except Exception as e:
    logger.fatal(f"Failed to initialize RAG Pipeline on startup: {e}")
    pipeline = None


# --- Pydantic Request & Response Schemas ---
class SourceCitation(BaseModel):
    file: str = Field(..., description="Document source filename")
    page: int = Field(1, description="Page number where the cited text was found")
    chunk_text: str = Field(..., description="Excerpt from the source document")
    score: float = Field(..., description="Relevance similarity / rerank score")

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question or insurance query", min_length=3)
    document_ids: Optional[List[str]] = Field(None, description="Optional document IDs to scope the search")
    score_threshold: Optional[float] = Field(None, description="Optional custom similarity cutoff (default 0.3)")

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    threshold_applied: bool

class ClaimDecisionRequest(BaseModel):
    query: str = Field(..., description="Natural language insurance claim details")
    sum_insured: Optional[int] = Field(None, description="Total sum insured in INR under the policy")
    document_ids: Optional[List[str]] = Field(None, description="Optional document IDs to restrict policy scope")
    preferred_model: Optional[str] = Field(None, description="Optional preferred LLM model")

class ClaimDecisionResponse(BaseModel):
    claim_id: str
    query: str
    decision: str
    payout: int
    currency: str
    justification: str
    model_used: Optional[str] = None
    clause_ids: List[str] = Field(default_factory=list, description="IDs of clauses relied on")
    evidence: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size_bytes: int
    chunk_count: int
    status: str
    message: str


# --- Document Management Endpoints ---

@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    authorized: bool = Depends(verify_auth)
):
    """
    Upload an insurance document (PDF, Word, Email, or TXT) to index into the vector store.
    Validates file format and size, then indexes non-blockingly via executor.
    """
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Pipeline is not initialized"
        )

    allowed_exts = pipeline.document_processor.get_supported_formats()
    ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {allowed_exts}"
        )

    # Save to temporary file with size check
    max_bytes = pipeline.document_processor.max_file_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {pipeline.document_processor.max_file_size_mb} MB"
        )

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        custom_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        index_res = await loop.run_in_executor(
            None, pipeline.index_documents, [tmp_path], custom_id
        )
        
        if index_res['failed_files'] > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document indexing failed: {index_res['errors']}"
            )
        
        METRICS['total_documents_uploaded'] += 1
        return DocumentUploadResponse(
            document_id=custom_id,
            filename=file.filename,
            file_size_bytes=len(content),
            chunk_count=index_res['total_chunks'],
            status="ready",
            message=f"Successfully indexed {index_res['total_chunks']} chunks with page citations."
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/documents")
async def list_documents(authorized: bool = Depends(verify_auth)):
    """List all indexed documents with metadata, chunk count, and pages."""
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")
    return {"documents": pipeline.list_documents()}


@app.get("/documents/{document_id}")
async def get_document(document_id: str, authorized: bool = Depends(verify_auth)):
    """Get metadata for a specific document."""
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")
    doc = pipeline.vector_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found")
    return doc


@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str, authorized: bool = Depends(verify_auth)):
    """Check indexing and readiness status for a document."""
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")
    doc = pipeline.vector_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found")
    return {
        "document_id": document_id,
        "filename": doc.get("filename"),
        "status": doc.get("status", "ready"),
        "chunk_count": doc.get("chunk_count", 0),
        "indexed_at": doc.get("indexed_at")
    }


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, authorized: bool = Depends(verify_auth)):
    """Delete a document and its chunks from the vector store."""
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")
    success = pipeline.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Failed to delete document '{document_id}'")
    return {"status": "deleted", "document_id": document_id}


# --- Query Endpoints ---

@app.post("/query", response_model=QueryResponse)
async def query_document(
    request: QueryRequest,
    authorized: bool = Depends(verify_auth)
):
    """
    Answers an insurance query using Hybrid Retrieval + Cross-Encoder Reranking
    and returns verified source citations with file names and page numbers.
    """
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")

    METRICS['total_queries'] += 1
    result = await pipeline.answer_with_citations(
        question=request.question,
        document_ids=request.document_ids,
        score_threshold=request.score_threshold
    )
    return QueryResponse(
        answer=result['answer'],
        sources=[SourceCitation(**s) for s in result['sources']],
        threshold_applied=result['threshold_applied']
    )


@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    authorized: bool = Depends(verify_auth)
):
    """
    Token-by-token Server-Sent Events (SSE) streaming.
    Sends source citations in the initial event, followed by live generated tokens.
    """
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")

    METRICS['total_stream_queries'] += 1
    generator = pipeline.stream_answer(
        question=request.question,
        document_ids=request.document_ids,
        score_threshold=request.score_threshold
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/query/claim", response_model=ClaimDecisionResponse)
async def query_claim_decision(
    request: ClaimDecisionRequest,
    authorized: bool = Depends(verify_auth)
):
    """
    Evaluates an insurance claim based on retrieved policy clauses.
    Returns structured decision (APPROVED / REJECTED / NEEDS_REVIEW), payout amount,
    justification, model used, and cited clause IDs.
    """
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")

    METRICS['total_claim_evaluations'] += 1
    result = await pipeline.query(
        user_query=request.query,
        preferred_model=request.preferred_model,
        document_ids=request.document_ids,
        sum_insured=request.sum_insured
    )
    json_out = result.get('json_output', {})
    model_used = result.get('metadata', {}).get('model_used') or json_out.get('processing_metadata', {}).get('model_used', 'unknown')

    return ClaimDecisionResponse(
        claim_id=json_out.get('claim_id', 'CLM_UNKNOWN'),
        query=request.query,
        decision=result.get('decision', 'NEEDS_REVIEW'),
        payout=result.get('payout', 0),
        currency=json_out.get('currency', 'INR'),
        justification=result.get('justification', ''),
        model_used=model_used,
        clause_ids=result.get('clause_ids', []),
        evidence=json_out.get('evidence', []),
        processing_metadata=result.get('metadata', {})
    )


# --- Observability Endpoints ---

@app.get("/metrics")
def get_metrics():
    """Returns runtime latency, request counters, cache stats, and per-stage latencies."""
    avg_latency = (sum(METRICS['latencies']) / len(METRICS['latencies'])) if METRICS['latencies'] else 0.0
    cache_stats = pipeline.embedding_service.cache_service.get_stats() if pipeline else {}
    stage_latencies = pipeline.get_stage_latencies() if pipeline else {}
    
    return {
        'uptime_seconds': round(time.time() - METRICS['start_time'], 1),
        'total_requests': METRICS['total_requests'],
        'total_queries': METRICS['total_queries'],
        'total_stream_queries': METRICS['total_stream_queries'],
        'total_claim_evaluations': METRICS['total_claim_evaluations'],
        'total_documents_uploaded': METRICS['total_documents_uploaded'],
        'avg_latency_ms': round(avg_latency, 2),
        'stage_latencies': stage_latencies,
        'cache': cache_stats
    }


@app.get("/health")
def health_check():
    """Health check endpoint reporting pipeline status. Returns 503 if unhealthy."""
    if not pipeline:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": "Pipeline not initialized"}
        )
    status_info = pipeline.get_system_status()
    if status_info.get("pipeline_status") != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=status_info
        )
    return status_info
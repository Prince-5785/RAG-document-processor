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
from typing import List, Optional, Dict, Any

import httpx
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

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
    Verifies Bearer token against environment variable INSUREQUERY_API_KEY.
    If no key is configured in environment, requests are permitted in development mode.
    """
    configured_key = os.environ.get("INSUREQUERY_API_KEY") or os.environ.get("API_KEY")
    if not configured_key:
        return True  # Dev mode: permissive when no key configured

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required"
        )
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer' or token != configured_key:
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
    document_ids: Optional[List[str]] = Field(None, description="Optional document IDs to restrict policy scope")
    preferred_model: Optional[str] = Field(None, description="Optional preferred LLM model")

class ClaimDecisionResponse(BaseModel):
    claim_id: str
    query: str
    decision: str
    payout: int
    currency: str
    justification: str
    evidence: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size_bytes: int
    chunk_count: int
    status: str
    message: str

class DocumentUrlUploadRequest(BaseModel):
    url: str
    document_id: Optional[str] = None

# Backward compatibility request/response for HackRX
class HackRXRunRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXRunResponse(BaseModel):
    answers: List[str]


# --- Document Management Endpoints ---

@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    authorized: bool = Depends(verify_auth)
):
    """
    Upload an insurance document (PDF, Word, Email, or TXT) to index into the vector store.
    Validates file format and size.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline is not initialized")

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
        # Preserve original filename
        custom_id = str(uuid.uuid4())
        index_res = pipeline.index_documents([tmp_path], document_id=custom_id)
        
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

@app.post("/documents/from-url", response_model=DocumentUploadResponse)
async def upload_document_from_url(
    req: DocumentUrlUploadRequest,
    authorized: bool = Depends(verify_auth)
):
    """Downloads a document from a URL and indexes it."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not initialized")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(req.url, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL: {str(e)}")

    content = resp.content
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc_id = req.document_id or str(uuid.uuid4())
        index_res = pipeline.index_documents([tmp_path], document_id=doc_id)
        if index_res['failed_files'] > 0:
            raise HTTPException(status_code=400, detail=f"Indexing failed: {index_res['errors']}")

        METRICS['total_documents_uploaded'] += 1
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=os.path.basename(req.url),
            file_size_bytes=len(content),
            chunk_count=index_res['total_chunks'],
            status="ready",
            message="Document downloaded and indexed successfully."
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/documents")
async def list_documents(authorized: bool = Depends(verify_auth)):
    """List all indexed documents with metadata, chunk count, and pages."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    return {"documents": pipeline.list_documents()}

@app.get("/documents/{document_id}")
async def get_document(document_id: str, authorized: bool = Depends(verify_auth)):
    """Get metadata for a specific document."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    doc = pipeline.vector_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return doc

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, authorized: bool = Depends(verify_auth)):
    """Delete a document and its chunks from the vector store."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    success = pipeline.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Failed to delete document '{document_id}'")
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
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

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
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

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
    Returns structured decision (APPROVED / REJECTED), payout amount, justification, and cited clauses.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    METRICS['total_claim_evaluations'] += 1
    result = await pipeline.query(
        user_query=request.query,
        preferred_model=request.preferred_model,
        document_ids=request.document_ids
    )
    json_out = result.get('json_output', {})
    return ClaimDecisionResponse(
        claim_id=json_out.get('claim_id', 'CLM_UNKNOWN'),
        query=request.query,
        decision=result.get('decision', 'REJECTED'),
        payout=result.get('payout', 0),
        currency=json_out.get('currency', 'INR'),
        justification=result.get('justification', ''),
        evidence=json_out.get('evidence', []),
        processing_metadata=result.get('metadata', {})
    )


# --- Backward Compatibility Route for HackRX ---

@app.post("/hackrx/run", response_model=HackRXRunResponse)
async def run_hackrx_processing(
    request: HackRXRunRequest, 
    authorized: bool = Depends(verify_auth)
):
    """
    Backward-compatible contest endpoint.
    Downloads the document URL, indexes it (skipping re-index if content hash seen),
    and answers all questions concurrently.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.documents, follow_redirects=True, timeout=30.0)
            resp.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
            temp_file.write(resp.content)
            temp_file.flush()
            
            # Index document with content-hash check (no more global reset!)
            indexing_result = pipeline.index_documents([temp_file.name])
            if indexing_result.get('failed_files', 0) > 0 and indexing_result.get('processed_files', 0) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process document: {indexing_result['errors']}"
                )

            tasks = [pipeline.answer_question(q) for q in request.questions]
            answers = await asyncio.gather(*tasks)

    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        logger.error(f"Error in /hackrx/run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        
    return HackRXRunResponse(answers=answers)


# --- Observability Endpoints ---

@app.get("/metrics")
def get_metrics():
    """Returns runtime latency, request counters, and cache stats."""
    avg_latency = (sum(METRICS['latencies']) / len(METRICS['latencies'])) if METRICS['latencies'] else 0.0
    cache_stats = pipeline.embedding_service.cache_service.get_stats() if pipeline else {}
    
    return {
        'uptime_seconds': round(time.time() - METRICS['start_time'], 1),
        'total_requests': METRICS['total_requests'],
        'total_queries': METRICS['total_queries'],
        'total_stream_queries': METRICS['total_stream_queries'],
        'total_claim_evaluations': METRICS['total_claim_evaluations'],
        'total_documents_uploaded': METRICS['total_documents_uploaded'],
        'avg_latency_ms': round(avg_latency, 2),
        'cache': cache_stats
    }

@app.get("/health")
def health_check():
    """Health check endpoint reporting pipeline status."""
    if not pipeline:
        return {"status": "unhealthy", "error": "Pipeline not initialized"}
    status_info = pipeline.get_system_status()
    return status_info
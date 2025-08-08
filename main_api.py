import os
import logging
import tempfile
import asyncio
from typing import List, Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline

COMPETITION_API_KEY = "2535b8fdb73ad498683ac089fc149e94d8fce3adf097c44187ecb3fc6cd5552f"

app = FastAPI(
    title="CodersHub RAG API for HackRX",
    description="API for processing insurance documents and answering questions.",
    version="1.0.0"
)

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verifies the Bearer token matches the competition's specific key."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header is missing")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer' or token != COMPETITION_API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format. Use 'Bearer <key>'." )
    return True

try:
    pipeline = RAGPipeline()
except Exception as e:
    logging.fatal(f"Failed to initialize RAG Pipeline on startup: {e}")
    pipeline = None

@app.post("/hackrx/run", response_model=RunResponse)
async def run_rag_processing(
    request: RunRequest, 
    authorized: bool = Depends(verify_api_key)
):
    """Processes a document and concurrently answers all questions about it."""
    if not pipeline:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG Pipeline is not initialized. Check server logs.")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.documents, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
            temp_file.write(response.content)
            pipeline.reset_index()
            indexing_result = pipeline.index_documents([temp_file.name])
            if indexing_result.get('failed_files', 0) > 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to process document from URL: {indexing_result['errors']}")

            tasks = [pipeline.answer_question(q) for q in request.questions]
            answers = await asyncio.gather(*tasks)

    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document from URL: {e.request.url}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during /hackrx/run: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {str(e)}")
        
    return RunResponse(answers=answers)

@app.get("/health")
def health_check():
    """A simple health check endpoint."""
    return {"status": "ok"}
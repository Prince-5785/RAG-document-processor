import os
import logging
import tempfile
from typing import List, Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel

# Import your existing RAG pipeline from the src directory
from src.rag_pipeline import RAGPipeline

# --- API Configuration ---
# The specific, hardcoded API key required by the competition guidelines.
COMPETITION_API_KEY = "2535b8fdb73ad498683ac089fc149e94d8fce3adf097c44187ecb3fc6cd5552f"

# --- FastAPI Application ---
app = FastAPI(
    title="CodersHub RAG API for HackRX",
    description="API for processing insurance documents and answering questions.",
    version="1.0.0"
)

# --- Pydantic Models for Request and Response Validation ---
# This ensures the incoming and outgoing data match the required format.
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- Dependency for API Key Authentication ---
async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verifies the Bearer token matches the competition's specific key."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is missing"
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer' or token != COMPETITION_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Bearer <key>'."
        )
    return True

# --- Initialize RAG Pipeline ---
# The pipeline is initialized once when the server starts for efficiency.
try:
    pipeline = RAGPipeline()
except Exception as e:
    logging.fatal(f"Failed to initialize RAG Pipeline on startup: {e}")
    pipeline = None

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=RunResponse)
async def run_rag_processing(
    request: RunRequest, 
    authorized: bool = Depends(verify_api_key)
):
    """
    Processes a document from a URL and answers a list of questions about it.
    """
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG Pipeline is not initialized. Check server logs."
        )

    document_url = request.documents
    questions = request.questions
    answers = []

    try:
        # 1. Download the document from the URL
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status() # Raises an exception for 4xx or 5xx status codes
        
        # 2. Save the document to a temporary file
        # Using a context manager ensures the file is deleted automatically
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

            # 3. Index the document for this request (resets the store to ensure isolation)
            pipeline.reset_index()
            indexing_result = pipeline.index_documents([temp_file_path])
            if indexing_result.get('failed_files', 0) > 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process document from URL: {indexing_result['errors']}"
                )

            # 4. Process each question against the newly indexed document
            for question in questions:
                answer_result = pipeline.answer_question(question)
                answers.append(answer_result)

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document from URL: {e.request.url}"
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during /hackrx/run: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )
        
    return RunResponse(answers=answers)

@app.get("/health")
def health_check():
    """A simple health check endpoint to verify the service is running."""
    return {"status": "ok"}
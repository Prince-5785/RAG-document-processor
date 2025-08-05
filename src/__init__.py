"""
LLM-Powered Document Processing System

A Retrieval-Augmented Generation (RAG) system for processing natural-language 
insurance queries against large, unstructured documents.
"""

__version__ = "1.0.0"
__author__ = "CodersHub Team"
__email__ = "team@codershub.com"

from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .llm_service import LLMService
from .rag_pipeline import RAGPipeline

__all__ = [
    "DocumentProcessor",
    "EmbeddingService", 
    "VectorStore",
    "LLMService",
    "RAGPipeline"
]

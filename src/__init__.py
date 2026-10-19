"""
InsureQuery - LLM Powered Insurance Document Intelligence System.

Production RAG system combining Dense Vector Search, BM25 Lexical Retrieval,
Cross-Encoder Re-ranking, Similarity Threshold Guardrails, Redis Caching,
and SSE Streaming for insurance policy analysis and claim decisioning.
"""

__version__ = "2.0.0"
__author__ = "InsureQuery Team"

from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .llm_service import LLMService
from .rag_pipeline import RAGPipeline
from .cache_service import CacheService
from .hybrid_retriever import HybridRetriever

__all__ = [
    "DocumentProcessor",
    "EmbeddingService", 
    "VectorStore",
    "LLMService",
    "RAGPipeline",
    "CacheService",
    "HybridRetriever"
]

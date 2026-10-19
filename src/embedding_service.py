import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install it for embedding functionality.")

from .utils import Timer
from .cache_service import CacheService


class EmbeddingService:
    """Service for generating dense embeddings and cross-encoder re-ranking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_config = config.get('embedding', {})
        
        self.model_name = self.embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        configured_device = self.embedding_config.get('device', 'cpu')
        if configured_device == 'cpu' and SENTENCE_TRANSFORMERS_AVAILABLE:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = configured_device
        self.batch_size = self.embedding_config.get('batch_size', 32)
        
        self.reranker_config = self.embedding_config.get('reranker', {})
        self.reranker_model_name = self.reranker_config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.cache_service = CacheService(config)
        
        self.model = None
        self.cross_encoder = None
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> None:
        """Load the sentence transformer model for retrieval."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available")
        if self.model is not None:
            return
        
        self.logger.info(f"Loading embedding model: {self.model_name}")
        try:
            with Timer(f"Loading model {self.model_name}"):
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.logger.info(f"Embedding model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def load_cross_encoder_model(self) -> None:
        """Loads the cross-encoder model for re-ranking."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.reranker_model_name:
            self.logger.warning("No re-ranker model configured or sentence-transformers not available.")
            return
        if self.cross_encoder is not None:
            return
            
        self.logger.info(f"Loading re-ranker model: {self.reranker_model_name}")
        try:
            with Timer(f"Loading re-ranker model {self.reranker_model_name}"):
                self.cross_encoder = CrossEncoder(self.reranker_model_name, device=self.device)
                self.logger.info(f"Re-ranker model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load re-ranker model {self.reranker_model_name}: {e}")
            self.cross_encoder = None

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of documents against a query using a cross-encoder.
        Computes relevance score for each (query, document) pair.
        """
        if not documents:
            return []
            
        if self.cross_encoder is None:
            self.load_cross_encoder_model()
        
        if not self.cross_encoder:
            # Fallback if cross-encoder is unavailable: preserve existing scores
            for doc in documents:
                if 'rerank_score' not in doc:
                    doc['rerank_score'] = doc.get('score', 0.5)
            return sorted(documents, key=lambda x: x.get('rerank_score', 0.0), reverse=True)
            
        self.logger.info(f"Re-ranking {len(documents)} documents for query: '{query[:40]}...'")
        
        pairs = [[query, doc.get('text', '')] for doc in documents]
        raw_scores = np.asarray(self.cross_encoder.predict(pairs, show_progress_bar=False), dtype=np.float32)
        # Apply sigmoid normalization to convert logits to [0.0, 1.0] probability scale
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -25.0, 25.0)))
        
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            
        return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        if self.model is None:
            self.load_model()
        
        self.logger.info(f"Encoding {len(texts)} texts")
        try:
            with Timer(f"Encoding {len(texts)} texts"):
                embeddings = self.model.encode(
                    texts, batch_size=self.batch_size, show_progress_bar=show_progress,
                    convert_to_tensor=False, normalize_embeddings=True
                )
                return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using cache when available."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
            
        cached_embedding = self.cache_service.get_embedding(self.model_name, text)
        if cached_embedding is not None:
            return cached_embedding
        
        embeddings = self.encode_texts([text], show_progress=False)
        embedding = embeddings[0] if len(embeddings) > 0 else np.array([], dtype=np.float32)
        
        if len(embedding) > 0:
            self.cache_service.set_embedding(self.model_name, text, embedding)
        return embedding
    
    def encode_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks, reusing cache where possible."""
        if not chunks:
            return []
            
        texts_to_encode = []
        indices_to_encode = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            cached = self.cache_service.get_embedding(self.model_name, text)
            if cached is not None:
                chunk['embedding'] = cached
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                
        if texts_to_encode:
            new_embeddings = self.encode_texts(texts_to_encode, show_progress=False)
            for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                chunks[idx]['embedding'] = emb
                self.cache_service.set_embedding(self.model_name, text, emb)
                
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model and cache status."""
        return {
            'model_loaded': self.model is not None,
            'model_name': self.model_name,
            'reranker_model': self.reranker_model_name,
            'device': self.device,
            'embedding_dimension': self.get_embedding_dimension() if self.model is not None else 384,
            'cache_stats': self.cache_service.get_stats()
        }
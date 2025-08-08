import os
import logging
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install it for embedding functionality.")

from .utils import Timer, create_directories

class EmbeddingService:
    """Service for generating embeddings and re-ranking documents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_config = config.get('embedding', {})
        
        self.model_name = self.embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.device = self.embedding_config.get('device', 'cpu')
        self.batch_size = self.embedding_config.get('batch_size', 32)
        
        self.reranker_config = self.embedding_config.get('reranker', {})
        self.reranker_model_name = self.reranker_config.get('model_name')
        
        self.cache_config = config.get('cache', {})
        self.enable_cache = self.cache_config.get('enable_embedding_cache', True)
        self.cache_dir = self.cache_config.get('cache_directory', './data/cache')
        
        self.model = None
        self.cross_encoder = None
        self.logger = logging.getLogger(__name__)
        
        if self.enable_cache:
            create_directories([self.cache_dir])
    
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
        """Re-ranks a list of documents against a query using a cross-encoder."""
        if self.cross_encoder is None:
            self.load_cross_encoder_model()
        
        if not self.cross_encoder or not documents:
            return documents
            
        self.logger.info(f"Re-ranking {len(documents)} documents.")
        
        pairs = [[query, doc['text']] for doc in documents]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
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
                return self.model.encode(
                    texts, batch_size=self.batch_size, show_progress_bar=show_progress,
                    convert_to_tensor=False, normalize_embeddings=True
                )
        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text.strip():
            return np.array([])
        if self.enable_cache:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                return cached_embedding
        
        embeddings = self.encode_texts([text], show_progress=False)
        embedding = embeddings[0] if len(embeddings) > 0 else np.array([])
        
        if self.enable_cache and len(embedding) > 0:
            self._cache_embedding(text, embedding)
        return embedding
    
    def encode_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks."""
        if not chunks:
            return []
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encode_texts(texts)
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i] if i < len(embeddings) else np.array([])
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def _get_cache_path(self, text: str) -> str:
        """Generate cache file path for a text."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"embedding_{text_hash}.pkl")
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding for text."""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding for text."""
        cache_path = self._get_cache_path(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {'model_loaded': False}
        return {
            'model_loaded': True, 'model_name': self.model_name,
            'device': str(self.model.device),
            'embedding_dimension': self.get_embedding_dimension()
        }
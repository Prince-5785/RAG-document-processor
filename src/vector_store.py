import os
import logging
import uuid
import json
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install chromadb for vector storage functionality.")

from .utils import Timer, create_directories


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_config = config.get('vector_store', {})
        self.persist_directory = self.vector_config.get('persist_directory', './data/chroma_index')
        self.collection_name = self.vector_config.get('collection_name', 'insurance_documents')
        self.distance_metric = self.vector_config.get('distance_metric', 'cosine')
        
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
        
        create_directories([self.persist_directory])
    
    def initialize(self) -> None:
        """Initializes the ChromaDB client and gets or creates the collection."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
        if self.collection is not None:
            return
        
        self.logger.info("Initializing ChromaDB vector store")
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            self.logger.info(f"Loaded or created collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            self.collection = None
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Adds document chunks to the vector store."""
        if not chunks:
            return
        if self.collection is None:
            self.initialize()
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        try:
            with Timer(f"Adding {len(chunks)} chunks"):
                ids, embeddings, documents, metadatas = [], [], [], []
                for chunk in chunks:
                    ids.append(str(uuid.uuid4()))
                    embedding = chunk.get('embedding', [])
                    embeddings.append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
                    documents.append(chunk.get('text', ''))
                    metadatas.append(self._serialize_metadata(chunk.get('metadata', {})))
                
                self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                self.logger.info(f"Successfully added {len(chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
            raise
            
    # --- THIS IS THE MISSING METHOD ---
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for the top_k most similar documents without a score threshold.
        """
        if self.collection is None:
            self.initialize()
        if query_embedding.size == 0:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            documents = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    meta = results['metadatas'][0][i] or {}
                    # Normalize a few keys for later explainability
                    if 'page_hint' in meta and isinstance(meta['page_hint'], str):
                        try:
                            meta['page_hint'] = int(meta['page_hint'])
                        except Exception:
                            pass
                    documents.append({
                        'text': results['documents'][0][i],
                        'metadata': meta,
                        'score': 1 - results['distances'][0][i] if results.get('distances') else 0.0
                    })
            return documents
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {e}", exc_info=True)
            return []

    def similarity_search_with_threshold(self, query_embedding: np.ndarray, top_k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Searches for similar documents and filters them by a minimum similarity threshold."""
        results = self.similarity_search(query_embedding, top_k)
        return [doc for doc in results if doc['score'] >= score_threshold]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Gets statistics about the collection."""
        if self.collection is None:
            self.initialize()
        try:
            count = self.collection.count()
            return {'collection_name': self.collection_name, 'document_count': count, 'distance_metric': self.distance_metric}
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}", exc_info=True)
            return {}
    
    def reset_collection(self) -> None:
        """Resets the collection by deleting and recreating it."""
        if self.client is None:
            self.initialize()
        
        try:
            self.logger.info(f"Resetting collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            self.logger.info(f"Collection '{self.collection_name}' reset successfully.")
        except Exception:
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.logger.info(f"Collection '{self.collection_name}' ensured to exist after reset attempt.")
            except Exception as final_e:
                self.logger.error(f"Failed to reset or create collection: {final_e}", exc_info=True)
                self.collection = None
                raise
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata values to types accepted by Chroma (Bool, Int, Float, Str).
        - Drops keys with None values (Chroma Rust binding rejects None)
        - Casts numpy scalars/arrays to Python primitives
        - Stringifies any unsupported types as a safe fallback
        """
        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                # Skip None values entirely to avoid Chroma type errors
                continue
            try:
                if isinstance(value, np.bool_):
                    serialized[key] = bool(value)
                elif isinstance(value, (bool,)):
                    serialized[key] = value
                elif isinstance(value, np.integer):
                    serialized[key] = int(value)
                elif isinstance(value, int):
                    serialized[key] = value
                elif isinstance(value, np.floating):
                    serialized[key] = float(value)
                elif isinstance(value, float):
                    serialized[key] = value
                elif isinstance(value, np.ndarray):
                    # Not supported directly in metadata; stringify
                    serialized[key] = json.dumps(value.tolist())
                elif isinstance(value, (list, dict, tuple)):
                    # Chroma expects primitive types; stringify complex types
                    serialized[key] = json.dumps(value)
                else:
                    # Assume string-compatible
                    serialized[key] = str(value)
            except Exception:
                # As a last resort, stringify to avoid failing the batch
                try:
                    serialized[key] = str(value)
                except Exception:
                    # If even str() fails, drop the key
                    self.logger.warning(f"Dropping unserializable metadata key: {key}")
        return serialized
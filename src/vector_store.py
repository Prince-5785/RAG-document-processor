"""
ChromaDB Vector Store with document-level scoping, metadata filtering,
document registry tracking, and deduplication by content hash.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install chromadb for vector storage functionality.")

from .utils import Timer, create_directories, save_json, load_json


class VectorStore:
    """ChromaDB-based vector store supporting multi-document indexing and document_id filtering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_config = config.get('vector_store', {})
        self.persist_directory = self.vector_config.get('persist_directory', './data/chroma_index')
        self.collection_name = self.vector_config.get('collection_name', 'insurequery_documents')
        self.distance_metric = self.vector_config.get('distance_metric', 'cosine')
        self.registry_file = os.path.join(self.persist_directory, 'document_registry.json')
        
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
        
        create_directories([self.persist_directory])
        self.document_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads registered documents metadata."""
        if os.path.exists(self.registry_file):
            return load_json(self.registry_file)
        return {}

    def _save_registry(self) -> None:
        """Persists registered documents metadata."""
        try:
            save_json(self.document_registry, self.registry_file)
        except Exception as e:
            self.logger.warning(f"Could not save document registry: {e}")

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

    def has_document_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Checks if a document with this content hash is already indexed."""
        for doc_id, meta in self.document_registry.items():
            if meta.get('file_hash') == file_hash:
                return meta
        return None

    def register_document(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Records a document in the local registry."""
        clean_metadata = {k: v for k, v in metadata.items() if k != 'file_path'}
        clean_metadata['indexed_at'] = clean_metadata.get('indexed_at') or datetime.utcnow().isoformat()
        clean_metadata['id'] = doc_id
        self.document_registry[doc_id] = clean_metadata
        self._save_registry()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all document chunks with their texts and metadatas from ChromaDB."""
        if not CHROMADB_AVAILABLE:
            return []
        if self.collection is None:
            try:
                self.initialize()
            except Exception as e:
                self.logger.warning(f"Could not initialize ChromaDB to fetch all chunks: {e}")
                return []
        try:
            results = self.collection.get(include=['documents', 'metadatas'])
            documents = results.get('documents', []) or []
            metadatas = results.get('metadatas', []) or []
            chunks = []
            for doc_text, meta in zip(documents, metadatas):
                chunks.append({
                    'text': doc_text,
                    'metadata': meta or {}
                })
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to fetch all chunks from ChromaDB: {e}", exc_info=True)
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns all currently indexed documents."""
        return list(self.document_registry.values())

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Gets metadata for a specific document."""
        return self.document_registry.get(document_id)

    def delete_document(self, document_id: str) -> bool:
        """Deletes all chunks belonging to a document from ChromaDB and registry."""
        if self.collection is None:
            self.initialize()

        try:
            # Delete from ChromaDB by metadata filter
            self.collection.delete(where={"document_id": document_id})
            if document_id in self.document_registry:
                del self.document_registry[document_id]
                self._save_registry()
            self.logger.info(f"Deleted document {document_id} from vector store.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}", exc_info=True)
            return False

    def add_documents(self, chunks: List[Dict[str, Any]], document_id: Optional[str] = None) -> None:
        """Adds document chunks to the vector store with document_id tagging."""
        if not chunks:
            return
        if self.collection is None:
            self.initialize()
        
        doc_id = document_id or str(uuid.uuid4())
        self.logger.info(f"Adding {len(chunks)} chunks to vector store for doc_id: {doc_id}")
        
        try:
            with Timer(f"Adding {len(chunks)} chunks"):
                ids, embeddings, documents, metadatas = [], [], [], []
                for chunk in chunks:
                    chunk_id = str(uuid.uuid4())
                    ids.append(chunk_id)
                    
                    embedding = chunk.get('embedding', [])
                    embeddings.append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
                    documents.append(chunk.get('text', ''))
                    
                    meta = dict(chunk.get('metadata', {}))
                    meta['document_id'] = doc_id
                    metadatas.append(self._serialize_metadata(meta))
                
                self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                self.logger.info(f"Successfully added {len(chunks)} chunks to collection")
        except Exception as e:
            self.logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
            raise

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for the top_k most similar documents, optionally filtered by document_ids.
        """
        if self.collection is None:
            self.initialize()
        if query_embedding.size == 0:
            return []
        
        # Build ChromaDB metadata filter if document_ids provided
        where_clause = None
        if document_ids:
            if len(document_ids) == 1:
                where_clause = {"document_id": document_ids[0]}
            else:
                where_clause = {"document_id": {"$in": document_ids}}
        
        try:
            query_kwargs = {
                'query_embeddings': [query_embedding.tolist()],
                'n_results': top_k,
                'include': ['documents', 'metadatas', 'distances']
            }
            if where_clause:
                query_kwargs['where'] = where_clause

            results = self.collection.query(**query_kwargs)
            
            documents = []
            if results and results.get('documents') and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results.get('distances') else 0.0
                    # For cosine distance, similarity = 1 - distance
                    similarity = max(0.0, 1.0 - distance)
                    documents.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': float(similarity)
                    })
            return documents
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {e}", exc_info=True)
            return []

    def similarity_search_with_threshold(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.3,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Searches for similar documents and filters them by a minimum similarity threshold."""
        results = self.similarity_search(query_embedding, top_k, document_ids=document_ids)
        return [doc for doc in results if doc.get('score', 0.0) >= score_threshold]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Gets statistics about the collection."""
        if self.collection is None:
            self.initialize()
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'registered_documents': len(self.document_registry),
                'distance_metric': self.distance_metric
            }
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
            self.document_registry.clear()
            self._save_registry()
            self.logger.info(f"Collection '{self.collection_name}' reset successfully.")
        except Exception:
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.document_registry.clear()
                self._save_registry()
            except Exception as final_e:
                self.logger.error(f"Failed to reset collection: {final_e}", exc_info=True)
                self.collection = None
                raise
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Converts metadata values to types supported by ChromaDB (str, int, float, bool)."""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (np.integer, int)):
                serialized[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                serialized[key] = float(value)
            elif isinstance(value, bool):
                serialized[key] = bool(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = json.dumps(value)
            elif value is None:
                serialized[key] = ""
            else:
                serialized[key] = str(value)
        return serialized
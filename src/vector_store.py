"""
Vector store implementation using ChromaDB for storing and retrieving document embeddings.
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
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
        
        # Create directories
        create_directories([self.persist_directory])
    
    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
        
        if self.client is not None:
            return
        
        self.logger.info("Initializing ChromaDB vector store")
        
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'embedding', and 'metadata' keys
        """
        if not chunks:
            return
        
        if self.collection is None:
            self.initialize()
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        try:
            with Timer(f"Adding {len(chunks)} chunks to vector store"):
                # Prepare data for ChromaDB
                ids = []
                embeddings = []
                documents = []
                metadatas = []
                
                for chunk in chunks:
                    # Generate unique ID
                    chunk_id = str(uuid.uuid4())
                    ids.append(chunk_id)
                    
                    # Add embedding
                    embedding = chunk.get('embedding', [])
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    embeddings.append(embedding)
                    
                    # Add document text
                    documents.append(chunk.get('text', ''))
                    
                    # Add metadata
                    metadata = chunk.get('metadata', {})
                    # Convert numpy types to Python types for JSON serialization
                    metadata = self._serialize_metadata(metadata)
                    metadatas.append(metadata)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                self.logger.info(f"Successfully added {len(chunks)} chunks")
                
        except Exception as e:
            self.logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with metadata and scores
        """
        if self.collection is None:
            self.initialize()
        
        if len(query_embedding) == 0:
            return []
        
        try:
            # Convert numpy array to list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            documents = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1 - results['distances'][0][i] if results['distances'] else 0.0  # Convert distance to similarity
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def similarity_search_with_threshold(self, query_embedding: np.ndarray, top_k: int = 5, 
                                       score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents with a minimum similarity threshold.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar documents above the threshold
        """
        results = self.similarity_search(query_embedding, top_k)
        
        # Filter by threshold
        filtered_results = [doc for doc in results if doc['score'] >= score_threshold]
        
        return filtered_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if self.collection is None:
            self.initialize()
        
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'distance_metric': self.distance_metric
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self.client is None:
            self.initialize()
        
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
    
    def reset_collection(self) -> None:
        """Reset the collection (delete and recreate)."""
        try:
            self.delete_collection()
            self.initialize()
            self.logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata to JSON-serializable format."""
        serialized = {}
        
        for key, value in metadata.items():
            if isinstance(value, np.integer):
                serialized[key] = int(value)
            elif isinstance(value, np.floating):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if self.collection is None:
            self.initialize()
        
        try:
            # ChromaDB where clause format
            where_clause = {}
            for key, value in metadata_filter.items():
                where_clause[key] = {"$eq": value}
            
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            documents = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    doc = {
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to search by metadata: {e}")
            return []

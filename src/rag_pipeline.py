import logging
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .llm_service import LLMService
from .utils import Timer, load_config, setup_logging


class RAGPipeline:
    """Main pipeline for Retrieval-Augmented Generation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the RAG pipeline by loading YAML config and injecting secrets from .env.
        """
        # Load base configuration from YAML
        self.config = load_config(config_path)
        
        # Load .env and inject the API key
        load_dotenv() 
        groq_api_key = os.environ.get("GROQ_API_KEY")

        if groq_api_key:
            self.config.setdefault('llm', {}).setdefault('api', {})['groq_api_key'] = groq_api_key
        
        # Now that config is complete, set up logging
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Log the result of the key loading
        if groq_api_key:
            self.logger.info("GROQ API key successfully loaded from .env file.")
        else:
            self.logger.warning("GROQ_API_KEY not found in .env file. API calls will fail.")

        # Initialize components with the now-complete config
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = VectorStore(self.config)
        self.llm_service = LLMService(self.config)
        
        # Retrieval configuration
        self.retrieval_config = self.config.get('retrieval', {})
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.score_threshold = self.retrieval_config.get('score_threshold', 0.7)
        
        self.logger.info("RAG Pipeline initialized")
    
    def index_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Index documents into the vector store.
        """
        self.logger.info(f"Starting document indexing for {len(file_paths)} files")
        
        results = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        try:
            with Timer("Document indexing"):
                processed_docs = self.document_processor.process_multiple_files(file_paths)
                all_chunks = []
                
                for doc in processed_docs:
                    if doc['metadata'].get('processed_successfully', False):
                        results['processed_files'] += 1
                        chunks = self.document_processor.chunk_text(
                            doc['content'], 
                            metadata={
                                'file_path': doc['file_path'],
                                'file_name': doc['file_name'],
                                'file_type': doc['file_type'],
                                'file_hash': doc['file_hash']
                            }
                        )
                        all_chunks.extend(chunks)
                        results['total_chunks'] += len(chunks)
                    else:
                        results['failed_files'] += 1
                        error_msg = doc['metadata'].get('error', 'Unknown error')
                        results['errors'].append(f"{doc['file_name']}: {error_msg}")
                
                if all_chunks:
                    self.logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
                    chunks_with_embeddings = self.embedding_service.encode_chunks(all_chunks)
                    
                    self.logger.info("Storing chunks in vector database")
                    self.vector_store.add_documents(chunks_with_embeddings)
                    
                    self.logger.info(f"Successfully indexed {results['processed_files']} files with {results['total_chunks']} chunks")
                else:
                    self.logger.warning("No chunks generated from documents")
                
        except Exception as e:
            self.logger.error(f"Error during document indexing: {e}", exc_info=True)
            results['errors'].append(f"Pipeline error: {str(e)}")
        
        return results
    
    def query(self, user_query: str, preferred_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query for the Streamlit App (Decision Making Task).
        """
        self.logger.info(f"Processing decision query: '{user_query}' with preferred_model: {preferred_model}")
        
        try:
            with Timer("Decision query processing"):
                parsed_query = self.llm_service.parse_query(user_query, preferred_model=preferred_model)
                query_embedding = self.embedding_service.encode_single_text(user_query)
                retrieved_docs = self.vector_store.similarity_search_with_threshold(
                    query_embedding, 
                    top_k=self.top_k,
                    score_threshold=self.score_threshold
                )
                decision_result = self.llm_service.make_decision(parsed_query, retrieved_docs, preferred_model=preferred_model)
                
                result = {
                    'query': user_query,
                    'parsed_query': parsed_query,
                    'decision': decision_result.get('decision', 'REJECTED'),
                    'payout': decision_result.get('payout', 0),
                    'justification': decision_result.get('justification', 'No justification provided'),
                    'retrieved_documents': len(retrieved_docs),
                    'relevant_clauses': [
                        {
                            'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                            'score': doc['score'],
                            'metadata': doc.get('metadata', {})
                        }
                        for doc in retrieved_docs
                    ],
                    'metadata': {
                        'processing_successful': True,
                        'embedding_dimension': self.embedding_service.get_embedding_dimension(),
                        'retrieval_threshold': self.score_threshold
                    }
                }
                
                result['json_output'] = self._generate_json_output(result)
                self.logger.info(f"Query processed successfully: {result['decision']}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                'query': user_query,
                'decision': 'ERROR',
                'payout': 0,
                'justification': f'An internal error occurred: {str(e)}',
                'metadata': { 'processing_successful': False, 'error': str(e) }
            }

    def answer_question(self, question: str) -> str:
        """
        Processes a single question for the API (Question Answering Task).
        """
        self.logger.info(f"Answering question: '{question}'")
        try:
            query_embedding = self.embedding_service.encode_single_text(question)
            
            # Use a slightly lower threshold for QA to gather more context
            retrieved_chunks = self.vector_store.similarity_search_with_threshold(
                query_embedding, 
                top_k=self.top_k,
                score_threshold=0.3 
            )
            
            answer = self.llm_service.answer_question(question, retrieved_chunks)
            return answer

        except Exception as e:
            self.logger.error(f"Error answering question '{question}': {e}", exc_info=True)
            return "I am sorry, but an error occurred while processing your question."

    def _generate_json_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured JSON output for audit purposes."""
        return {
            'claim_id': f"CLM_{hash(result['query']) % 1000000:06d}",
            'query': result['query'],
            'parsed_information': result['parsed_query'],
            'decision': result['decision'],
            'payout_amount': result['payout'],
            'currency': 'INR',
            'justification': result['justification'],
            'evidence': [
                {
                    'clause_id': f"CL_{i+1:03d}",
                    'text': clause['text'],
                    'relevance_score': clause['score'],
                    'source': clause['metadata'].get('file_name', 'Unknown')
                }
                for i, clause in enumerate(result.get('relevant_clauses', []))
            ],
            'processing_metadata': {
                'timestamp': None,
                'model_version': '1.0.0',
                'retrieval_method': 'semantic_similarity',
                'documents_retrieved': result.get('retrieved_documents', 0)
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            embedding_info = self.embedding_service.get_model_info()
            
            return {
                'pipeline_status': 'healthy',
                'vector_store': vector_stats,
                'embedding_service': embedding_info,
                'configuration': {
                    'top_k': self.top_k,
                    'score_threshold': self.score_threshold,
                    'supported_formats': self.document_processor.get_supported_formats()
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'pipeline_status': 'error', 'error': str(e)}
    
    def reset_index(self) -> Dict[str, Any]:
        """Reset the vector store index."""
        try:
            self.vector_store.reset_collection()
            return {'status': 'success', 'message': 'Vector store index reset successfully'}
        except Exception as e:
            self.logger.error(f"Error resetting index: {e}")
            return {'status': 'error', 'message': f'Failed to reset index: {str(e)}'}
    
    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        for i, query in enumerate(queries):
            self.logger.info(f"Processing batch query {i+1}/{len(queries)}")
            result = self.query(query)
            results.append(result)
        return results
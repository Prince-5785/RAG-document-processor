"""
InsureQuery Main RAG Pipeline.
Implements:
- Hybrid Retrieval (Dense Vector + BM25 Sparse Search via Reciprocal Rank Fusion)
- Cross-Encoder Re-ranking
- Similarity & Rerank Score Threshold Guardrails
- Non-blocking CPU execution via ThreadPool Executor
- Source Citations with file, page number, and chunk text
- Token-by-token SSE streaming support
- Multi-document indexing with content hash deduplication
"""

import logging
import json
import os
import uuid
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, AsyncGenerator

from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .llm_service import LLMService
from .hybrid_retriever import HybridRetriever
from .utils import Timer, load_config, setup_logging


class RAGPipeline:
    """Enterprise RAG Pipeline for Insurance Document Intelligence."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load base configuration from YAML
        self.config = load_config(config_path)
        
        # Load .env and inject secrets
        load_dotenv() 
        groq_api_keys_str = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
        
        if groq_api_keys_str:
            groq_api_keys_list = [k.strip() for k in groq_api_keys_str.split(',') if k.strip()]
            self.config.setdefault('llm', {}).setdefault('api', {})['groq_api_keys'] = groq_api_keys_list
        
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-services
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.vector_store = VectorStore(self.config)
        self.llm_service = LLMService(self.config)
        self.hybrid_retriever = HybridRetriever(self.config)
        
        # Retrieval parameters
        self.retrieval_config = self.config.get('retrieval', {})
        self.candidate_k = self.retrieval_config.get('candidate_k', 25)
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.score_threshold = float(self.retrieval_config.get('score_threshold', 0.3))
        
        # Thread pool executor for non-blocking CPU-bound math/model inferences
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # In-memory chunk cache for BM25 index
        self._all_indexed_chunks: List[Dict[str, Any]] = []
        
        self.logger.info(f"InsureQuery RAG Pipeline initialized (threshold={self.score_threshold}, top_k={self.top_k})")

    def index_documents(
        self,
        file_paths: List[str],
        document_id: Optional[str] = None,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents into vector store and BM25 index with deduplication.
        Does NOT wipe the global index.
        """
        self.logger.info(f"Indexing {len(file_paths)} documents (doc_id={document_id})")
        
        results = {
            'total_files': len(file_paths),
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'document_ids': [],
            'errors': []
        }
        
        new_chunks_to_index: List[Dict[str, Any]] = []

        try:
            with Timer(f"Indexing {len(file_paths)} files"):
                for file_path in file_paths:
                    if not os.path.exists(file_path):
                        results['failed_files'] += 1
                        results['errors'].append(f"File not found: {file_path}")
                        continue
                    
                    try:
                        doc = self.document_processor.process_file(file_path)
                    except Exception as e:
                        results['failed_files'] += 1
                        results['errors'].append(f"{os.path.basename(file_path)}: {str(e)}")
                        continue
                    
                    if not doc['metadata'].get('processed_successfully', False):
                        results['failed_files'] += 1
                        results['errors'].append(f"{doc['file_name']}: {doc['metadata'].get('error', 'Processing failed')}")
                        continue
                    
                    file_hash = doc['file_hash']
                    
                    # Deduplication check
                    if not force_reindex:
                        existing = self.vector_store.has_document_hash(file_hash)
                        if existing:
                            self.logger.info(f"Document {doc['file_name']} with hash {file_hash[:8]} already indexed. Skipping.")
                            results['skipped_files'] += 1
                            results['document_ids'].append(existing['id'])
                            continue

                    assigned_doc_id = document_id or str(uuid.uuid4())
                    results['document_ids'].append(assigned_doc_id)
                    
                    # Chunk with page numbers retained
                    chunks = self.document_processor.chunk_document(doc, extra_metadata={'document_id': assigned_doc_id})
                    
                    if not chunks:
                        results['failed_files'] += 1
                        results['errors'].append(f"{doc['file_name']}: No text content extracted")
                        continue

                    new_chunks_to_index.extend(chunks)
                    results['processed_files'] += 1
                    results['total_chunks'] += len(chunks)

                    # Register document in vector store registry
                    self.vector_store.register_document(assigned_doc_id, {
                        'filename': doc['file_name'],
                        'file_path': doc['file_path'],
                        'file_hash': file_hash,
                        'file_size': doc['metadata'].get('file_size', 0),
                        'total_pages': doc['metadata'].get('total_pages', 1),
                        'chunk_count': len(chunks),
                        'status': 'ready'
                    })

                if new_chunks_to_index:
                    self.logger.info(f"Encoding {len(new_chunks_to_index)} chunks...")
                    chunks_with_embeddings = self.embedding_service.encode_chunks(new_chunks_to_index)
                    
                    self.logger.info("Persisting chunks to vector store...")
                    # We pass assigned doc id if single file
                    single_doc_id = results['document_ids'][0] if len(results['document_ids']) == 1 else None
                    self.vector_store.add_documents(chunks_with_embeddings, document_id=single_doc_id)
                    
                    # Update BM25 hybrid index
                    self._all_indexed_chunks.extend(new_chunks_to_index)
                    self.hybrid_retriever.build_bm25_index(self._all_indexed_chunks)
                    
                    self.logger.info(f"Successfully indexed {results['processed_files']} files with {results['total_chunks']} chunks")

        except Exception as e:
            self.logger.error(f"Error indexing documents: {e}", exc_info=True)
            results['errors'].append(f"Pipeline error: {str(e)}")
            
        return results

    async def retrieve_relevant_chunks(
        self,
        query_text: str,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Retrieval (Vector + BM25) and Cross-Encoder Re-ranking
        in a non-blocking thread executor, applying the threshold guardrail.
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        loop = asyncio.get_running_loop()

        def _do_hybrid_retrieve() -> List[Dict[str, Any]]:
            # 1. Dense vector candidate retrieval
            query_emb = self.embedding_service.encode_single_text(query_text)
            dense_docs = self.vector_store.similarity_search(
                query_emb,
                top_k=self.candidate_k,
                document_ids=document_ids
            )

            # 2. BM25 sparse candidate retrieval
            bm25_docs = self.hybrid_retriever.search_bm25(
                query_text,
                top_k=self.candidate_k,
                document_ids=document_ids
            )

            # 3. Reciprocal Rank Fusion
            fused_candidates = self.hybrid_retriever.fuse_results(
                dense_results=dense_docs,
                sparse_results=bm25_docs,
                top_k=self.candidate_k
            )

            if not fused_candidates:
                return []

            # 4. Cross-Encoder Re-ranking
            reranked = self.embedding_service.rerank_documents(query_text, fused_candidates)
            return reranked

        # Run CPU-bound retrieval & reranking in executor
        reranked_docs = await loop.run_in_executor(self.executor, _do_hybrid_retrieve)

        if not reranked_docs:
            return []

        # 5. Threshold Guardrail:
        # Check if top candidate meets the score threshold
        top_score = reranked_docs[0].get('rerank_score', reranked_docs[0].get('score', 0.0))
        if top_score < threshold:
            self.logger.info(
                f"Threshold Guardrail Triggered: Top score {top_score:.3f} below threshold {threshold:.3f}. Dropping all candidates."
            )
            return []

        # Keep candidates meeting threshold, up to top_k
        filtered_docs = [
            doc for doc in reranked_docs[:self.top_k]
            if doc.get('rerank_score', doc.get('score', 0.0)) >= threshold
        ]
        return filtered_docs

    async def answer_question(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> str:
        """
        Answers a single question with threshold guardrails and hybrid retrieval.
        Returns 'not found' immediately without calling LLM if below threshold.
        """
        self.logger.info(f"Answering question: '{question}'")
        try:
            final_context = await self.retrieve_relevant_chunks(
                question,
                document_ids=document_ids,
                score_threshold=score_threshold
            )
            
            # Guardrail check: no documents passed threshold
            if not final_context:
                return "The answer to this question could not be found in the provided document."

            answer = await self.llm_service.answer_question(question, final_context)
            return answer

        except Exception as e:
            self.logger.error(f"Error answering question '{question}': {e}", exc_info=True)
            return "I am sorry, but an error occurred while processing your question."

    async def answer_with_citations(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Returns an answer along with formal source citations:
        {answer, sources: [{file, page, chunk_text, score}], threshold_applied: bool}
        """
        final_context = await self.retrieve_relevant_chunks(
            question,
            document_ids=document_ids,
            score_threshold=score_threshold
        )

        sources = []
        for ctx in final_context:
            meta = ctx.get('metadata', {})
            score = ctx.get('rerank_score', ctx.get('score', 0.0))
            sources.append({
                'file': meta.get('file_name', 'Unknown'),
                'page': int(meta.get('page', 1)),
                'chunk_text': ctx.get('text', ''),
                'score': round(float(score), 4)
            })

        if not final_context:
            return {
                'answer': "The answer to this question could not be found in the provided document.",
                'sources': [],
                'threshold_applied': True
            }

        answer = await self.llm_service.answer_question(question, final_context)
        return {
            'answer': answer,
            'sources': sources,
            'threshold_applied': False
        }

    async def stream_answer(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Server-Sent Events (SSE) generator.
        Emits:
        1. Source citations event: data: {"type": "sources", "sources": [...]}\n\n
        2. Token events: data: {"type": "token", "content": "..."}\n\n
        3. Done event: data: {"type": "done"}\n\n
        """
        final_context = await self.retrieve_relevant_chunks(
            question,
            document_ids=document_ids,
            score_threshold=score_threshold
        )

        sources = [
            {
                'file': ctx.get('metadata', {}).get('file_name', 'Unknown'),
                'page': int(ctx.get('metadata', {}).get('page', 1)),
                'chunk_text': ctx.get('text', '')[:200],
                'score': round(float(ctx.get('rerank_score', ctx.get('score', 0.0))), 4)
            }
            for ctx in final_context
        ]

        # 1. Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # 2. Check guardrail
        if not final_context:
            not_found_msg = "The answer to this question could not be found in the provided document."
            yield f"data: {json.dumps({'type': 'token', 'content': not_found_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # 3. Stream tokens from LLM
        async for token in self.llm_service.stream_answer_question(question, final_context):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # 4. Send done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def query(
        self,
        user_query: str,
        preferred_model: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates an insurance query for structured decision-making (Approval/Rejection with Payout).
        """
        self.logger.info(f"Processing structured insurance claim query: '{user_query}'")
        
        try:
            with Timer("Insurance claim query processing"):
                # Run query parsing and hybrid retrieval concurrently
                parsed_query_task = self.llm_service.parse_query(user_query, preferred_model=preferred_model)
                retrieval_task = self.retrieve_relevant_chunks(user_query, document_ids=document_ids)

                parsed_query, final_docs = await asyncio.gather(parsed_query_task, retrieval_task)

                # Threshold guardrail check
                if not final_docs:
                    return {
                        'query': user_query,
                        'parsed_query': parsed_query,
                        'decision': 'REJECTED',
                        'payout': 0,
                        'justification': 'No relevant policy clauses met the similarity threshold to support this claim.',
                        'retrieved_documents': 0,
                        'relevant_clauses': [],
                        'metadata': {
                            'processing_successful': True,
                            'threshold_triggered': True
                        },
                        'json_output': self._generate_json_output({
                            'query': user_query,
                            'parsed_query': parsed_query,
                            'decision': 'REJECTED',
                            'payout': 0,
                            'justification': 'No relevant policy clauses met the similarity threshold to support this claim.',
                            'relevant_clauses': []
                        })
                    }

                # LLM makes decision on retrieved clauses
                decision_result = await self.llm_service.make_decision(
                    parsed_query, final_docs, preferred_model=preferred_model
                )
                
                relevant_clauses = [{
                    'text': doc['text'][:250] + '...' if len(doc['text']) > 250 else doc['text'],
                    'score': doc.get('rerank_score', doc.get('score', 0.0)),
                    'metadata': doc.get('metadata', {})
                } for doc in final_docs]

                result = {
                    'query': user_query,
                    'parsed_query': parsed_query,
                    'decision': decision_result.get('decision', 'REJECTED'),
                    'payout': decision_result.get('payout', 0),
                    'justification': decision_result.get('justification', 'No justification provided'),
                    'retrieved_documents': len(final_docs),
                    'relevant_clauses': relevant_clauses,
                    'metadata': {
                        'processing_successful': True,
                        'threshold_triggered': False,
                        'model_used': decision_result.get('model_used', 'unknown')
                    }
                }
                
                result['json_output'] = self._generate_json_output(result)
                return result
                
        except Exception as e:
            self.logger.error(f"Error processing claim query: {e}", exc_info=True)
            return {
                'query': user_query,
                'decision': 'ERROR',
                'payout': 0,
                'justification': f'An internal error occurred: {str(e)}',
                'metadata': {'processing_successful': False, 'error': str(e)}
            }

    def _generate_json_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deterministic structured JSON audit output."""
        # Use MD5 hash for stable claim_id across restarts
        query_hash = hashlib.md5(result['query'].encode('utf-8')).hexdigest()[:8].upper()
        
        evidence = []
        for i, clause in enumerate(result.get('relevant_clauses', [])):
            meta = clause.get('metadata', {})
            evidence.append({
                'clause_id': f"CL_{i+1:03d}",
                'text': clause.get('text', ''),
                'relevance_score': round(float(clause.get('score', 0.0)), 4),
                'source': meta.get('file_name', 'Unknown'),
                'page': meta.get('page', 1)
            })

        return {
            'claim_id': f"CLM_{query_hash}",
            'query': result['query'],
            'parsed_information': result.get('parsed_query', {}),
            'decision': result.get('decision', 'REJECTED'),
            'payout_amount': result.get('payout', 0),
            'currency': 'INR',
            'justification': result.get('justification', ''),
            'evidence': evidence,
            'processing_metadata': {
                'model_version': '2.0.0',
                'retrieval_method': 'hybrid_bm25_dense_reranked',
                'documents_retrieved': len(evidence)
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete health status of all pipeline components."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            embedding_info = self.embedding_service.get_model_info()
            
            return {
                'pipeline_status': 'healthy',
                'project': 'InsureQuery',
                'version': '2.0.0',
                'vector_store': vector_stats,
                'embedding_service': embedding_info,
                'configuration': {
                    'top_k': self.top_k,
                    'candidate_k': self.candidate_k,
                    'score_threshold': self.score_threshold,
                    'hybrid_enabled': self.hybrid_retriever.enabled,
                    'supported_formats': self.document_processor.get_supported_formats()
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'pipeline_status': 'error', 'error': str(e)}

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        return self.vector_store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from vector store and BM25 index."""
        success = self.vector_store.delete_document(document_id)
        if success:
            self._all_indexed_chunks = [
                c for c in self._all_indexed_chunks
                if c.get('metadata', {}).get('document_id') != document_id
            ]
            self.hybrid_retriever.build_bm25_index(self._all_indexed_chunks)
        return success
    
    def reset_index(self) -> Dict[str, Any]:
        """Reset the vector store and BM25 index."""
        try:
            self.vector_store.reset_collection()
            self._all_indexed_chunks = []
            self.hybrid_retriever.build_bm25_index([])
            return {'status': 'success', 'message': 'Vector store and BM25 index reset successfully'}
        except Exception as e:
            self.logger.error(f"Error resetting index: {e}")
            return {'status': 'error', 'message': f'Failed to reset index: {str(e)}'}
    
    async def batch_query(self, queries: List[str], document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently."""
        tasks = [self.query(q, document_ids=document_ids) for q in queries]
        results = await asyncio.gather(*tasks)
        return results
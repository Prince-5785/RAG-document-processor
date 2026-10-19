"""
Unit and integration tests for the InsureQuery RAG Pipeline.
Tests:
- Pipeline initialization
- Hybrid Retrieval (Dense Vector + BM25)
- Cross-Encoder Re-ranking
- Threshold Guardrail (dropping irrelevant candidates & preventing hallucinations)
- Citations with page numbers
- Server-Sent Events (SSE) streaming output
- Structured Insurance Claim Decisions (Pydantic / JSON schema)
- Redis / In-memory embedding cache
- Document registry and deletion
"""

import pytest
import tempfile
import os
import json
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from src.rag_pipeline import RAGPipeline
from src.cache_service import CacheService
from src.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'document_processing': {
            'chunk_size': 500,
            'chunk_overlap': 100,
            'max_file_size_mb': 25,
            'allowed_extensions': ['txt', 'pdf', 'docx', 'eml']
        },
        'embedding': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'device': 'cpu',
            'reranker': {
                'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'device': 'cpu'
            }
        },
        'vector_store': {
            'persist_directory': './test_chroma_index',
            'collection_name': 'test_insurequery_documents',
            'distance_metric': 'cosine'
        },
        'retrieval': {
            'top_k': 3,
            'candidate_k': 10,
            'score_threshold': 0.3,
            'hybrid_search': {
                'enabled': True,
                'vector_weight': 0.6,
                'bm25_weight': 0.4,
                'rrf_k': 60
            }
        },
        'cache': {
            'enable_embedding_cache': True,
            'backend': 'in_memory',
            'ttl_seconds': 3600
        },
        'ui': {
            'supported_formats': ['txt']
        },
        'logging': {
            'level': 'INFO'
        }
    }


@pytest.fixture
def sample_documents():
    """Create sample policy text documents for testing."""
    documents = []
    
    doc1_content = """[Page 1]
COMPREHENSIVE HEALTH INSURANCE POLICY
Sum Insured: ₹5,00,000 per policy year
Room rent sub-limit: Up to ₹5,000 per day
ICU charges: Up to ₹10,000 per day

[Page 2]
Exclusions:
- Cosmetic or plastic surgery
- Dental treatment (unless due to accident)
- Alcoholism or drug addiction treatment

Waiting Periods:
- Pre-existing diseases: 48 months
- Specific diseases (diabetes, hypertension): 24 months"""
    
    doc2_content = """[Page 1]
CLAIMS PROCESSING GUIDELINES
Reimbursement Claims:
- Claim intimation: Within 30 days of discharge
- Required documents: Original bills and discharge summary
- Settlement time: 15-30 days"""
    
    for i, content in enumerate([doc1_content, doc2_content], 1):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            documents.append(f.name)
    
    yield documents
    
    for doc_path in documents:
        try:
            os.unlink(doc_path)
        except Exception:
            pass


class TestCacheService:
    """Test cases for CacheService."""
    
    def test_in_memory_cache(self, sample_config):
        cache = CacheService(sample_config)
        test_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Initially missing
        assert cache.get_embedding("test_model", "hello world") is None
        
        # Store
        cache.set_embedding("test_model", "hello world", test_vec)
        
        # Retrieve
        cached = cache.get_embedding("test_model", "hello world")
        assert cached is not None
        assert np.allclose(cached, test_vec)
        
        # Stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1


class TestHybridRetriever:
    """Test cases for BM25 and Reciprocal Rank Fusion."""
    
    def test_bm25_and_rrf(self, sample_config):
        retriever = HybridRetriever(sample_config)
        chunks = [
            {'text': 'The room rent limit is ₹5,000 per day.', 'metadata': {'file_name': 'policy.txt', 'page': 1, 'chunk_index': 0}},
            {'text': 'Pre-existing diseases have a 48 months waiting period.', 'metadata': {'file_name': 'policy.txt', 'page': 2, 'chunk_index': 1}},
            {'text': 'Cosmetic surgery is an exclusion.', 'metadata': {'file_name': 'policy.txt', 'page': 2, 'chunk_index': 2}}
        ]
        retriever.build_bm25_index(chunks)
        
        # BM25 search for exact term 'waiting period'
        bm25_results = retriever.search_bm25('waiting period', top_k=2)
        assert len(bm25_results) > 0
        assert 'waiting period' in bm25_results[0]['text'].lower()
        
        # RRF Fusion
        fused = retriever.fuse_results(
            dense_results=[chunks[1], chunks[0]],
            sparse_results=[chunks[1], chunks[2]],
            top_k=2
        )
        assert len(fused) == 2
        assert fused[0]['text'] == chunks[1]['text']


class TestRAGPipeline:
    """Test cases for RAGPipeline."""
    
    @patch('src.rag_pipeline.load_config')
    def test_initialization(self, mock_load_config, sample_config):
        mock_load_config.return_value = sample_config
        pipeline = RAGPipeline()
        
        assert pipeline.top_k == 3
        assert pipeline.score_threshold == 0.3
        assert pipeline.document_processor is not None
        assert pipeline.embedding_service is not None
        assert pipeline.vector_store is not None
        assert pipeline.hybrid_retriever is not None

    @pytest.mark.asyncio
    @patch('src.rag_pipeline.load_config')
    async def test_threshold_guardrail_below_cutoff(self, mock_load_config, sample_config):
        """Tests that queries below the threshold return 'not found' immediately without calling LLM."""
        mock_load_config.return_value = sample_config
        pipeline = RAGPipeline()
        
        # Mock retrieval returning candidates with low score < 0.3
        pipeline.embedding_service.encode_single_text = Mock(return_value=np.array([0.1, 0.2]))
        pipeline.vector_store.similarity_search = Mock(return_value=[
            {'text': 'Irrelevant text', 'score': 0.15, 'metadata': {'file_name': 'other.txt', 'page': 1}}
        ])
        pipeline.hybrid_retriever.search_bm25 = Mock(return_value=[])
        pipeline.embedding_service.rerank_documents = Mock(return_value=[
            {'text': 'Irrelevant text', 'rerank_score': 0.18, 'score': 0.18, 'metadata': {'file_name': 'other.txt', 'page': 1}}
        ])
        pipeline.llm_service.answer_question = AsyncMock()

        answer = await pipeline.answer_question("What is quantum entanglement?", score_threshold=0.3)
        
        assert "could not be found" in answer.lower()
        # Verify LLM was NOT called due to guardrail
        pipeline.llm_service.answer_question.assert_not_called()

    @pytest.mark.asyncio
    @patch('src.rag_pipeline.load_config')
    async def test_answer_with_citations(self, mock_load_config, sample_config):
        """Tests returning source citations with file name and page numbers."""
        mock_load_config.return_value = sample_config
        pipeline = RAGPipeline()

        pipeline.embedding_service.encode_single_text = Mock(return_value=np.array([0.1, 0.2]))
        pipeline.vector_store.similarity_search = Mock(return_value=[
            {'text': 'Room rent limit is ₹5,000 per day', 'score': 0.85, 'metadata': {'file_name': 'sample_policy.txt', 'page': 1}}
        ])
        pipeline.hybrid_retriever.search_bm25 = Mock(return_value=[])
        pipeline.embedding_service.rerank_documents = Mock(return_value=[
            {'text': 'Room rent limit is ₹5,000 per day', 'rerank_score': 0.89, 'score': 0.89, 'metadata': {'file_name': 'sample_policy.txt', 'page': 1}}
        ])
        pipeline.llm_service.answer_question = AsyncMock(return_value="The room rent limit is ₹5,000 per day.")

        result = await pipeline.answer_with_citations("What is the room rent limit?", score_threshold=0.3)
        
        assert result['threshold_applied'] is False
        assert "5,000" in result['answer']
        assert len(result['sources']) == 1
        source = result['sources'][0]
        assert source['file'] == 'sample_policy.txt'
        assert source['page'] == 1
        assert source['score'] == 0.89
        assert '5,000' in source['chunk_text']

    @pytest.mark.asyncio
    @patch('src.rag_pipeline.load_config')
    async def test_stream_answer_events(self, mock_load_config, sample_config):
        """Tests that stream_answer emits sources event first, followed by tokens and done."""
        mock_load_config.return_value = sample_config
        pipeline = RAGPipeline()

        pipeline.embedding_service.encode_single_text = Mock(return_value=np.array([0.1, 0.2]))
        pipeline.vector_store.similarity_search = Mock(return_value=[
            {'text': 'Waiting period is 48 months', 'score': 0.9, 'metadata': {'file_name': 'policy.txt', 'page': 2}}
        ])
        pipeline.hybrid_retriever.search_bm25 = Mock(return_value=[])
        pipeline.embedding_service.rerank_documents = Mock(return_value=[
            {'text': 'Waiting period is 48 months', 'rerank_score': 0.92, 'metadata': {'file_name': 'policy.txt', 'page': 2}}
        ])

        async def mock_stream(q, ctx):
            yield "48 "
            yield "months"

        pipeline.llm_service.stream_answer_question = mock_stream

        events = []
        async for chunk in pipeline.stream_answer("What is the waiting period?"):
            events.append(chunk)

        assert len(events) >= 3
        # First event must be sources
        first_event = json.loads(events[0].replace('data: ', '').strip())
        assert first_event['type'] == 'sources'
        assert len(first_event['sources']) == 1
        assert first_event['sources'][0]['page'] == 2

        # Last event must be done
        last_event = json.loads(events[-1].replace('data: ', '').strip())
        assert last_event['type'] == 'done'

    @pytest.mark.asyncio
    @patch('src.rag_pipeline.load_config')
    async def test_query_claim_decision(self, mock_load_config, sample_config):
        """Tests structured insurance claim evaluation."""
        mock_load_config.return_value = sample_config
        pipeline = RAGPipeline()

        pipeline.llm_service.parse_query = AsyncMock(return_value={'age': 45, 'procedure': 'knee surgery'})
        pipeline.embedding_service.encode_single_text = Mock(return_value=np.array([0.1, 0.2]))
        pipeline.vector_store.similarity_search = Mock(return_value=[
            {'text': 'Knee surgery covered up to ₹50,000.', 'score': 0.88, 'metadata': {'file_name': 'policy.txt', 'page': 1}}
        ])
        pipeline.hybrid_retriever.search_bm25 = Mock(return_value=[])
        pipeline.embedding_service.rerank_documents = Mock(return_value=[
            {'text': 'Knee surgery covered up to ₹50,000.', 'rerank_score': 0.91, 'score': 0.91, 'metadata': {'file_name': 'policy.txt', 'page': 1}}
        ])
        pipeline.llm_service.make_decision = AsyncMock(return_value={
            'decision': 'APPROVED',
            'payout': 50000,
            'justification': 'Knee surgery approved under policy terms.',
            'model_used': 'test-model'
        })

        result = await pipeline.query("45 year old male needs knee surgery")

        assert result['decision'] == 'APPROVED'
        assert result['payout'] == 50000
        assert 'CLM_' in result['json_output']['claim_id']
        assert result['json_output']['currency'] == 'INR'
        assert len(result['relevant_clauses']) == 1


if __name__ == "__main__":
    pytest.main([__file__])

"""
Integration tests for the RAG pipeline.
"""

import pytest
import tempfile
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.rag_pipeline import RAGPipeline


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'document_processing': {
            'chunk_size': 500,
            'chunk_overlap': 100
        },
        'embedding': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'device': 'cpu'
        },
        'vector_store': {
            'persist_directory': './test_chroma_index',
            'collection_name': 'test_documents'
        },
        'retrieval': {
            'top_k': 3,
            'score_threshold': 0.5
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
    """Create sample documents for testing."""
    documents = []
    
    # Document 1: Insurance policy
    doc1_content = """Health Insurance Policy

Coverage: This policy covers medical expenses up to ₹5,00,000 per year.
Age Limit: 18-65 years
Premium: ₹12,000 annually

Covered Services:
- Hospitalization
- Surgery
- Emergency treatment
- Diagnostic tests

Exclusions:
- Cosmetic surgery
- Dental treatment
- Pre-existing conditions (first 2 years)

Claims Process:
Submit claim within 30 days of treatment.
Provide medical reports and bills.
Approval within 7-14 business days."""
    
    # Document 2: Terms and conditions
    doc2_content = """Terms and Conditions

Eligibility:
- Minimum age: 18 years
- Maximum age: 65 years
- No pre-existing conditions for new policies

Premium Payment:
- Annual payment required
- Grace period: 30 days
- Policy lapses after grace period

Claim Settlement:
- Cashless treatment at network hospitals
- Reimbursement for non-network hospitals
- Maximum claim amount: ₹5,00,000 per year

Geographic Coverage:
- Valid across India
- Emergency coverage worldwide (up to ₹1,00,000)"""
    
    # Create temporary files
    for i, content in enumerate([doc1_content, doc2_content], 1):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            documents.append(f.name)
    
    yield documents
    
    # Cleanup
    for doc_path in documents:
        try:
            os.unlink(doc_path)
        except:
            pass


class TestRAGPipeline:
    """Test cases for RAGPipeline."""
    
    @patch('src.rag_pipeline.load_config')
    def test_initialization(self, mock_load_config, sample_config):
        """Test RAGPipeline initialization."""
        mock_load_config.return_value = sample_config
        
        pipeline = RAGPipeline()
        
        assert pipeline.config == sample_config
        assert pipeline.top_k == 3
        assert pipeline.score_threshold == 0.5
        assert pipeline.document_processor is not None
        assert pipeline.embedding_service is not None
        assert pipeline.vector_store is not None
        assert pipeline.llm_service is not None
    
    @patch('src.rag_pipeline.load_config')
    @patch('src.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    @patch('src.vector_store.CHROMADB_AVAILABLE', False)
    def test_initialization_without_dependencies(self, mock_load_config, sample_config):
        """Test initialization when optional dependencies are not available."""
        mock_load_config.return_value = sample_config
        
        # Should still initialize without errors
        pipeline = RAGPipeline()
        assert pipeline is not None
    
    @patch('src.rag_pipeline.load_config')
    def test_index_documents_empty_list(self, mock_load_config, sample_config):
        """Test indexing with empty document list."""
        mock_load_config.return_value = sample_config
        
        pipeline = RAGPipeline()
        results = pipeline.index_documents([])
        
        assert results['total_files'] == 0
        assert results['processed_files'] == 0
        assert results['total_chunks'] == 0
    
    @patch('src.rag_pipeline.load_config')
    def test_index_documents_nonexistent_files(self, mock_load_config, sample_config):
        """Test indexing with non-existent files."""
        mock_load_config.return_value = sample_config
        
        pipeline = RAGPipeline()
        results = pipeline.index_documents(['nonexistent1.txt', 'nonexistent2.txt'])
        
        assert results['total_files'] == 2
        assert results['failed_files'] == 2
        assert results['processed_files'] == 0
        assert len(results['errors']) == 2
    
    @patch('src.rag_pipeline.load_config')
    @pytest.mark.asyncio
    async def test_query_basic(self, mock_load_config, sample_config):
        """Test basic query processing."""
        mock_load_config.return_value = sample_config

        pipeline = RAGPipeline()

        # Mock the components to avoid actual model loading
        pipeline.llm_service.parse_query = AsyncMock(return_value={
            'age': 30,
            'gender': 'male',
            'procedure': 'surgery'
        })

        pipeline.embedding_service.encode_single_text = Mock(return_value=[0.1, 0.2, 0.3])

        pipeline.vector_store.similarity_search = Mock(return_value=[
            {
                'text': 'Sample policy clause',
                'score': 0.8,
                'metadata': {'file_name': 'policy.txt'}
            }
        ])

        pipeline.embedding_service.rerank_documents = Mock(return_value=[
            {
                'text': 'Sample policy clause',
                'score': 0.8,
                'metadata': {'file_name': 'policy.txt'}
            }
        ])

        pipeline.llm_service.make_decision = AsyncMock(return_value={
            'decision': 'APPROVED',
            'payout': 50000,
            'justification': 'Claim approved based on policy terms.'
        })

        result = await pipeline.query("30-year-old male needs surgery")

        assert result['decision'] == 'APPROVED'
        assert result['payout'] == 50000
        assert 'justification' in result
        assert result['metadata']['processing_successful'] is True
    
    @patch('src.rag_pipeline.load_config')
    @pytest.mark.asyncio
    async def test_query_with_error(self, mock_load_config, sample_config):
        """Test query processing with error."""
        mock_load_config.return_value = sample_config

        pipeline = RAGPipeline()

        # Mock an error in LLM service
        pipeline.llm_service.parse_query = AsyncMock(side_effect=Exception("LLM error"))

        result = await pipeline.query("Test query")

        assert result['decision'] == 'ERROR'
        assert result['payout'] == 0
        assert 'An internal error occurred' in result['justification']
        assert result['metadata']['processing_successful'] is False
    
    @patch('src.rag_pipeline.load_config')
    def test_get_system_status(self, mock_load_config, sample_config):
        """Test getting system status."""
        mock_load_config.return_value = sample_config
        
        pipeline = RAGPipeline()
        
        # Mock component methods
        pipeline.vector_store.get_collection_stats = Mock(return_value={
            'document_count': 10,
            'collection_name': 'test_documents'
        })
        
        pipeline.embedding_service.get_model_info = Mock(return_value={
            'model_loaded': True,
            'embedding_dimension': 384
        })
        
        status = pipeline.get_system_status()
        
        assert status['pipeline_status'] == 'healthy'
        assert 'vector_store' in status
        assert 'embedding_service' in status
        assert 'configuration' in status
    
    @patch('src.rag_pipeline.load_config')
    def test_reset_index(self, mock_load_config, sample_config):
        """Test resetting the index."""
        mock_load_config.return_value = sample_config
        
        pipeline = RAGPipeline()
        
        # Mock vector store reset
        pipeline.vector_store.reset_collection = Mock()
        
        result = pipeline.reset_index()
        
        assert result['status'] == 'success'
        assert 'reset successfully' in result['message']
        pipeline.vector_store.reset_collection.assert_called_once()
    
    @patch('src.rag_pipeline.load_config')
    @pytest.mark.asyncio
    async def test_batch_query(self, mock_load_config, sample_config):
        """Test batch query processing."""
        mock_load_config.return_value = sample_config

        pipeline = RAGPipeline()

        # Mock the query method
        pipeline.query = AsyncMock(side_effect=[
            {'decision': 'APPROVED', 'payout': 30000},
            {'decision': 'REJECTED', 'payout': 0}
        ])

        queries = ["Query 1", "Query 2"]
        results = await pipeline.batch_query(queries)

        assert len(results) == 2
        assert results[0]['decision'] == 'APPROVED'
        assert results[1]['decision'] == 'REJECTED'
        assert pipeline.query.call_count == 2
    
    @patch('src.rag_pipeline.load_config')
    def test_generate_json_output(self, mock_load_config, sample_config):
        """Test JSON output generation."""
        mock_load_config.return_value = sample_config
        
        pipeline = RAGPipeline()
        
        result = {
            'query': 'Test query',
            'parsed_query': {'age': 30},
            'decision': 'APPROVED',
            'payout': 50000,
            'justification': 'Test justification',
            'retrieved_documents': 2,
            'relevant_clauses': [
                {
                    'text': 'Clause 1 text',
                    'score': 0.9,
                    'metadata': {'file_name': 'policy.txt'}
                }
            ]
        }
        
        json_output = pipeline._generate_json_output(result)
        
        assert 'claim_id' in json_output
        assert json_output['decision'] == 'APPROVED'
        assert json_output['payout_amount'] == 50000
        assert json_output['currency'] == 'INR'
        assert len(json_output['evidence']) == 1
        assert 'processing_metadata' in json_output


if __name__ == "__main__":
    pytest.main([__file__])

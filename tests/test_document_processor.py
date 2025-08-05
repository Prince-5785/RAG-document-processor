"""
Tests for the document processor module.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.utils import load_config


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'document_processing': {
            'chunk_size': 500,
            'chunk_overlap': 100,
            'separators': ["\n\n", "\n", " ", ""]
        },
        'ui': {
            'supported_formats': ['pdf', 'docx', 'eml', 'txt']
        }
    }


@pytest.fixture
def document_processor(config):
    """Document processor instance."""
    return DocumentProcessor(config)


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    content = """This is a sample insurance policy document.

Section 1: Coverage Details
This policy covers medical expenses up to ₹5,00,000 per year.
Age limit: 18-65 years.
Pre-existing conditions are covered after 2 years.

Section 2: Exclusions
The following are not covered:
- Cosmetic surgery
- Dental treatment (unless due to accident)
- Mental health conditions

Section 3: Claims Process
To file a claim:
1. Submit claim form within 30 days
2. Provide medical reports
3. Wait for approval (7-14 days)

Contact: support@insurance.com
Phone: +91-9876543210"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_initialization(self, config):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(config)
        assert processor.config == config
        assert processor.supported_formats == ['pdf', 'docx', 'eml', 'txt']
    
    def test_get_supported_formats(self, document_processor):
        """Test getting supported formats."""
        formats = document_processor.get_supported_formats()
        assert isinstance(formats, list)
        assert 'txt' in formats
        assert 'pdf' in formats
    
    def test_process_text_file(self, document_processor, sample_text_file):
        """Test processing a text file."""
        result = document_processor.process_file(sample_text_file)
        
        assert result['file_path'] == sample_text_file
        assert result['file_name'] == Path(sample_text_file).name
        assert result['file_type'] == 'txt'
        assert result['metadata']['processed_successfully'] is True
        assert len(result['content']) > 0
        assert 'insurance policy' in result['content'].lower()
    
    def test_process_nonexistent_file(self, document_processor):
        """Test processing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            document_processor.process_file('nonexistent.txt')
    
    def test_process_unsupported_format(self, document_processor):
        """Test processing an unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                document_processor.process_file(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def test_chunk_text(self, document_processor):
        """Test text chunking functionality."""
        text = "This is a long text. " * 100  # Create long text
        
        chunks = document_processor.chunk_text(text)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all(len(chunk['text']) <= 500 for chunk in chunks)  # Respect chunk size
    
    def test_chunk_text_with_metadata(self, document_processor):
        """Test text chunking with metadata."""
        text = "Sample text for chunking."
        metadata = {'file_name': 'test.txt', 'file_type': 'txt'}
        
        chunks = document_processor.chunk_text(text, metadata)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk['metadata']['file_name'] == 'test.txt'
            assert chunk['metadata']['file_type'] == 'txt'
            assert 'chunk_index' in chunk['metadata']
    
    def test_chunk_empty_text(self, document_processor):
        """Test chunking empty text."""
        chunks = document_processor.chunk_text("")
        assert chunks == []
        
        chunks = document_processor.chunk_text("   ")
        assert chunks == []
    
    def test_basic_text_chunking(self, document_processor):
        """Test basic text chunking fallback."""
        text = "This is a test sentence. " * 50
        
        chunks = document_processor._basic_text_chunking(text, 100, 20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + some overlap
    
    def test_process_multiple_files(self, document_processor, sample_text_file):
        """Test processing multiple files."""
        # Create another temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Another test document.")
            temp_path2 = f.name
        
        try:
            results = document_processor.process_multiple_files([sample_text_file, temp_path2])
            
            assert len(results) == 2
            assert all(result['metadata']['processed_successfully'] for result in results)
            assert all(len(result['content']) > 0 for result in results)
        finally:
            try:
                os.unlink(temp_path2)
            except:
                pass
    
    def test_process_multiple_files_with_errors(self, document_processor, sample_text_file):
        """Test processing multiple files with some errors."""
        nonexistent_file = 'nonexistent.txt'
        
        results = document_processor.process_multiple_files([sample_text_file, nonexistent_file])
        
        assert len(results) == 2
        assert results[0]['metadata']['processed_successfully'] is True
        assert results[1]['metadata']['processed_successfully'] is False
        assert 'error' in results[1]['metadata']


if __name__ == "__main__":
    pytest.main([__file__])

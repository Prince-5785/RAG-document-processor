"""
Document processing module for handling various file formats including PDFs, Word documents, and emails.
Extracts per-page text and assigns page-level metadata to chunks for precise source citations.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF processing libraries not available. Install PyPDF2 and pdfplumber.")

# Word document processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("Word document processing not available. Install python-docx.")

from .utils import clean_text, get_file_hash, validate_file_type

# Text chunking
try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Text chunking will use basic splitting.")


class DocumentProcessor:
    """Handles document loading, text extraction, page tracking, and chunking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ui_config = config.get('ui', {})
        dp_config = config.get('document_processing', {})
        self.supported_formats = dp_config.get('allowed_extensions') or ui_config.get('supported_formats', ['pdf', 'docx', 'eml', 'txt'])
        self.max_file_size_mb = dp_config.get('max_file_size_mb', 25)
        self.logger = logging.getLogger(__name__)

    def validate_file(self, file_path: str) -> None:
        """Validates that a file exists, is within the size limit, and has an allowed format."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed limit of {self.max_file_size_mb} MB")
            
        if not validate_file_type(file_path, self.supported_formats):
            raise ValueError(f"Unsupported file format: {Path(file_path).suffix}. Allowed: {self.supported_formats}")
        
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file and extract text content with page-level tracking and metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing extracted text, pages, metadata, and processing info
        """
        self.validate_file(file_path)
            
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        file_hash = get_file_hash(file_path)
        
        self.logger.info(f"Processing file: {file_path} ({file_extension})")
        
        try:
            pages: List[Dict[str, Any]] = []
            if file_extension == 'pdf':
                content, pages = self._process_pdf(file_path)
            elif file_extension == 'docx':
                content = self._process_docx(file_path)
                pages = [{'page_number': 1, 'text': content}]
            elif file_extension == 'eml':
                content = self._process_email(file_path)
                pages = [{'page_number': 1, 'text': content}]
            elif file_extension == 'txt':
                content = self._process_text(file_path)
                pages = self._detect_text_pages(content)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_hash': file_hash,
                'file_type': file_extension,
                'content': content,
                'pages': pages,
                'metadata': {
                    'file_size': os.path.getsize(file_path),
                    'total_pages': len(pages),
                    'processed_successfully': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'file_hash': file_hash,
                'file_type': file_extension,
                'content': "",
                'pages': [],
                'metadata': {
                    'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    'total_pages': 0,
                    'processed_successfully': False,
                    'error': str(e)
                }
            }
    
    def _process_pdf(self, file_path: str) -> tuple[str, List[Dict[str, Any]]]:
        """Extract text from PDF file along with per-page text objects."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries (pdfplumber/PyPDF2) not available")
            
        pages = []
        full_text_parts = []
        
        try:
            # Try pdfplumber first (high fidelity layout and table parsing)
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    raw_text = page.extract_text() or ""
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        pages.append({'page_number': page_num, 'text': cleaned})
                        full_text_parts.append(f"[Page {page_num}]\n{cleaned}")
                        
        except Exception as e:
            self.logger.warning(f"pdfplumber failed for {file_path}, falling back to PyPDF2: {e}")
            pages.clear()
            full_text_parts.clear()
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        raw_text = page.extract_text() or ""
                        cleaned = clean_text(raw_text)
                        if cleaned:
                            pages.append({'page_number': page_num, 'text': cleaned})
                            full_text_parts.append(f"[Page {page_num}]\n{cleaned}")
            except Exception as e2:
                self.logger.error(f"Both PDF processors failed for {file_path}: {e2}")
                raise e2
        
        full_text = '\n\n'.join(full_text_parts)
        return full_text, pages
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        if not DOCX_AVAILABLE:
            raise ImportError("Word document processing not available")
            
        doc = Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_content.append(' | '.join(row_text))
        
        return clean_text('\n\n'.join(text_content))
    
    def _process_email(self, file_path: str) -> str:
        """Extract text from email file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            msg = email.message_from_file(file)
        
        text_content = []
        subject = msg.get('Subject', '')
        sender = msg.get('From', '')
        recipient = msg.get('To', '')
        date = msg.get('Date', '')
        
        if subject: text_content.append(f"Subject: {subject}")
        if sender: text_content.append(f"From: {sender}")
        if recipient: text_content.append(f"To: {recipient}")
        if date: text_content.append(f"Date: {date}")
        text_content.append("")
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True)
                    if body:
                        text_content.append(body.decode('utf-8', errors='ignore'))
        else:
            body = msg.get_payload(decode=True)
            if body:
                text_content.append(body.decode('utf-8', errors='ignore'))
        
        return clean_text('\n\n'.join(text_content))
    
    def _process_text(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        return clean_text(content)

    def _detect_text_pages(self, content: str) -> List[Dict[str, Any]]:
        """Parses [Page X] markers if present in text, otherwise treats as page 1."""
        page_splits = re.split(r'\[Page\s+(\d+)\]', content)
        if len(page_splits) > 1:
            pages = []
            # format: [preamble, page_num_1, text_1, page_num_2, text_2, ...]
            for i in range(1, len(page_splits), 2):
                p_num = int(page_splits[i])
                p_text = clean_text(page_splits[i+1])
                if p_text:
                    pages.append({'page_number': p_num, 'text': p_text})
            if pages:
                return pages
        return [{'page_number': 1, 'text': content}]
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files and return their extracted content with page tracking."""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'file_hash': get_file_hash(file_path) if os.path.exists(file_path) else "",
                    'file_type': Path(file_path).suffix.lower().lstrip('.'),
                    'content': "",
                    'pages': [],
                    'metadata': {
                        'processed_successfully': False,
                        'error': str(e)
                    }
                })
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        return self.supported_formats.copy()

    def chunk_document(self, doc_data: Dict[str, Any], extra_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks an entire processed document, preserving page numbers for every chunk.
        """
        pages = doc_data.get('pages', [])
        base_meta = {
            'file_name': doc_data.get('file_name', ''),
            'file_path': doc_data.get('file_path', ''),
            'file_hash': doc_data.get('file_hash', ''),
            'file_type': doc_data.get('file_type', ''),
        }
        if extra_metadata:
            base_meta.update(extra_metadata)

        all_chunks = []
        if pages:
            for page_info in pages:
                p_num = page_info.get('page_number', 1)
                p_text = page_info.get('text', '')
                if p_text.strip():
                    page_meta = dict(base_meta)
                    page_meta['page'] = p_num
                    chunks = self.chunk_text(p_text, metadata=page_meta)
                    all_chunks.extend(chunks)
        else:
            content = doc_data.get('content', '')
            if content.strip():
                base_meta['page'] = 1
                all_chunks = self.chunk_text(content, metadata=base_meta)

        # Re-index total chunks across the whole document
        for idx, chunk in enumerate(all_chunks):
            chunk['metadata']['chunk_index'] = idx
            chunk['metadata']['total_chunks'] = len(all_chunks)

        return all_chunks

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks for embedding and retrieval.
        """
        if not text or not text.strip():
            return []

        doc_config = self.config.get('document_processing', {})
        chunk_size = doc_config.get('chunk_size', 800)
        chunk_overlap = doc_config.get('chunk_overlap', 200)
        separators = doc_config.get('separators', ["\n\n", "\n", ". ", " ", ""])

        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
            )
            chunks = text_splitter.split_text(text)
        else:
            chunks = self._basic_text_chunking(text, chunk_size, chunk_overlap)

        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks),
                'page': (metadata.get('page') if metadata else 1) or 1
            }

            if metadata:
                chunk_metadata.update(metadata)

            chunk_objects.append({
                'text': chunk,
                'metadata': chunk_metadata
            })

        return chunk_objects

    def _basic_text_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Basic text chunking implementation when LangChain is not available."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            chunk = text[start:end]
            last_period = chunk.rfind('. ')
            last_newline = chunk.rfind('\n')

            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                end = start + break_point + 1

            chunks.append(text[start:end].strip())
            start = end - chunk_overlap

        return [c for c in chunks if c]

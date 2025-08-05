# LLM-Powered Document Processing System

A Retrieval-Augmented Generation (RAG) system for processing natural-language insurance queries against large, unstructured documents (PDFs, Word, emails). This fully local, open-source solution provides transparent, explainable AI for insurance claims processing.

## 🚀 Features

- **Semantic Document Search**: Uses Hugging Face's all-MiniLM-L6-v2 for embeddings
- **Local LLM Reasoning**: Meta Llama 2 7B Chat or Mistral 7B Instruct
- **Vector Storage**: ChromaDB with HNSW indexing
- **Web Interface**: Streamlit-based UI for easy interaction
- **Multi-format Support**: PDFs, Word documents, and emails
- **Transparent Decisions**: Clear justifications with clause citations
- **Fully Local**: No API costs or vendor lock-in

## 🏗️ Architecture

```
User Query → Query Parsing → Embedding → Vector Search → LLM Reasoning → Decision + Justification
                ↓
            ChromaDB Vector Store ← Document Indexing ← Document Processing
```

## 📋 Requirements

- Python 3.8+
- 8GB+ RAM (for local LLM)
- GPU recommended (for faster inference)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd LLM-Powered-Document-Processing-System
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Start the Streamlit app**:

```bash
streamlit run app.py
```

2. **Upload documents**: Use the file uploader to add your insurance policy documents

3. **Query the system**: Enter natural language queries like:

   - "46-year-old male, knee surgery in Pune, 3-month policy"
   - "Female, 35 years, dental treatment in Mumbai, annual policy"

4. **Get results**: Receive approval/rejection decisions with clear justifications

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # Document ingestion and processing
│   ├── embedding_service.py     # Embedding generation
│   ├── vector_store.py          # ChromaDB integration
│   ├── llm_service.py           # LLM integration and prompting
│   ├── rag_pipeline.py          # Main RAG pipeline
│   └── utils.py                 # Utility functions
├── config/
│   └── config.yaml             # Configuration settings
├── data/
│   ├── documents/              # Sample documents
│   └── chroma_index/           # ChromaDB persistence
├── tests/
│   ├── __init__.py
│   ├── test_document_processor.py
│   ├── test_embedding_service.py
│   ├── test_vector_store.py
│   └── test_rag_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Model settings
- Chunk sizes
- Retrieval parameters
- UI preferences

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests with coverage
python run_tests.py

# Run quick tests only (no model loading)
python run_tests.py quick

# Check dependencies
python run_tests.py deps
```

Or use pytest directly:

```bash
pytest tests/ -v --cov=src
```

## 📊 Performance

- **Embedding Model**: 384-dimensional vectors
- **Chunk Size**: ~1,000 characters with 200-character overlap
- **Retrieval**: Top-5 similar chunks
- **Response Time**: ~2-5 seconds per query

## 🚀 Deployment

For production deployment options:

```bash
# Docker deployment
docker-compose up -d

# Cloud deployment
# See DEPLOYMENT.md for detailed instructions
```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the embedding models
- Meta for Llama 2
- ChromaDB team for the vector database
- Streamlit for the web framework

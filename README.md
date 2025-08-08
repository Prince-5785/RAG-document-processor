# LLM-Powered Document Processing System

A Retrieval-Augmented Generation (RAG) system for processing natural-language insurance queries against large, unstructured documents. This solution provides transparent, explainable AI for insurance claims processing.

## 🚀 Features

-   **Multi-Modal Reasoning**: Processes complex queries against PDF policy documents.
-   **Advanced RAG Architecture**: Implements a retrieve-and-rerank strategy for high-accuracy context retrieval.
-   **High-Performance API**: Built with FastAPI and `asyncio` to handle concurrent requests for fast, scalable processing.
-   **Resilient LLM Service**: Features automatic key rotation and exponential backoff to robustly handle API rate limits and errors.
-   **Interactive UI**: A Streamlit web app for demonstrations and direct interaction.
-   **Multi-format Support**: Natively handles PDFs, Word documents, and emails.
-   **Secure Exposure**: Instructions for secure, public deployment using Cloudflare Tunnel.

## 🏗️ Architecture
```

User Query → [FastAPI Server] → Embedding → Vector Search → Re-ranking → LLM Reasoning → Answer
↓
ChromaDB Vector Store ← Document Indexing ← Document Processing

````

## 📋 Requirements

-   Python 3.9+
-   8GB+ RAM
-   A `.env` file in the project root with your `GROQ_API_KEYS`.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd your-project-directory
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` file:**
    Create a `.env` file in the project root and add your Groq API keys:
    ```env
    GROQ_API_KEYS="gsk_key1...,gsk_key2...,gsk_key3..."
    ```

## 🚀 Quick Start

You can run the interactive Streamlit UI or the backend API server.

### 1. Run the Streamlit Web App (for demos)

```bash
streamlit run app.py
````

  - **Upload documents** in the UI to build the knowledge base.
  - **Ask questions** and receive detailed decisions with justifications.

### 2\. Run the Backend API Server (for competition)

This exposes a high-performance JSON API for programmatic access.

#### **Step A: Start the API Server**

Run the Uvicorn server, which will host your FastAPI application.

```bash
uvicorn main_api:app --host 127.0.0.1 --port 8000 --reload
```

Your API is now running locally at `http://127.0.0.1:8000`.

#### **Step B: Expose the API with Cloudflare Tunnel**

To make your local server publicly accessible with a secure HTTPS URL, use `cloudflared`.

1.  [Download and install `cloudflared`](https://www.google.com/search?q=https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/install-and-setup/installation/).
2.  In a **new terminal window**, run the following command:
    ```bash
    cloudflared tunnel --url http://localhost:8000
    ```
3.  Cloudflare will generate a random public URL (e.g., `https://random-words.trycloudflare.com`). You can now use this URL to send requests to your local API from anywhere.

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── main_api.py            # Main FastAPI application
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── embedding_service.py
│   ├── vector_store.py
│   ├── llm_service.py
│   ├── rag_pipeline.py
│   └── utils.py
├── config/
│   └── config.yaml
├── .env                   # For storing API keys
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

  - Embedding and re-ranker model names
  - Chunk sizes and overlap
  - Retrieval parameters (`candidate_k`, `top_k`)
  - LLM API retry settings

## 🧪 Testing

Run the comprehensive test suite:

```bash
pytest tests/ -v --cov=src
```

## 🚀 Deployment

For production deployment options, see the detailed [DEPLOYMENT.md](https://www.google.com/search?q=DEPLOYMENT.md) guide, which includes instructions for Docker and major cloud providers.

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Make your changes and add tests
4.  Commit your changes (`git commit -m 'Add some amazing feature'`)
5.  Push to the branch (`git push origin feature/amazing-feature`)
6.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```
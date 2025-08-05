"""
Streamlit web application for the LLM-Powered Document Processing System.
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import tempfile
from pathlib import Path

# Import our RAG pipeline
from src.rag_pipeline import RAGPipeline
from src.utils import format_file_size

# Page configuration
st.set_page_config(
    page_title="CodersHub Insurance Query Solver",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .decision-approved {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .decision-rejected {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .payout-amount {
        color: #007bff;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = False
if 'indexing_results' not in st.session_state:
    st.session_state.indexing_results = None

def initialize_pipeline():
    """Initialize the RAG pipeline."""
    try:
        if st.session_state.pipeline is None:
            with st.spinner("Initializing RAG pipeline..."):
                st.session_state.pipeline = RAGPipeline()
        return True
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return False

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files to temporary directory and return file paths."""
    file_paths = []
    
    if not uploaded_files:
        return file_paths
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        # Save file to temporary location
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths

def display_indexing_results(results: Dict[str, Any]):
    """Display document indexing results."""
    if results['processed_files'] > 0:
        st.markdown(f"""
        <div class="success-box">
            ✅ Successfully processed {results['processed_files']} out of {results['total_files']} files<br>
            📄 Generated {results['total_chunks']} text chunks for indexing
        </div>
        """, unsafe_allow_html=True)
    
    if results['failed_files'] > 0:
        st.markdown(f"""
        <div class="error-box">
            ❌ Failed to process {results['failed_files']} files
        </div>
        """, unsafe_allow_html=True)
        
        if results['errors']:
            with st.expander("View Errors"):
                for error in results['errors']:
                    st.error(error)

def display_query_result(result: Dict[str, Any]):
    """Display query processing results."""
    # Main decision
    decision = result.get('decision', 'ERROR')
    payout = result.get('payout', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if decision == 'APPROVED':
            st.markdown(f'<p class="decision-approved">✅ APPROVED</p>', unsafe_allow_html=True)
        elif decision == 'REJECTED':
            st.markdown(f'<p class="decision-rejected">❌ REJECTED</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="decision-rejected">⚠️ ERROR</p>', unsafe_allow_html=True)
    
    with col2:
        if payout > 0:
            st.markdown(f'<p class="payout-amount">💰 Payout: ₹{payout:,}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="payout-amount">💰 Payout: ₹0</p>', unsafe_allow_html=True)
    
    # Justification
    st.subheader("📋 Justification")
    justification = result.get('justification', 'No justification provided')
    st.write(justification)
    
    # Retrieved documents info
    retrieved_count = result.get('retrieved_documents', 0)
    if retrieved_count > 0:
        st.markdown(f"""
        <div class="info-box">
            📚 Found {retrieved_count} relevant policy clauses
        </div>
        """, unsafe_allow_html=True)
        
        # Show relevant clauses
        with st.expander("View Relevant Policy Clauses"):
            for i, clause in enumerate(result.get('relevant_clauses', []), 1):
                st.markdown(f"**Clause {i}** (Relevance: {clause['score']:.2f})")
                st.text(clause['text'])
                st.markdown("---")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 CodersHub Insurance Query Solver</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI-powered insurance claims processing system. Upload your policy documents 
    and ask natural language questions to get instant approval decisions with clear justifications.
    """)
    
    # Initialize pipeline
    if not initialize_pipeline():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        if st.button("Check System Status"):
            with st.spinner("Checking system status..."):
                status = st.session_state.pipeline.get_system_status()
                
                if status.get('pipeline_status') == 'healthy':
                    st.success("✅ System is healthy")
                    
                    # Show vector store stats
                    vector_stats = status.get('vector_store', {})
                    if vector_stats:
                        st.metric("Documents Indexed", vector_stats.get('document_count', 0))
                else:
                    st.error("❌ System error")
                    st.error(status.get('error', 'Unknown error'))
        
        st.markdown("---")
        
        # Reset index option
        st.header("🔄 Reset Index")
        if st.button("Reset Document Index", type="secondary"):
            if st.session_state.pipeline:
                with st.spinner("Resetting index..."):
                    reset_result = st.session_state.pipeline.reset_index()
                    if reset_result['status'] == 'success':
                        st.success("Index reset successfully")
                        st.session_state.documents_indexed = False
                        st.session_state.indexing_results = None
                    else:
                        st.error(f"Reset failed: {reset_result['message']}")
    
    # Main content
    tab1, tab2 = st.tabs(["📄 Document Upload & Indexing", "❓ Query Processing"])
    
    with tab1:
        st.header("📄 Document Upload & Indexing")
        
        st.markdown("""
        Upload your insurance policy documents (PDF, Word, Email, Text files) to build the knowledge base.
        The system will process and index these documents for semantic search.
        """)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'eml', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word (.docx), Email (.eml), Text (.txt)"
        )
        
        if uploaded_files:
            st.write(f"📁 Selected {len(uploaded_files)} files:")
            for file in uploaded_files:
                file_size = len(file.getbuffer())
                st.write(f"- {file.name} ({format_file_size(file_size)})")
        
        # Index documents button
        if st.button("🔍 Index Documents", type="primary", disabled=not uploaded_files):
            if uploaded_files:
                with st.spinner("Processing and indexing documents..."):
                    # Save uploaded files
                    file_paths = save_uploaded_files(uploaded_files)
                    
                    # Index documents
                    results = st.session_state.pipeline.index_documents(file_paths)
                    st.session_state.indexing_results = results
                    
                    if results['processed_files'] > 0:
                        st.session_state.documents_indexed = True
                    
                    # Clean up temporary files
                    for file_path in file_paths:
                        try:
                            os.unlink(file_path)
                        except:
                            pass
        
        # Display indexing results
        if st.session_state.indexing_results:
            st.subheader("📊 Indexing Results")
            display_indexing_results(st.session_state.indexing_results)
    
    with tab2:
        st.header("❓ Query Processing")
        
        if not st.session_state.documents_indexed:
            st.markdown("""
            <div class="info-box">
                ℹ️ Please upload and index documents first before querying the system.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        Ask natural language questions about insurance claims. The system will analyze your query 
        against the indexed policy documents and provide approval decisions with justifications.
        """)
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - "46-year-old male, knee surgery in Pune, 3-month policy"
            - "Female, 35 years, dental treatment in Mumbai, annual policy"
            - "60-year-old patient, heart surgery, premium policy holder"
            - "Emergency appendix operation for 25-year-old in Delhi"
            """)
        
        # Query input
        user_query = st.text_area(
            "Enter your insurance query:",
            placeholder="e.g., 46-year-old male, knee surgery in Pune, 3-month policy",
            height=100
        )
        
        # Query button
        if st.button("🔍 Process Query", type="primary", disabled=not user_query.strip()):
            if user_query.strip():
                with st.spinner("Processing your query..."):
                    start_time = time.time()
                    result = st.session_state.pipeline.query(user_query.strip())
                    processing_time = time.time() - start_time
                    
                    st.subheader("📋 Query Results")
                    display_query_result(result)
                    
                    # Processing info
                    st.markdown(f"""
                    <div class="info-box">
                        ⏱️ Processing time: {processing_time:.2f} seconds
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # JSON output for audit
                    with st.expander("📄 JSON Output (for audit)"):
                        json_output = result.get('json_output', {})
                        if json_output:
                            # Add timestamp
                            json_output['processing_metadata']['timestamp'] = datetime.now().isoformat()
                            st.json(json_output)
                        else:
                            st.write("No JSON output available")

if __name__ == "__main__":
    main()

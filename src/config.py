# config.py

import os
from dotenv import load_dotenv

# This line loads the variables from your .env file
load_dotenv()

# --- Application Configuration ---

APP_CONFIG = {
    'llm': {
        'api': {
            # This safely gets the API key from the .env file.
            # If it's missing, the value will be None, and the API client won't initialize.
            'groq_api_key': os.environ.get('GROQ_API_KEY'),
            
            # This is the critical list that was likely missing or empty.
            # Ensure this structure ('llm' -> 'api' -> 'model_sequence') exists.
            'model_sequence': [
                'meta-llama/llama-4-maverick-17b-128e-instruct',
                'llama-3.3-70b-versatile',
                'gemma2-9b-it',
                'llama-3.1-8b-instant'
            ]
        },
        'local_model': {
            # Set to None if you don't want a local fallback.
            # The log "No local model configured" is normal if this is None or empty.
            'model_name': None, 
            'device': 'cpu'
        },
        'generation_params': {
            'temperature': 0.1,
            'top_p': 0.9,
            'max_new_tokens_parsing': 250,
            'max_new_tokens_decision': 500
        }
    },
    'logging': {
        'level': os.environ.get('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    'vector_store': {
        'persist_directory': './data/chroma_index',
        'collection_name': 'insurance_documents',
        'distance_metric': 'cosine' # 'l2' (Euclidean) is also an option
    },
    'document_processing': {
        'chunk_size': 1000,
        'chunk_overlap': 200
    }
}
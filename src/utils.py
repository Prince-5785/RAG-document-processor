"""
Utility functions for the LLM-Powered Document Processing System
"""

import os
import json
import yaml
import logging
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        return {}

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )

def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return ""

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")

def load_json(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"JSON file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from {file_path}: {e}")
        return {}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_file_type(file_path: str, allowed_extensions: List[str]) -> bool:
    """Validate if file type is allowed."""
    file_extension = Path(file_path).suffix.lower().lstrip('.')
    return file_extension in [ext.lower() for ext in allowed_extensions]

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    import re
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts the first valid JSON object from a string.
    It can handle JSON embedded within markdown code blocks or plain text.
    """
    if not isinstance(text, str):
        return None

    # Pattern to find JSON within ```json ... ``` markdown block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback to finding the first occurrence of a curly brace
        first_brace = text.find('{')
        if first_brace == -1:
            logging.debug("No JSON object found in text.")
            return None
        json_str = text[first_brace:]

    # Attempt to parse the found string into JSON
    try:
        # Clean up potential LLM artifacts like trailing commas before closing bracket/brace
        cleaned_json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        # If cleaning fails, try to find a valid JSON object by balancing braces
        open_braces = 0
        end_index = -1
        for i, char in enumerate(json_str):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            if open_braces == 0 and i > 0:
                end_index = i + 1
                break

        if end_index != -1:
            potential_json = json_str[:end_index]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON substring: {e}")
                logging.debug(f"Invalid JSON string: {potential_json}")
                return None

    logging.error("Could not extract a valid JSON object from the provided text.")
    return None

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        logging.info(f"{self.description} completed in {elapsed:.2f} seconds")

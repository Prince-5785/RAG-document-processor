#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Document Processing System.
"""

import sys
import subprocess
import os
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'sentence_transformers',
        'chromadb',
        'langchain'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    directories = [
        'data/documents',
        'data/chroma_index', 
        'data/cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_dependencies():
    """Install missing dependencies."""
    print("\n🔧 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_application():
    """Start the Streamlit application."""
    print("\n🚀 Starting the application...")
    print("   The application will open in your browser")
    print("   Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

def main():
    """Main startup function."""
    print("🏥 LLM-Powered Document Processing System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        response = input("\n❓ Install missing dependencies? (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                sys.exit(1)
            print("\n✅ Please restart the application")
            sys.exit(0)
        else:
            print("❌ Cannot start without required dependencies")
            sys.exit(1)
    
    print("\n✅ All dependencies available!")
    
    # Check if sample documents exist
    sample_docs = Path('data/documents')
    if not any(sample_docs.iterdir()):
        print("\n📄 Sample documents not found in data/documents/")
        print("   You can upload documents through the web interface")
    else:
        doc_count = len(list(sample_docs.glob('*.txt')))
        print(f"\n📄 Found {doc_count} sample documents")
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()

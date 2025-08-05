#!/usr/bin/env python3
"""
Test runner for the LLM-Powered Document Processing System.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all tests and generate coverage report."""
    
    print("🧪 Running LLM-Powered Document Processing System Tests")
    print("=" * 60)
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if pytest is available
    try:
        import pytest
        print("✅ pytest found")
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        import pytest
    
    # Run tests with coverage
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--cov=src",  # Coverage for src directory
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html:htmlcov",  # Generate HTML coverage report
        "tests/"  # Test directory
    ]
    
    print("\n🚀 Starting test execution...")
    print("-" * 40)
    
    # Run pytest
    exit_code = pytest.main(test_args)
    
    print("\n" + "=" * 60)
    
    if exit_code == 0:
        print("✅ All tests passed!")
        print("\n📊 Coverage report generated in 'htmlcov/' directory")
        print("   Open 'htmlcov/index.html' in your browser to view detailed coverage")
    else:
        print("❌ Some tests failed!")
        print(f"   Exit code: {exit_code}")
    
    print("\n📋 Test Summary:")
    print("   - Unit tests: Document processing, embedding service")
    print("   - Integration tests: RAG pipeline")
    print("   - Sample data: Insurance policy documents")
    
    return exit_code

def run_quick_tests():
    """Run only quick tests (no model loading)."""
    
    print("⚡ Running Quick Tests (No Model Loading)")
    print("=" * 50)
    
    # Run only document processor tests
    test_args = [
        "-v",
        "--tb=short",
        "tests/test_document_processor.py"
    ]
    
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("✅ Quick tests passed!")
    else:
        print("❌ Quick tests failed!")
    
    return exit_code

def check_dependencies():
    """Check if all required dependencies are available."""
    
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("chromadb", "ChromaDB"),
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("PyPDF2", "PyPDF2"),
        ("docx", "python-docx"),
        ("pdfplumber", "pdfplumber")
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (missing)")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def main():
    """Main test runner."""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            return run_quick_tests()
        elif command == "deps":
            return 0 if check_dependencies() else 1
        elif command == "help":
            print("Test Runner Commands:")
            print("  python run_tests.py        - Run all tests")
            print("  python run_tests.py quick  - Run quick tests only")
            print("  python run_tests.py deps   - Check dependencies")
            print("  python run_tests.py help   - Show this help")
            return 0
        else:
            print(f"Unknown command: {command}")
            print("Use 'python run_tests.py help' for available commands")
            return 1
    
    # Default: run all tests
    return run_tests()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

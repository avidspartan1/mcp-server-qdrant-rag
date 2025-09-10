"""
Comprehensive test fixtures for CLI file ingestion testing.

This module provides reusable test fixtures including:
- Sample file structures with various content types
- Mock configurations and settings
- Test data for different scenarios
- Helper functions for test setup and teardown
"""

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
    IngestConfig,
    CLISettings,
    FileInfo,
    OperationResult,
    ProgressReporter,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings
from src.mcp_server_qdrant_rag.qdrant import Entry


class TestFileStructure:
    """Creates and manages test file structures for CLI testing."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.created_files: List[Path] = []
        self.created_dirs: List[Path] = []
    
    def create_basic_structure(self) -> Dict[str, Path]:
        """Create a basic file structure with common file types."""
        files = {}
        
        # Text files
        files['readme'] = self._create_file("README.md", """
# Test Project

This is a test project for CLI ingestion testing.

## Features

- Document processing
- Vector storage
- Semantic search

## Usage

Run the CLI tool to ingest documents.
        """.strip())
        
        files['guide'] = self._create_file("guide.txt", """
User Guide

This guide explains how to use the system effectively.
It covers basic operations and advanced features.

Getting Started:
1. Install the software
2. Configure your settings
3. Run your first ingestion

Advanced Topics:
- Pattern matching
- Batch processing
- Error handling
        """.strip())
        
        # Code files
        files['main_py'] = self._create_file("main.py", """
#!/usr/bin/env python3
\"\"\"
Main application entry point.
\"\"\"

import sys
import asyncio
from pathlib import Path

def main():
    \"\"\"Main function.\"\"\"
    print("Hello, world!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
        """.strip())
        
        files['utils_js'] = self._create_file("utils.js", """
/**
 * Utility functions for the application.
 */

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function validateEmail(email) {
    const regex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return regex.test(email);
}

module.exports = {
    formatDate,
    validateEmail
};
        """.strip())
        
        # Configuration files
        files['config_json'] = self._create_file("config.json", """
{
    "app_name": "Test Application",
    "version": "1.0.0",
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "testdb"
    },
    "features": {
        "logging": true,
        "metrics": false,
        "debug": true
    }
}
        """.strip())
        
        files['settings_yaml'] = self._create_file("settings.yaml", """
server:
  host: localhost
  port: 8080
  ssl: false

database:
  url: postgresql://localhost/testdb
  pool_size: 10
  timeout: 30

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: app.log
        """.strip())
        
        return files
    
    def create_nested_structure(self) -> Dict[str, Path]:
        """Create a nested directory structure with files."""
        files = {}
        
        # Create directories
        docs_dir = self._create_dir("docs")
        api_dir = self._create_dir("docs/api")
        src_dir = self._create_dir("src")
        tests_dir = self._create_dir("tests")
        
        # Documentation files
        files['docs_index'] = self._create_file("docs/index.md", """
# Documentation

Welcome to the project documentation.

## Sections

- [API Reference](api/README.md)
- [User Guide](guide.md)
- [Examples](examples.md)
        """.strip())
        
        files['api_readme'] = self._create_file("docs/api/README.md", """
# API Reference

## Endpoints

### GET /api/health
Returns the health status of the service.

### POST /api/ingest
Ingests new documents into the system.

### GET /api/search
Searches for documents using semantic similarity.
        """.strip())
        
        files['api_spec'] = self._create_file("docs/api/openapi.yaml", """
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
paths:
  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Service is healthy
  /ingest:
    post:
      summary: Ingest documents
      responses:
        '201':
          description: Documents ingested successfully
        """.strip())
        
        # Source code files
        files['src_main'] = self._create_file("src/main.py", """
\"\"\"Main application module.\"\"\"

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    \"\"\"Processes documents for ingestion.\"\"\"
    
    def __init__(self, config: dict):
        self.config = config
        
    async def process(self, documents: List[str]) -> List[dict]:
        \"\"\"Process a list of documents.\"\"\"
        results = []
        for doc in documents:
            result = await self._process_single(doc)
            results.append(result)
        return results
        
    async def _process_single(self, document: str) -> dict:
        \"\"\"Process a single document.\"\"\"
        return {
            'content': document,
            'tokens': len(document.split()),
            'processed_at': datetime.now().isoformat()
        }
        """.strip())
        
        # Test files
        files['test_main'] = self._create_file("tests/test_main.py", """
\"\"\"Tests for main module.\"\"\"

import pytest
from src.main import DocumentProcessor

class TestDocumentProcessor:
    \"\"\"Test cases for DocumentProcessor.\"\"\"
    
    def setup_method(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.processor = DocumentProcessor({'debug': True})
    
    @pytest.mark.asyncio
    async def test_process_single_document(self):
        \"\"\"Test processing a single document.\"\"\"
        result = await self.processor._process_single("test document")
        assert result['content'] == "test document"
        assert result['tokens'] == 2
        """.strip())
        
        return files
    
    def create_large_files(self) -> Dict[str, Path]:
        """Create large files for performance testing."""
        files = {}
        
        # Large text file
        large_content = []
        for i in range(1000):
            large_content.append(f"""
Paragraph {i}: This is a large document created for performance testing.
It contains many paragraphs with repeated content to simulate real-world
documents that need to be chunked and processed efficiently. The content
includes various topics and concepts that would be found in technical
documentation, user guides, and reference materials.

Key points for paragraph {i}:
- Performance testing is important
- Large documents need efficient processing
- Chunking strategies affect retrieval quality
- Memory usage should be monitored
- Processing time should be reasonable

This paragraph discusses topic {i % 10} which relates to the overall
theme of the document. The content is designed to test the system's
ability to handle large volumes of text while maintaining good
performance characteristics.
            """.strip())
        
        files['large_doc'] = self._create_file("large_document.txt", "\n\n".join(large_content))
        
        # Large code file
        large_code = []
        for i in range(500):
            large_code.append(f"""
def function_{i}(param1, param2, param3):
    \"\"\"
    Function {i} performs operation {i % 5}.
    
    Args:
        param1: First parameter for operation
        param2: Second parameter for operation  
        param3: Third parameter for operation
        
    Returns:
        Result of operation {i % 5}
    \"\"\"
    if param1 is None:
        raise ValueError("param1 cannot be None")
    
    result = param1 + param2 * param3
    
    # Process based on operation type
    if {i % 5} == 0:
        return result * 2
    elif {i % 5} == 1:
        return result + 10
    elif {i % 5} == 2:
        return result - 5
    elif {i % 5} == 3:
        return result / 2
    else:
        return result ** 2

class Class_{i}:
    \"\"\"Class {i} for testing purposes.\"\"\"
    
    def __init__(self, value):
        self.value = value
        self.id = {i}
    
    def process(self):
        \"\"\"Process the value.\"\"\"
        return function_{i}(self.value, self.id, 1)
            """.strip())
        
        files['large_code'] = self._create_file("large_code.py", "\n\n".join(large_code))
        
        return files
    
    def create_problematic_files(self) -> Dict[str, Path]:
        """Create files that might cause processing issues."""
        files = {}
        
        # Empty file
        files['empty'] = self._create_file("empty.txt", "")
        
        # Whitespace only file
        files['whitespace'] = self._create_file("whitespace.txt", "   \n\t\n   \n")
        
        # Single word file
        files['single_word'] = self._create_file("single.txt", "word")
        
        # Very long line file
        files['long_line'] = self._create_file("long_line.txt", "x" * 10000)
        
        # File with special characters
        files['special_chars'] = self._create_file("special.txt", """
Special characters test: áéíóú àèìòù âêîôû ãñõ çß
Unicode symbols: ★ ♠ ♣ ♥ ♦ ← → ↑ ↓ ∞ ≈ ≠ ≤ ≥
Mathematical: ∑ ∏ ∫ √ ∂ ∆ π α β γ δ ε
Currency: $ € £ ¥ ₹ ₽ ₩ ₪ ₫ ₡
        """.strip())
        
        # File with mixed encodings (simulate encoding issues)
        files['mixed_encoding'] = self._create_file("mixed.txt", """
Regular ASCII text
Some Latin-1 characters: café résumé naïve
UTF-8 characters: 世界 こんにちは مرحبا
        """.strip())
        
        # Binary file (should be skipped)
        binary_content = bytes(range(256))
        files['binary'] = self._create_binary_file("binary.bin", binary_content)
        
        return files
    
    def create_pattern_test_files(self) -> Dict[str, Path]:
        """Create files for testing include/exclude patterns."""
        files = {}
        
        # Create files with different extensions
        files['doc_txt'] = self._create_file("document.txt", "Text document content")
        files['doc_md'] = self._create_file("document.md", "# Markdown document")
        files['script_py'] = self._create_file("script.py", "print('Python script')")
        files['script_js'] = self._create_file("script.js", "console.log('JavaScript');")
        files['config_json'] = self._create_file("config.json", '{"key": "value"}')
        files['config_yaml'] = self._create_file("config.yaml", "key: value")
        files['readme'] = self._create_file("README", "Plain readme file")
        files['license'] = self._create_file("LICENSE", "MIT License")
        
        # Create files in subdirectories
        self._create_dir("src")
        self._create_dir("tests")
        self._create_dir("docs")
        
        files['src_main'] = self._create_file("src/main.py", "# Main module")
        files['src_utils'] = self._create_file("src/utils.py", "# Utilities")
        files['test_main'] = self._create_file("tests/test_main.py", "# Test main")
        files['docs_guide'] = self._create_file("docs/guide.md", "# Guide")
        
        # Create hidden files (should be excluded by default)
        files['hidden_txt'] = self._create_file(".hidden.txt", "Hidden file")
        files['hidden_config'] = self._create_file(".gitignore", "*.pyc\n__pycache__/")
        
        # Create files with special naming patterns
        files['temp_file'] = self._create_file("temp_file.tmp", "Temporary file")
        files['backup_file'] = self._create_file("backup.bak", "Backup file")
        files['log_file'] = self._create_file("app.log", "Log entries")
        
        return files
    
    def _create_file(self, relative_path: str, content: str) -> Path:
        """Create a file with the given content."""
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        self.created_files.append(file_path)
        return file_path
    
    def _create_binary_file(self, relative_path: str, content: bytes) -> Path:
        """Create a binary file with the given content."""
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        self.created_files.append(file_path)
        return file_path
    
    def _create_dir(self, relative_path: str) -> Path:
        """Create a directory."""
        dir_path = self.base_path / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        self.created_dirs.append(dir_path)
        return dir_path
    
    def cleanup(self):
        """Clean up created files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        
        for dir_path in sorted(self.created_dirs, reverse=True):
            if dir_path.exists() and not any(dir_path.iterdir()):
                dir_path.rmdir()


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture
def basic_file_structure(temp_workspace):
    """Create a basic file structure for testing."""
    structure = TestFileStructure(temp_workspace)
    files = structure.create_basic_structure()
    yield temp_workspace, files
    structure.cleanup()


@pytest.fixture
def nested_file_structure(temp_workspace):
    """Create a nested file structure for testing."""
    structure = TestFileStructure(temp_workspace)
    files = structure.create_nested_structure()
    yield temp_workspace, files
    structure.cleanup()


@pytest.fixture
def large_file_structure(temp_workspace):
    """Create large files for performance testing."""
    structure = TestFileStructure(temp_workspace)
    files = structure.create_large_files()
    yield temp_workspace, files
    structure.cleanup()


@pytest.fixture
def problematic_file_structure(temp_workspace):
    """Create problematic files for error testing."""
    structure = TestFileStructure(temp_workspace)
    files = structure.create_problematic_files()
    yield temp_workspace, files
    structure.cleanup()


@pytest.fixture
def pattern_test_structure(temp_workspace):
    """Create files for pattern testing."""
    structure = TestFileStructure(temp_workspace)
    files = structure.create_pattern_test_files()
    yield temp_workspace, files
    structure.cleanup()


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = IngestConfig(
        cli_settings=CLISettings(
            operation_mode="ingest",
            verbose=False,
            dry_run=False,
            force_operation=False,
        ),
        qdrant_settings=QdrantSettings(),
        embedding_settings=EmbeddingProviderSettings(
            embedding_model="nomic-ai/nomic-embed-text-v1.5-Q",
        ),
        target_path=Path("/tmp/test"),
        knowledgebase_name="test_collection",
    )
    return config


@pytest.fixture
def mock_progress_reporter():
    """Create a mock progress reporter for testing."""
    reporter = MagicMock(spec=ProgressReporter)
    reporter.log_info = MagicMock()
    reporter.log_warning = MagicMock()
    reporter.log_error = MagicMock()
    reporter.log_success = MagicMock()
    reporter.show_progress = MagicMock()
    reporter.update_progress = MagicMock()
    reporter.finish_progress = MagicMock()
    return reporter


@pytest.fixture
def mock_qdrant_connector():
    """Create a mock Qdrant connector for testing."""
    connector = AsyncMock()
    connector.get_collection_names.return_value = []
    connector.create_collection.return_value = None
    connector.store.return_value = None
    connector.store_entries.return_value = None
    connector.search.return_value = []
    connector.delete_collection.return_value = None
    connector.get_collection_info.return_value = {
        "vectors_count": 0,
        "indexed_vectors_count": 0,
        "points_count": 0,
        "segments_count": 1,
        "config": {
            "params": {
                "vectors": {
                    "size": 768,
                    "distance": "Cosine"
                }
            }
        }
    }
    return connector


@pytest.fixture
def sample_file_infos(temp_workspace):
    """Create sample FileInfo objects for testing."""
    # Create actual files
    file1 = temp_workspace / "test1.txt"
    file2 = temp_workspace / "test2.py"
    file3 = temp_workspace / "test3.md"
    
    file1.write_text("Test content 1")
    file2.write_text("print('test')")
    file3.write_text("# Test")
    
    # Create FileInfo objects
    infos = [
        FileInfo(
            path=file1,
            size=file1.stat().st_size,
            modified_time=datetime.fromtimestamp(file1.stat().st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=3
        ),
        FileInfo(
            path=file2,
            size=file2.stat().st_size,
            modified_time=datetime.fromtimestamp(file2.stat().st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=2
        ),
        FileInfo(
            path=file3,
            size=file3.stat().st_size,
            modified_time=datetime.fromtimestamp(file3.stat().st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=1
        ),
    ]
    
    return infos


@pytest.fixture
def sample_entries():
    """Create sample Entry objects for testing."""
    entries = [
        Entry(
            content="This is test content for entry 1",
            metadata={"source": "test1.txt", "type": "text"},
        ),
        Entry(
            content="This is test content for entry 2",
            metadata={"source": "test2.py", "type": "code"},
        ),
        Entry(
            content="# Test Header\n\nThis is markdown content",
            metadata={"source": "test3.md", "type": "markdown"},
        ),
    ]
    return entries


def create_operation_result(
    success: bool = True,
    files_processed: int = 0,
    files_failed: int = 0,
    files_skipped: int = 0,
    chunks_created: int = 0,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    execution_time: float = 0.0,
) -> OperationResult:
    """Helper function to create OperationResult objects for testing."""
    return OperationResult(
        success=success,
        files_processed=files_processed,
        files_failed=files_failed,
        files_skipped=files_skipped,
        chunks_created=chunks_created,
        errors=errors or [],
        warnings=warnings or [],
        execution_time=execution_time,
    )


def create_test_config(
    operation_mode: str = "ingest",
    target_path: Optional[Path] = None,
    knowledgebase_name: str = "test_collection",
    dry_run: bool = False,
    verbose: bool = False,
    force_operation: bool = False,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    update_mode: str = "add-only",
    batch_size: int = 10,
    show_progress: bool = True,
    **kwargs
) -> IngestConfig:
    """Helper function to create test configurations."""
    return IngestConfig(
        cli_settings=CLISettings(
            operation_mode=operation_mode,
            dry_run=dry_run,
            verbose=verbose,
            force_operation=force_operation,
            include_patterns=include_patterns or [],
            exclude_patterns=exclude_patterns or [],
            update_mode=update_mode,
            batch_size=batch_size,
            show_progress=show_progress,
            **kwargs
        ),
        qdrant_settings=QdrantSettings(),
        embedding_settings=EmbeddingProviderSettings(),
        target_path=target_path or Path("/tmp/test"),
        knowledgebase_name=knowledgebase_name,
    )
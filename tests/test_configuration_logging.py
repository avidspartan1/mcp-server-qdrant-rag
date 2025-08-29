"""Tests for configuration logging and debugging functionality."""

import logging
import pytest
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO

from mcp_server_qdrant_rag.settings import EmbeddingProviderSettings, QdrantSettings, ToolSettings
from mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from mcp_server_qdrant_rag.embeddings.base import EmbeddingProvider
from mcp_server_qdrant_rag.chunking.chunker import DocumentChunker
from mcp_server_qdrant_rag.common.exceptions import ConfigurationValidationError


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, model_name: str = "test-model"):
        self.model_name = model_name
    
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in documents]
    
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]
    
    def get_vector_name(self) -> str:
        return "test-vector"
    
    def get_vector_size(self) -> int:
        return 3
    
    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "vector_size": 3,
            "vector_name": "test-vector",
            "description": "Test model",
            "status": "loaded"
        }


class TestChunkingOperationLogging:
    """Test logging for chunking operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.log_buffer = StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Set up logger for chunker
        self.logger = logging.getLogger("mcp_server_qdrant_rag.chunking.chunker")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:-1]:
            self.logger.removeHandler(handler)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()
    
    def get_log_output(self) -> str:
        """Get the captured log output."""
        return self.log_buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_chunking_decision_logging(self):
        """Test that chunking decisions are logged."""
        chunker = DocumentChunker(max_tokens=50, overlap_tokens=10)
        
        # Test short document (no chunking needed)
        short_text = "This is a short document."
        chunks = await chunker.chunk_document(short_text)
        
        log_output = self.get_log_output()
        assert "Starting document chunking" in log_output
        assert "Document fits in single chunk, no chunking needed" in log_output
        
        # Clear log buffer
        self.log_buffer.seek(0)
        self.log_buffer.truncate(0)
        
        # Test long document (chunking needed)
        long_text = " ".join(["This is a longer document that will need chunking."] * 10)
        chunks = await chunker.chunk_document(long_text)
        
        log_output = self.get_log_output()
        assert "Starting document chunking" in log_output
        assert "will chunk with 10 token overlap" in log_output
        assert "Hybrid chunking produced" in log_output
        assert "Successfully created" in log_output
    
    @pytest.mark.asyncio
    async def test_empty_content_logging(self):
        """Test logging for empty content."""
        chunker = DocumentChunker(max_tokens=50, overlap_tokens=10)
        
        chunks = await chunker.chunk_document("")
        
        log_output = self.get_log_output()
        assert "Empty content provided for chunking" in log_output
    
    @pytest.mark.asyncio
    async def test_hybrid_chunking_logging(self):
        """Test detailed logging during hybrid chunking."""
        chunker = DocumentChunker(max_tokens=20, overlap_tokens=5)
        
        # Create text that will require sentence splitting and chunking
        text = "First sentence is here. Second sentence follows. Third sentence continues. Fourth sentence ends."
        
        chunks = await chunker.chunk_document(text)
        
        log_output = self.get_log_output()
        
        # Verify detailed chunking logs
        assert "Split text into" in log_output
        assert "sentences" in log_output
        assert "Hybrid chunking complete:" in log_output
        assert "chunks created" in log_output


class TestConfigurationValidationLogging:
    """Test configuration validation logging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.log_buffer = StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Set up logger for settings
        self.logger = logging.getLogger("mcp_server_qdrant_rag.settings")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:-1]:
            self.logger.removeHandler(handler)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()
    
    def get_log_output(self) -> str:
        """Get the captured log output."""
        return self.log_buffer.getvalue()
    
    def test_valid_configuration_logging(self):
        """Test that valid configuration values are logged."""
        import os
        old_env = {}
        env_vars = {
            "MAX_CHUNK_SIZE": "1024",
            "CHUNK_OVERLAP": "100",
            "CHUNK_STRATEGY": "semantic"
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            settings = EmbeddingProviderSettings()
            
            # Verify validation logging for valid values
            log_output = self.get_log_output()
            assert "Validating max_chunk_size: 1024" in log_output
            assert "max_chunk_size validation passed: 1024" in log_output
            assert "Validating chunk_overlap: 100" in log_output
            assert "chunk_overlap validation passed: 100" in log_output
            assert "Validating chunk_strategy: semantic" in log_output
            assert "chunk_strategy validation passed: semantic" in log_output
            assert "chunk_overlap validation passed: 9.8% overlap ratio" in log_output
        
        finally:
            # Restore environment variables
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
    
    def test_invalid_max_chunk_size_logging(self):
        """Test logging for invalid max_chunk_size values."""
        import os
        
        # Test minimum value
        old_env = os.environ.get("MAX_CHUNK_SIZE")
        os.environ["MAX_CHUNK_SIZE"] = "10"
        
        try:
            with pytest.raises(ConfigurationValidationError):
                EmbeddingProviderSettings()
            
            log_output = self.get_log_output()
            assert "Validating max_chunk_size: 10" in log_output
            assert "max_chunk_size validation failed: 10 is below minimum (50)" in log_output
        finally:
            if old_env is None:
                os.environ.pop("MAX_CHUNK_SIZE", None)
            else:
                os.environ["MAX_CHUNK_SIZE"] = old_env
        
        # Clear log buffer
        self.log_buffer.seek(0)
        self.log_buffer.truncate(0)
        
        # Test maximum value
        os.environ["MAX_CHUNK_SIZE"] = "10000"
        
        try:
            with pytest.raises(ConfigurationValidationError):
                EmbeddingProviderSettings()
            
            log_output = self.get_log_output()
            assert "Validating max_chunk_size: 10000" in log_output
            assert "max_chunk_size validation failed: 10000 exceeds maximum (8192)" in log_output
        finally:
            if old_env is None:
                os.environ.pop("MAX_CHUNK_SIZE", None)
            else:
                os.environ["MAX_CHUNK_SIZE"] = old_env


class TestQdrantConnectorLogging:
    """Test logging for QdrantConnector chunking decisions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.log_buffer = StringIO()
        self.log_handler = logging.StreamHandler(self.log_buffer)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Set up logger for qdrant connector
        self.logger = logging.getLogger("mcp_server_qdrant_rag.qdrant")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:-1]:
            self.logger.removeHandler(handler)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()
    
    def get_log_output(self) -> str:
        """Get the captured log output."""
        return self.log_buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_chunking_decision_logging(self):
        """Test that chunking decisions are logged in QdrantConnector."""
        from mcp_server_qdrant_rag.qdrant import QdrantConnector, Entry
        
        # Create a mock embedding provider
        mock_provider = MockEmbeddingProvider()
        
        # Create a mock chunker
        mock_chunker = Mock()
        mock_chunker.max_tokens = 100
        mock_chunker._count_tokens = Mock(return_value=50)  # Short document
        
        with patch('mcp_server_qdrant_rag.qdrant.AsyncQdrantClient'):
            connector = QdrantConnector(
                qdrant_url=None,
                qdrant_api_key=None,
                collection_name="test",
                embedding_provider=mock_provider,
                enable_chunking=True,
                max_chunk_size=100,
                chunk_overlap=10,
                chunk_strategy="semantic"
            )
            connector._chunker = mock_chunker
            
            # Mock the _ensure_collection_exists and _store_single_entry methods
            connector._ensure_collection_exists = AsyncMock()
            connector._store_single_entry = AsyncMock()
            
            # Test with short document (no chunking)
            entry = Entry(content="Short document")
            await connector.store(entry)
            
            log_output = self.get_log_output()
            assert "Chunking decision: 50 tokens vs 100 max -> NO CHUNK" in log_output
            assert "Storing single entry in collection 'test' - 14 chars (below chunk threshold)" in log_output
        
        # Clear log buffer
        self.log_buffer.seek(0)
        self.log_buffer.truncate(0)
        
        # Test with long document (chunking needed)
        mock_chunker._count_tokens = Mock(return_value=150)  # Long document
        
        with patch('mcp_server_qdrant_rag.qdrant.AsyncQdrantClient'):
            connector = QdrantConnector(
                qdrant_url=None,
                qdrant_api_key=None,
                collection_name="test",
                embedding_provider=mock_provider,
                enable_chunking=True,
                max_chunk_size=100,
                chunk_overlap=10,
                chunk_strategy="semantic"
            )
            connector._chunker = mock_chunker
            connector._ensure_collection_exists = AsyncMock()
            connector._store_chunked_document = AsyncMock()
            
            # Test with long document
            entry = Entry(content="This is a much longer document that will need chunking")
            await connector.store(entry)
            
            log_output = self.get_log_output()
            assert "Chunking decision: 150 tokens vs 100 max -> CHUNK" in log_output
            assert "Document will be chunked before storage" in log_output
    
    @pytest.mark.asyncio
    async def test_chunking_disabled_logging(self):
        """Test logging when chunking is disabled."""
        from mcp_server_qdrant_rag.qdrant import QdrantConnector, Entry
        
        mock_provider = MockEmbeddingProvider()
        
        with patch('mcp_server_qdrant_rag.qdrant.AsyncQdrantClient'):
            connector = QdrantConnector(
                qdrant_url=None,
                qdrant_api_key=None,
                collection_name="test",
                embedding_provider=mock_provider,
                enable_chunking=False,  # Chunking disabled
                max_chunk_size=100,
                chunk_overlap=10,
                chunk_strategy="semantic"
            )
            
            connector._ensure_collection_exists = AsyncMock()
            connector._store_single_entry = AsyncMock()
            
            entry = Entry(content="Long document that would normally be chunked")
            await connector.store(entry)
            
            log_output = self.get_log_output()
            assert "Storing single entry in collection 'test' - 44 chars (chunking disabled)" in log_output
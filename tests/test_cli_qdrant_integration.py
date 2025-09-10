"""
Integration tests for CLI operations with QdrantConnector and chunking system.

This module tests the integration between CLI operations, QdrantConnector,
and the document chunking system to ensure proper file ingestion workflow.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime
from typing import List

from src.mcp_server_qdrant_rag.cli_ingest import (
    IngestOperation,
    UpdateOperation,
    RemoveOperation,
    ListOperation,
    IngestConfig,
    CLISettings,
    FileInfo,
    ContentProcessor,
)
from src.mcp_server_qdrant_rag.qdrant import QdrantConnector, Entry
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings
from src.mcp_server_qdrant_rag.embeddings.base import EmbeddingProvider
from src.mcp_server_qdrant_rag.chunking.models import DocumentChunk


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self):
        self.model_name = "test-model"
        self._vector_size = 384
        self._vector_name = "test_vector"
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Return mock embeddings."""
        return [[0.1] * self._vector_size for _ in documents]
    
    async def embed_query(self, query: str) -> List[float]:
        """Return mock query embedding."""
        return [0.1] * self._vector_size
    
    def get_vector_size(self) -> int:
        return self._vector_size
    
    def get_vector_name(self) -> str:
        return self._vector_name


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content for ingestion")
    
    cli_settings = CLISettings()
    cli_settings.operation_mode = "ingest"
    cli_settings.supported_extensions = [".txt", ".md"]
    cli_settings.verbose = True
    cli_settings.dry_run = False
    
    qdrant_settings = QdrantSettings()
    qdrant_settings.location = "http://localhost:6333"
    qdrant_settings.collection_name = "test_collection"
    
    embedding_settings = EmbeddingProviderSettings()
    embedding_settings.model_name = "test-model"
    embedding_settings.enable_chunking = True
    embedding_settings.max_chunk_size = 100
    embedding_settings.chunk_overlap = 20
    
    return IngestConfig(
        cli_settings=cli_settings,
        qdrant_settings=qdrant_settings,
        embedding_settings=embedding_settings,
        target_path=test_file,
        knowledgebase_name="test_kb"
    )


class TestQdrantConnectorIntegration:
    """Test QdrantConnector integration with CLI operations."""
    
    @pytest.mark.asyncio
    async def test_connector_initialization_with_chunking(self, test_config, mock_embedding_provider):
        """Test QdrantConnector initialization with chunking configuration."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = IngestOperation(test_config)
            connector = await operation.get_connector()
            
            assert connector is not None
            assert connector._enable_chunking == test_config.embedding_settings.enable_chunking
            assert connector._chunker is not None
            assert connector._chunker.max_tokens == test_config.embedding_settings.max_chunk_size
            assert connector._chunker.overlap_tokens == test_config.embedding_settings.chunk_overlap 
   
    @pytest.mark.asyncio
    async def test_collection_existence_checking(self, test_config, mock_embedding_provider):
        """Test collection existence checking functionality."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = IngestOperation(test_config)
            
            # Mock the connector's get_collection_names method
            with patch.object(QdrantConnector, 'get_collection_names', new_callable=AsyncMock) as mock_get_names:
                mock_get_names.return_value = ["existing_collection", "another_collection"]
                
                exists = await operation.check_collection_exists("existing_collection")
                assert exists is True
                
                exists = await operation.check_collection_exists("nonexistent_collection")
                assert exists is False
    
    @pytest.mark.asyncio
    async def test_collection_creation_logic(self, test_config, mock_embedding_provider):
        """Test collection creation through QdrantConnector."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = IngestOperation(test_config)
            connector = await operation.get_connector()
            
            # Mock the Qdrant client methods
            with patch.object(connector._client, 'collection_exists', new_callable=AsyncMock) as mock_exists, \
                 patch.object(connector._client, 'create_collection', new_callable=AsyncMock) as mock_create:
                
                mock_exists.return_value = False
                
                # Test that _ensure_collection_exists creates collection when it doesn't exist
                await connector._ensure_collection_exists("new_collection")
                
                mock_exists.assert_called_once_with("new_collection")
                mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_entry_creation_with_file_metadata(self, test_config, tmp_path):
        """Test Entry object creation with file-specific metadata."""
        # Create test file
        test_file = tmp_path / "sample.py"
        test_content = "def hello():\n    print('Hello, World!')"
        test_file.write_text(test_content)
        
        # Create FileInfo
        file_info = FileInfo(
            path=test_file,
            size=len(test_content),
            modified_time=datetime.now(),
            encoding="utf-8",
            is_binary=False,
            estimated_tokens=10
        )
        
        # Process file to create Entry
        processor = ContentProcessor()
        entry = await processor.process_file(file_info)
        
        assert entry is not None
        assert entry.content == test_content
        assert entry.metadata is not None
        
        # Check file-specific metadata
        assert entry.metadata["file_path"] == str(test_file)
        assert entry.metadata["file_name"] == "sample.py"
        assert entry.metadata["file_extension"] == ".py"
        assert entry.metadata["file_type"] == "python"
        assert entry.metadata["source_type"] == "file_ingestion"
        assert entry.metadata["ingestion_method"] == "cli_tool"


class TestChunkingSystemIntegration:
    """Test integration with the chunking system."""
    
    @pytest.mark.asyncio
    async def test_chunking_integration_with_large_document(self, test_config, mock_embedding_provider):
        """Test that large documents are properly chunked through QdrantConnector."""
        # Create a large document that should be chunked
        large_content = "This is a test sentence. " * 50  # Should exceed chunk size
        
        entry = Entry(
            content=large_content,
            metadata={"test": "metadata"},
            is_chunk=False
        )
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = IngestOperation(test_config)
            connector = await operation.get_connector()
            
            # Mock the client methods
            with patch.object(connector._client, 'collection_exists', new_callable=AsyncMock) as mock_exists, \
                 patch.object(connector._client, 'create_collection', new_callable=AsyncMock), \
                 patch.object(connector._client, 'upsert', new_callable=AsyncMock) as mock_upsert:
                
                mock_exists.return_value = False
                
                # Store the entry - should trigger chunking
                await connector.store(entry, collection_name="test_collection")
                
                # Verify that upsert was called multiple times (indicating chunks were created)
                assert mock_upsert.call_count > 1
                
                # Check that the stored points have chunk metadata
                # Get the first call to check chunk metadata
                first_call_args = mock_upsert.call_args_list[0]
                points = first_call_args[1]['points']  # Get points from keyword arguments
                
                # Each call should have one point (chunk)
                assert len(points) == 1
                
                # Each point should have chunk metadata
                for point in points:
                    payload = point.payload
                    assert payload.get("is_chunk") is True
                    assert "source_document_id" in payload
                    assert "chunk_index" in payload
                    assert "total_chunks" in payload
    
    @pytest.mark.asyncio
    async def test_small_document_no_chunking(self, test_config, mock_embedding_provider):
        """Test that small documents are not chunked."""
        # Create a small document that should not be chunked
        small_content = "This is a small test document."
        
        entry = Entry(
            content=small_content,
            metadata={"test": "metadata"},
            is_chunk=False
        )
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = IngestOperation(test_config)
            connector = await operation.get_connector()
            
            # Mock the client methods
            with patch.object(connector._client, 'collection_exists', new_callable=AsyncMock) as mock_exists, \
                 patch.object(connector._client, 'create_collection', new_callable=AsyncMock), \
                 patch.object(connector._client, 'upsert', new_callable=AsyncMock) as mock_upsert:
                
                mock_exists.return_value = False
                
                # Store the entry - should not trigger chunking
                await connector.store(entry, collection_name="test_collection")
                
                # Verify that upsert was called once (single entry)
                assert mock_upsert.call_count == 1
                
                # Check that the stored point does not have chunk metadata
                call_args = mock_upsert.call_args
                points = call_args[1]['points']
                
                assert len(points) == 1
                payload = points[0].payload
                assert payload.get("is_chunk", False) is False
                assert "source_document_id" not in payload


class TestOperationIntegration:
    """Test integration of CLI operations with QdrantConnector."""
    
    @pytest.mark.asyncio
    async def test_ingest_operation_end_to_end(self, test_config, mock_embedding_provider, tmp_path):
        """Test complete ingest operation workflow."""
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"test_{i}.txt"
            test_file.write_text(f"Test content for file {i}")
            files.append(test_file)
        
        # Update config to point to directory
        test_config.target_path = tmp_path
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = IngestOperation(test_config)
            
            # Mock Qdrant connector methods
            with patch.object(QdrantConnector, 'get_collection_names', new_callable=AsyncMock) as mock_get_names, \
                 patch.object(QdrantConnector, 'store', new_callable=AsyncMock) as mock_store:
                
                mock_get_names.return_value = []
                
                result = await operation.execute()
                
                assert result.success is True
                assert result.files_processed >= 3  # At least 3 files (may include fixture file)
                assert result.files_failed == 0
                assert mock_store.call_count >= 3  # At least one call per file
    
    @pytest.mark.asyncio
    async def test_update_operation_replace_mode(self, test_config, mock_embedding_provider):
        """Test update operation in replace mode."""
        test_config.cli_settings.operation_mode = "update"
        test_config.cli_settings.update_mode = "replace"
        test_config.cli_settings.force_operation = True  # Skip confirmation
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = UpdateOperation(test_config)
            
            with patch.object(QdrantConnector, 'get_collection_names', new_callable=AsyncMock) as mock_get_names, \
                 patch.object(QdrantConnector, 'clear_collection', new_callable=AsyncMock) as mock_clear, \
                 patch.object(QdrantConnector, 'store', new_callable=AsyncMock) as mock_store:
                
                mock_get_names.return_value = ["test_kb"]  # Collection exists
                mock_clear.return_value = 10  # 10 points cleared
                
                result = await operation.execute()
                
                assert result.success is True
                # Verify collection was cleared
                mock_clear.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_operation_integration(self, test_config, mock_embedding_provider):
        """Test remove operation integration."""
        test_config.cli_settings.operation_mode = "remove"
        test_config.cli_settings.force_operation = True  # Skip confirmation
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = RemoveOperation(test_config)
            
            with patch.object(QdrantConnector, 'get_collection_names', new_callable=AsyncMock) as mock_get_names, \
                 patch.object(QdrantConnector, 'delete_collection', new_callable=AsyncMock) as mock_delete:
                
                mock_get_names.return_value = ["test_kb"]  # Collection exists
                mock_delete.return_value = True  # Successfully deleted
                
                result = await operation.execute()
                
                assert result.success is True
                assert result.files_processed == 1  # Indicates collection was deleted
                mock_delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_operation_integration(self, test_config, mock_embedding_provider):
        """Test list operation integration."""
        test_config.cli_settings.operation_mode = "list"
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', return_value=mock_embedding_provider):
            operation = ListOperation(test_config)
            
            with patch.object(QdrantConnector, 'get_collection_names', new_callable=AsyncMock) as mock_get_names, \
                 patch.object(QdrantConnector, 'analyze_collection_compatibility', new_callable=AsyncMock) as mock_analyze:
                
                mock_get_names.return_value = ["collection1", "collection2"]
                mock_analyze.return_value = {
                    "exists": True,
                    "points_count": 100,
                    "available_vectors": ["test_vector"],
                    "expected_dimensions": 384,
                    "current_model": "test-model"
                }
                
                result = await operation.execute()
                
                assert result.success is True
                assert result.files_processed == 2  # Number of collections listed
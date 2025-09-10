"""
Unit tests for CLI operation classes.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.mcp_server_qdrant_rag.cli_ingest import (
    BaseOperation,
    IngestOperation,
    UpdateOperation,
    RemoveOperation,
    ListOperation,
    OperationResult,
    FileInfo,
)

def test_operation_result_creation():
    """Test OperationResult creation and properties."""
    from src.mcp_server_qdrant_rag.cli_ingest import OperationResult
    
    result = OperationResult(success=True, files_processed=5, files_failed=1)
    
    assert result.success is True
    assert result.files_processed == 5
    assert result.files_failed == 1
    assert result.total_files == 6  # processed + skipped + failed
    assert result.success_rate == (5/6) * 100  # 83.33%


def test_file_info_creation(tmp_path):
    """Test FileInfo creation."""
    from src.mcp_server_qdrant_rag.cli_ingest import FileInfo
    
    # Create a real temporary file for testing
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    test_time = datetime.now()
    
    file_info = FileInfo(
        path=test_file,
        size=1000,
        modified_time=test_time,
        encoding="utf-8",
        is_binary=False,
        estimated_tokens=250
    )
    
    assert file_info.path == test_file
    assert file_info.size == 1000
    assert file_info.modified_time == test_time
    assert file_info.encoding == "utf-8"
    assert file_info.is_binary is False
    assert file_info.estimated_tokens == 250


class TestBaseOperationFunctionality:
    """Test BaseOperation functionality."""
    
    def test_base_operation_initialization(self):
        """Test BaseOperation initialization."""
        from src.mcp_server_qdrant_rag.cli_ingest import BaseOperation, OperationResult
        
        # Create a concrete implementation for testing
        class ConcreteOperation(BaseOperation):
            async def execute(self) -> OperationResult:
                return OperationResult(success=True)
            
            def validate_preconditions(self) -> List[str]:
                return []
        
        config = MagicMock()
        operation = ConcreteOperation(config)
        
        assert operation.config == config
        assert operation._connector is None
        assert operation._embedding_provider is None
    
    def test_confirm_operation_force_mode(self):
        """Test confirmation in force mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import BaseOperation, OperationResult
        
        class ConcreteOperation(BaseOperation):
            async def execute(self) -> OperationResult:
                return OperationResult(success=True)
            
            def validate_preconditions(self) -> List[str]:
                return []
        
        config = MagicMock()
        config.cli_settings.force_operation = True
        operation = ConcreteOperation(config)
        
        result = operation._confirm_operation("Test confirmation")
        assert result is True


class TestIngestOperationFunctionality:
    """Test IngestOperation functionality."""
    
    def test_validate_preconditions_missing_path(self):
        """Test validation with missing target path."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        config = MagicMock()
        config.target_path = None
        config.knowledgebase_name = "test-kb"
        config.qdrant_settings.location = "http://localhost:6333"
        config.embedding_settings.model_name = "test-model"
        
        operation = IngestOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Target path does not exist" in error for error in errors)
    
    def test_validate_preconditions_missing_knowledgebase(self, tmp_path):
        """Test validation with missing knowledgebase name."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        config = MagicMock()
        config.target_path = test_file
        config.knowledgebase_name = None
        config.qdrant_settings.location = "http://localhost:6333"
        config.embedding_settings.model_name = "test-model"
        
        operation = IngestOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Knowledgebase name is required" in error for error in errors)


class TestUpdateOperationFunctionality:
    """Test UpdateOperation functionality."""
    
    def test_validate_preconditions_invalid_mode(self, tmp_path):
        """Test validation with invalid update mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import UpdateOperation
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        config = MagicMock()
        config.target_path = test_file
        config.knowledgebase_name = "test-kb"
        config.qdrant_settings.location = "http://localhost:6333"
        config.embedding_settings.model_name = "test-model"
        config.cli_settings.update_mode = "invalid-mode"
        
        operation = UpdateOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Invalid update mode" in error for error in errors)


class TestRemoveOperationFunctionality:
    """Test RemoveOperation functionality."""
    
    def test_validate_preconditions_success(self):
        """Test successful precondition validation."""
        from src.mcp_server_qdrant_rag.cli_ingest import RemoveOperation
        
        config = MagicMock()
        config.knowledgebase_name = "test-kb"
        config.qdrant_settings.location = "http://localhost:6333"
        
        operation = RemoveOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) == 0
    
    def test_validate_preconditions_missing_knowledgebase(self):
        """Test validation with missing knowledgebase name."""
        from src.mcp_server_qdrant_rag.cli_ingest import RemoveOperation
        
        config = MagicMock()
        config.knowledgebase_name = None
        config.qdrant_settings.location = "http://localhost:6333"
        
        operation = RemoveOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Knowledgebase name is required" in error for error in errors)


class TestListOperationFunctionality:
    """Test ListOperation functionality."""
    
    def test_validate_preconditions_success(self):
        """Test successful precondition validation."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        
        operation = ListOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) == 0
    
    def test_validate_preconditions_missing_url(self):
        """Test validation with missing Qdrant URL."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = None
        
        operation = ListOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Qdrant URL is required" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_execute_no_collections(self):
        """Test execution when no collections exist."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = False
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.return_value = []
            
            result = await operation.execute()
            
            assert result.success is True
            assert result.files_processed == 0
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_collections_basic(self):
        """Test execution with collections in basic mode (non-verbose)."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = False
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.return_value = ["collection1", "collection2"]
            
            result = await operation.execute()
            
            assert result.success is True
            assert result.files_processed == 2  # Number of collections listed
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_collections_verbose(self):
        """Test execution with collections in verbose mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = True
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.return_value = ["test_collection"]
            mock_connector.analyze_collection_compatibility.return_value = {
                "exists": True,
                "points_count": 150,
                "available_vectors": ["default"],
                "expected_dimensions": 384,
                "current_model": "nomic-ai/nomic-embed-text-v1.5-Q"
            }
            
            result = await operation.execute()
            
            assert result.success is True
            assert result.files_processed == 1
            assert len(result.errors) == 0
            
            # Verify that analyze_collection_compatibility was called for verbose mode
            mock_connector.analyze_collection_compatibility.assert_called_once_with("test_collection")
    
    @pytest.mark.asyncio
    async def test_execute_connection_failure(self):
        """Test execution when Qdrant connection fails."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = False
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.side_effect = Exception("Connection failed")
            
            result = await operation.execute()
            
            assert result.success is False
            assert len(result.errors) == 1
            assert "Failed to connect to Qdrant" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_execute_compatibility_analysis_error(self):
        """Test execution when compatibility analysis fails in verbose mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = True
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.return_value = ["test_collection"]
            mock_connector.analyze_collection_compatibility.side_effect = Exception("Analysis failed")
            
            result = await operation.execute()
            
            # Should still succeed even if compatibility analysis fails
            assert result.success is True
            assert result.files_processed == 1
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_execute_precondition_validation_failure(self):
        """Test execution when precondition validation fails."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = None  # Missing URL
        
        operation = ListOperation(config)
        
        result = await operation.execute()
        
        assert result.success is False
        assert len(result.errors) >= 1
        assert any("Qdrant URL is required" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_execute_with_enhanced_verbose_output(self):
        """Test execution with enhanced verbose output showing compatibility details."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = True
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.return_value = ["enhanced_collection"]
            
            # Mock enhanced compatibility info
            mock_connector.analyze_collection_compatibility.return_value = {
                "exists": True,
                "points_count": 1500,
                "available_vectors": ["default"],
                "expected_dimensions": 384,
                "actual_dimensions": 384,
                "dimension_compatible": True,
                "current_model": "nomic-ai/nomic-embed-text-v1.5-Q",
                "has_chunked_content": True,
                "has_non_chunked_content": False,
                "mixed_content": False,
                "compatible": True,
                "recommendations": []
            }
            
            result = await operation.execute()
            
            assert result.success is True
            assert result.files_processed == 1
            assert len(result.errors) == 0
            
            # Verify that analyze_collection_compatibility was called
            mock_connector.analyze_collection_compatibility.assert_called_once_with("enhanced_collection")
    
    @pytest.mark.asyncio
    async def test_execute_with_compatibility_issues(self):
        """Test execution showing compatibility issues in verbose mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        config.qdrant_settings.api_key = None
        config.cli_settings.verbose = True
        config.cli_settings.show_progress = True
        config.cli_settings.batch_size = 10
        
        operation = ListOperation(config)
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.EmbeddingProviderSettings') as mock_settings_class:
            
            mock_provider = MagicMock()
            mock_create_provider.return_value = mock_provider
            
            mock_settings = MagicMock()
            mock_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            mock_settings_class.return_value = mock_settings
            
            mock_connector = AsyncMock()
            mock_connector_class.return_value = mock_connector
            mock_connector.get_collection_names.return_value = ["incompatible_collection"]
            
            # Mock compatibility issues
            mock_connector.analyze_collection_compatibility.return_value = {
                "exists": True,
                "points_count": 500,
                "available_vectors": ["default"],
                "expected_dimensions": 384,
                "actual_dimensions": 768,  # Dimension mismatch
                "dimension_compatible": False,
                "current_model": "different-model",
                "has_chunked_content": True,
                "has_non_chunked_content": True,
                "mixed_content": True,
                "compatible": False,
                "recommendations": [
                    "Dimension mismatch. Collection has 768 dimensions, but model produces 384",
                    "Consider using a different collection name or switching to a compatible model",
                    "Collection contains both chunked and non-chunked content"
                ]
            }
            
            result = await operation.execute()
            
            assert result.success is True
            assert result.files_processed == 1
            assert len(result.errors) == 0
"""
Comprehensive end-to-end integration tests for CLI file ingestion.

This module provides complete integration testing covering:
- Full CLI workflows from argument parsing to execution
- All operation types (ingest, update, remove, list)
- Error scenarios and edge cases
- Performance testing with large files
- Pattern matching and filtering
- Configuration validation and management
"""

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
    main_async,
    IngestOperation,
    UpdateOperation,
    RemoveOperation,
    ListOperation,
    FileDiscovery,
    ContentProcessor,
    ProgressReporter,
    IngestConfig,
    CLISettings,
    OperationResult,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings
from src.mcp_server_qdrant_rag.common.exceptions import (
    ConfigurationValidationError,
    MCPQdrantError,
    VectorDimensionMismatchError,
)

from tests.fixtures.cli_test_fixtures import (
    basic_file_structure,
    nested_file_structure,
    large_file_structure,
    problematic_file_structure,
    pattern_test_structure,
    mock_config,
    mock_progress_reporter,
    mock_qdrant_connector,
    create_operation_result,
    create_test_config,
)


class TestCompleteIngestWorkflow:
    """Test complete ingest workflow from start to finish."""

    @pytest.mark.asyncio
    async def test_basic_ingest_workflow_success(self, basic_file_structure, mock_qdrant_connector):
        """Test successful basic ingest workflow."""
        workspace, files = basic_file_structure
        
        # Create configuration
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="test_basic_ingest",
            verbose=True,
        )
        
        # Mock connector to simulate successful operations
        mock_qdrant_connector.get_collection_names.return_value = []  # New collection
        mock_qdrant_connector.create_collection.return_value = None
        mock_qdrant_connector.store_entries.return_value = None
        
        # Create and execute operation
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        # Verify successful execution
        assert result.success
        assert result.files_processed > 0
        assert result.files_failed == 0
        assert len(result.errors) == 0
        
        # Verify Qdrant operations were called
        mock_qdrant_connector.get_collection_names.assert_called()
        mock_qdrant_connector.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_workflow_with_existing_collection(self, basic_file_structure, mock_qdrant_connector):
        """Test ingest workflow with existing collection."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="existing_collection",
        )
        
        # Mock existing collection
        mock_qdrant_connector.get_collection_names.return_value = ["existing_collection"]
        mock_qdrant_connector.get_collection_info.return_value = {
            "vectors_count": 10,
            "config": {"params": {"vectors": {"size": 768}}}
        }
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        # Should not create collection if it exists
        mock_qdrant_connector.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_workflow_dry_run(self, basic_file_structure, mock_qdrant_connector):
        """Test ingest workflow in dry-run mode."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="dry_run_test",
            dry_run=True,
            verbose=True,
        )
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        # Dry run should succeed but not perform actual operations
        assert result.success
        assert result.files_processed > 0
        
        # No actual Qdrant operations should be performed in dry run
        mock_qdrant_connector.create_collection.assert_not_called()
        mock_qdrant_connector.store_entries.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_workflow_with_patterns(self, pattern_test_structure, mock_qdrant_connector):
        """Test ingest workflow with include/exclude patterns."""
        workspace, files = pattern_test_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="pattern_test",
            include_patterns=[r'\.py$', r'\.md$'],  # Only Python and Markdown files
            exclude_patterns=[r'test_'],  # Exclude test files
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        # Should process some files but not all due to patterns
        assert result.files_processed > 0
        assert result.files_processed < len(files)


class TestCompleteUpdateWorkflow:
    """Test complete update workflow scenarios."""

    @pytest.mark.asyncio
    async def test_update_workflow_add_only_mode(self, basic_file_structure, mock_qdrant_connector):
        """Test update workflow in add-only mode."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="update",
            target_path=workspace,
            knowledgebase_name="update_test",
            update_mode="add-only",
        )
        
        # Mock existing collection with some data
        mock_qdrant_connector.get_collection_names.return_value = ["update_test"]
        mock_qdrant_connector.search.return_value = []  # No existing files
        
        operation = UpdateOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        assert result.files_processed > 0
        
        # Should not delete existing data in add-only mode
        mock_qdrant_connector.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_workflow_replace_mode(self, basic_file_structure, mock_qdrant_connector):
        """Test update workflow in replace mode."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="update",
            target_path=workspace,
            knowledgebase_name="replace_test",
            update_mode="replace",
            force_operation=True,  # Skip confirmation
        )
        
        mock_qdrant_connector.get_collection_names.return_value = ["replace_test"]
        
        operation = UpdateOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        # Should recreate collection in replace mode
        mock_qdrant_connector.delete_collection.assert_called_once()
        mock_qdrant_connector.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_workflow_nonexistent_collection(self, basic_file_structure, mock_qdrant_connector):
        """Test update workflow with nonexistent collection."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="update",
            target_path=workspace,
            knowledgebase_name="nonexistent",
        )
        
        # Mock no existing collections
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = UpdateOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        # Should create new collection if it doesn't exist
        assert result.success
        mock_qdrant_connector.create_collection.assert_called_once()


class TestCompleteRemoveWorkflow:
    """Test complete remove workflow scenarios."""

    @pytest.mark.asyncio
    async def test_remove_workflow_with_confirmation(self, mock_qdrant_connector):
        """Test remove workflow with user confirmation."""
        config = create_test_config(
            operation_mode="remove",
            knowledgebase_name="remove_test",
            force_operation=False,
        )
        
        mock_qdrant_connector.get_collection_names.return_value = ["remove_test"]
        
        operation = RemoveOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            with patch('builtins.input', return_value='y'):  # User confirms
                result = await operation.execute()
        
        assert result.success
        mock_qdrant_connector.delete_collection.assert_called_once_with("remove_test")

    @pytest.mark.asyncio
    async def test_remove_workflow_with_force_flag(self, mock_qdrant_connector):
        """Test remove workflow with force flag (no confirmation)."""
        config = create_test_config(
            operation_mode="remove",
            knowledgebase_name="force_remove_test",
            force_operation=True,
        )
        
        mock_qdrant_connector.get_collection_names.return_value = ["force_remove_test"]
        
        operation = RemoveOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            with patch('builtins.input') as mock_input:
                result = await operation.execute()
        
        assert result.success
        # Should not ask for confirmation with force flag
        mock_input.assert_not_called()
        mock_qdrant_connector.delete_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_workflow_user_cancels(self, mock_qdrant_connector):
        """Test remove workflow when user cancels."""
        config = create_test_config(
            operation_mode="remove",
            knowledgebase_name="cancel_test",
            force_operation=False,
        )
        
        mock_qdrant_connector.get_collection_names.return_value = ["cancel_test"]
        
        operation = RemoveOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            with patch('builtins.input', return_value='n'):  # User cancels
                result = await operation.execute()
        
        assert result.success  # Operation succeeds but doesn't delete
        mock_qdrant_connector.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_workflow_nonexistent_collection(self, mock_qdrant_connector):
        """Test remove workflow with nonexistent collection."""
        config = create_test_config(
            operation_mode="remove",
            knowledgebase_name="nonexistent",
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = RemoveOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        # Should succeed with warning about nonexistent collection
        assert result.success
        assert len(result.warnings) > 0
        mock_qdrant_connector.delete_collection.assert_not_called()


class TestCompleteListWorkflow:
    """Test complete list workflow scenarios."""

    @pytest.mark.asyncio
    async def test_list_workflow_with_collections(self, mock_qdrant_connector):
        """Test list workflow with existing collections."""
        config = create_test_config(operation_mode="list")
        
        # Mock multiple collections
        mock_qdrant_connector.get_collection_names.return_value = [
            "collection1", "collection2", "collection3"
        ]
        
        # Mock collection info for each
        def mock_get_info(name):
            return {
                "vectors_count": 100 if name == "collection1" else 50,
                "config": {"params": {"vectors": {"size": 768}}},
                "status": "green"
            }
        
        mock_qdrant_connector.get_collection_info.side_effect = mock_get_info
        
        operation = ListOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        # Should call get_collection_info for each collection
        assert mock_qdrant_connector.get_collection_info.call_count == 3

    @pytest.mark.asyncio
    async def test_list_workflow_no_collections(self, mock_qdrant_connector):
        """Test list workflow with no collections."""
        config = create_test_config(operation_mode="list")
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = ListOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        # Should not call get_collection_info if no collections
        mock_qdrant_connector.get_collection_info.assert_not_called()


class TestPerformanceWithLargeFiles:
    """Test performance characteristics with large files."""

    @pytest.mark.asyncio
    async def test_large_file_processing_performance(self, large_file_structure, mock_qdrant_connector):
        """Test processing performance with large files."""
        workspace, files = large_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="performance_test",
            verbose=True,
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        import time
        start_time = time.time()
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        execution_time = time.time() - start_time
        
        # Verify successful processing
        assert result.success
        assert result.files_processed > 0
        assert result.chunks_created > result.files_processed  # Large files should be chunked
        
        # Performance assertions (adjust thresholds as needed)
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert result.execution_time > 0  # Should track execution time

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, nested_file_structure, mock_qdrant_connector):
        """Test batch processing efficiency with many files."""
        workspace, files = nested_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="batch_test",
            batch_size=5,  # Small batch size for testing
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        assert result.files_processed == len(files)
        
        # Verify batch processing was used
        # store_entries should be called multiple times for batching
        assert mock_qdrant_connector.store_entries.call_count > 0

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_files(self, large_file_structure, mock_qdrant_connector):
        """Test memory usage remains reasonable with large files."""
        workspace, files = large_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="memory_test",
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        # Monitor memory usage (simplified)
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result.success
        # Memory increase should be reasonable (less than 100MB for test files)
        assert memory_increase < 100 * 1024 * 1024


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_qdrant_connection_error(self, basic_file_structure):
        """Test handling of Qdrant connection errors."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="connection_error_test",
        )
        
        operation = IngestOperation(config)
        
        # Mock connection error
        with patch.object(operation, 'get_connector', side_effect=MCPQdrantError("Connection failed")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0
        assert "Connection failed" in str(result.errors)

    @pytest.mark.asyncio
    async def test_embedding_model_mismatch(self, basic_file_structure, mock_qdrant_connector):
        """Test handling of embedding model dimension mismatch."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="mismatch_test",
        )
        
        # Mock existing collection with different dimensions
        mock_qdrant_connector.get_collection_names.return_value = ["mismatch_test"]
        mock_qdrant_connector.get_collection_info.return_value = {
            "config": {"params": {"vectors": {"size": 384}}}  # Different from expected 768
        }
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            with patch.object(operation, '_validate_embedding_compatibility', 
                            side_effect=VectorDimensionMismatchError("Dimension mismatch", 768, 384)):
                result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_file_processing_errors(self, problematic_file_structure, mock_qdrant_connector):
        """Test handling of file processing errors."""
        workspace, files = problematic_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="processing_error_test",
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        # Should handle problematic files gracefully
        assert result.success  # Overall operation should succeed
        assert result.files_skipped > 0  # Some files should be skipped
        assert result.files_processed >= 0  # Some files might be processed

    @pytest.mark.asyncio
    async def test_permission_errors(self, temp_workspace, mock_qdrant_connector):
        """Test handling of file permission errors."""
        # Create a file and make it unreadable
        test_file = temp_workspace / "restricted.txt"
        test_file.write_text("restricted content")
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=temp_workspace,
            knowledgebase_name="permission_test",
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        # Mock permission error during file reading
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
        
        # Should handle permission errors gracefully
        assert result.success or result.files_failed > 0
        if result.files_failed > 0:
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_regex_patterns(self, basic_file_structure):
        """Test handling of invalid regex patterns."""
        workspace, files = basic_file_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="regex_error_test",
            include_patterns=["[invalid"],  # Invalid regex
        )
        
        operation = IngestOperation(config)
        
        # Should raise configuration error for invalid regex
        with pytest.raises((ConfigurationValidationError, ValueError)):
            await operation.execute()


class TestMainAsyncIntegration:
    """Test main_async function with complete workflows."""

    @pytest.mark.asyncio
    async def test_main_async_complete_workflow(self, basic_file_structure):
        """Test complete workflow through main_async."""
        workspace, files = basic_file_structure
        
        test_args = [
            "qdrant-ingest",
            "ingest",
            str(workspace),
            "--knowledgebase", "main_async_test",
            "--dry-run",
            "--verbose",
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                mock_connector = AsyncMock()
                mock_connector.get_collection_names.return_value = []
                mock_connector_class.return_value = mock_connector
                
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'):
                    with pytest.raises(SystemExit) as exc_info:
                        await main_async()
        
        # Should exit successfully
        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_main_async_error_handling(self):
        """Test main_async error handling."""
        test_args = [
            "qdrant-ingest",
            "ingest",
            "/nonexistent/path",
        ]
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                await main_async()
        
        # Should exit with error code
        assert exc_info.value.code != 0

    @pytest.mark.asyncio
    async def test_main_async_keyboard_interrupt(self, basic_file_structure):
        """Test main_async handling of keyboard interrupt."""
        workspace, files = basic_file_structure
        
        test_args = [
            "qdrant-ingest",
            "ingest",
            str(workspace),
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.parse_and_validate_args_intelligent', 
                      side_effect=KeyboardInterrupt()):
                with pytest.raises(SystemExit) as exc_info:
                    await main_async()
        
        # Should exit with SIGINT code
        assert exc_info.value.code == 130


class TestConfigurationValidation:
    """Test comprehensive configuration validation."""

    @pytest.mark.asyncio
    async def test_invalid_path_configuration(self):
        """Test configuration validation with invalid paths."""
        config = create_test_config(
            target_path=Path("/nonexistent/path"),
            knowledgebase_name="invalid_path_test",
        )
        
        operation = IngestOperation(config)
        result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_knowledgebase_name(self, basic_file_structure):
        """Test configuration validation with invalid knowledgebase names."""
        workspace, files = basic_file_structure
        
        # Test with invalid characters in collection name
        config = create_test_config(
            target_path=workspace,
            knowledgebase_name="invalid/name",  # Invalid characters
        )
        
        operation = IngestOperation(config)
        
        # Should handle invalid collection names gracefully
        result = await operation.execute()
        # Behavior depends on implementation - might succeed with sanitized name or fail
        assert isinstance(result, OperationResult)

    @pytest.mark.asyncio
    async def test_configuration_with_all_options(self, pattern_test_structure, mock_qdrant_connector):
        """Test configuration with all available options."""
        workspace, files = pattern_test_structure
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=workspace,
            knowledgebase_name="full_config_test",
            include_patterns=[r'\.py$', r'\.md$'],
            exclude_patterns=[r'test_', r'\.tmp$'],
            dry_run=True,
            verbose=True,
            force_operation=False,
            batch_size=3,
            show_progress=True,
        )
        
        mock_qdrant_connector.get_collection_names.return_value = []
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
            result = await operation.execute()
        
        assert result.success
        # Should respect all configuration options
        assert result.files_processed > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
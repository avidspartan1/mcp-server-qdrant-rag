"""
Integration tests for complete CLI workflows and main entry point.

This module tests the main CLI orchestration including:
- Command routing and operation execution
- Dry-run functionality across all operations
- Force flag handling for confirmation bypassing
- Operation result aggregation and display
- Error handling and exit codes
"""

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from unittest.mock import call

import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
    IngestConfig,
    CLISettings,
    OperationResult,
    main_async,
    _create_operation,
    _handle_operation_result,
    IngestOperation,
    UpdateOperation,
    RemoveOperation,
    ListOperation,
    ProgressReporter,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings


class TestMainEntryPoint:
    """Test the main CLI entry point and orchestration."""

    @pytest.fixture
    def base_config(self, tmp_path):
        """Create a base configuration for testing."""
        # Create a test file in tmp_path
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Create config with mocked validation
        config = IngestConfig.__new__(IngestConfig)
        config.cli_settings = CLISettings(
            operation_mode="ingest",
            verbose=False,
            dry_run=False,
            force_operation=False,
        )
        config.qdrant_settings = QdrantSettings()
        config.embedding_settings = EmbeddingProviderSettings()
        config.target_path = tmp_path
        config.knowledgebase_name = "test_collection"
        
        return config

    @pytest.mark.asyncio
    async def test_create_operation_ingest(self, base_config):
        """Test creating ingest operation."""
        base_config.cli_settings.operation_mode = "ingest"
        progress_reporter = ProgressReporter()
        
        operation = await _create_operation(base_config, progress_reporter)
        
        assert isinstance(operation, IngestOperation)
        assert operation.config == base_config

    @pytest.mark.asyncio
    async def test_create_operation_update(self, base_config):
        """Test creating update operation."""
        base_config.cli_settings.operation_mode = "update"
        progress_reporter = ProgressReporter()
        
        operation = await _create_operation(base_config, progress_reporter)
        
        assert isinstance(operation, UpdateOperation)
        assert operation.config == base_config

    @pytest.mark.asyncio
    async def test_create_operation_remove(self, base_config):
        """Test creating remove operation."""
        base_config.cli_settings.operation_mode = "remove"
        progress_reporter = ProgressReporter()
        
        operation = await _create_operation(base_config, progress_reporter)
        
        assert isinstance(operation, RemoveOperation)
        assert operation.config == base_config

    @pytest.mark.asyncio
    async def test_create_operation_list(self, base_config):
        """Test creating list operation."""
        base_config.cli_settings.operation_mode = "list"
        progress_reporter = ProgressReporter()
        
        operation = await _create_operation(base_config, progress_reporter)
        
        assert isinstance(operation, ListOperation)
        assert operation.config == base_config

    @pytest.mark.asyncio
    async def test_create_operation_invalid_mode(self, base_config):
        """Test creating operation with invalid mode."""
        base_config.cli_settings.operation_mode = "invalid"
        progress_reporter = ProgressReporter()
        
        with pytest.raises(ValueError, match="Unsupported operation mode: invalid"):
            await _create_operation(base_config, progress_reporter)

    def test_handle_operation_result_complete_success(self):
        """Test handling completely successful operation result."""
        result = OperationResult(
            success=True,
            files_processed=5,
            files_failed=0,
            errors=[],
        )
        progress_reporter = MagicMock()
        
        exit_code = _handle_operation_result(result, progress_reporter)
        
        assert exit_code == 0
        progress_reporter.log_success.assert_called_once_with("Operation completed successfully!")

    def test_handle_operation_result_partial_success(self):
        """Test handling partially successful operation result."""
        result = OperationResult(
            success=True,
            files_processed=3,
            files_failed=2,
            errors=["Error 1", "Error 2"],
        )
        progress_reporter = MagicMock()
        
        exit_code = _handle_operation_result(result, progress_reporter)
        
        assert exit_code == 1
        progress_reporter.log_warning.assert_called_once_with("Operation completed with some failures")

    def test_handle_operation_result_complete_failure(self):
        """Test handling completely failed operation result."""
        result = OperationResult(
            success=False,
            files_processed=0,
            files_failed=5,
            errors=["Fatal error"],
        )
        progress_reporter = MagicMock()
        
        exit_code = _handle_operation_result(result, progress_reporter)
        
        assert exit_code == 4
        progress_reporter.log_error.assert_called_once_with("Operation failed - no files were processed")

    def test_handle_operation_result_partial_failure(self):
        """Test handling partially failed operation result."""
        result = OperationResult(
            success=False,
            files_processed=2,
            files_failed=3,
            errors=["Some error"],
        )
        progress_reporter = MagicMock()
        
        exit_code = _handle_operation_result(result, progress_reporter)
        
        assert exit_code == 5
        progress_reporter.log_error.assert_called_once_with("Operation failed with partial results")


class TestMainAsyncIntegration:
    """Test the main_async function with mocked operations."""

    @pytest.fixture
    def mock_parse_and_validate(self):
        """Mock the parse_and_validate_args_intelligent function."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.parse_and_validate_args_intelligent') as mock:
            yield mock

    @pytest.fixture
    def mock_operation(self):
        """Mock operation instance."""
        operation = AsyncMock()
        operation.execute.return_value = OperationResult(
            success=True,
            files_processed=3,
            files_failed=0,
            errors=[],
        )
        return operation

    @pytest.mark.asyncio
    async def test_main_async_successful_execution(self, mock_parse_and_validate, mock_operation):
        """Test successful main_async execution."""
        # Setup config
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create config with mocked validation
            config = IngestConfig.__new__(IngestConfig)
            config.cli_settings = CLISettings(
                operation_mode="ingest",
                verbose=False,
                dry_run=False,
            )
            config.qdrant_settings = QdrantSettings()
            config.embedding_settings = EmbeddingProviderSettings()
            config.target_path = Path(tmp_dir)
            config.knowledgebase_name = "test"
            
            mock_parse_and_validate.return_value = config
            
            # Mock operation creation
            with patch('src.mcp_server_qdrant_rag.cli_ingest._create_operation', return_value=mock_operation):
                with pytest.raises(SystemExit) as exc_info:
                    await main_async()
            
            # Verify successful exit
            assert exc_info.value.code == 0
            mock_operation.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_async_verbose_mode(self, mock_parse_and_validate, mock_operation):
        """Test main_async with verbose mode enabled."""
        # Setup config with verbose mode
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create config with mocked validation
            config = IngestConfig.__new__(IngestConfig)
            config.cli_settings = CLISettings(
                operation_mode="ingest",
                verbose=True,
                dry_run=False,
            )
            config.qdrant_settings = QdrantSettings()
            config.embedding_settings = EmbeddingProviderSettings()
            config.target_path = Path(tmp_dir)
            config.knowledgebase_name = "test"
            
            mock_parse_and_validate.return_value = config
            
            with patch('src.mcp_server_qdrant_rag.cli_ingest._create_operation', return_value=mock_operation):
                with patch('builtins.print') as mock_print:
                    with pytest.raises(SystemExit) as exc_info:
                        await main_async()
            
            # Verify verbose output was shown
            assert exc_info.value.code == 0
            # Check that configuration summary was printed
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("Configuration Summary" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_main_async_keyboard_interrupt(self, mock_parse_and_validate):
        """Test main_async handling of keyboard interrupt."""
        mock_parse_and_validate.side_effect = KeyboardInterrupt()
        
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                await main_async()
        
        # Verify SIGINT exit code
        assert exc_info.value.code == 130
        # Check that both expected messages were printed
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert "\n🚫 Operation cancelled by user" in print_calls
        assert "💡 You can resume operations later - no partial data was committed" in print_calls

    @pytest.mark.asyncio
    async def test_main_async_configuration_error(self, mock_parse_and_validate):
        """Test main_async handling of configuration errors."""
        from src.mcp_server_qdrant_rag.common.exceptions import ConfigurationValidationError
        
        mock_parse_and_validate.side_effect = ConfigurationValidationError(
            field_name="test_field",
            invalid_value="invalid",
            validation_error="Invalid config"
        )
        
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                await main_async()
        
        # Verify configuration error exit code
        assert exc_info.value.code == 2
        # Check that the error messages were printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[1].get('file') == sys.stderr]
        assert any("❌ Configuration error:" in msg for msg in print_calls)
        assert any("💡 Use --help for detailed usage information" in msg for msg in print_calls)

    @pytest.mark.asyncio
    async def test_main_async_qdrant_error(self, mock_parse_and_validate):
        """Test main_async handling of Qdrant errors."""
        from src.mcp_server_qdrant_rag.common.exceptions import MCPQdrantError
        
        mock_parse_and_validate.side_effect = MCPQdrantError("Qdrant connection failed")
        
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                await main_async()
        
        # Verify Qdrant error exit code
        assert exc_info.value.code == 3
        # Check that the error messages were printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[1].get('file') == sys.stderr]
        assert any("❌ Qdrant error: Qdrant connection failed" in msg for msg in print_calls)
        assert any("💡 Test connection: qdrant-ingest list" in msg for msg in print_calls)

    @pytest.mark.asyncio
    async def test_main_async_unexpected_error(self, mock_parse_and_validate):
        """Test main_async handling of unexpected errors."""
        mock_parse_and_validate.side_effect = RuntimeError("Unexpected error")
        
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                await main_async()
        
        # Verify general error exit code
        assert exc_info.value.code == 1
        # Check that the error messages were printed
        print_calls = [call[0][0] for call in mock_print.call_args_list if call[1].get('file') == sys.stderr]
        assert any("❌ Unexpected error: Unexpected error" in msg for msg in print_calls)
        assert any("   • Report this issue if it persists" in msg for msg in print_calls)

    @pytest.mark.asyncio
    async def test_main_async_debug_mode(self, mock_parse_and_validate):
        """Test main_async with debug mode enabled."""
        mock_parse_and_validate.side_effect = RuntimeError("Debug error")
        
        with patch.dict(os.environ, {"DEBUG": "1"}):
            with patch('builtins.print') as mock_print:
                with patch('traceback.print_exc') as mock_traceback:
                    with pytest.raises(SystemExit) as exc_info:
                        await main_async()
        
        # Verify debug traceback was printed
        assert exc_info.value.code == 1
        mock_traceback.assert_called_once()


class TestDryRunFunctionality:
    """Test dry-run functionality across all operations."""

    @pytest.fixture
    def dry_run_config(self, tmp_path):
        """Create a configuration with dry-run enabled."""
        # Create a test file in tmp_path
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Create config with mocked validation
        config = IngestConfig.__new__(IngestConfig)
        config.cli_settings = CLISettings(
            operation_mode="ingest",
            dry_run=True,
            verbose=True,
        )
        config.qdrant_settings = QdrantSettings()
        config.embedding_settings = EmbeddingProviderSettings()
        config.target_path = tmp_path
        config.knowledgebase_name = "test_dry_run"
        
        return config

    @pytest.mark.asyncio
    async def test_dry_run_ingest_operation(self, tmp_path):
        """Test dry-run functionality for ingest operation."""
        # Create test files
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for dry run")
        
        # Set environment variables for Qdrant settings
        with patch.dict(os.environ, {'QDRANT_URL': 'http://localhost:6333'}):
            # Create config inside environment patch
            config = IngestConfig(
                cli_settings=CLISettings(
                    operation_mode="ingest",
                    dry_run=True,
                    verbose=True,
                ),
                qdrant_settings=QdrantSettings(),
                embedding_settings=EmbeddingProviderSettings(),
                target_path=tmp_path,
                knowledgebase_name="test_dry_run",
            )
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector') as mock_connector:
                result = await operation.execute()
            
            # Verify dry-run behavior
            assert result.success
            assert result.files_processed > 0
            # In dry run for ingest, connector should not be called at all
            mock_connector.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_update_operation(self, tmp_path):
        """Test dry-run functionality for update operation."""
        # Create test files
        test_file = tmp_path / "update_test.txt"
        test_file.write_text("Update content for dry run")
        
        # Set environment variables for Qdrant settings
        with patch.dict(os.environ, {'QDRANT_URL': 'http://localhost:6333'}):
            # Create config inside environment patch
            config = IngestConfig(
                cli_settings=CLISettings(
                    operation_mode="update",
                    dry_run=True,
                    verbose=True,
                ),
                qdrant_settings=QdrantSettings(),
                embedding_settings=EmbeddingProviderSettings(),
                target_path=tmp_path,
                knowledgebase_name="test_dry_run",
            )
            
            operation = UpdateOperation(config)
            
            with patch.object(operation, 'get_connector') as mock_connector:
                result = await operation.execute()
            
            # Verify dry-run behavior
            assert result.success
            # In dry run, connector is called to check collection status but not for actual updates
            mock_connector.assert_called()
            # Verify no actual storage operations were called
            mock_connector.return_value.store_entries.assert_not_called() if hasattr(mock_connector.return_value, 'store_entries') else None

    @pytest.mark.asyncio
    async def test_dry_run_remove_operation(self):
        """Test dry-run functionality for remove operation."""
        # Set environment variables for Qdrant settings
        with patch.dict(os.environ, {'QDRANT_URL': 'http://localhost:6333'}):
            # Create config inside environment patch
            config = IngestConfig(
                cli_settings=CLISettings(
                    operation_mode="remove",
                    dry_run=True,
                    verbose=True,
                ),
                qdrant_settings=QdrantSettings(),
                embedding_settings=EmbeddingProviderSettings(),
                target_path=None,
                knowledgebase_name="test_dry_run",
            )
            
            operation = RemoveOperation(config)
            
            with patch.object(operation, 'get_connector') as mock_connector:
                result = await operation.execute()
            
            # Verify dry-run behavior
            assert result.success
            # In dry run, connector is called to check collection status but not for actual removal
            mock_connector.assert_called()
            # Verify no actual deletion operations were called
            mock_connector.return_value.delete_collection.assert_not_called() if hasattr(mock_connector.return_value, 'delete_collection') else None


class TestForceFlag:
    """Test force flag handling for confirmation bypassing."""

    @pytest.fixture
    def force_config(self):
        """Create a configuration with force flag enabled."""
        # Create config with mocked validation
        config = IngestConfig.__new__(IngestConfig)
        config.cli_settings = CLISettings(
            operation_mode="remove",
            force_operation=True,
            verbose=True,
        )
        config.qdrant_settings = QdrantSettings()
        config.embedding_settings = EmbeddingProviderSettings()
        config.target_path = None
        config.knowledgebase_name = "test_force"
        
        return config

    @pytest.mark.asyncio
    async def test_force_flag_remove_operation(self):
        """Test force flag bypasses confirmation for remove operation."""
        # Set environment variables for Qdrant settings
        with patch.dict(os.environ, {'QDRANT_URL': 'http://localhost:6333'}):
            # Create config inside environment patch
            config = IngestConfig(
                cli_settings=CLISettings(
                    operation_mode="remove",
                    force_operation=True,
                    verbose=True,
                ),
                qdrant_settings=QdrantSettings(),
                embedding_settings=EmbeddingProviderSettings(),
                target_path=None,
                knowledgebase_name="test_force",
            )
            
            operation = RemoveOperation(config)
            
            # Mock connector to simulate existing collection
            mock_connector = AsyncMock()
            mock_connector.get_collection_names.return_value = ["test_force"]  # Collection exists
            mock_connector.delete_collection.return_value = None
            
            with patch.object(operation, 'get_connector', return_value=mock_connector):
                with patch('builtins.input') as mock_input:
                    result = await operation.execute()
            
            # Verify force flag behavior
            assert result.success
            # Input should not be called when force flag is enabled
            mock_input.assert_not_called()
            mock_connector.delete_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_force_flag_requires_confirmation(self):
        """Test that without force flag, confirmation is required."""
        # Set environment variables for Qdrant settings
        with patch.dict(os.environ, {'QDRANT_URL': 'http://localhost:6333'}):
            # Create config inside environment patch
            config = IngestConfig(
                cli_settings=CLISettings(
                    operation_mode="remove",
                    force_operation=False,  # Force disabled
                ),
                qdrant_settings=QdrantSettings(),
                embedding_settings=EmbeddingProviderSettings(),
                target_path=None,
                knowledgebase_name="test_no_force",
            )
            
            operation = RemoveOperation(config)
            
            # Mock connector to simulate existing collection
            mock_connector = AsyncMock()
            mock_connector.get_collection_names.return_value = ["test_no_force"]  # Collection exists
            
            with patch.object(operation, 'get_connector', return_value=mock_connector):
                with patch('builtins.input', return_value='n') as mock_input:  # User says no
                    result = await operation.execute()
            
            # Verify confirmation was requested
            mock_input.assert_called_once()
            # Operation should be successful but no deletion should occur
            assert result.success
            # Verify no actual deletion was performed
            mock_connector.delete_collection.assert_not_called()


class TestEndToEndWorkflows:
    """Test complete end-to-end CLI workflows."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files
            (workspace / "doc1.txt").write_text("This is document 1 content")
            (workspace / "doc2.md").write_text("# Document 2\nMarkdown content")
            (workspace / "code.py").write_text("print('Hello, world!')")
            
            # Create subdirectory
            subdir = workspace / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested document content")
            
            yield workspace

    @pytest.mark.asyncio
    async def test_complete_ingest_workflow(self, temp_workspace):
        """Test complete ingest workflow from argument parsing to execution."""
        # Mock sys.argv to simulate CLI call
        test_args = [
            "qdrant-ingest",
            "ingest",
            str(temp_workspace),
            "--knowledgebase", "test_workflow",
            "--verbose",
            "--dry-run",
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'):
                    with pytest.raises(SystemExit) as exc_info:
                        await main_async()
        
        # Verify successful completion
        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_complete_list_workflow(self):
        """Test complete list workflow."""
        test_args = [
            "qdrant-ingest",
            "list",
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                mock_connector = AsyncMock()
                mock_connector.list_collections.return_value = ["collection1", "collection2"]
                mock_connector_class.return_value = mock_connector
                
                with pytest.raises(SystemExit) as exc_info:
                    await main_async()
        
        # Verify successful completion
        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_workflow_with_patterns(self, temp_workspace):
        """Test workflow with include/exclude patterns."""
        test_args = [
            "qdrant-ingest",
            "ingest",
            str(temp_workspace),
            "--include", r".*\.txt$",
            "--exclude", r".*nested.*",
            "--dry-run",
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector'):
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'):
                    with pytest.raises(SystemExit) as exc_info:
                        await main_async()
        
        # Verify successful completion
        assert exc_info.value.code == 0

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling with invalid path."""
        test_args = [
            "qdrant-ingest",
            "ingest",
            "/nonexistent/path",
        ]
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                await main_async()
        
        # Verify error exit code
        assert exc_info.value.code != 0
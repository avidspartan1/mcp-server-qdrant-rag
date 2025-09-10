"""
Unit tests for CLI error handling functionality.

This module tests the comprehensive error handling system including:
- CLIErrorHandler class functionality
- Error categorization and recovery strategies
- Retry mechanisms for transient failures
- Error reporting and user guidance
"""

import pytest
import asyncio
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Optional

from src.mcp_server_qdrant_rag.cli_ingest import (
    CLIErrorHandler,
    ProgressReporter,
    IngestConfig,
    CLISettings,
    IngestOperation,
    FileInfo,
    OperationResult,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings
from src.mcp_server_qdrant_rag.common.exceptions import (
    MCPQdrantError,
    ModelValidationError,
    VectorDimensionMismatchError,
    ChunkingError,
    ConfigurationValidationError,
    CollectionAccessError,
)


class TestCLIErrorHandler:
    """Test cases for the CLIErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.progress_reporter = Mock(spec=ProgressReporter)
        self.error_handler = CLIErrorHandler(self.progress_reporter)
    
    def test_initialization(self):
        """Test error handler initialization."""
        assert self.error_handler.progress_reporter == self.progress_reporter
        assert self.error_handler.max_retry_attempts == 3
        assert self.error_handler.retry_delay == 1.0
        assert all(count == 0 for count in self.error_handler.error_counts.values())
        assert len(self.error_handler.recovery_attempts) == 0
    
    def test_handle_configuration_error_validation_error(self):
        """Test handling of configuration validation errors."""
        error = ConfigurationValidationError(
            field_name="embedding_model",
            invalid_value="invalid-model",
            validation_error="Model not supported",
            valid_options=["model1", "model2"],
            suggested_value="model1"
        )
        
        result = self.error_handler.handle_configuration_error(error, "test context")
        
        assert not result  # Configuration errors are fatal
        assert self.error_handler.error_counts['configuration'] == 1
        self.progress_reporter.log_error.assert_called()
        self.progress_reporter.log_info.assert_called()
    
    def test_handle_configuration_error_value_error(self):
        """Test handling of ValueError in configuration."""
        error = ValueError("Target path does not exist: /invalid/path")
        
        result = self.error_handler.handle_configuration_error(error)
        
        assert not result  # Configuration errors are fatal
        assert self.error_handler.error_counts['configuration'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_connection_error_first_attempt(self):
        """Test handling of connection error on first attempt."""
        error = ConnectionError("Connection refused")
        qdrant_url = "http://localhost:6333"
        
        result = self.error_handler.handle_connection_error(error, qdrant_url, "test operation")
        
        assert result  # Should retry
        assert self.error_handler.error_counts['connection'] == 1
        assert self.error_handler.recovery_attempts[f"connection_{qdrant_url}"] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_connection_error_max_retries_exceeded(self):
        """Test handling of connection error after max retries."""
        error = ConnectionError("Connection refused")
        qdrant_url = "http://localhost:6333"
        error_key = f"connection_{qdrant_url}"
        
        # Simulate max retries already reached
        self.error_handler.recovery_attempts[error_key] = 3
        
        result = self.error_handler.handle_connection_error(error, qdrant_url)
        
        assert not result  # Should not retry
        assert self.error_handler.error_counts['connection'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_connection_error_specific_patterns(self):
        """Test handling of specific connection error patterns."""
        test_cases = [
            ("Connection refused", "Qdrant server may not be running"),
            ("timeout", "Connection timeout"),
            ("authentication failed", "Authentication failed"),
            ("not found", "Server not found"),
        ]
        
        for error_msg, expected_guidance in test_cases:
            error_handler = CLIErrorHandler(Mock(spec=ProgressReporter))
            error = ConnectionError(error_msg)
            
            result = error_handler.handle_connection_error(error, "http://localhost:6333")
            
            assert result  # Should retry on first attempt
            error_handler.progress_reporter.log_info.assert_called()
    
    def test_handle_file_processing_error_permission_denied(self):
        """Test handling of permission denied errors."""
        error = PermissionError("Permission denied")
        file_path = Path("/test/file.txt")
        
        result = self.error_handler.handle_file_processing_error(error, file_path, "reading")
        
        assert result  # Should continue with other files
        assert self.error_handler.error_counts['file_processing'] == 1
        self.progress_reporter.log_error.assert_called()
        self.progress_reporter.log_warning.assert_called()
    
    def test_handle_file_processing_error_file_not_found(self):
        """Test handling of file not found errors."""
        error = FileNotFoundError("File not found")
        file_path = Path("/test/missing.txt")
        
        result = self.error_handler.handle_file_processing_error(error, file_path)
        
        assert result  # Should continue with other files
        assert self.error_handler.error_counts['file_processing'] == 1
        self.progress_reporter.log_warning.assert_called()
    
    def test_handle_file_processing_error_unicode_decode(self):
        """Test handling of Unicode decode errors."""
        error = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
        file_path = Path("/test/binary.bin")
        
        result = self.error_handler.handle_file_processing_error(error, file_path)
        
        assert result  # Should continue with other files
        assert self.error_handler.error_counts['file_processing'] == 1
        self.progress_reporter.log_warning.assert_called()
    
    def test_handle_storage_error_dimension_mismatch(self):
        """Test handling of vector dimension mismatch errors."""
        error = VectorDimensionMismatchError(
            collection_name="test_collection",
            expected_dimensions=768,
            actual_dimensions=384,
            model_name="test-model",
            vector_name="test-vector"
        )
        
        result = self.error_handler.handle_storage_error(error, "test entry", "test_collection")
        
        assert not result  # Dimension mismatches are fatal
        assert self.error_handler.error_counts['storage'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_storage_error_collection_access(self):
        """Test handling of collection access errors."""
        error = CollectionAccessError(
            collection_name="test_collection",
            operation="store",
            original_error=Exception("Access denied"),
            available_collections=["other_collection"]
        )
        
        result = self.error_handler.handle_storage_error(error, "test entry", "test_collection")
        
        assert not result  # Collection access errors are fatal
        assert self.error_handler.error_counts['storage'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_storage_error_chunking_with_fallback(self):
        """Test handling of chunking errors with fallback."""
        error = ChunkingError(
            original_error=Exception("Chunking failed"),
            document_length=1000,
            chunking_config={},
            fallback_used=True
        )
        
        result = self.error_handler.handle_storage_error(error, "test entry", "test_collection")
        
        assert result  # Should continue with fallback
        assert self.error_handler.error_counts['storage'] == 1
        self.progress_reporter.log_warning.assert_called()
    
    def test_handle_storage_error_transient_with_retry(self):
        """Test handling of transient storage errors with retry logic."""
        error = Exception("Connection timeout - temporary failure")
        entry_info = "test entry"
        collection_name = "test_collection"
        
        # First attempt should allow retry
        result = self.error_handler.handle_storage_error(error, entry_info, collection_name, retry_possible=True)
        
        assert result  # Should retry
        assert self.error_handler.error_counts['transient'] == 1
        error_key = f"storage_{collection_name}_{entry_info}"
        assert self.error_handler.recovery_attempts[error_key] == 1
    
    def test_handle_validation_error_model_validation(self):
        """Test handling of model validation errors."""
        error = ModelValidationError(
            model_name="invalid-model",
            available_models=["model1", "model2", "model3"],
            suggestion="Use model1 instead"
        )
        
        result = self.error_handler.handle_validation_error(error, "embedding model selection")
        
        assert not result  # Model validation errors are fatal
        assert self.error_handler.error_counts['validation'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_validation_error_regex_error(self):
        """Test handling of regex validation errors."""
        error = re.error("Invalid regex pattern")
        
        result = self.error_handler.handle_validation_error(error, "include patterns")
        
        assert not result  # Regex errors are fatal
        assert self.error_handler.error_counts['validation'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_embedding_error_model_not_found(self):
        """Test handling of embedding model not found errors."""
        error = Exception("Model 'invalid-model' not found or not available")
        model_name = "invalid-model"
        
        result = self.error_handler.handle_embedding_error(error, model_name)
        
        assert not result  # Model not found is fatal
        assert self.error_handler.error_counts['unknown'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_embedding_error_download_failure(self):
        """Test handling of embedding model download failures."""
        error = Exception("Failed to download model due to network error")
        model_name = "test-model"
        
        result = self.error_handler.handle_embedding_error(error, model_name)
        
        assert not result  # Download failures are fatal
        self.progress_reporter.log_error.assert_called()
    
    def test_handle_unknown_error(self):
        """Test handling of unknown/unexpected errors."""
        error = RuntimeError("Unexpected runtime error")
        context = "test operation"
        
        result = self.error_handler.handle_unknown_error(error, context)
        
        assert not result  # Unknown errors are treated as fatal
        assert self.error_handler.error_counts['unknown'] == 1
        self.progress_reporter.log_error.assert_called()
    
    def test_should_continue_processing_with_configuration_errors(self):
        """Test should_continue_processing with configuration errors."""
        self.error_handler.error_counts['configuration'] = 1
        
        result = self.error_handler.should_continue_processing()
        
        assert not result  # Should stop with configuration errors
    
    def test_should_continue_processing_with_many_connection_errors(self):
        """Test should_continue_processing with many connection errors."""
        self.error_handler.error_counts['connection'] = 6
        
        result = self.error_handler.should_continue_processing()
        
        assert not result  # Should stop with too many connection errors
    
    def test_should_continue_processing_with_many_unknown_errors(self):
        """Test should_continue_processing with many unknown errors."""
        self.error_handler.error_counts['unknown'] = 11
        
        result = self.error_handler.should_continue_processing()
        
        assert not result  # Should stop with too many unknown errors
    
    def test_should_continue_processing_with_file_processing_errors(self):
        """Test should_continue_processing with file processing errors."""
        self.error_handler.error_counts['file_processing'] = 5
        
        result = self.error_handler.should_continue_processing()
        
        assert result  # Should continue with file processing errors
    
    def test_get_error_summary(self):
        """Test getting error summary with recommendations."""
        # Simulate various error types
        self.error_handler.error_counts['configuration'] = 1
        self.error_handler.error_counts['connection'] = 2
        self.error_handler.error_counts['file_processing'] = 8
        self.error_handler.error_counts['storage'] = 1
        self.error_handler.error_counts['transient'] = 3
        self.error_handler.recovery_attempts = {"test1": 1, "test2": 2}
        
        summary = self.error_handler.get_error_summary()
        
        assert summary['total_errors'] == 15
        assert summary['error_counts']['configuration'] == 1
        assert summary['error_counts']['connection'] == 2
        assert summary['retry_attempts'] == 2
        assert len(summary['recommendations']) > 0
        
        # Check that recommendations are appropriate
        recommendations = summary['recommendations']
        assert any("configuration" in rec.lower() for rec in recommendations)
        assert any("qdrant" in rec.lower() or "connectivity" in rec.lower() for rec in recommendations)
        assert any("file" in rec.lower() for rec in recommendations)
        assert any("qdrant collection" in rec.lower() or "available space" in rec.lower() for rec in recommendations)
        assert any("transient" in rec.lower() for rec in recommendations)
    
    @patch('time.sleep')
    def test_connection_error_retry_with_backoff(self, mock_sleep):
        """Test connection error retry with exponential backoff."""
        error = Exception("Connection timeout")
        qdrant_url = "http://localhost:6333"
        
        # First retry
        result1 = self.error_handler.handle_connection_error(error, qdrant_url)
        assert result1
        mock_sleep.assert_called_with(1.0)
        
        # Second retry
        result2 = self.error_handler.handle_connection_error(error, qdrant_url)
        assert result2
        mock_sleep.assert_called_with(1.0)
        
        # Third retry (should fail)
        result3 = self.error_handler.handle_connection_error(error, qdrant_url)
        assert not result3
    
    def test_error_handler_without_progress_reporter(self):
        """Test error handler functionality without progress reporter."""
        error_handler = CLIErrorHandler(None)
        
        # Should not raise exceptions when logging
        error_handler._log_error("test error")
        error_handler._log_warning("test warning")
        error_handler._log_info("test info")
        
        # Should still handle errors correctly
        error = ValueError("test error")
        result = error_handler.handle_configuration_error(error)
        assert not result
        assert error_handler.error_counts['configuration'] == 1


class TestErrorHandlerIntegration:
    """Integration tests for error handler with CLI operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        import tempfile
        import os
        
        self.cli_settings = CLISettings()
        # Use environment variable approach for QdrantSettings
        os.environ["QDRANT_URL"] = "http://localhost:6333"
        self.qdrant_settings = QdrantSettings()
        self.embedding_settings = EmbeddingProviderSettings()
        
        # Create a temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.config = IngestConfig(
            cli_settings=self.cli_settings,
            qdrant_settings=self.qdrant_settings,
            embedding_settings=self.embedding_settings,
            target_path=self.temp_dir,
            knowledgebase_name="test_kb"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        import shutil
        
        if "QDRANT_URL" in os.environ:
            del os.environ["QDRANT_URL"]
        
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_ingest_operation_with_connection_error(self):
        """Test IngestOperation handling connection errors."""
        operation = IngestOperation(self.config)
        
        # Create a test file in the temp directory
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # Mock the connector creation to raise connection error
        with patch.object(operation, 'get_connector', side_effect=ConnectionError("Connection refused")):
            result = await operation.execute()
            
            assert not result.success
            assert len(result.errors) > 0
            # The connection error should be handled by the error handler
            assert "Connection refused" in str(result.errors)
    
    @pytest.mark.asyncio
    async def test_ingest_operation_with_file_processing_errors(self):
        """Test IngestOperation handling file processing errors."""
        operation = IngestOperation(self.config)
        
        # Create test files in the temp directory
        test_file1 = self.temp_dir / "file1.txt"
        test_file2 = self.temp_dir / "file2.txt"
        test_file1.write_text("test content 1")
        test_file2.write_text("test content 2")
        
        # Mock content processor to raise errors for some files
        with patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor') as mock_processor:
            mock_processor.return_value.process_file.side_effect = [
                PermissionError("Permission denied"),  # First file fails
                Mock(content="test content", metadata={})  # Second file succeeds
            ]
            mock_processor.return_value.get_processing_stats.return_value = {
                'total_entries': 1, 'average_content_length': 12,
                'file_types': {'text': 1}, 'size_categories': {'small': 1}, 'encodings': {'utf-8': 1}
            }
            
            result = await operation.execute()
            
            # Should continue processing despite file errors
            assert result.files_failed == 1
            assert result.files_processed == 1
            assert operation.error_handler.error_counts['file_processing'] == 1
    
    @pytest.mark.asyncio
    async def test_ingest_operation_with_storage_errors(self):
        """Test IngestOperation handling storage errors."""
        operation = IngestOperation(self.config)
        
        # Create a test file in the temp directory
        test_file = self.temp_dir / "file.txt"
        test_file.write_text("test content")
        
        test_entry = Mock()
        test_entry.content = "test content"
        test_entry.metadata = {"file_path": str(test_file)}
        
        with patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor') as mock_processor, \
             patch.object(operation, 'get_connector') as mock_get_connector:
            
            mock_processor.return_value.process_file.return_value = test_entry
            mock_processor.return_value.get_processing_stats.return_value = {
                'total_entries': 1, 'average_content_length': 12,
                'file_types': {'text': 1}, 'size_categories': {'small': 1}, 'encodings': {'utf-8': 1}
            }
            
            # Mock connector to raise storage error
            mock_connector = AsyncMock()
            mock_connector.store.side_effect = VectorDimensionMismatchError(
                collection_name="test_kb",
                expected_dimensions=768,
                actual_dimensions=384,
                model_name="test-model",
                vector_name="test-vector"
            )
            mock_get_connector.return_value = mock_connector
            
            result = await operation.execute()
            
            # Should handle storage error gracefully
            assert not result.success  # Fatal storage error
            assert len(result.errors) > 0
            assert operation.error_handler.error_counts['storage'] == 1


if __name__ == "__main__":
    pytest.main([__file__])
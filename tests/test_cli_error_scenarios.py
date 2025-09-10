"""
Comprehensive error scenario tests for CLI file ingestion.

This module tests error handling and edge cases including:
- File system errors (permissions, missing files, etc.)
- Qdrant connection and operation errors
- Configuration validation errors
- Encoding and processing errors
- Network and timeout errors
- Resource exhaustion scenarios
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
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
    MCPQdrantError,
    ConfigurationValidationError,
    VectorDimensionMismatchError,
    ChunkingError,
    ModelValidationError,
    CollectionAccessError,
    TokenizerError,
    SentenceSplitterError,
    BackwardCompatibilityError,
)

from tests.fixtures.cli_test_fixtures import (
    problematic_file_structure,
    mock_qdrant_connector,
    create_test_config,
    create_operation_result,
)


class TestFileSystemErrors:
    """Test handling of file system related errors."""

    @pytest.mark.asyncio
    async def test_nonexistent_path_error(self):
        """Test handling of nonexistent file paths."""
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")
        
        config = create_test_config(
            operation_mode="ingest",
            target_path=nonexistent_path,
            knowledgebase_name="nonexistent_test",
        )
        
        operation = IngestOperation(config)
        result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0
        assert "does not exist" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_permission_denied_directory(self):
        """Test handling of permission denied errors for directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            restricted_dir = workspace / "restricted"
            restricted_dir.mkdir()
            
            # Create a file in the restricted directory
            test_file = restricted_dir / "test.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=restricted_dir,
                knowledgebase_name="permission_test",
            )
            
            operation = IngestOperation(config)
            
            # Mock permission error during directory scanning
            with patch('os.walk', side_effect=PermissionError("Permission denied")):
                result = await operation.execute()
            
            # Should handle permission error gracefully
            assert not result.success or result.files_failed > 0
            if not result.success:
                assert len(result.errors) > 0
                assert "permission" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_permission_denied_file_read(self, mock_qdrant_connector):
        """Test handling of permission denied errors for file reading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "restricted.txt"
            test_file.write_text("restricted content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="file_permission_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Mock permission error during file reading
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            
            # Should handle file permission errors gracefully
            assert result.files_failed > 0 or result.files_skipped > 0
            if result.files_failed > 0:
                assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_disk_full_error(self, mock_qdrant_connector):
        """Test handling of disk full errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "test.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="disk_full_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Mock disk full error during processing
            with patch('pathlib.Path.write_text', side_effect=OSError("No space left on device")):
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            
            # Should handle disk full errors gracefully
            assert isinstance(result, OperationResult)
            # Behavior depends on when the error occurs

    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self, mock_qdrant_connector):
        """Test handling of corrupted or unreadable files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a file that will cause encoding errors
            corrupted_file = workspace / "corrupted.txt"
            with open(corrupted_file, 'wb') as f:
                # Write invalid UTF-8 sequence
                f.write(b'\xff\xfe\x00\x00invalid\x80\x81\x82')
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="corrupted_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle corrupted files gracefully
            assert result.files_skipped > 0 or result.files_failed > 0


class TestQdrantConnectionErrors:
    """Test handling of Qdrant connection and operation errors."""

    @pytest.mark.asyncio
    async def test_qdrant_connection_timeout(self):
        """Test handling of Qdrant connection timeout."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="timeout_test",
        )
        
        operation = IngestOperation(config)
        
        # Mock connection timeout
        with patch.object(operation, 'get_connector', side_effect=asyncio.TimeoutError("Connection timeout")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0
        assert "timeout" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_qdrant_server_unavailable(self):
        """Test handling of Qdrant server unavailable."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="unavailable_test",
        )
        
        operation = IngestOperation(config)
        
        # Mock server unavailable error
        with patch.object(operation, 'get_connector', side_effect=MCPQdrantError("Server unavailable")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0
        assert "unavailable" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_qdrant_authentication_error(self):
        """Test handling of Qdrant authentication errors."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="auth_test",
        )
        
        operation = IngestOperation(config)
        
        # Mock authentication error
        with patch.object(operation, 'get_connector', side_effect=MCPQdrantError("Authentication failed")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0
        assert "authentication" in str(result.errors).lower()

    @pytest.mark.asyncio
    async def test_qdrant_collection_creation_error(self, mock_qdrant_connector):
        """Test handling of collection creation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "test.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="creation_error_test",
            )
            
            # Mock collection creation error
            mock_qdrant_connector.get_collection_names.return_value = []
            mock_qdrant_connector.create_collection.side_effect = MCPQdrantError("Collection creation failed")
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            assert not result.success
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_qdrant_storage_error(self, mock_qdrant_connector):
        """Test handling of storage operation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "test.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="storage_error_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            mock_qdrant_connector.create_collection.return_value = None
            mock_qdrant_connector.store_entries.side_effect = MCPQdrantError("Storage failed")
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            assert not result.success
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_vector_dimension_mismatch(self, mock_qdrant_connector):
        """Test handling of vector dimension mismatch errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "test.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="dimension_mismatch_test",
            )
            
            # Mock existing collection with different dimensions
            mock_qdrant_connector.get_collection_names.return_value = ["dimension_mismatch_test"]
            mock_qdrant_connector.get_collection_info.return_value = {
                "config": {"params": {"vectors": {"size": 384}}}  # Different from expected
            }
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                with patch.object(operation, '_validate_embedding_compatibility',
                                side_effect=VectorDimensionMismatchError("Dimension mismatch", 768, 384)):
                    result = await operation.execute()
            
            assert not result.success
            assert len(result.errors) > 0
            assert "dimension" in str(result.errors).lower()


class TestConfigurationErrors:
    """Test handling of configuration validation errors."""

    @pytest.mark.asyncio
    async def test_invalid_embedding_model(self):
        """Test handling of invalid embedding model configuration."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="invalid_model_test",
        )
        
        # Override with invalid embedding model
        config.embedding_settings.embedding_model = "invalid/model/name"
        
        operation = IngestOperation(config)
        
        # Mock embedding provider creation error
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider',
                  side_effect=ModelValidationError("Invalid model", "invalid/model/name")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_qdrant_url(self):
        """Test handling of invalid Qdrant URL configuration."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="invalid_url_test",
        )
        
        # Override with invalid URL
        config.qdrant_settings.qdrant_url = "invalid://url:format"
        
        operation = IngestOperation(config)
        
        with patch.object(operation, 'get_connector', side_effect=ConfigurationValidationError(
            "qdrant_url", "invalid://url:format", "Invalid URL format")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_collection_name(self):
        """Test handling of invalid collection names."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="invalid/collection/name",  # Invalid characters
        )
        
        operation = IngestOperation(config)
        
        # Mock collection name validation error
        with patch.object(operation, 'get_connector', side_effect=ConfigurationValidationError(
            "collection_name", "invalid/collection/name", "Invalid collection name")):
            result = await operation.execute()
        
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_invalid_regex_patterns(self):
        """Test handling of invalid regex patterns."""
        config = create_test_config(
            operation_mode="ingest",
            target_path=Path("/tmp"),
            knowledgebase_name="regex_error_test",
            include_patterns=["[invalid"],  # Invalid regex
        )
        
        operation = IngestOperation(config)
        
        # Should raise configuration error for invalid regex
        with pytest.raises((ConfigurationValidationError, ValueError)):
            await operation.execute()

    @pytest.mark.asyncio
    async def test_conflicting_configuration_options(self):
        """Test handling of conflicting configuration options."""
        config = create_test_config(
            operation_mode="update",
            target_path=Path("/tmp"),
            knowledgebase_name="conflict_test",
            update_mode="replace",
            dry_run=True,  # Conflicting: replace mode with dry run
        )
        
        operation = UpdateOperation(config)
        
        # Should handle conflicting options gracefully
        result = await operation.execute()
        
        # Dry run should take precedence
        assert result.success


class TestProcessingErrors:
    """Test handling of content processing errors."""

    @pytest.mark.asyncio
    async def test_chunking_error_handling(self, mock_qdrant_connector):
        """Test handling of chunking errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "chunking_error.txt"
            test_file.write_text("test content for chunking error")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="chunking_error_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Mock chunking error
            with patch.object(operation, '_process_file_content',
                            side_effect=ChunkingError("Chunking failed")):
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            
            # Should handle chunking errors gracefully
            assert result.files_failed > 0 or result.files_skipped > 0
            if result.files_failed > 0:
                assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_tokenizer_error_handling(self, mock_qdrant_connector):
        """Test handling of tokenizer errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "tokenizer_error.txt"
            test_file.write_text("test content for tokenizer error")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="tokenizer_error_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Mock tokenizer error
            with patch('src.mcp_server_qdrant_rag.cli_ingest.DocumentChunker.chunk_document',
                      side_effect=TokenizerError("Tokenizer failed")):
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            
            # Should handle tokenizer errors gracefully
            assert result.files_failed > 0 or result.files_skipped > 0

    @pytest.mark.asyncio
    async def test_sentence_splitter_error(self, mock_qdrant_connector):
        """Test handling of sentence splitter errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "splitter_error.txt"
            test_file.write_text("test content for sentence splitter error")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="splitter_error_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Mock sentence splitter error
            with patch('src.mcp_server_qdrant_rag.chunking.chunker.DocumentChunker._split_into_sentences',
                      side_effect=SentenceSplitterError("Sentence splitting failed")):
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            
            # Should handle sentence splitter errors gracefully
            assert result.files_failed > 0 or result.files_skipped > 0

    @pytest.mark.asyncio
    async def test_embedding_generation_error(self, mock_qdrant_connector):
        """Test handling of embedding generation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "embedding_error.txt"
            test_file.write_text("test content for embedding error")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="embedding_error_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            mock_qdrant_connector.store_entries.side_effect = Exception("Embedding generation failed")
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle embedding errors gracefully
            assert not result.success
            assert len(result.errors) > 0


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_directory_processing(self, mock_qdrant_connector):
        """Test processing of empty directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            # Create empty subdirectories
            (workspace / "empty1").mkdir()
            (workspace / "empty2").mkdir()
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="empty_dir_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle empty directories gracefully
            assert result.success
            assert result.files_processed == 0
            assert len(result.warnings) > 0  # Should warn about no files found

    @pytest.mark.asyncio
    async def test_very_deep_directory_structure(self, mock_qdrant_connector):
        """Test processing of very deep directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a very deep directory structure
            current_dir = workspace
            for i in range(50):  # 50 levels deep
                current_dir = current_dir / f"level_{i}"
                current_dir.mkdir()
            
            # Create a file at the deepest level
            deep_file = current_dir / "deep_file.txt"
            deep_file.write_text("Content at the deepest level")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="deep_structure_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle deep structures gracefully
            assert result.success
            assert result.files_processed == 1

    @pytest.mark.asyncio
    async def test_circular_symlink_handling(self, mock_qdrant_connector):
        """Test handling of circular symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a regular file
            regular_file = workspace / "regular.txt"
            regular_file.write_text("Regular file content")
            
            # Create circular symlinks (if supported by the system)
            try:
                link1 = workspace / "link1"
                link2 = workspace / "link2"
                link1.symlink_to(link2)
                link2.symlink_to(link1)
            except (OSError, NotImplementedError):
                # Skip test if symlinks are not supported
                pytest.skip("Symbolic links not supported on this system")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="circular_link_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle circular symlinks gracefully
            assert result.success
            assert result.files_processed >= 1  # Should process the regular file

    @pytest.mark.asyncio
    async def test_unicode_filename_handling(self, mock_qdrant_connector):
        """Test handling of Unicode filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create files with Unicode names
            unicode_files = [
                "测试文件.txt",  # Chinese
                "файл_тест.txt",  # Russian
                "αρχείο_δοκιμή.txt",  # Greek
                "ファイル_テスト.txt",  # Japanese
                "🌍_emoji_file.txt",  # Emoji
            ]
            
            for filename in unicode_files:
                try:
                    file_path = workspace / filename
                    file_path.write_text(f"Content of {filename}")
                except (OSError, UnicodeError):
                    # Skip files that can't be created on this system
                    continue
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="unicode_filename_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle Unicode filenames gracefully
            assert result.success
            assert result.files_processed > 0

    @pytest.mark.asyncio
    async def test_extremely_long_filename(self, mock_qdrant_connector):
        """Test handling of extremely long filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a file with a very long name (close to filesystem limit)
            long_name = "a" * 200 + ".txt"  # 200+ character filename
            
            try:
                long_file = workspace / long_name
                long_file.write_text("Content of file with very long name")
            except OSError:
                # Skip test if long filename is not supported
                pytest.skip("Long filenames not supported on this system")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="long_filename_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle long filenames gracefully
            assert result.success
            assert result.files_processed == 1


class TestResourceExhaustion:
    """Test handling of resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, mock_qdrant_connector):
        """Test handling of memory pressure scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a file that would consume significant memory if loaded entirely
            large_content = "Large content line. " * 100000  # ~2MB
            large_file = workspace / "memory_pressure.txt"
            large_file.write_text(large_content)
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="memory_pressure_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Mock memory error during processing
            with patch('builtins.open', side_effect=MemoryError("Out of memory")):
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            
            # Should handle memory errors gracefully
            assert result.files_failed > 0 or result.files_skipped > 0

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, mock_qdrant_connector):
        """Test handling of network timeout scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "network_timeout.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="network_timeout_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            mock_qdrant_connector.store_entries.side_effect = asyncio.TimeoutError("Network timeout")
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle network timeouts gracefully
            assert not result.success
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_concurrent_operation_conflicts(self, mock_qdrant_connector):
        """Test handling of concurrent operation conflicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            test_file = workspace / "concurrent_conflict.txt"
            test_file.write_text("test content")
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="concurrent_conflict_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            mock_qdrant_connector.create_collection.side_effect = CollectionAccessError(
                "Collection is being modified by another operation"
            )
            
            operation = IngestOperation(config)
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            # Should handle concurrent operation conflicts gracefully
            assert not result.success
            assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
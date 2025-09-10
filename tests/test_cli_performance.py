"""
Performance tests for CLI file ingestion.

This module provides performance testing for:
- Large file processing and chunking
- Batch operations with many files
- Memory usage monitoring
- Processing time benchmarks
- Concurrent processing capabilities
"""

import asyncio
import gc
import os
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
    IngestOperation,
    UpdateOperation,
    FileDiscovery,
    ContentProcessor,
    IngestConfig,
    CLISettings,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings

from tests.fixtures.cli_test_fixtures import (
    large_file_structure,
    mock_qdrant_connector,
    create_test_config,
)


class TestLargeFileProcessing:
    """Test performance with large files."""

    @pytest.mark.asyncio
    async def test_large_document_processing_time(self, mock_qdrant_connector):
        """Test processing time for large documents."""
        # Create a very large document
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a 1MB text file
            large_content = []
            for i in range(5000):  # ~1MB of text
                large_content.append(f"""
This is paragraph {i} of a very large document created for performance testing.
The document contains extensive content that will be processed by the CLI tool.
Each paragraph discusses different aspects of document processing, chunking strategies,
and performance optimization techniques. The content is designed to simulate
real-world documents that users might want to ingest into their vector database.

Key performance considerations for paragraph {i}:
- Memory usage should remain constant regardless of document size
- Processing time should scale linearly with content size
- Chunking should preserve semantic boundaries
- Embedding generation should be efficient
- Storage operations should be batched for optimal performance

This paragraph covers topic {i % 20} which relates to various aspects of
information retrieval, natural language processing, and vector database operations.
The system should handle this content efficiently while maintaining high quality
semantic representations for accurate search and retrieval.
                """.strip())
            
            large_file = workspace / "large_document.txt"
            large_file.write_text("\n\n".join(large_content))
            
            # Verify file size
            file_size = large_file.stat().st_size
            assert file_size > 1024 * 1024  # At least 1MB
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="large_file_test",
                verbose=True,
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Measure processing time
            start_time = time.time()
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            processing_time = time.time() - start_time
            
            # Verify successful processing
            assert result.success
            assert result.files_processed == 1
            assert result.chunks_created > 1  # Large file should be chunked
            
            # Performance assertions
            assert processing_time < 60.0  # Should complete within 1 minute
            
            # Calculate processing rate
            processing_rate = file_size / processing_time  # bytes per second
            assert processing_rate > 50000  # At least 50KB/s processing rate
            
            print(f"Processed {file_size} bytes in {processing_time:.2f}s "
                  f"({processing_rate:.0f} bytes/s)")

    @pytest.mark.asyncio
    async def test_memory_usage_large_files(self, mock_qdrant_connector):
        """Test memory usage remains reasonable with large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create multiple large files
            for i in range(3):
                content = "Large file content. " * 50000  # ~1MB each
                (workspace / f"large_file_{i}.txt").write_text(content)
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="memory_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            # Monitor memory usage
            if not HAS_PSUTIL:
                pytest.skip("psutil not available for memory monitoring")
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            peak_memory = initial_memory
            
            def memory_monitor():
                nonlocal peak_memory
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)
            
            # Start memory monitoring
            monitor_task = asyncio.create_task(self._monitor_memory_usage(memory_monitor))
            
            try:
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            final_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory
            
            assert result.success
            
            # Memory increase should be reasonable (less than 200MB for 3MB of files)
            assert memory_increase < 200 * 1024 * 1024
            
            # Memory should not grow linearly with file size
            total_file_size = sum((workspace / f"large_file_{i}.txt").stat().st_size for i in range(3))
            memory_ratio = memory_increase / total_file_size
            assert memory_ratio < 10  # Memory increase should be less than 10x file size
            
            print(f"Memory increase: {memory_increase / 1024 / 1024:.1f}MB for "
                  f"{total_file_size / 1024 / 1024:.1f}MB of files "
                  f"(ratio: {memory_ratio:.2f})")

    async def _monitor_memory_usage(self, callback, interval=0.1):
        """Monitor memory usage during operation."""
        while True:
            callback()
            await asyncio.sleep(interval)

    @pytest.mark.asyncio
    async def test_chunking_performance(self, mock_qdrant_connector):
        """Test chunking performance with various chunk sizes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a document that will require chunking
            content = "This is a test sentence. " * 10000  # ~250KB
            test_file = workspace / "chunking_test.txt"
            test_file.write_text(content)
            
            # Test different chunk sizes
            chunk_sizes = [100, 500, 1000, 2000]
            results = {}
            
            for chunk_size in chunk_sizes:
                config = create_test_config(
                    operation_mode="ingest",
                    target_path=workspace,
                    knowledgebase_name=f"chunk_test_{chunk_size}",
                )
                
                mock_qdrant_connector.get_collection_names.return_value = []
                
                operation = IngestOperation(config)
                
                start_time = time.time()
                
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
                
                processing_time = time.time() - start_time
                
                assert result.success
                
                results[chunk_size] = {
                    'time': processing_time,
                    'chunks': result.chunks_created,
                }
                
                print(f"Chunk size {chunk_size}: {processing_time:.2f}s, "
                      f"{result.chunks_created} chunks")
            
            # Verify that smaller chunks create more chunks but don't take exponentially longer
            assert results[100]['chunks'] > results[2000]['chunks']
            
            # Processing time should not increase dramatically with smaller chunks
            time_ratio = results[100]['time'] / results[2000]['time']
            assert time_ratio < 5  # Should not be more than 5x slower


class TestBatchProcessing:
    """Test performance with batch operations."""

    @pytest.mark.asyncio
    async def test_many_small_files_performance(self, mock_qdrant_connector):
        """Test performance with many small files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create many small files
            num_files = 100
            for i in range(num_files):
                content = f"This is test file {i} with some content for testing batch processing."
                (workspace / f"file_{i:03d}.txt").write_text(content)
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="batch_test",
                batch_size=10,  # Process in batches of 10
                verbose=True,
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            start_time = time.time()
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            processing_time = time.time() - start_time
            
            assert result.success
            assert result.files_processed == num_files
            
            # Performance assertions
            assert processing_time < 30.0  # Should complete within 30 seconds
            
            # Calculate files per second
            files_per_second = num_files / processing_time
            assert files_per_second > 5  # At least 5 files per second
            
            print(f"Processed {num_files} files in {processing_time:.2f}s "
                  f"({files_per_second:.1f} files/s)")

    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, mock_qdrant_connector):
        """Test optimal batch size for processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files
            num_files = 50
            for i in range(num_files):
                content = f"Test file {i} content for batch size optimization testing."
                (workspace / f"batch_file_{i}.txt").write_text(content)
            
            # Test different batch sizes
            batch_sizes = [1, 5, 10, 25, 50]
            results = {}
            
            for batch_size in batch_sizes:
                config = create_test_config(
                    operation_mode="ingest",
                    target_path=workspace,
                    knowledgebase_name=f"batch_size_test_{batch_size}",
                    batch_size=batch_size,
                )
                
                mock_qdrant_connector.reset_mock()
                mock_qdrant_connector.get_collection_names.return_value = []
                
                operation = IngestOperation(config)
                
                start_time = time.time()
                
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
                
                processing_time = time.time() - start_time
                
                assert result.success
                assert result.files_processed == num_files
                
                results[batch_size] = {
                    'time': processing_time,
                    'store_calls': mock_qdrant_connector.store_entries.call_count,
                }
                
                print(f"Batch size {batch_size}: {processing_time:.2f}s, "
                      f"{mock_qdrant_connector.store_entries.call_count} store calls")
            
            # Verify that larger batch sizes result in fewer store calls
            assert results[1]['store_calls'] > results[50]['store_calls']
            
            # Find optimal batch size (should be faster than batch size 1)
            optimal_time = min(results[bs]['time'] for bs in batch_sizes if bs > 1)
            assert optimal_time < results[1]['time']

    @pytest.mark.asyncio
    async def test_concurrent_file_processing(self, mock_qdrant_connector):
        """Test concurrent processing of multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create files of varying sizes
            file_sizes = [1000, 5000, 10000, 2000, 8000]  # Different sizes
            for i, size in enumerate(file_sizes):
                content = f"File {i} content. " * size
                (workspace / f"concurrent_file_{i}.txt").write_text(content)
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="concurrent_test",
            )
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            operation = IngestOperation(config)
            
            start_time = time.time()
            
            with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                result = await operation.execute()
            
            concurrent_time = time.time() - start_time
            
            assert result.success
            assert result.files_processed == len(file_sizes)
            
            # Compare with sequential processing
            config_sequential = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="sequential_test",
            )
            
            operation_sequential = IngestOperation(config_sequential)
            
            mock_qdrant_connector.reset_mock()
            mock_qdrant_connector.get_collection_names.return_value = []
            
            start_time = time.time()
            
            with patch.object(operation_sequential, 'get_connector', return_value=mock_qdrant_connector):
                result_sequential = await operation_sequential.execute()
            
            sequential_time = time.time() - start_time
            
            assert result_sequential.success
            
            # Concurrent processing should be faster (or at least not significantly slower)
            speedup = sequential_time / concurrent_time
            print(f"Concurrent: {concurrent_time:.2f}s, Sequential: {sequential_time:.2f}s, "
                  f"Speedup: {speedup:.2f}x")
            
            # Allow for some overhead, but concurrent should not be much slower
            assert speedup > 0.8  # At least 80% of sequential performance


class TestFileDiscoveryPerformance:
    """Test performance of file discovery operations."""

    @pytest.mark.asyncio
    async def test_large_directory_scanning(self):
        """Test performance of scanning large directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a large directory structure
            num_dirs = 20
            files_per_dir = 50
            
            for dir_i in range(num_dirs):
                dir_path = workspace / f"dir_{dir_i:02d}"
                dir_path.mkdir()
                
                for file_i in range(files_per_dir):
                    file_path = dir_path / f"file_{file_i:03d}.txt"
                    file_path.write_text(f"Content for file {file_i} in directory {dir_i}")
            
            total_files = num_dirs * files_per_dir
            
            discovery = FileDiscovery(['.txt'])
            
            start_time = time.time()
            files = await discovery.discover_files(workspace, recursive=True)
            discovery_time = time.time() - start_time
            
            assert len(files) == total_files
            
            # Performance assertions
            assert discovery_time < 10.0  # Should complete within 10 seconds
            
            files_per_second = total_files / discovery_time
            assert files_per_second > 100  # At least 100 files per second
            
            print(f"Discovered {total_files} files in {discovery_time:.2f}s "
                  f"({files_per_second:.0f} files/s)")

    @pytest.mark.asyncio
    async def test_pattern_matching_performance(self):
        """Test performance of pattern matching with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create files with different extensions
            extensions = ['.txt', '.py', '.js', '.md', '.json', '.yaml', '.xml', '.csv']
            files_per_ext = 100
            
            for ext in extensions:
                for i in range(files_per_ext):
                    file_path = workspace / f"file_{i:03d}{ext}"
                    file_path.write_text(f"Content for {ext} file {i}")
            
            total_files = len(extensions) * files_per_ext
            
            discovery = FileDiscovery(extensions)
            
            # Test complex pattern matching
            include_patterns = [r'\.py$', r'\.js$', r'\.md$']
            exclude_patterns = [r'file_0[0-4][0-9]']  # Exclude files 000-049
            
            start_time = time.time()
            files = await discovery.discover_files(
                workspace,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                recursive=True
            )
            pattern_time = time.time() - start_time
            
            # Should find Python, JavaScript, and Markdown files, minus excluded ones
            expected_files = 3 * files_per_ext - 3 * 50  # 3 extensions, minus 50 excluded per extension
            assert len(files) == expected_files
            
            # Performance assertions
            assert pattern_time < 5.0  # Should complete within 5 seconds
            
            files_per_second = total_files / pattern_time  # Total files scanned
            assert files_per_second > 200  # At least 200 files per second
            
            print(f"Pattern matched {total_files} files in {pattern_time:.2f}s "
                  f"({files_per_second:.0f} files/s), found {len(files)} matches")


class TestContentProcessingPerformance:
    """Test performance of content processing operations."""

    @pytest.mark.asyncio
    async def test_encoding_detection_performance(self):
        """Test performance of encoding detection with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create files with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            files_per_encoding = 50
            
            for encoding in encodings:
                for i in range(files_per_encoding):
                    file_path = workspace / f"{encoding}_file_{i}.txt"
                    content = f"Content with encoding {encoding} - file {i}"
                    
                    # Add some special characters for each encoding
                    if encoding == 'utf-8':
                        content += " - Unicode: 世界 🌍"
                    elif encoding == 'latin-1':
                        content += " - Latin: café résumé"
                    elif encoding == 'cp1252':
                        content += " - CP1252: \"smart quotes\""
                    
                    with open(file_path, 'w', encoding=encoding) as f:
                        f.write(content)
            
            total_files = len(encodings) * files_per_encoding
            
            discovery = FileDiscovery(['.txt'])
            
            start_time = time.time()
            files = await discovery.discover_files(workspace, recursive=True)
            detection_time = time.time() - start_time
            
            assert len(files) == total_files
            
            # Verify encoding detection worked
            detected_encodings = {f.encoding for f in files}
            assert len(detected_encodings) > 1  # Should detect multiple encodings
            
            # Performance assertions
            assert detection_time < 10.0  # Should complete within 10 seconds
            
            files_per_second = total_files / detection_time
            assert files_per_second > 20  # At least 20 files per second (encoding detection is slower)
            
            print(f"Detected encodings for {total_files} files in {detection_time:.2f}s "
                  f"({files_per_second:.1f} files/s)")
            print(f"Detected encodings: {sorted(detected_encodings)}")

    @pytest.mark.asyncio
    async def test_token_estimation_performance(self):
        """Test performance of token estimation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create files with varying content lengths
            content_lengths = [100, 500, 1000, 5000, 10000]  # words
            files_per_length = 20
            
            for length in content_lengths:
                for i in range(files_per_length):
                    file_path = workspace / f"tokens_{length}_{i}.txt"
                    # Create content with approximately the specified number of words
                    words = [f"word{j}" for j in range(length)]
                    content = " ".join(words)
                    file_path.write_text(content)
            
            total_files = len(content_lengths) * files_per_length
            
            discovery = FileDiscovery(['.txt'])
            
            start_time = time.time()
            files = await discovery.discover_files(workspace, recursive=True)
            estimation_time = time.time() - start_time
            
            assert len(files) == total_files
            
            # Verify token estimation
            total_estimated_tokens = sum(f.estimated_tokens for f in files)
            assert total_estimated_tokens > 0
            
            # Performance assertions
            assert estimation_time < 15.0  # Should complete within 15 seconds
            
            files_per_second = total_files / estimation_time
            assert files_per_second > 10  # At least 10 files per second
            
            tokens_per_second = total_estimated_tokens / estimation_time
            assert tokens_per_second > 1000  # At least 1000 tokens per second
            
            print(f"Estimated tokens for {total_files} files in {estimation_time:.2f}s "
                  f"({files_per_second:.1f} files/s, {tokens_per_second:.0f} tokens/s)")
            print(f"Total estimated tokens: {total_estimated_tokens}")


class TestMemoryLeakDetection:
    """Test for memory leaks during long-running operations."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, mock_qdrant_connector):
        """Test for memory leaks during repeated operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files
            for i in range(10):
                content = f"Test content for memory leak detection - file {i}"
                (workspace / f"leak_test_{i}.txt").write_text(content)
            
            mock_qdrant_connector.get_collection_names.return_value = []
            
            # Perform multiple operations and monitor memory
            if not HAS_PSUTIL:
                pytest.skip("psutil not available for memory monitoring")
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            memory_samples = [initial_memory]
            
            for iteration in range(10):
                config = create_test_config(
                    operation_mode="ingest",
                    target_path=workspace,
                    knowledgebase_name=f"leak_test_{iteration}",
                )
                
                operation = IngestOperation(config)
                
                with patch.object(operation, 'get_connector', return_value=mock_qdrant_connector):
                    result = await operation.execute()
                
                assert result.success
                
                # Force garbage collection
                gc.collect()
                
                # Sample memory usage
                current_memory = process.memory_info().rss
                memory_samples.append(current_memory)
                
                # Reset mock for next iteration
                mock_qdrant_connector.reset_mock()
                mock_qdrant_connector.get_collection_names.return_value = []
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Check for memory leaks
            # Memory should not increase significantly over iterations
            max_acceptable_increase = 50 * 1024 * 1024  # 50MB
            assert memory_increase < max_acceptable_increase, \
                f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB, " \
                f"which may indicate a memory leak"
            
            # Check that memory usage is relatively stable
            memory_growth_rate = memory_increase / len(memory_samples)
            max_growth_per_iteration = 5 * 1024 * 1024  # 5MB per iteration
            assert memory_growth_rate < max_growth_per_iteration, \
                f"Memory growing at {memory_growth_rate / 1024 / 1024:.1f}MB per iteration"
            
            print(f"Memory usage over {len(memory_samples)} iterations:")
            for i, memory in enumerate(memory_samples):
                print(f"  Iteration {i}: {memory / 1024 / 1024:.1f}MB")
            print(f"Total increase: {memory_increase / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
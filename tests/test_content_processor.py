"""
Unit tests for ContentProcessor class in CLI file ingestion.

Tests cover file reading, encoding detection, binary file detection,
metadata extraction, and error handling scenarios.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.mcp_server_qdrant_rag.cli_ingest import ContentProcessor, FileInfo
from src.mcp_server_qdrant_rag.qdrant import Entry


class TestContentProcessor:
    """Test cases for ContentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a ContentProcessor instance for testing."""
        return ContentProcessor(max_file_size=1024 * 1024)  # 1MB for testing
    
    @pytest.fixture
    def sample_file_info(self, tmp_path):
        """Create a sample FileInfo object for testing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding='utf-8')
        
        stat = test_file.stat()
        return FileInfo(
            path=test_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=3
        )
    
    @pytest.mark.asyncio
    async def test_process_file_success(self, processor, sample_file_info):
        """Test successful file processing."""
        entry = await processor.process_file(sample_file_info)
        
        assert entry is not None
        assert isinstance(entry, Entry)
        assert entry.content == "Hello, world!"
        assert entry.is_chunk is False
        assert entry.source_document_id is None
        assert entry.metadata is not None
        assert entry.metadata["file_name"] == "test.txt"
        assert entry.metadata["source_type"] == "file_ingestion"
    
    @pytest.mark.asyncio
    async def test_process_binary_file_returns_none(self, processor, tmp_path):
        """Test that binary files return None."""
        # Create a binary file
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')
        
        file_info = FileInfo(
            path=binary_file,
            size=4,
            modified_time=datetime.now(),
            encoding='binary',
            is_binary=True,
            estimated_tokens=1
        )
        
        entry = await processor.process_file(file_info)
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_process_empty_file_returns_none(self, processor, tmp_path):
        """Test that empty files return None."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding='utf-8')
        
        file_info = FileInfo(
            path=empty_file,
            size=0,
            modified_time=datetime.now(),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=0
        )
        
        entry = await processor.process_file(file_info)
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_process_whitespace_only_file_returns_none(self, processor, tmp_path):
        """Test that files with only whitespace return None."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n\t  \n  ", encoding='utf-8')
        
        stat = whitespace_file.stat()
        file_info = FileInfo(
            path=whitespace_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=2
        )
        
        entry = await processor.process_file(file_info)
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_process_large_file_raises_error(self, processor, tmp_path):
        """Test that files exceeding size limit raise ValueError."""
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * (2 * 1024 * 1024), encoding='utf-8')  # 2MB
        
        stat = large_file.stat()
        file_info = FileInfo(
            path=large_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=500000
        )
        
        with pytest.raises(ValueError, match="File too large"):
            await processor.process_file(file_info)
    
    @pytest.mark.asyncio
    async def test_read_file_content_utf8(self, processor, tmp_path):
        """Test reading UTF-8 encoded file."""
        utf8_file = tmp_path / "utf8.txt"
        content = "Hello, 世界! 🌍"
        utf8_file.write_text(content, encoding='utf-8')
        
        file_info = FileInfo(
            path=utf8_file,
            size=utf8_file.stat().st_size,
            modified_time=datetime.now(),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=5
        )
        
        result = await processor._read_file_content(file_info)
        assert result == content
    
    @pytest.mark.asyncio
    async def test_read_file_content_encoding_fallback(self, processor, tmp_path):
        """Test encoding fallback when detected encoding fails."""
        # Create a file with latin-1 encoding but mark it as utf-8
        latin1_file = tmp_path / "latin1.txt"
        content = "Café résumé"
        latin1_file.write_text(content, encoding='latin-1')
        
        file_info = FileInfo(
            path=latin1_file,
            size=latin1_file.stat().st_size,
            modified_time=datetime.now(),
            encoding='utf-8',  # Wrong encoding detected
            is_binary=False,
            estimated_tokens=3
        )
        
        result = await processor._read_file_content(file_info)
        # Should successfully read with fallback encoding
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_read_file_content_nonexistent_file(self, processor, tmp_path):
        """Test reading non-existent file raises OSError."""
        # Create a file and then delete it to bypass FileInfo validation
        temp_file = tmp_path / "temp.txt"
        temp_file.write_text("content", encoding='utf-8')
        
        stat = temp_file.stat()
        file_info = FileInfo(
            path=temp_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=25
        )
        
        # Now delete the file to simulate non-existent file
        temp_file.unlink()
        
        with pytest.raises((OSError, UnicodeDecodeError)):
            await processor._read_file_content(file_info)
    
    def test_is_mostly_control_chars_binary_content(self, processor):
        """Test detection of binary content with control characters."""
        binary_content = "\x00\x01\x02\x03\x04"
        assert processor._is_mostly_control_chars(binary_content) is True
    
    def test_is_mostly_control_chars_text_content(self, processor):
        """Test detection of normal text content."""
        text_content = "This is normal text content with some numbers 123."
        assert processor._is_mostly_control_chars(text_content) is False
    
    def test_is_mostly_control_chars_mixed_content(self, processor):
        """Test detection of mixed content with some control chars."""
        mixed_content = "Normal text\x00\x01more text"
        # Should be False because control chars are less than 30%
        assert processor._is_mostly_control_chars(mixed_content) is False
    
    def test_is_mostly_control_chars_empty_content(self, processor):
        """Test detection of empty content."""
        assert processor._is_mostly_control_chars("") is True
    
    def test_extract_metadata_basic(self, processor, tmp_path):
        """Test basic metadata extraction."""
        test_file = tmp_path / "example.py"
        test_file.write_text("print('hello')", encoding='utf-8')
        
        stat = test_file.stat()
        file_info = FileInfo(
            path=test_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=4
        )
        
        metadata = processor._extract_metadata(file_info)
        
        assert metadata["file_name"] == "example.py"
        assert metadata["file_extension"] == ".py"
        assert metadata["file_stem"] == "example"
        assert metadata["file_type"] == "python"
        assert metadata["source_type"] == "file_ingestion"
        assert metadata["ingestion_method"] == "cli_tool"
        assert metadata["encoding"] == "utf-8"
        assert metadata["estimated_tokens"] == 4
        assert "ingestion_time" in metadata
        assert "modified_time" in metadata
    
    def test_extract_metadata_nested_path(self, processor, tmp_path):
        """Test metadata extraction for nested file paths."""
        nested_dir = tmp_path / "src" / "components"
        nested_dir.mkdir(parents=True)
        
        nested_file = nested_dir / "Button.tsx"
        nested_file.write_text("export const Button = () => <button />;", encoding='utf-8')
        
        stat = nested_file.stat()
        file_info = FileInfo(
            path=nested_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=10
        )
        
        metadata = processor._extract_metadata(file_info)
        
        assert metadata["file_name"] == "Button.tsx"
        assert metadata["file_extension"] == ".tsx"
        assert metadata["file_type"] == "typescript"
        assert str(nested_dir) in metadata["parent_directory"]
    
    def test_classify_file_type_known_extensions(self, processor):
        """Test file type classification for known extensions."""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.md"), "markdown"),
            (Path("test.json"), "json"),
            (Path("test.yaml"), "yaml"),
            (Path("test.yml"), "yaml"),
            (Path("test.tf"), "terraform"),
            (Path("test.sql"), "sql"),
            (Path("test.html"), "html"),
            (Path("test.css"), "css"),
        ]
        
        for file_path, expected_type in test_cases:
            result = processor._classify_file_type(file_path)
            assert result == expected_type, f"Expected {expected_type} for {file_path}, got {result}"
    
    def test_classify_file_type_unknown_extension(self, processor):
        """Test file type classification for unknown extensions."""
        unknown_file = Path("test.unknown")
        result = processor._classify_file_type(unknown_file)
        assert result == "unknown"
    
    def test_classify_file_type_no_extension(self, processor):
        """Test file type classification for files without extension."""
        no_ext_file = Path("README")
        result = processor._classify_file_type(no_ext_file)
        assert result == "unknown"
    
    def test_categorize_file_size(self, processor):
        """Test file size categorization."""
        test_cases = [
            (500, "tiny"),           # < 1KB
            (5000, "small"),         # < 10KB
            (50000, "medium"),       # < 100KB
            (500000, "large"),       # < 1MB
            (5000000, "very_large"), # >= 1MB
        ]
        
        for size, expected_category in test_cases:
            result = processor._categorize_file_size(size)
            assert result == expected_category, f"Expected {expected_category} for size {size}, got {result}"
    
    def test_can_process_file_valid(self, processor, tmp_path):
        """Test can_process_file with valid file."""
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("content", encoding='utf-8')
        
        assert processor.can_process_file(valid_file) is True
    
    def test_can_process_file_nonexistent(self, processor):
        """Test can_process_file with non-existent file."""
        nonexistent = Path("/nonexistent/file.txt")
        assert processor.can_process_file(nonexistent) is False
    
    def test_can_process_file_directory(self, processor, tmp_path):
        """Test can_process_file with directory."""
        directory = tmp_path / "dir"
        directory.mkdir()
        
        assert processor.can_process_file(directory) is False
    
    def test_can_process_file_hidden(self, processor, tmp_path):
        """Test can_process_file with hidden file."""
        hidden_file = tmp_path / ".hidden"
        hidden_file.write_text("content", encoding='utf-8')
        
        assert processor.can_process_file(hidden_file) is False
    
    def test_can_process_file_too_large(self, processor, tmp_path):
        """Test can_process_file with file exceeding size limit."""
        # Create processor with small size limit
        small_processor = ContentProcessor(max_file_size=100)
        
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 200, encoding='utf-8')  # 200 bytes
        
        assert small_processor.can_process_file(large_file) is False
    
    def test_get_processing_stats_empty_list(self, processor):
        """Test processing stats with empty entry list."""
        stats = processor.get_processing_stats([])
        
        assert stats["total_entries"] == 0
        assert stats["total_content_length"] == 0
        assert stats["average_content_length"] == 0
        assert stats["file_types"] == {}
        assert stats["size_categories"] == {}
        assert stats["encodings"] == {}
    
    def test_get_processing_stats_with_entries(self, processor):
        """Test processing stats with sample entries."""
        entries = [
            Entry(
                content="Short content",
                metadata={
                    "file_type": "python",
                    "size_category": "small",
                    "encoding": "utf-8"
                }
            ),
            Entry(
                content="Longer content with more text",
                metadata={
                    "file_type": "javascript",
                    "size_category": "medium",
                    "encoding": "utf-8"
                }
            ),
            Entry(
                content="Another python file",
                metadata={
                    "file_type": "python",
                    "size_category": "small",
                    "encoding": "latin-1"
                }
            )
        ]
        
        stats = processor.get_processing_stats(entries)
        
        assert stats["total_entries"] == 3
        assert stats["total_content_length"] == sum(len(e.content) for e in entries)
        assert stats["average_content_length"] == stats["total_content_length"] / 3
        assert stats["file_types"]["python"] == 2
        assert stats["file_types"]["javascript"] == 1
        assert stats["size_categories"]["small"] == 2
        assert stats["size_categories"]["medium"] == 1
        assert stats["encodings"]["utf-8"] == 2
        assert stats["encodings"]["latin-1"] == 1


class TestContentProcessorIntegration:
    """Integration tests for ContentProcessor with real files."""
    
    @pytest.fixture
    def processor(self):
        """Create a ContentProcessor instance for integration testing."""
        return ContentProcessor()
    
    @pytest.mark.asyncio
    async def test_process_various_file_types(self, processor, tmp_path):
        """Test processing various file types end-to-end."""
        # Create test files of different types
        files_to_create = [
            ("script.py", "#!/usr/bin/env python3\nprint('Hello, world!')\n", "python"),
            ("config.json", '{"name": "test", "version": "1.0"}', "json"),
            ("README.md", "# Test Project\n\nThis is a test.", "markdown"),
            ("style.css", "body { margin: 0; padding: 0; }", "css"),
            ("data.sql", "SELECT * FROM users WHERE active = 1;", "sql"),
        ]
        
        processed_entries = []
        
        for filename, content, expected_type in files_to_create:
            # Create file
            test_file = tmp_path / filename
            test_file.write_text(content, encoding='utf-8')
            
            # Create FileInfo
            stat = test_file.stat()
            file_info = FileInfo(
                path=test_file,
                size=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                encoding='utf-8',
                is_binary=False,
                estimated_tokens=len(content) // 4
            )
            
            # Process file
            entry = await processor.process_file(file_info)
            assert entry is not None
            assert entry.content == content
            assert entry.metadata["file_type"] == expected_type
            
            processed_entries.append(entry)
        
        # Check processing stats
        stats = processor.get_processing_stats(processed_entries)
        assert stats["total_entries"] == 5
        assert len(stats["file_types"]) == 5
    
    @pytest.mark.asyncio
    async def test_process_file_with_special_characters(self, processor, tmp_path):
        """Test processing files with special characters and Unicode."""
        special_file = tmp_path / "special.txt"
        content = "Special chars: àáâãäå ñ 中文 🚀 emoji\n\tTabbed content\n"
        special_file.write_text(content, encoding='utf-8')
        
        stat = special_file.stat()
        file_info = FileInfo(
            path=special_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=len(content) // 4
        )
        
        entry = await processor.process_file(file_info)
        assert entry is not None
        assert entry.content == content
        assert "🚀" in entry.content  # Verify emoji preserved
        assert "中文" in entry.content  # Verify Chinese characters preserved
    
    @pytest.mark.asyncio
    async def test_error_handling_permission_denied(self, processor, tmp_path):
        """Test error handling when file permissions are denied."""
        # This test might not work on all systems, so we'll mock the permission error
        restricted_file = tmp_path / "restricted.txt"
        restricted_file.write_text("content", encoding='utf-8')
        
        stat = restricted_file.stat()
        file_info = FileInfo(
            path=restricted_file,
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=2
        )
        
        # Mock file opening to raise PermissionError
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(ValueError, match="Failed to process file"):
                await processor.process_file(file_info)


if __name__ == "__main__":
    pytest.main([__file__])
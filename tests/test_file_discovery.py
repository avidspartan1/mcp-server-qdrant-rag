"""
Unit tests for file discovery functionality in CLI ingestion.

Tests cover file discovery, pattern matching, encoding detection,
and metadata extraction functionality.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.mcp_server_qdrant_rag.cli_ingest import FileDiscovery, FileInfo


class TestFileDiscovery:
    """Test cases for FileDiscovery class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.supported_extensions = ['.txt', '.md', '.py', '.js', '.json']
        self.discovery = FileDiscovery(self.supported_extensions)

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test.txt").write_text("Hello world", encoding='utf-8')
            (temp_path / "readme.md").write_text("# README", encoding='utf-8')
            (temp_path / "script.py").write_text("print('hello')", encoding='utf-8')
            (temp_path / "config.json").write_text('{"key": "value"}', encoding='utf-8')
            (temp_path / "binary.bin").write_bytes(b'\x00\x01\x02\x03')
            (temp_path / "unsupported.xyz").write_text("unsupported", encoding='utf-8')
            (temp_path / ".hidden.txt").write_text("hidden file", encoding='utf-8')
            
            # Create subdirectory
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("nested content", encoding='utf-8')
            (subdir / "nested.py").write_text("# nested python", encoding='utf-8')
            
            # Create hidden subdirectory
            hidden_dir = temp_path / ".hidden_dir"
            hidden_dir.mkdir()
            (hidden_dir / "hidden_nested.txt").write_text("hidden nested", encoding='utf-8')
            
            yield temp_path

    @pytest.mark.asyncio
    async def test_discover_single_file(self, temp_directory):
        """Test discovering a single file."""
        test_file = temp_directory / "test.txt"
        
        files = await self.discovery.discover_files(test_file)
        
        assert len(files) == 1
        assert files[0].path == test_file
        assert files[0].size > 0
        assert files[0].encoding == 'utf-8'
        assert not files[0].is_binary

    @pytest.mark.asyncio
    async def test_discover_directory_recursive(self, temp_directory):
        """Test recursive directory discovery."""
        files = await self.discovery.discover_files(temp_directory, recursive=True)
        
        # Should find supported files but not binary, unsupported, or hidden files
        expected_files = {'test.txt', 'readme.md', 'script.py', 'config.json', 'nested.txt', 'nested.py'}
        found_files = {f.path.name for f in files}
        
        assert found_files == expected_files
        assert len(files) == 6

    @pytest.mark.asyncio
    async def test_discover_directory_non_recursive(self, temp_directory):
        """Test non-recursive directory discovery."""
        files = await self.discovery.discover_files(temp_directory, recursive=False)
        
        # Should only find files in root directory
        expected_files = {'test.txt', 'readme.md', 'script.py', 'config.json'}
        found_files = {f.path.name for f in files}
        
        assert found_files == expected_files
        assert len(files) == 4

    @pytest.mark.asyncio
    async def test_include_patterns(self, temp_directory):
        """Test include pattern filtering."""
        # Include only Python files
        include_patterns = [r'\.py$']
        
        files = await self.discovery.discover_files(
            temp_directory, 
            include_patterns=include_patterns,
            recursive=True
        )
        
        found_files = {f.path.name for f in files}
        expected_files = {'script.py', 'nested.py'}
        
        assert found_files == expected_files

    @pytest.mark.asyncio
    async def test_exclude_patterns(self, temp_directory):
        """Test exclude pattern filtering."""
        # Exclude Python files
        exclude_patterns = [r'\.py$']
        
        files = await self.discovery.discover_files(
            temp_directory,
            exclude_patterns=exclude_patterns,
            recursive=True
        )
        
        found_files = {f.path.name for f in files}
        # Should find all supported files except Python files
        expected_files = {'test.txt', 'readme.md', 'config.json', 'nested.txt'}
        
        assert found_files == expected_files

    @pytest.mark.asyncio
    async def test_include_and_exclude_patterns(self, temp_directory):
        """Test combined include and exclude patterns."""
        # Include text files but exclude nested ones
        include_patterns = [r'\.txt$']
        exclude_patterns = [r'nested']
        
        files = await self.discovery.discover_files(
            temp_directory,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            recursive=True
        )
        
        found_files = {f.path.name for f in files}
        expected_files = {'test.txt'}  # nested.txt should be excluded
        
        assert found_files == expected_files

    @pytest.mark.asyncio
    async def test_multiple_include_patterns(self, temp_directory):
        """Test multiple include patterns (OR logic)."""
        # Include Python OR markdown files
        include_patterns = [r'\.py$', r'\.md$']
        
        files = await self.discovery.discover_files(
            temp_directory,
            include_patterns=include_patterns,
            recursive=True
        )
        
        found_files = {f.path.name for f in files}
        expected_files = {'readme.md', 'script.py', 'nested.py'}
        
        assert found_files == expected_files

    @pytest.mark.asyncio
    async def test_multiple_exclude_patterns(self, temp_directory):
        """Test multiple exclude patterns (OR logic)."""
        # Exclude Python OR JSON files
        exclude_patterns = [r'\.py$', r'\.json$']
        
        files = await self.discovery.discover_files(
            temp_directory,
            exclude_patterns=exclude_patterns,
            recursive=True
        )
        
        found_files = {f.path.name for f in files}
        expected_files = {'test.txt', 'readme.md', 'nested.txt'}
        
        assert found_files == expected_files

    @pytest.mark.asyncio
    async def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            await self.discovery.discover_files(
                Path("."),
                include_patterns=["[invalid"]
            )

    @pytest.mark.asyncio
    async def test_nonexistent_path(self):
        """Test handling of nonexistent paths."""
        nonexistent_path = Path("/nonexistent/path")
        
        with pytest.raises(ValueError, match="Path does not exist"):
            await self.discovery.discover_files(nonexistent_path)

    def test_supported_extensions_filtering(self):
        """Test supported extensions filtering."""
        # Test with different extensions
        assert self.discovery._is_supported_extension(Path("test.txt"))
        assert self.discovery._is_supported_extension(Path("test.py"))
        assert not self.discovery._is_supported_extension(Path("test.xyz"))
        assert not self.discovery._is_supported_extension(Path("test.bin"))

    def test_supported_extensions_case_insensitive(self):
        """Test that extension matching is case insensitive."""
        assert self.discovery._is_supported_extension(Path("test.TXT"))
        assert self.discovery._is_supported_extension(Path("test.Py"))
        assert self.discovery._is_supported_extension(Path("test.JSON"))

    def test_no_supported_extensions(self):
        """Test behavior when no supported extensions are specified."""
        discovery = FileDiscovery([])
        
        # Should allow all files when no extensions specified
        assert discovery._is_supported_extension(Path("test.anything"))
        assert discovery._is_supported_extension(Path("test.xyz"))

    def test_hidden_file_filtering(self):
        """Test that hidden files are filtered out."""
        include_regexes = []
        exclude_regexes = []
        
        # Hidden files should be excluded
        assert not self.discovery._should_include_file(
            Path(".hidden.txt"), include_regexes, exclude_regexes
        )
        
        # Regular files should be included
        assert self.discovery._should_include_file(
            Path("regular.txt"), include_regexes, exclude_regexes
        )

    @pytest.mark.asyncio
    async def test_file_info_creation(self, temp_directory):
        """Test FileInfo object creation with metadata."""
        test_file = temp_directory / "test.txt"
        
        file_info = await self.discovery._create_file_info(test_file)
        
        assert file_info is not None
        assert file_info.path == test_file
        assert file_info.size > 0
        assert isinstance(file_info.modified_time, datetime)
        assert file_info.encoding == 'utf-8'
        assert not file_info.is_binary
        assert file_info.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_binary_file_detection(self, temp_directory):
        """Test binary file detection and exclusion."""
        binary_file = temp_directory / "binary.bin"
        
        file_info = await self.discovery._create_file_info(binary_file)
        
        # Binary files should return None (excluded)
        assert file_info is None

    @pytest.mark.asyncio
    async def test_encoding_detection_utf8(self, temp_directory):
        """Test UTF-8 encoding detection."""
        utf8_file = temp_directory / "utf8.txt"
        utf8_file.write_text("Hello 世界", encoding='utf-8')
        
        encoding, is_binary = await self.discovery._detect_file_encoding(utf8_file)
        
        assert encoding == 'utf-8'
        assert not is_binary

    @pytest.mark.asyncio
    async def test_encoding_detection_latin1(self, temp_directory):
        """Test Latin-1 encoding detection."""
        latin1_file = temp_directory / "latin1.txt"
        with open(latin1_file, 'w', encoding='latin-1') as f:
            f.write("Café résumé")
        
        encoding, is_binary = await self.discovery._detect_file_encoding(latin1_file)
        
        # Should detect as latin-1 or similar encoding
        assert encoding in ['latin-1', 'cp1252', 'iso-8859-1']
        assert not is_binary

    @pytest.mark.asyncio
    async def test_encoding_detection_binary(self, temp_directory):
        """Test binary file detection."""
        binary_file = temp_directory / "binary.bin"
        
        encoding, is_binary = await self.discovery._detect_file_encoding(binary_file)
        
        assert encoding == 'binary'
        assert is_binary

    def test_discovery_stats_empty(self):
        """Test statistics for empty file list."""
        stats = self.discovery.get_discovery_stats([])
        
        expected_stats = {
            'total_files': 0,
            'total_size': 0,
            'estimated_tokens': 0,
            'extensions': {},
            'encodings': {}
        }
        
        assert stats == expected_stats

    def test_discovery_stats_with_files(self, tmp_path):
        """Test statistics calculation with files."""
        # Create actual test files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.py"
        file3 = tmp_path / "test3.txt"
        
        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")
        
        # Create FileInfo objects with real files
        files = [
            FileInfo(
                path=file1,
                size=100,
                modified_time=datetime.now(),
                encoding='utf-8',
                estimated_tokens=25
            ),
            FileInfo(
                path=file2,
                size=200,
                modified_time=datetime.now(),
                encoding='utf-8',
                estimated_tokens=50
            ),
            FileInfo(
                path=file3,
                size=150,
                modified_time=datetime.now(),
                encoding='latin-1',
                estimated_tokens=37
            )
        ]
        
        stats = self.discovery.get_discovery_stats(files)
        
        assert stats['total_files'] == 3
        assert stats['total_size'] == 450
        assert stats['estimated_tokens'] == 112
        assert stats['extensions'] == {'.txt': 2, '.py': 1}
        assert stats['encodings'] == {'utf-8': 2, 'latin-1': 1}

    @pytest.mark.asyncio
    async def test_permission_error_handling(self, temp_directory):
        """Test handling of permission errors during file access."""
        # Create a file and make it unreadable
        restricted_file = temp_directory / "restricted.txt"
        restricted_file.write_text("restricted content")
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            file_info = await self.discovery._create_file_info(restricted_file)
            
            # Should return None for inaccessible files
            assert file_info is None

    @pytest.mark.asyncio
    async def test_directory_permission_error_handling(self, temp_directory):
        """Test handling of permission errors during directory scanning."""
        # Create subdirectory
        subdir = temp_directory / "restricted_dir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")
        
        # Mock os.walk to raise permission error
        with patch('os.walk', side_effect=PermissionError("Access denied")):
            files = await self.discovery._scan_directory(
                temp_directory, [], [], recursive=True
            )
            
            # Should return empty list but not crash
            assert files == []

    @pytest.mark.asyncio
    async def test_pattern_matching_with_full_paths(self, temp_directory):
        """Test pattern matching works with full file paths."""
        # Create nested structure
        subdir = temp_directory / "docs" / "api"
        subdir.mkdir(parents=True)
        (subdir / "endpoint.md").write_text("API docs")
        
        # Pattern should match full path
        include_patterns = [r'docs/api/.*\.md$']
        
        files = await self.discovery.discover_files(
            temp_directory,
            include_patterns=include_patterns,
            recursive=True
        )
        
        assert len(files) == 1
        assert files[0].path.name == "endpoint.md"

    @pytest.mark.asyncio
    async def test_large_file_token_estimation(self, temp_directory):
        """Test token estimation for files of different sizes."""
        # Create files of different sizes
        small_file = temp_directory / "small.txt"
        large_file = temp_directory / "large.txt"
        
        small_content = "small"
        large_content = "x" * 10000
        
        small_file.write_text(small_content)
        large_file.write_text(large_content)
        
        small_info = await self.discovery._create_file_info(small_file)
        large_info = await self.discovery._create_file_info(large_file)
        
        # Token estimation should be roughly size / 4
        assert small_info.estimated_tokens == max(1, len(small_content) // 4)
        assert large_info.estimated_tokens == len(large_content) // 4
        assert large_info.estimated_tokens > small_info.estimated_tokens

    def test_compile_patterns_valid(self):
        """Test compilation of valid regex patterns."""
        patterns = [r'\.txt$', r'test.*', r'^/path/']
        compiled = self.discovery._compile_patterns(patterns)
        
        assert len(compiled) == 3
        assert all(hasattr(p, 'search') for p in compiled)

    def test_compile_patterns_invalid(self):
        """Test handling of invalid regex patterns."""
        patterns = [r'\.txt$', r'[invalid', r'valid.*']
        
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            self.discovery._compile_patterns(patterns)


class TestFileInfo:
    """Test cases for FileInfo data class."""

    def test_file_info_creation_valid(self, tmp_path):
        """Test FileInfo creation with valid data."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        file_info = FileInfo(
            path=test_file,
            size=100,
            modified_time=datetime.now(),
            encoding='utf-8',
            is_binary=False,
            estimated_tokens=25
        )
        
        assert file_info.path == test_file
        assert file_info.size == 100
        assert file_info.encoding == 'utf-8'
        assert not file_info.is_binary
        assert file_info.estimated_tokens == 25

    def test_file_info_validation_nonexistent_file(self):
        """Test FileInfo validation with nonexistent file."""
        nonexistent_file = Path("/nonexistent/file.txt")
        
        with pytest.raises(ValueError, match="File does not exist"):
            FileInfo(
                path=nonexistent_file,
                size=100,
                modified_time=datetime.now()
            )

    def test_file_info_validation_negative_size(self, tmp_path):
        """Test FileInfo validation with negative size."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        with pytest.raises(ValueError, match="Invalid file size"):
            FileInfo(
                path=test_file,
                size=-1,
                modified_time=datetime.now()
            )

    def test_file_info_defaults(self, tmp_path):
        """Test FileInfo default values."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        file_info = FileInfo(
            path=test_file,
            size=100,
            modified_time=datetime.now()
        )
        
        # Test default values
        assert file_info.encoding == 'utf-8'
        assert not file_info.is_binary
        assert file_info.estimated_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__])
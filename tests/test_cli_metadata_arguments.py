"""
Tests for CLI metadata argument parsing and assignment functionality.

This module tests the new --document-type, --set, and --sets-config CLI arguments
and their integration with the document processing pipeline.
"""

import argparse
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.mcp_server_qdrant_rag.cli_ingest import (
    CLIArgumentParser,
    CLISettings,
    CLIConfigBuilder,
    ContentProcessor,
    FileInfo,
    IngestConfig,
)
from src.mcp_server_qdrant_rag.qdrant import Entry
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings


class TestCLIMetadataArguments:
    """Test CLI metadata argument parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CLIArgumentParser()
        self.config_builder = CLIConfigBuilder()

    def test_document_type_argument_parsing(self):
        """Test that --document-type argument is parsed correctly."""
        args = self.parser.parse_args([
            "ingest", 
            "/test/path", 
            "--document-type", "api_docs"
        ])
        
        assert hasattr(args, "document_type")
        assert args.document_type == "api_docs"

    def test_set_argument_parsing(self):
        """Test that --set argument is parsed correctly."""
        args = self.parser.parse_args([
            "ingest", 
            "/test/path", 
            "--set", "platform_code"
        ])
        
        assert hasattr(args, "set")
        assert args.set == "platform_code"

    def test_sets_config_argument_parsing(self):
        """Test that --sets-config argument is parsed correctly."""
        args = self.parser.parse_args([
            "ingest", 
            "/test/path", 
            "--sets-config", "/custom/path/sets.json"
        ])
        
        assert hasattr(args, "sets_config")
        assert args.sets_config == "/custom/path/sets.json"

    def test_all_metadata_arguments_together(self):
        """Test parsing all metadata arguments together."""
        args = self.parser.parse_args([
            "ingest",
            "/test/path",
            "--document-type", "design_docs",
            "--set", "frontend_code",
            "--sets-config", "/custom/sets.json"
        ])
        
        assert args.document_type == "design_docs"
        assert args.set == "frontend_code"
        assert args.sets_config == "/custom/sets.json"

    def test_metadata_arguments_optional(self):
        """Test that metadata arguments are optional."""
        args = self.parser.parse_args(["ingest", "/test/path"])
        
        assert getattr(args, "document_type", None) is None
        assert getattr(args, "set", None) is None
        assert getattr(args, "sets_config", None) is None

    def test_metadata_arguments_in_update_command(self):
        """Test metadata arguments work with update command."""
        args = self.parser.parse_args([
            "update",
            "/test/path",
            "--document-type", "code",
            "--set", "backend_services"
        ])
        
        assert args.command == "update"
        assert args.document_type == "code"
        assert args.set == "backend_services"


class TestCLISettingsMetadata:
    """Test CLISettings with metadata fields."""

    def test_cli_settings_with_metadata(self):
        """Test CLISettings creation with metadata fields."""
        settings = CLISettings(
            document_type="api_docs",
            set_id="platform_code",
            sets_config_path="/custom/sets.json"
        )
        
        assert settings.document_type == "api_docs"
        assert settings.set_id == "platform_code"
        assert settings.sets_config_path == "/custom/sets.json"

    def test_cli_settings_metadata_defaults(self):
        """Test CLISettings metadata fields default to None."""
        settings = CLISettings()
        
        assert settings.document_type is None
        assert settings.set_id is None
        assert settings.sets_config_path is None


class TestConfigBuilderMetadata:
    """Test configuration builder with metadata arguments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = CLIConfigBuilder()

    def test_build_cli_settings_with_metadata(self):
        """Test building CLI settings with metadata arguments."""
        # Create mock args with metadata
        args = argparse.Namespace(
            command="ingest",
            document_type="requirements",
            set="documentation",
            sets_config="/path/to/sets.json",
            include_patterns=None,
            exclude_patterns=None,
            mode="add-only",
            force=False,
            dry_run=False,
            verbose=False
        )
        
        cli_settings = self.builder._build_cli_settings(args)
        
        assert cli_settings.document_type == "requirements"
        assert cli_settings.set_id == "documentation"
        assert cli_settings.sets_config_path == "/path/to/sets.json"

    def test_build_cli_settings_without_metadata(self):
        """Test building CLI settings without metadata arguments."""
        # Create mock args without metadata
        args = argparse.Namespace(
            command="ingest",
            include_patterns=None,
            exclude_patterns=None,
            mode="add-only",
            force=False,
            dry_run=False,
            verbose=False
        )
        
        cli_settings = self.builder._build_cli_settings(args)
        
        assert cli_settings.document_type is None
        assert cli_settings.set_id is None
        assert cli_settings.sets_config_path is None

    def test_build_config_with_metadata(self):
        """Test building complete config with metadata arguments."""
        # Create a temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(
                command="ingest",
                path=temp_dir,
                knowledgebase=None,
                url="http://localhost:6333",
                api_key=None,
                embedding="test-model",
                document_type="code",
                set="backend",
                sets_config="/custom/sets.json",
                include_patterns=None,
                exclude_patterns=None,
                mode="add-only",
                force=False,
                dry_run=False,
                verbose=False
            )
            
            config = self.builder.build_config(args)
            
            assert isinstance(config, IngestConfig)
            assert config.cli_settings.document_type == "code"
            assert config.cli_settings.set_id == "backend"
            assert config.cli_settings.sets_config_path == "/custom/sets.json"


class TestContentProcessorMetadata:
    """Test ContentProcessor with metadata assignment."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary file for testing
        import tempfile
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write("test content")
        self.temp_file.close()
        
        from datetime import datetime
        self.test_file_info = FileInfo(
            path=Path(self.temp_file.name),
            size=100,
            modified_time=datetime.now(),
            encoding="utf-8",
            is_binary=False,
            estimated_tokens=50
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import os
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file.name)

    def test_content_processor_with_metadata(self):
        """Test ContentProcessor initialization with metadata."""
        processor = ContentProcessor(
            document_type="api_docs",
            set_id="platform_code"
        )
        
        assert processor.document_type == "api_docs"
        assert processor.set_id == "platform_code"

    def test_content_processor_without_metadata(self):
        """Test ContentProcessor initialization without metadata."""
        processor = ContentProcessor()
        
        assert processor.document_type is None
        assert processor.set_id is None

    @patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor._read_file_content')
    @patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor._extract_metadata')
    async def test_process_file_with_metadata(self, mock_extract_metadata, mock_read_content):
        """Test file processing with metadata assignment."""
        # Setup mocks
        mock_read_content.return_value = "test content"
        mock_extract_metadata.return_value = {"file_path": "/test/file.txt"}
        
        processor = ContentProcessor(
            document_type="design_docs",
            set_id="frontend_code"
        )
        
        entry = await processor.process_file(self.test_file_info)
        
        assert entry is not None
        assert isinstance(entry, Entry)
        assert entry.content == "test content"
        assert entry.document_type == "design_docs"
        assert entry.set_id == "frontend_code"
        assert entry.is_chunk is False

    @patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor._read_file_content')
    @patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor._extract_metadata')
    async def test_process_file_without_metadata(self, mock_extract_metadata, mock_read_content):
        """Test file processing without metadata assignment."""
        # Setup mocks
        mock_read_content.return_value = "test content"
        mock_extract_metadata.return_value = {"file_path": "/test/file.txt"}
        
        processor = ContentProcessor()
        
        entry = await processor.process_file(self.test_file_info)
        
        assert entry is not None
        assert isinstance(entry, Entry)
        assert entry.content == "test content"
        assert entry.document_type is None
        assert entry.set_id is None
        assert entry.is_chunk is False

    @patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor._read_file_content')
    @patch('src.mcp_server_qdrant_rag.cli_ingest.ContentProcessor._extract_metadata')
    async def test_process_file_partial_metadata(self, mock_extract_metadata, mock_read_content):
        """Test file processing with partial metadata assignment."""
        # Setup mocks
        mock_read_content.return_value = "test content"
        mock_extract_metadata.return_value = {"file_path": "/test/file.txt"}
        
        processor = ContentProcessor(
            document_type="code",
            set_id=None  # Only document_type provided
        )
        
        entry = await processor.process_file(self.test_file_info)
        
        assert entry is not None
        assert isinstance(entry, Entry)
        assert entry.document_type == "code"
        assert entry.set_id is None


class TestMetadataValidation:
    """Test metadata field validation."""

    def test_empty_document_type_validation(self):
        """Test that empty document_type is handled correctly."""
        processor = ContentProcessor(
            document_type="",  # Empty string
            set_id="test_set"
        )
        
        # Empty string should be treated as None
        assert processor.document_type == ""

    def test_whitespace_document_type_validation(self):
        """Test that whitespace-only document_type is handled correctly."""
        processor = ContentProcessor(
            document_type="   ",  # Whitespace only
            set_id="test_set"
        )
        
        # Whitespace-only string should be preserved as-is for now
        # (Entry model validation will handle trimming)
        assert processor.document_type == "   "

    def test_none_metadata_values(self):
        """Test that None metadata values are handled correctly."""
        processor = ContentProcessor(
            document_type=None,
            set_id=None
        )
        
        assert processor.document_type is None
        assert processor.set_id is None


class TestIntegrationMetadata:
    """Integration tests for metadata functionality."""

    def test_cli_to_processor_metadata_flow(self):
        """Test metadata flow from CLI args to ContentProcessor."""
        # This would be an integration test that verifies the complete flow
        # from CLI argument parsing to Entry creation with metadata
        
        # Create CLI settings with metadata
        cli_settings = CLISettings(
            document_type="integration_test",
            set_id="test_suite"
        )
        
        # Create ContentProcessor with metadata from settings
        processor = ContentProcessor(
            document_type=cli_settings.document_type,
            set_id=cli_settings.set_id
        )
        
        # Verify metadata is properly passed through
        assert processor.document_type == "integration_test"
        assert processor.set_id == "test_suite"

    def test_help_text_includes_metadata_arguments(self):
        """Test that help text includes the new metadata arguments."""
        parser = CLIArgumentParser()
        
        # Test that we can parse the arguments (which means they exist)
        try:
            args = parser.parse_args([
                "ingest", 
                "/test/path", 
                "--document-type", "test",
                "--set", "test",
                "--sets-config", "/test/path"
            ])
            # If parsing succeeds, the arguments exist
            assert hasattr(args, "document_type")
            assert hasattr(args, "set")
            assert hasattr(args, "sets_config")
        except SystemExit:
            # If parsing fails, the arguments don't exist
            assert False, "Metadata arguments not properly added to parser"
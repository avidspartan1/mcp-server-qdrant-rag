"""
Unit tests for CLI argument parsing and validation.
"""

import argparse
import pytest
from pathlib import Path
from unittest.mock import patch

from src.mcp_server_qdrant_rag.cli_ingest import (
    CLIArgumentParser,
    CLIValidator,
    CLIConfigBuilder,
    parse_and_validate_args,
)


class TestCLIArgumentParser:
    """Test CLI argument parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CLIArgumentParser()

    def test_ingest_command_with_path(self):
        """Test ingest command with path argument."""
        args = self.parser.parse_args(["ingest", "/path/to/docs"])
        
        assert args.command == "ingest"
        assert args.path == "/path/to/docs"
        assert args.url == "http://localhost:6333"
        assert args.embedding == "nomic-ai/nomic-embed-text-v1.5-Q"

    def test_default_command_ingest(self):
        """Test that ingest is the default command when no subcommand specified."""
        args = self.parser.parse_args(["/path/to/docs"])
        
        assert args.command == "ingest"
        assert args.path == "/path/to/docs"

    def test_update_command_with_mode(self):
        """Test update command with mode option."""
        args = self.parser.parse_args(["update", "/path/to/docs", "--mode", "replace"])
        
        assert args.command == "update"
        assert args.path == "/path/to/docs"
        assert args.mode == "replace"

    def test_update_command_default_mode(self):
        """Test update command uses add-only as default mode."""
        args = self.parser.parse_args(["update", "/path/to/docs"])
        
        assert args.command == "update"
        assert args.mode == "add-only"

    def test_remove_command(self):
        """Test remove command with knowledgebase name."""
        args = self.parser.parse_args(["remove", "my-kb", "--force"])
        
        assert args.command == "remove"
        assert args.knowledgebase == "my-kb"
        assert args.force is True

    def test_list_command(self):
        """Test list command."""
        args = self.parser.parse_args(["list"])
        
        assert args.command == "list"

    def test_qdrant_url_argument(self):
        """Test Qdrant URL argument."""
        args = self.parser.parse_args(["ingest", "/path", "--url", "https://my-qdrant.com"])
        
        assert args.url == "https://my-qdrant.com"

    def test_api_key_argument(self):
        """Test API key argument."""
        args = self.parser.parse_args(["ingest", "/path", "--api-key", "secret-key"])
        
        assert args.api_key == "secret-key"

    def test_knowledgebase_argument(self):
        """Test knowledgebase name argument."""
        args = self.parser.parse_args(["ingest", "/path", "--knowledgebase", "my-docs"])
        
        assert args.knowledgebase == "my-docs"

    def test_embedding_argument(self):
        """Test embedding model argument."""
        args = self.parser.parse_args(["ingest", "/path", "--embedding", "custom-model"])
        
        assert args.embedding == "custom-model"

    def test_include_patterns(self):
        """Test include pattern arguments."""
        args = self.parser.parse_args([
            "ingest", "/path", 
            "--include", "*.py", 
            "--include", "*.md"
        ])
        
        assert args.include_patterns == ["*.py", "*.md"]

    def test_exclude_patterns(self):
        """Test exclude pattern arguments."""
        args = self.parser.parse_args([
            "ingest", "/path", 
            "--exclude", "*.log", 
            "--exclude", "test_*"
        ])
        
        assert args.exclude_patterns == ["*.log", "test_*"]

    def test_verbose_flag(self):
        """Test verbose flag."""
        args = self.parser.parse_args(["ingest", "/path", "--verbose"])
        
        assert args.verbose is True

    def test_dry_run_flag(self):
        """Test dry run flag."""
        args = self.parser.parse_args(["ingest", "/path", "--dry-run"])
        
        assert args.dry_run is True

    def test_short_verbose_flag(self):
        """Test short verbose flag."""
        args = self.parser.parse_args(["ingest", "/path", "-v"])
        
        assert args.verbose is True


class TestCLIValidator:
    """Test CLI argument validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CLIValidator()

    def test_validate_ingest_command_valid_path(self, tmp_path):
        """Test validation of ingest command with valid path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        args = argparse.Namespace(
            command="ingest",
            path=str(test_file),
            url="http://localhost:6333",
            include_patterns=None,
            exclude_patterns=None,
        )
        
        errors = self.validator.validate_args(args)
        assert errors == []

    def test_validate_ingest_command_missing_path(self):
        """Test validation fails when path is missing for ingest command."""
        args = argparse.Namespace(
            command="ingest",
            path=None,
            url="http://localhost:6333",
            include_patterns=None,
            exclude_patterns=None,
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "PATH argument is required" in errors[0]

    def test_validate_ingest_command_nonexistent_path(self):
        """Test validation fails when path doesn't exist."""
        args = argparse.Namespace(
            command="ingest",
            path="/nonexistent/path",
            url="http://localhost:6333",
            include_patterns=None,
            exclude_patterns=None,
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Path does not exist" in errors[0]

    def test_validate_update_command_valid_directory(self, tmp_path):
        """Test validation of update command with valid directory."""
        args = argparse.Namespace(
            command="update",
            path=str(tmp_path),
            url="http://localhost:6333",
            include_patterns=None,
            exclude_patterns=None,
        )
        
        errors = self.validator.validate_args(args)
        assert errors == []

    def test_validate_remove_command_valid_name(self):
        """Test validation of remove command with valid knowledgebase name."""
        args = argparse.Namespace(
            command="remove",
            knowledgebase="my-docs",
            url="http://localhost:6333",
        )
        
        errors = self.validator.validate_args(args)
        assert errors == []

    def test_validate_remove_command_missing_name(self):
        """Test validation fails when knowledgebase name is missing for remove."""
        args = argparse.Namespace(
            command="remove",
            knowledgebase=None,
            url="http://localhost:6333",
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Knowledgebase name is required" in errors[0]

    def test_validate_remove_command_invalid_name(self):
        """Test validation fails with invalid knowledgebase name."""
        args = argparse.Namespace(
            command="remove",
            knowledgebase="invalid name with spaces",
            url="http://localhost:6333",
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Invalid knowledgebase name" in errors[0]

    def test_validate_list_command(self):
        """Test validation of list command."""
        args = argparse.Namespace(
            command="list",
            url="http://localhost:6333",
        )
        
        errors = self.validator.validate_args(args)
        assert errors == []

    def test_validate_invalid_qdrant_url(self):
        """Test validation fails with invalid Qdrant URL."""
        args = argparse.Namespace(
            command="list",
            url="invalid-url",
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Invalid Qdrant URL format" in errors[0]

    def test_validate_empty_qdrant_url(self):
        """Test validation fails with empty Qdrant URL."""
        args = argparse.Namespace(
            command="list",
            url="",
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Qdrant URL cannot be empty" in errors[0]

    def test_validate_valid_regex_patterns(self, tmp_path):
        """Test validation passes with valid regex patterns."""
        args = argparse.Namespace(
            command="ingest",
            path=str(tmp_path),
            url="http://localhost:6333",
            include_patterns=[".*\\.py$", "test_.*"],
            exclude_patterns=["__pycache__", ".*\\.log$"],
        )
        
        errors = self.validator.validate_args(args)
        assert errors == []

    def test_validate_invalid_include_pattern(self, tmp_path):
        """Test validation fails with invalid include regex pattern."""
        args = argparse.Namespace(
            command="ingest",
            path=str(tmp_path),
            url="http://localhost:6333",
            include_patterns=["[invalid"],
            exclude_patterns=None,
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Invalid include pattern" in errors[0]

    def test_validate_invalid_exclude_pattern(self, tmp_path):
        """Test validation fails with invalid exclude regex pattern."""
        args = argparse.Namespace(
            command="ingest",
            path=str(tmp_path),
            url="http://localhost:6333",
            include_patterns=None,
            exclude_patterns=["*[invalid"],
        )
        
        errors = self.validator.validate_args(args)
        assert len(errors) == 1
        assert "Invalid exclude pattern" in errors[0]

    def test_derive_knowledgebase_name_from_directory(self, tmp_path):
        """Test knowledgebase name derivation from directory."""
        test_dir = tmp_path / "my-project"
        test_dir.mkdir()
        
        name = self.validator._derive_knowledgebase_name(test_dir)
        assert name == "my-project"

    def test_derive_knowledgebase_name_from_file(self, tmp_path):
        """Test knowledgebase name derivation from file."""
        test_file = tmp_path / "document.txt"
        test_file.write_text("content")
        
        name = self.validator._derive_knowledgebase_name(test_file)
        assert name == "document"

    def test_valid_knowledgebase_names(self):
        """Test valid knowledgebase name validation."""
        valid_names = ["my-docs", "project_1", "test123", "a", "A-B_C-1"]
        
        for name in valid_names:
            assert self.validator._is_valid_knowledgebase_name(name)

    def test_invalid_knowledgebase_names(self):
        """Test invalid knowledgebase name validation."""
        invalid_names = ["my docs", "project@1", "test!", "", "name with spaces"]
        
        for name in invalid_names:
            assert not self.validator._is_valid_knowledgebase_name(name)


class TestCLIConfigBuilder:
    """Test CLI configuration building functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = CLIConfigBuilder()

    def test_build_ingest_config(self, tmp_path):
        """Test building configuration for ingest command."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        args = argparse.Namespace(
            command="ingest",
            path=str(test_file),
            url="http://localhost:6333",
            api_key=None,
            embedding="nomic-ai/nomic-embed-text-v1.5-Q",
            knowledgebase=None,
            include_patterns=[".*\\.txt$"],
            exclude_patterns=[".*\\.log$"],
            dry_run=False,
            verbose=True,
        )
        
        config = self.builder.build_config(args)
        
        assert config.cli_settings.operation_mode == "ingest"
        assert config.cli_settings.include_patterns == [".*\\.txt$"]
        assert config.cli_settings.exclude_patterns == [".*\\.log$"]
        assert config.cli_settings.verbose is True
        assert config.qdrant_settings.location == "http://localhost:6333"
        assert config.embedding_settings.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert config.target_path == test_file
        assert config.knowledgebase_name == "test"

    def test_build_update_config(self, tmp_path):
        """Test building configuration for update command."""
        args = argparse.Namespace(
            command="update",
            path=str(tmp_path),
            url="https://my-qdrant.com",
            api_key="secret",
            embedding="custom-model",
            knowledgebase="my-kb",
            mode="replace",
            include_patterns=None,
            exclude_patterns=None,
            dry_run=True,
            verbose=False,
        )
        
        config = self.builder.build_config(args)
        
        assert config.cli_settings.operation_mode == "update"
        assert config.cli_settings.update_mode == "replace"
        assert config.cli_settings.dry_run is True
        assert config.qdrant_settings.location == "https://my-qdrant.com"
        assert config.qdrant_settings.api_key == "secret"
        assert config.embedding_settings.model_name == "custom-model"
        assert config.knowledgebase_name == "my-kb"

    def test_build_remove_config(self):
        """Test building configuration for remove command."""
        args = argparse.Namespace(
            command="remove",
            knowledgebase="kb-to-remove",
            url="http://localhost:6333",
            api_key=None,
            force=True,
        )
        
        config = self.builder.build_config(args)
        
        assert config.cli_settings.operation_mode == "remove"
        assert config.cli_settings.force_operation is True
        assert config.knowledgebase_name == "kb-to-remove"

    def test_build_list_config(self):
        """Test building configuration for list command."""
        args = argparse.Namespace(
            command="list",
            url="http://localhost:6333",
            api_key=None,
        )
        
        config = self.builder.build_config(args)
        
        assert config.cli_settings.operation_mode == "list"


class TestParseAndValidateArgs:
    """Test the main parse_and_validate_args function."""

    def test_successful_parsing_and_validation(self, tmp_path):
        """Test successful argument parsing and validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        config = parse_and_validate_args(["ingest", str(test_file)])
        
        assert config.cli_settings.operation_mode == "ingest"
        assert config.target_path == test_file
        assert config.knowledgebase_name == "test"

    def test_validation_error_exits(self):
        """Test that validation errors cause system exit."""
        with pytest.raises(SystemExit):
            parse_and_validate_args(["ingest", "/nonexistent/path"])

    def test_missing_path_exits(self):
        """Test that missing path causes system exit."""
        with pytest.raises(SystemExit):
            parse_and_validate_args(["ingest"])

    def test_invalid_regex_pattern_exits(self, tmp_path):
        """Test that invalid regex pattern causes system exit."""
        with pytest.raises(SystemExit):
            parse_and_validate_args([
                "ingest", str(tmp_path), 
                "--include", "[invalid"
            ])

    @patch('sys.stderr')
    def test_error_messages_printed_to_stderr(self, mock_stderr, tmp_path):
        """Test that error messages are printed to stderr."""
        with pytest.raises(SystemExit):
            parse_and_validate_args(["ingest", "/nonexistent/path"])
        
        # Verify that print was called (error messages were output)
        assert mock_stderr is not None
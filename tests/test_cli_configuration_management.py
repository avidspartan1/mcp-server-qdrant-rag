"""
Unit tests for CLI configuration management system.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import argparse

from src.mcp_server_qdrant_rag.cli_ingest import (
    CLISettings,
    IngestConfig,
    CLIConfigBuilder,
    ConfigurationManager,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings, EmbeddingProviderSettings


class TestCLISettings:
    """Test CLI settings functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli_settings = CLISettings()

    def test_default_values(self):
        """Test that CLI settings have correct default values."""
        assert self.cli_settings.operation_mode == "ingest"
        assert self.cli_settings.update_mode == "add-only"
        assert self.cli_settings.force_operation is False
        assert self.cli_settings.dry_run is False
        assert self.cli_settings.show_progress is True
        assert self.cli_settings.batch_size == 10
        assert self.cli_settings.verbose is False
        assert self.cli_settings.include_patterns == []
        assert self.cli_settings.exclude_patterns == []
        assert len(self.cli_settings.supported_extensions) > 0

    def test_derive_knowledgebase_name_from_directory(self, tmp_path):
        """Test knowledgebase name derivation from directory."""
        test_dir = tmp_path / "my-project"
        test_dir.mkdir()
        
        name = self.cli_settings.derive_knowledgebase_name(test_dir)
        assert name == "my-project"

    def test_derive_knowledgebase_name_from_file(self, tmp_path):
        """Test knowledgebase name derivation from file."""
        test_file = tmp_path / "document.txt"
        test_file.write_text("content")
        
        name = self.cli_settings.derive_knowledgebase_name(test_file)
        assert name == "document"

    def test_derive_knowledgebase_name_with_special_characters(self, tmp_path):
        """Test knowledgebase name derivation with special characters."""
        test_dir = tmp_path / "my project with spaces!"
        test_dir.mkdir()
        
        name = self.cli_settings.derive_knowledgebase_name(test_dir)
        assert name == "my-project-with-spaces"

    def test_derive_knowledgebase_name_with_multiple_hyphens(self, tmp_path):
        """Test knowledgebase name derivation removes multiple consecutive hyphens."""
        test_dir = tmp_path / "my---project"
        test_dir.mkdir()
        
        name = self.cli_settings.derive_knowledgebase_name(test_dir)
        assert name == "my-project"

    def test_derive_knowledgebase_name_removes_leading_trailing_hyphens(self, tmp_path):
        """Test knowledgebase name derivation removes leading/trailing hyphens."""
        test_dir = tmp_path / "-my-project-"
        test_dir.mkdir()
        
        name = self.cli_settings.derive_knowledgebase_name(test_dir)
        assert name == "my-project"

    def test_derive_knowledgebase_name_invalid_path(self):
        """Test knowledgebase name derivation with invalid path."""
        with pytest.raises(ValueError, match="Path must be a valid Path object"):
            self.cli_settings.derive_knowledgebase_name(None)

    def test_derive_knowledgebase_name_empty_name(self, tmp_path):
        """Test knowledgebase name derivation with path that results in empty name."""
        test_dir = tmp_path / "---"
        test_dir.mkdir()
        
        with pytest.raises(ValueError, match="cannot be sanitized to valid knowledgebase name"):
            self.cli_settings.derive_knowledgebase_name(test_dir)

    def test_sanitize_knowledgebase_name(self):
        """Test knowledgebase name sanitization."""
        test_cases = [
            ("my-project", "my-project"),
            ("my_project", "my_project"),
            ("my project", "my-project"),
            ("my@project#123", "my-project-123"),
            ("---test---", "test"),
            ("test--name", "test-name"),
        ]
        
        for input_name, expected in test_cases:
            result = self.cli_settings._sanitize_knowledgebase_name(input_name)
            assert result == expected


class TestIngestConfig:
    """Test IngestConfig functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli_settings = CLISettings()
        self.qdrant_settings = QdrantSettings()
        self.qdrant_settings.location = "http://localhost:6333"
        self.embedding_settings = EmbeddingProviderSettings()

    def test_valid_ingest_config(self, tmp_path):
        """Test creating valid ingest configuration."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        self.cli_settings.operation_mode = "ingest"
        
        config = IngestConfig(
            cli_settings=self.cli_settings,
            qdrant_settings=self.qdrant_settings,
            embedding_settings=self.embedding_settings,
            target_path=test_file,
        )
        
        assert config.knowledgebase_name == "test"
        assert config.qdrant_settings.collection_name == "test"

    def test_valid_remove_config(self):
        """Test creating valid remove configuration."""
        self.cli_settings.operation_mode = "remove"
        
        config = IngestConfig(
            cli_settings=self.cli_settings,
            qdrant_settings=self.qdrant_settings,
            embedding_settings=self.embedding_settings,
            knowledgebase_name="test-kb",
        )
        
        assert config.knowledgebase_name == "test-kb"
        assert config.qdrant_settings.collection_name == "test-kb"

    def test_valid_list_config(self):
        """Test creating valid list configuration."""
        self.cli_settings.operation_mode = "list"
        
        config = IngestConfig(
            cli_settings=self.cli_settings,
            qdrant_settings=self.qdrant_settings,
            embedding_settings=self.embedding_settings,
        )
        
        # List operation doesn't require target path or knowledgebase name
        assert config.target_path is None
        assert config.knowledgebase_name is None

    def test_ingest_config_missing_target_path(self):
        """Test ingest config validation fails without target path."""
        self.cli_settings.operation_mode = "ingest"
        
        with pytest.raises(ValueError, match="Target path is required"):
            IngestConfig(
                cli_settings=self.cli_settings,
                qdrant_settings=self.qdrant_settings,
                embedding_settings=self.embedding_settings,
            )

    def test_ingest_config_nonexistent_target_path(self):
        """Test ingest config validation fails with nonexistent target path."""
        self.cli_settings.operation_mode = "ingest"
        
        with pytest.raises(ValueError, match="Target path does not exist"):
            IngestConfig(
                cli_settings=self.cli_settings,
                qdrant_settings=self.qdrant_settings,
                embedding_settings=self.embedding_settings,
                target_path=Path("/nonexistent/path"),
            )

    def test_remove_config_missing_knowledgebase_name(self):
        """Test remove config validation fails without knowledgebase name."""
        self.cli_settings.operation_mode = "remove"
        
        with pytest.raises(ValueError, match="Knowledgebase name is required"):
            IngestConfig(
                cli_settings=self.cli_settings,
                qdrant_settings=self.qdrant_settings,
                embedding_settings=self.embedding_settings,
            )

    def test_config_missing_qdrant_url(self):
        """Test config validation fails without Qdrant URL."""
        self.cli_settings.operation_mode = "list"
        self.qdrant_settings.location = None
        
        with pytest.raises(ValueError, match="Qdrant URL is required"):
            IngestConfig(
                cli_settings=self.cli_settings,
                qdrant_settings=self.qdrant_settings,
                embedding_settings=self.embedding_settings,
            )

    def test_config_missing_embedding_model(self):
        """Test config validation fails without embedding model."""
        self.cli_settings.operation_mode = "list"
        self.embedding_settings.model_name = None
        
        with pytest.raises(ValueError, match="Embedding model name is required"):
            IngestConfig(
                cli_settings=self.cli_settings,
                qdrant_settings=self.qdrant_settings,
                embedding_settings=self.embedding_settings,
            )

    def test_get_effective_settings_summary(self, tmp_path):
        """Test getting effective settings summary."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        self.cli_settings.operation_mode = "ingest"
        self.cli_settings.verbose = True
        self.cli_settings.include_patterns = ["*.py"]
        
        config = IngestConfig(
            cli_settings=self.cli_settings,
            qdrant_settings=self.qdrant_settings,
            embedding_settings=self.embedding_settings,
            target_path=test_file,
        )
        
        summary = config.get_effective_settings_summary()
        
        assert summary["operation_mode"] == "ingest"
        assert summary["target_path"] == str(test_file)
        assert summary["knowledgebase_name"] == "test"
        assert summary["qdrant_url"] == "http://localhost:6333"
        assert summary["verbose"] is True
        assert summary["include_patterns"] == ["*.py"]

    def test_validate_for_operation_with_invalid_regex(self, tmp_path):
        """Test operation validation with invalid regex patterns."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        self.cli_settings.operation_mode = "ingest"
        self.cli_settings.include_patterns = ["[invalid"]
        
        config = IngestConfig(
            cli_settings=self.cli_settings,
            qdrant_settings=self.qdrant_settings,
            embedding_settings=self.embedding_settings,
            target_path=test_file,
        )
        
        errors = config.validate_for_operation()
        assert len(errors) == 1
        assert "Invalid include pattern" in errors[0]


class TestCLIConfigBuilder:
    """Test CLI configuration builder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = CLIConfigBuilder()

    def test_build_ingest_config(self, tmp_path):
        """Test building ingest configuration."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        args = argparse.Namespace(
            command="ingest",
            path=str(test_file),
            url="http://localhost:6333",
            api_key=None,
            embedding="nomic-ai/nomic-embed-text-v1.5-Q",
            knowledgebase=None,
            include_patterns=["*.txt"],
            exclude_patterns=["*.log"],
            dry_run=False,
            verbose=True,
        )
        
        config = self.builder.build_config(args)
        
        assert config.cli_settings.operation_mode == "ingest"
        assert config.cli_settings.include_patterns == ["*.txt"]
        assert config.cli_settings.exclude_patterns == ["*.log"]
        assert config.cli_settings.verbose is True
        assert config.qdrant_settings.location == "http://localhost:6333"
        assert config.embedding_settings.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert config.target_path == test_file
        assert config.knowledgebase_name == "test"

    def test_build_update_config_with_explicit_knowledgebase(self, tmp_path):
        """Test building update configuration with explicit knowledgebase name."""
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
        """Test building remove configuration."""
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
        assert config.target_path is None

    def test_build_list_config(self):
        """Test building list configuration."""
        args = argparse.Namespace(
            command="list",
            url="http://localhost:6333",
            api_key=None,
        )
        
        config = self.builder.build_config(args)
        
        assert config.cli_settings.operation_mode == "list"
        assert config.target_path is None
        assert config.knowledgebase_name is None

    def test_build_config_missing_path_for_ingest(self):
        """Test building config fails when path is missing for ingest."""
        args = argparse.Namespace(
            command="ingest",
            path=None,
            url="http://localhost:6333",
        )
        
        with pytest.raises(ValueError, match="Path is required for 'ingest' command"):
            self.builder.build_config(args)

    def test_build_config_missing_knowledgebase_for_remove(self):
        """Test building config fails when knowledgebase is missing for remove."""
        args = argparse.Namespace(
            command="remove",
            url="http://localhost:6333",
        )
        
        with pytest.raises(ValueError, match="Knowledgebase name is required for 'remove' command"):
            self.builder.build_config(args)


class TestConfigurationManager:
    """Test configuration manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()

    def test_create_config_from_valid_args(self, tmp_path):
        """Test creating configuration from valid arguments."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        args = argparse.Namespace(
            command="ingest",
            path=str(test_file),
            url="http://localhost:6333",
            api_key=None,
            embedding="nomic-ai/nomic-embed-text-v1.5-Q",
            knowledgebase=None,
            include_patterns=None,
            exclude_patterns=None,
            dry_run=False,
            verbose=False,
        )
        
        config = self.config_manager.create_config_from_args(args)
        
        assert config.cli_settings.operation_mode == "ingest"
        assert config.target_path == test_file
        assert config.knowledgebase_name == "test"

    def test_create_config_from_invalid_args(self):
        """Test creating configuration from invalid arguments fails."""
        args = argparse.Namespace(
            command="ingest",
            path="/nonexistent/path",
            url="http://localhost:6333",
        )
        
        with pytest.raises(ValueError, match="Invalid arguments"):
            self.config_manager.create_config_from_args(args)

    def test_create_config_with_overrides(self, tmp_path):
        """Test creating configuration with overrides."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        # Create base configuration
        cli_settings = CLISettings(operation_mode="ingest")
        qdrant_settings = QdrantSettings()
        qdrant_settings.location = "http://localhost:6333"
        embedding_settings = EmbeddingProviderSettings()
        
        base_config = IngestConfig(
            cli_settings=cli_settings,
            qdrant_settings=qdrant_settings,
            embedding_settings=embedding_settings,
            target_path=test_file,
        )
        
        # Create config with overrides
        new_config = self.config_manager.create_config_with_overrides(
            base_config,
            knowledgebase_name="overridden-name"
        )
        
        assert new_config.knowledgebase_name == "overridden-name"
        assert new_config.target_path == test_file  # Original value preserved

    def test_create_config_with_invalid_override(self, tmp_path):
        """Test creating configuration with invalid override fails."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        cli_settings = CLISettings(operation_mode="ingest")
        qdrant_settings = QdrantSettings()
        qdrant_settings.location = "http://localhost:6333"
        embedding_settings = EmbeddingProviderSettings()
        
        base_config = IngestConfig(
            cli_settings=cli_settings,
            qdrant_settings=qdrant_settings,
            embedding_settings=embedding_settings,
            target_path=test_file,
        )
        
        with pytest.raises(ValueError, match="Unknown configuration field"):
            self.config_manager.create_config_with_overrides(
                base_config,
                invalid_field="value"
            )

    def test_create_config_with_overrides_validation_failure(self, tmp_path):
        """Test creating configuration with overrides that fail validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        cli_settings = CLISettings(operation_mode="ingest")
        qdrant_settings = QdrantSettings()
        qdrant_settings.location = "http://localhost:6333"
        embedding_settings = EmbeddingProviderSettings()
        
        base_config = IngestConfig(
            cli_settings=cli_settings,
            qdrant_settings=qdrant_settings,
            embedding_settings=embedding_settings,
            target_path=test_file,
        )
        
        # Override with invalid path
        with pytest.raises(ValueError, match="Invalid configuration after overrides"):
            self.config_manager.create_config_with_overrides(
                base_config,
                target_path=Path("/nonexistent/path")
            )
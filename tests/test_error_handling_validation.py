"""
Unit tests for comprehensive error handling and validation.

This module tests the error handling and validation functionality added in task 9,
including configuration file validation, metadata validation, and graceful degradation.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.mcp_server_qdrant_rag.settings import (
    SetSettings,
    SetConfiguration,
    SetConfigurationError,
    SetConfigurationFileError,
    SetConfigurationValidationError,
    MetadataValidationError,
    validate_document_type,
    validate_set_id,
    validate_metadata_dict,
)
from src.mcp_server_qdrant_rag.semantic_matcher import (
    SemanticSetMatcher,
    NoMatchFoundError,
    AmbiguousMatchError,
)


class TestSetConfigurationValidation:
    """Test set configuration validation and error handling."""
    
    def test_valid_set_configuration(self):
        """Test that valid set configurations are accepted."""
        config = SetConfiguration(
            slug="test_set",
            description="Test Set Description",
            aliases=["test", "testing"]
        )
        assert config.slug == "test_set"
        assert config.description == "Test Set Description"
        assert config.aliases == ["test", "testing"]
    
    def test_empty_slug_validation(self):
        """Test that empty slugs are rejected."""
        with pytest.raises(ValueError, match="slug cannot be empty"):
            SetConfiguration(
                slug="",
                description="Test Description",
                aliases=[]
            )
    
    def test_whitespace_slug_validation(self):
        """Test that whitespace-only slugs are rejected."""
        with pytest.raises(ValueError, match="slug cannot be empty"):
            SetConfiguration(
                slug="   ",
                description="Test Description",
                aliases=[]
            )
    
    def test_invalid_slug_characters(self):
        """Test that slugs with invalid characters are rejected."""
        with pytest.raises(ValueError, match="must contain only alphanumeric characters"):
            SetConfiguration(
                slug="test@set",
                description="Test Description",
                aliases=[]
            )
    
    def test_empty_description_validation(self):
        """Test that empty descriptions are rejected."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            SetConfiguration(
                slug="test_set",
                description="",
                aliases=[]
            )
    
    def test_whitespace_description_validation(self):
        """Test that whitespace-only descriptions are rejected."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            SetConfiguration(
                slug="test_set",
                description="   ",
                aliases=[]
            )


class TestSetSettingsFileHandling:
    """Test set settings file handling with error scenarios."""
    
    def test_missing_file_creates_default(self):
        """Test that missing configuration file creates default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_sets.json"
            settings = SetSettings()
            
            # Load from non-existent file should create default
            settings.load_from_file(config_path)
            
            # Should have created the file
            assert config_path.exists()
            
            # Should have loaded default sets
            assert len(settings.sets) > 0
            assert "platform_code" in settings.sets
    
    def test_permission_error_handling(self):
        """Test handling of permission errors when accessing configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "readonly_sets.json"
            
            # Create a file and make it unreadable
            config_path.write_text('{"sets": {}}')
            config_path.chmod(0o000)
            
            settings = SetSettings()
            
            try:
                with pytest.raises(SetConfigurationFileError) as exc_info:
                    settings.load_from_file(config_path)
                
                assert "access" in str(exc_info.value)
                assert isinstance(exc_info.value.original_error, PermissionError)
            finally:
                # Restore permissions for cleanup
                config_path.chmod(0o644)
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_sets.json"
            
            # Write invalid JSON
            config_path.write_text('{"sets": {invalid json}')
            
            settings = SetSettings()
            
            # Should handle invalid JSON gracefully by creating default config
            settings.load_from_file(config_path)
            
            # Should have created backup and default config
            backup_files = list(config_path.parent.glob("*.backup_*.json"))
            assert len(backup_files) == 1
            
            # Should have default sets loaded
            assert len(settings.sets) > 0
    
    def test_corrupted_config_backup_creation(self):
        """Test that corrupted configuration files are backed up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "corrupted_sets.json"
            
            # Write corrupted JSON
            original_content = '{"sets": {"test": invalid}}'
            config_path.write_text(original_content)
            
            settings = SetSettings()
            settings.load_from_file(config_path)
            
            # Should have created a backup
            backup_files = list(config_path.parent.glob("*.backup_*.json"))
            assert len(backup_files) == 1
            
            # Backup should contain original content
            backup_content = backup_files[0].read_text()
            assert backup_content == original_content
    
    def test_partial_configuration_loading(self):
        """Test loading configuration with some invalid sets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "partial_sets.json"
            
            # Configuration with one valid and one invalid set
            config_data = {
                "version": "1.0",
                "sets": {
                    "valid_set": {
                        "slug": "valid_set",
                        "description": "Valid Set",
                        "aliases": ["valid"]
                    },
                    "invalid_set": {
                        "slug": "invalid_set",
                        "description": "",  # Invalid empty description
                        "aliases": []
                    }
                }
            }
            
            config_path.write_text(json.dumps(config_data))
            
            settings = SetSettings()
            settings.load_from_file(config_path)
            
            # Should have loaded only the valid set
            assert len(settings.sets) == 1
            assert "valid_set" in settings.sets
            assert "invalid_set" not in settings.sets
    
    def test_configuration_structure_validation(self):
        """Test validation of overall configuration structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "bad_structure.json"
            
            # Configuration with wrong structure
            config_data = ["not", "an", "object"]
            config_path.write_text(json.dumps(config_data))
            
            settings = SetSettings()
            settings.load_from_file(config_path)
            
            # Should fall back to default configuration
            assert len(settings.sets) > 0
            assert "platform_code" in settings.sets


class TestMetadataValidation:
    """Test metadata field validation."""
    
    def test_valid_document_type(self):
        """Test that valid document types are accepted."""
        valid_types = ["code", "documentation", "api_docs", "config", "test_files"]
        
        for doc_type in valid_types:
            result = validate_document_type(doc_type)
            assert result == doc_type
    
    def test_none_document_type(self):
        """Test that None document type is accepted."""
        result = validate_document_type(None)
        assert result is None
    
    def test_empty_document_type(self):
        """Test that empty document type is rejected."""
        with pytest.raises(MetadataValidationError, match="cannot be empty"):
            validate_document_type("")
    
    def test_whitespace_document_type(self):
        """Test that whitespace-only document type is rejected."""
        with pytest.raises(MetadataValidationError, match="cannot be empty"):
            validate_document_type("   ")
    
    def test_non_string_document_type(self):
        """Test that non-string document type is rejected."""
        with pytest.raises(MetadataValidationError, match="must be a string"):
            validate_document_type(123)
    
    def test_long_document_type(self):
        """Test that overly long document type is rejected."""
        long_type = "a" * 101
        with pytest.raises(MetadataValidationError, match="cannot exceed 100 characters"):
            validate_document_type(long_type)
    
    def test_invalid_characters_document_type(self):
        """Test that document type with invalid characters is rejected."""
        with pytest.raises(MetadataValidationError, match="contains invalid characters"):
            validate_document_type("doc@type!")
    
    def test_valid_set_id(self):
        """Test that valid set IDs are accepted."""
        valid_ids = ["platform_code", "api-docs", "frontend_ui", "test123"]
        
        for set_id in valid_ids:
            result = validate_set_id(set_id)
            assert result == set_id
    
    def test_none_set_id(self):
        """Test that None set ID is accepted."""
        result = validate_set_id(None)
        assert result is None
    
    def test_empty_set_id(self):
        """Test that empty set ID is rejected."""
        with pytest.raises(MetadataValidationError, match="cannot be empty"):
            validate_set_id("")
    
    def test_non_string_set_id(self):
        """Test that non-string set ID is rejected."""
        with pytest.raises(MetadataValidationError, match="must be a string"):
            validate_set_id(456)
    
    def test_long_set_id(self):
        """Test that overly long set ID is rejected."""
        long_id = "a" * 51
        with pytest.raises(MetadataValidationError, match="cannot exceed 50 characters"):
            validate_set_id(long_id)
    
    def test_invalid_format_set_id(self):
        """Test that set ID with invalid format is rejected."""
        with pytest.raises(MetadataValidationError, match="must be a valid slug"):
            validate_set_id("invalid set id")
    
    def test_set_id_validation_against_available_sets(self):
        """Test set ID validation against available sets."""
        available_sets = {
            "platform_code": {"description": "Platform Code"},
            "api_docs": {"description": "API Documentation"}
        }
        
        # Valid set ID should pass
        result = validate_set_id("platform_code", available_sets)
        assert result == "platform_code"
        
        # Invalid set ID should fail with suggestions
        with pytest.raises(MetadataValidationError, match="is not a configured set"):
            validate_set_id("unknown_set", available_sets)
    
    def test_metadata_dict_validation(self):
        """Test metadata dictionary validation."""
        # Valid metadata
        valid_metadata = {
            "key1": "value1",
            "key2": 123,
            "key3": True,
            "key4": None
        }
        result = validate_metadata_dict(valid_metadata)
        assert result == valid_metadata
        
        # None metadata
        result = validate_metadata_dict(None)
        assert result is None
        
        # Empty metadata
        result = validate_metadata_dict({})
        assert result is None
    
    def test_invalid_metadata_dict(self):
        """Test invalid metadata dictionary handling."""
        # Non-dict metadata
        with pytest.raises(MetadataValidationError, match="must be a dictionary"):
            validate_metadata_dict("not a dict")
        
        # Invalid key type
        with pytest.raises(MetadataValidationError, match="must be a string"):
            validate_metadata_dict({123: "value"})
        
        # Empty key
        with pytest.raises(MetadataValidationError, match="cannot be empty"):
            validate_metadata_dict({"": "value"})
        
        # Invalid value type
        with pytest.raises(MetadataValidationError, match="must be a string, number, boolean, or null"):
            validate_metadata_dict({"key": ["list", "not", "allowed"]})
        
        # Overly long string value
        long_value = "a" * 1001
        with pytest.raises(MetadataValidationError, match="cannot exceed 1000 characters"):
            validate_metadata_dict({"key": long_value})


class TestSemanticMatcherErrorHandling:
    """Test semantic matcher error handling."""
    
    def test_no_match_found_error(self):
        """Test NoMatchFoundError with available sets."""
        sets = {
            "platform_code": SetConfiguration(
                slug="platform_code",
                description="Platform Code",
                aliases=["platform"]
            )
        }
        
        matcher = SemanticSetMatcher(sets)
        
        with pytest.raises(NoMatchFoundError) as exc_info:
            # Use asyncio.run to test async method
            import asyncio
            asyncio.run(matcher.match_set("nonexistent"))
        
        error = exc_info.value
        assert error.query == "nonexistent"
        assert "platform_code" in error.available_sets[0]
    
    def test_ambiguous_match_error(self):
        """Test AmbiguousMatchError with multiple equal matches."""
        sets = {
            "code1": SetConfiguration(
                slug="code1",
                description="Code Set One",
                aliases=["code"]
            ),
            "code2": SetConfiguration(
                slug="code2", 
                description="Code Set Two",
                aliases=["code"]
            )
        }
        
        matcher = SemanticSetMatcher(sets)
        
        with pytest.raises(AmbiguousMatchError) as exc_info:
            import asyncio
            asyncio.run(matcher.match_set("code"))
        
        error = exc_info.value
        assert error.query == "code"
        assert len(error.matches) == 2
    
    def test_empty_query_handling(self):
        """Test handling of empty queries."""
        sets = {
            "test": SetConfiguration(
                slug="test",
                description="Test Set",
                aliases=[]
            )
        }
        
        matcher = SemanticSetMatcher(sets)
        
        with pytest.raises(NoMatchFoundError):
            import asyncio
            asyncio.run(matcher.match_set(""))
    
    def test_no_configured_sets(self):
        """Test handling when no sets are configured."""
        matcher = SemanticSetMatcher({})
        
        with pytest.raises(NoMatchFoundError) as exc_info:
            import asyncio
            asyncio.run(matcher.match_set("anything"))
        
        error = exc_info.value
        assert error.available_sets == []


class TestCLIMetadataValidation:
    """Test CLI metadata validation."""
    
    def test_cli_validator_import(self):
        """Test that CLI validator can import validation functions."""
        from src.mcp_server_qdrant_rag.cli_ingest import CLIValidator
        
        validator = CLIValidator()
        
        # Create mock args with metadata
        args = Mock()
        args.command = "ingest"
        args.path = "/tmp/test"
        args.url = "http://localhost:6333"
        args.document_type = "valid_type"
        args.set = "valid_set"
        args.sets_config = None
        args.include_patterns = None
        args.exclude_patterns = None
        args.knowledgebase = "test"
        
        # Should not raise any errors for valid metadata
        errors = validator._validate_metadata_arguments(args)
        assert len(errors) == 0
    
    def test_cli_invalid_document_type_validation(self):
        """Test CLI validation of invalid document type."""
        from src.mcp_server_qdrant_rag.cli_ingest import CLIValidator
        
        validator = CLIValidator()
        
        args = Mock()
        args.document_type = ""  # Invalid empty document type
        args.set = None
        args.sets_config = None
        
        errors = validator._validate_metadata_arguments(args)
        assert len(errors) == 1
        assert "Invalid document type" in errors[0]
    
    def test_cli_invalid_set_validation(self):
        """Test CLI validation of invalid set."""
        from src.mcp_server_qdrant_rag.cli_ingest import CLIValidator
        
        validator = CLIValidator()
        
        args = Mock()
        args.document_type = None
        args.set = "invalid set with spaces"  # Invalid set format
        args.sets_config = None
        
        errors = validator._validate_metadata_arguments(args)
        assert len(errors) == 1
        assert "Invalid set identifier" in errors[0]
    
    def test_cli_sets_config_path_validation(self):
        """Test CLI validation of sets config path."""
        from src.mcp_server_qdrant_rag.cli_ingest import CLIValidator
        
        validator = CLIValidator()
        
        args = Mock()
        args.document_type = None
        args.set = None
        args.sets_config = "/nonexistent/path.json"
        
        errors = validator._validate_metadata_arguments(args)
        assert len(errors) == 1
        assert "Invalid sets configuration path" in errors[0]


if __name__ == "__main__":
    pytest.main([__file__])
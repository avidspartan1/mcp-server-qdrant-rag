"""Tests for default configuration creation and validation."""

import json
import tempfile
from pathlib import Path
import pytest

from mcp_server_qdrant_rag.settings import SetSettings, SetConfiguration


class TestDefaultConfigurationCreation:
    """Test default configuration file creation with examples and documentation."""
    
    def test_create_default_config_file_structure(self):
        """Test that default configuration file has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_default.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            # Verify file was created
            assert config_path.exists()
            
            # Load and verify structure
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Check required top-level fields
            assert "version" in config_data
            assert "description" in config_data
            assert "sets" in config_data
            assert "_documentation" in config_data
            
            # Verify version
            assert config_data["version"] == "1.0"
            
            # Verify description
            assert isinstance(config_data["description"], str)
            assert len(config_data["description"]) > 0
    
    def test_default_config_documentation_structure(self):
        """Test that default configuration includes comprehensive documentation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_docs.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            docs = config_data["_documentation"]
            
            # Check documentation sections
            assert "overview" in docs
            assert "usage" in docs
            assert "configuration" in docs
            assert "validation_rules" in docs
            assert "examples" in docs
            
            # Verify usage section
            usage = docs["usage"]
            assert "ingestion" in usage
            assert "search" in usage
            assert "examples" in usage
            assert isinstance(usage["examples"], list)
            assert len(usage["examples"]) > 0
            
            # Verify configuration section
            config_docs = docs["configuration"]
            assert "slug" in config_docs
            assert "description" in config_docs
            assert "aliases" in config_docs
            
            # Verify validation rules
            validation = docs["validation_rules"]
            assert "slug" in validation
            assert "description" in validation
            assert "aliases" in validation
            
            # Verify examples section
            examples = docs["examples"]
            assert "minimal_set" in examples
            assert "full_set" in examples
            
            # Validate example structures
            minimal = examples["minimal_set"]
            assert "slug" in minimal
            assert "description" in minimal
            assert "aliases" in minimal
            
            full = examples["full_set"]
            assert "slug" in full
            assert "description" in full
            assert "aliases" in full
            assert isinstance(full["aliases"], list)
            assert len(full["aliases"]) > 0
    
    def test_default_config_sets_content(self):
        """Test that default configuration includes comprehensive set examples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_sets.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            sets = config_data["sets"]
            
            # Check that we have a good variety of default sets
            expected_sets = [
                "platform_code",
                "api_docs", 
                "frontend_code",
                "database_schema",
                "deployment_config",
                "test_code",
                "project_docs",
                "configuration"
            ]
            
            for expected_set in expected_sets:
                assert expected_set in sets, f"Missing expected set: {expected_set}"
            
            # Verify each set has required fields
            for set_slug, set_data in sets.items():
                assert "slug" in set_data
                assert "description" in set_data
                assert "aliases" in set_data
                
                # Verify slug matches key
                assert set_data["slug"] == set_slug
                
                # Verify description is meaningful
                assert isinstance(set_data["description"], str)
                assert len(set_data["description"]) > 10  # Reasonable description length
                
                # Verify aliases are provided and meaningful
                assert isinstance(set_data["aliases"], list)
                assert len(set_data["aliases"]) > 0  # All default sets should have aliases
    
    def test_default_config_sets_validation(self):
        """Test that all default sets pass validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_validation.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            # Verify all sets in memory are valid SetConfiguration objects
            assert len(settings.sets) > 0
            
            for slug, set_config in settings.sets.items():
                assert isinstance(set_config, SetConfiguration)
                assert set_config.slug == slug
                assert len(set_config.description.strip()) > 0
                assert isinstance(set_config.aliases, list)
                
                # Test that slug follows validation rules
                import re
                assert re.match(r'^[a-zA-Z0-9_-]+$', set_config.slug)
                assert len(set_config.slug) <= 50
    
    def test_default_config_aliases_coverage(self):
        """Test that default sets have good alias coverage for common terms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_aliases.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            # Collect all aliases from all sets
            all_aliases = []
            for set_config in settings.sets.values():
                all_aliases.extend(set_config.aliases)
            
            # Check for coverage of common development terms
            common_terms = [
                "api", "frontend", "backend", "database", "config", 
                "docs", "tests", "deployment", "ui", "documentation"
            ]
            
            for term in common_terms:
                # Check if term appears in any alias (case insensitive)
                found = any(term.lower() in alias.lower() for alias in all_aliases)
                assert found, f"Common term '{term}' not found in any aliases"
    
    def test_default_config_file_encoding(self):
        """Test that default configuration file uses proper UTF-8 encoding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_encoding.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            # Read file with explicit UTF-8 encoding
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verify content is valid JSON
            config_data = json.loads(content)
            assert isinstance(config_data, dict)
            
            # Verify file can be read without encoding issues
            with open(config_path, 'rb') as f:
                raw_content = f.read()
            
            # Should be valid UTF-8
            decoded_content = raw_content.decode('utf-8')
            assert len(decoded_content) > 0
    
    def test_default_config_json_formatting(self):
        """Test that default configuration file is properly formatted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_formatting.json"
            settings = SetSettings()
            
            settings.create_default_config(config_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that JSON is properly indented (should contain newlines and spaces)
            assert '\n' in content
            assert '  ' in content  # 2-space indentation
            
            # Verify it's valid JSON
            config_data = json.loads(content)
            assert isinstance(config_data, dict)
            
            # Re-format and compare to ensure consistent formatting
            reformatted = json.dumps(config_data, indent=2, ensure_ascii=False)
            
            # Content should be similar (allowing for minor differences in ordering)
            assert len(content) == len(reformatted)
    
    def test_create_default_config_error_handling(self):
        """Test error handling in default configuration creation."""
        settings = SetSettings()
        
        # Test with invalid path (should raise SetConfigurationFileError)
        from mcp_server_qdrant_rag.settings import SetConfigurationFileError
        
        # Try to create config in non-existent directory with no permissions
        invalid_path = Path("/root/nonexistent/config.json")
        
        with pytest.raises(SetConfigurationFileError) as exc_info:
            settings.create_default_config(invalid_path)
        
        error = exc_info.value
        assert error.file_path == invalid_path
        assert error.operation in ["create parent directory", "write", "create"]
    
    def test_default_config_memory_fallback(self):
        """Test that sets are loaded in memory even if file creation fails."""
        settings = SetSettings()
        
        # Try to create config in invalid location
        invalid_path = Path("/root/nonexistent/config.json")
        
        try:
            settings.create_default_config(invalid_path)
        except Exception:
            pass  # Expected to fail
        
        # Even if file creation failed, sets should be loaded in memory
        assert len(settings.sets) > 0
        assert "platform_code" in settings.sets
        assert "api_docs" in settings.sets


class TestDefaultConfigurationIntegration:
    """Test integration of default configuration with other components."""
    
    def test_load_from_file_creates_default_when_missing(self):
        """Test that load_from_file creates default configuration when file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing_config.json"
            settings = SetSettings()
            
            # File doesn't exist initially
            assert not config_path.exists()
            
            # Load should create default configuration
            settings.load_from_file(config_path)
            
            # File should now exist
            assert config_path.exists()
            
            # Should have loaded default sets
            assert len(settings.sets) > 0
            assert "platform_code" in settings.sets
            
            # Verify file content includes documentation
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            assert "_documentation" in config_data
    
    def test_default_config_reloading(self):
        """Test that default configuration can be reloaded properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "reload_test.json"
            settings = SetSettings()
            
            # Create default configuration
            settings.create_default_config(config_path)
            original_sets_count = len(settings.sets)
            
            # Clear sets and reload
            settings.sets = {}
            settings.load_from_file(config_path)
            
            # Should have reloaded all sets
            assert len(settings.sets) == original_sets_count
            assert "platform_code" in settings.sets
    
    def test_default_config_with_custom_path(self):
        """Test default configuration creation with custom file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "custom" / "my_sets.json"
            settings = SetSettings()
            
            # Should create parent directories
            settings.create_default_config(custom_path)
            
            assert custom_path.exists()
            assert custom_path.parent.exists()
            
            # Verify content
            with open(custom_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            assert "sets" in config_data
            assert len(config_data["sets"]) > 0
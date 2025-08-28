"""Unit tests for configuration parsing and validation."""

import pytest
import os
from unittest.mock import patch

from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings, 
    QdrantSettings, 
    ToolSettings,
    FilterableField
)
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.common.exceptions import ConfigurationValidationError


class TestConfigurationParsing:
    """Test configuration parsing from environment variables."""

    def test_embedding_provider_settings_defaults(self):
        """Test that EmbeddingProviderSettings has correct defaults."""
        settings = EmbeddingProviderSettings()
        
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert settings.enable_chunking is True
        assert settings.max_chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.chunk_strategy == "semantic"

    def test_embedding_provider_settings_from_env(self, monkeypatch):
        """Test loading EmbeddingProviderSettings from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("ENABLE_CHUNKING", "false")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "1024")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed")
        
        settings = EmbeddingProviderSettings()
        
        assert settings.model_name == "custom-model"
        assert settings.enable_chunking is False
        assert settings.max_chunk_size == 1024
        assert settings.chunk_overlap == 100
        assert settings.chunk_strategy == "fixed"

    def test_qdrant_settings_defaults(self):
        """Test that QdrantSettings has correct defaults."""
        settings = QdrantSettings()
        
        assert settings.location is None
        assert settings.api_key is None
        assert settings.collection_name is None
        assert settings.local_path is None
        assert settings.search_limit == 10
        assert settings.read_only is False
        assert settings.filterable_fields is None
        assert settings.allow_arbitrary_filter is False

    def test_qdrant_settings_from_env(self, monkeypatch):
        """Test loading QdrantSettings from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        monkeypatch.setenv("COLLECTION_NAME", "test-collection")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "20")
        monkeypatch.setenv("QDRANT_READ_ONLY", "true")
        monkeypatch.setenv("QDRANT_ALLOW_ARBITRARY_FILTER", "1")
        
        settings = QdrantSettings()
        
        assert settings.location == "http://localhost:6333"
        assert settings.api_key == "test-key"
        assert settings.collection_name == "test-collection"
        assert settings.search_limit == 20
        assert settings.read_only is True
        assert settings.allow_arbitrary_filter is True

    def test_tool_settings_defaults(self):
        """Test that ToolSettings has correct defaults."""
        settings = ToolSettings()
        
        assert "Keep the memory for later use" in settings.tool_store_description
        assert "Look up memories in Qdrant" in settings.tool_find_description

    def test_tool_settings_from_env(self, monkeypatch):
        """Test loading ToolSettings from environment variables."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        
        settings = ToolSettings()
        
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"

    def test_boolean_parsing_variations(self, monkeypatch):
        """Test that boolean environment variables are parsed correctly."""
        # Test various truthy values
        for value in ["true", "True", "TRUE", "1", "yes", "Yes", "YES", "on", "On", "ON"]:
            monkeypatch.setenv("ENABLE_CHUNKING", value)
            settings = EmbeddingProviderSettings()
            assert settings.enable_chunking is True, f"Failed for value: {value}"
        
        # Test various falsy values
        for value in ["false", "False", "FALSE", "0", "no", "No", "NO", "off", "Off", "OFF"]:
            monkeypatch.setenv("ENABLE_CHUNKING", value)
            settings = EmbeddingProviderSettings()
            assert settings.enable_chunking is False, f"Failed for value: {value}"

    def test_integer_parsing(self, monkeypatch):
        """Test that integer environment variables are parsed correctly."""
        monkeypatch.setenv("MAX_CHUNK_SIZE", "256")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "15")
        
        embedding_settings = EmbeddingProviderSettings()
        qdrant_settings = QdrantSettings()
        
        assert embedding_settings.max_chunk_size == 256
        assert embedding_settings.chunk_overlap == 25
        assert qdrant_settings.search_limit == 15

    def test_invalid_integer_parsing(self, monkeypatch):
        """Test that invalid integer values raise appropriate errors."""
        monkeypatch.setenv("MAX_CHUNK_SIZE", "not-a-number")
        
        with pytest.raises(Exception):  # Pydantic will raise a validation error
            EmbeddingProviderSettings()

    def test_filterable_fields_dict(self):
        """Test the filterable_fields_dict method."""
        field1 = FilterableField(name="category", description="Category field", field_type="keyword")
        field2 = FilterableField(name="score", description="Score field", field_type="float", condition=">=")
        
        settings = QdrantSettings(filterable_fields=[field1, field2])
        
        fields_dict = settings.filterable_fields_dict()
        assert len(fields_dict) == 2
        assert "category" in fields_dict
        assert "score" in fields_dict
        assert fields_dict["category"] == field1
        assert fields_dict["score"] == field2

    def test_filterable_fields_dict_with_conditions(self):
        """Test the filterable_fields_dict_with_conditions method."""
        field1 = FilterableField(name="category", description="Category field", field_type="keyword")  # No condition
        field2 = FilterableField(name="score", description="Score field", field_type="float", condition=">=")
        field3 = FilterableField(name="active", description="Active field", field_type="boolean", condition="==")
        
        settings = QdrantSettings(filterable_fields=[field1, field2, field3])
        
        fields_dict = settings.filterable_fields_dict_with_conditions()
        assert len(fields_dict) == 2  # Only fields with conditions
        assert "category" not in fields_dict  # No condition
        assert "score" in fields_dict
        assert "active" in fields_dict

    def test_local_path_conflict_validation(self, monkeypatch):
        """Test that local_path conflicts with URL/API key are detected."""
        # Test local_path with URL
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        
        with pytest.raises(ValueError) as exc_info:
            QdrantSettings()
        assert "local_path" in str(exc_info.value)
        assert "location" in str(exc_info.value)

    def test_local_path_with_api_key_conflict(self, monkeypatch):
        """Test that local_path conflicts with API key."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local")
        monkeypatch.setenv("QDRANT_API_KEY", "test-key")
        
        with pytest.raises(ValueError) as exc_info:
            QdrantSettings()
        assert "local_path" in str(exc_info.value)
        assert "api_key" in str(exc_info.value)

    def test_valid_local_path_configuration(self, monkeypatch):
        """Test that local_path works when no URL or API key is set."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local")
        
        settings = QdrantSettings()
        assert settings.local_path == "/path/to/local"
        assert settings.location is None
        assert settings.api_key is None

    def test_environment_variable_aliases(self, monkeypatch):
        """Test that environment variable aliases work correctly."""
        # Test all the validation_alias mappings
        monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")  # Use lowercase enum value
        monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
        monkeypatch.setenv("ENABLE_CHUNKING", "false")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "256")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed")
        
        settings = EmbeddingProviderSettings()
        
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "test-model"
        assert settings.enable_chunking is False
        assert settings.max_chunk_size == 256
        assert settings.chunk_overlap == 25
        assert settings.chunk_strategy == "fixed"

    def test_enum_parsing(self, monkeypatch):
        """Test that enum values are parsed correctly."""
        # Test the correct enum value
        monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED

    def test_empty_environment_variables(self, monkeypatch):
        """Test behavior with empty environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        monkeypatch.setenv("COLLECTION_NAME", "")
        
        embedding_settings = EmbeddingProviderSettings()
        qdrant_settings = QdrantSettings()
        
        # Empty strings should be treated as None or use defaults
        assert embedding_settings.model_name == ""  # Pydantic keeps empty string
        assert qdrant_settings.collection_name == ""

    def test_whitespace_handling(self, monkeypatch):
        """Test that whitespace in environment variables is handled correctly."""
        monkeypatch.setenv("EMBEDDING_MODEL", "  test-model  ")
        monkeypatch.setenv("COLLECTION_NAME", "\ttest-collection\n")
        
        embedding_settings = EmbeddingProviderSettings()
        qdrant_settings = QdrantSettings()
        
        # Pydantic should preserve whitespace (application should handle trimming if needed)
        assert embedding_settings.model_name == "  test-model  "
        assert qdrant_settings.collection_name == "\ttest-collection\n"

    def test_configuration_field_descriptions(self):
        """Test that configuration fields have proper descriptions."""
        # Test that key fields have descriptions
        embedding_settings = EmbeddingProviderSettings()
        
        # Check that the model has field info with descriptions
        fields = EmbeddingProviderSettings.model_fields
        
        assert "enable_chunking" in fields
        assert fields["enable_chunking"].description is not None
        assert "automatic document chunking" in fields["enable_chunking"].description.lower()
        
        assert "max_chunk_size" in fields
        assert fields["max_chunk_size"].description is not None
        assert "maximum size" in fields["max_chunk_size"].description.lower()

    def test_filterable_field_validation(self):
        """Test FilterableField validation."""
        # Valid field
        field = FilterableField(
            name="category",
            description="Category field",
            field_type="keyword",
            condition="==",
            required=True
        )
        
        assert field.name == "category"
        assert field.description == "Category field"
        assert field.field_type == "keyword"
        assert field.condition == "=="
        assert field.required is True

    def test_filterable_field_defaults(self):
        """Test FilterableField default values."""
        field = FilterableField(
            name="test",
            description="Test field",
            field_type="keyword"
        )
        
        assert field.condition is None
        assert field.required is False

    def test_research_backed_chunking_defaults(self):
        """Test that chunking defaults match research recommendations."""
        settings = EmbeddingProviderSettings()
        
        # Based on requirements: 512 tokens max, 50 token overlap
        assert settings.max_chunk_size == 512
        assert settings.chunk_overlap == 50
        
        # Semantic chunking for better coherence
        assert settings.chunk_strategy == "semantic"
        
        # Chunking enabled by default
        assert settings.enable_chunking is True
        
        # Overlap should be about 10% of chunk size
        overlap_percentage = (settings.chunk_overlap / settings.max_chunk_size) * 100
        assert 8 <= overlap_percentage <= 12  # Allow some tolerance around 10%
import pytest

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
        QdrantSettings()

    def test_minimal_config(self, monkeypatch):
        """Test loading minimal configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")

        settings = QdrantSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    def test_full_config(self, monkeypatch):
        """Test loading full configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        monkeypatch.setenv("COLLECTION_NAME", "my_memories")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "15")
        monkeypatch.setenv("QDRANT_READ_ONLY", "1")

        settings = QdrantSettings()
        assert settings.location == "http://qdrant.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.search_limit == 15
        assert settings.read_only is True

    def test_local_path_config(self, monkeypatch):
        """Test loading local path configuration from environment variables."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        settings = QdrantSettings()
        assert settings.local_path == "/path/to/local/qdrant"

    def test_local_path_is_exclusive_with_url(self, monkeypatch):
        """Test that local path cannot be set if Qdrant URL is provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        with pytest.raises(ValueError):
            QdrantSettings()

        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        with pytest.raises(ValueError):
            QdrantSettings()


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
        # Test chunking defaults
        assert settings.enable_chunking is True
        assert settings.max_chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.chunk_strategy == "semantic"

    def test_custom_values(self, monkeypatch):
        """Test loading custom values from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom_model")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "custom_model"

    def test_chunking_configuration(self, monkeypatch):
        """Test loading chunking configuration from environment variables."""
        monkeypatch.setenv("ENABLE_CHUNKING", "false")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "1024")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed")
        
        settings = EmbeddingProviderSettings()
        assert settings.enable_chunking is False
        assert settings.max_chunk_size == 1024
        assert settings.chunk_overlap == 100
        assert settings.chunk_strategy == "fixed"

    def test_max_chunk_size_validation(self, monkeypatch):
        """Test validation of max_chunk_size field."""
        # Test minimum boundary
        monkeypatch.setenv("MAX_CHUNK_SIZE", "49")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")  # Set valid overlap
        with pytest.raises(ValueError, match="max_chunk_size must be at least 50"):
            EmbeddingProviderSettings()
        
        # Test maximum boundary
        monkeypatch.setenv("MAX_CHUNK_SIZE", "8193")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")  # Set valid overlap
        with pytest.raises(ValueError, match="max_chunk_size must not exceed 8192"):
            EmbeddingProviderSettings()
        
        # Test valid values
        monkeypatch.setenv("MAX_CHUNK_SIZE", "50")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")  # Set valid overlap
        settings = EmbeddingProviderSettings()
        assert settings.max_chunk_size == 50
        
        monkeypatch.setenv("MAX_CHUNK_SIZE", "8192")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")  # Set valid overlap
        settings = EmbeddingProviderSettings()
        assert settings.max_chunk_size == 8192

    def test_chunk_overlap_validation(self, monkeypatch):
        """Test validation of chunk_overlap field."""
        # Test negative value
        monkeypatch.setenv("CHUNK_OVERLAP", "-1")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "512")  # Set valid chunk size
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            EmbeddingProviderSettings()
        
        # Test maximum boundary
        monkeypatch.setenv("CHUNK_OVERLAP", "1001")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "2000")  # Set larger chunk size
        with pytest.raises(ValueError, match="chunk_overlap must not exceed 1000"):
            EmbeddingProviderSettings()
        
        # Test valid values
        monkeypatch.setenv("CHUNK_OVERLAP", "0")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "512")  # Set valid chunk size
        settings = EmbeddingProviderSettings()
        assert settings.chunk_overlap == 0
        
        monkeypatch.setenv("CHUNK_OVERLAP", "1000")
        monkeypatch.setenv("MAX_CHUNK_SIZE", "2000")  # Set larger chunk size
        settings = EmbeddingProviderSettings()
        assert settings.chunk_overlap == 1000

    def test_chunk_strategy_validation(self, monkeypatch):
        """Test validation of chunk_strategy field."""
        # Test invalid strategy
        monkeypatch.setenv("CHUNK_STRATEGY", "invalid")
        with pytest.raises(ValueError, match="chunk_strategy must be one of"):
            EmbeddingProviderSettings()
        
        # Test valid strategies
        for strategy in ["semantic", "fixed", "sentence"]:
            monkeypatch.setenv("CHUNK_STRATEGY", strategy)
            settings = EmbeddingProviderSettings()
            assert settings.chunk_strategy == strategy

    def test_chunk_overlap_vs_size_validation(self, monkeypatch):
        """Test validation that chunk_overlap is smaller than max_chunk_size."""
        # Test overlap equal to size (should fail)
        monkeypatch.setenv("MAX_CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        with pytest.raises(ValueError, match="chunk_overlap .* must be smaller than max_chunk_size"):
            EmbeddingProviderSettings()
        
        # Test overlap larger than size (should fail)
        monkeypatch.setenv("MAX_CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "150")
        with pytest.raises(ValueError, match="chunk_overlap .* must be smaller than max_chunk_size"):
            EmbeddingProviderSettings()
        
        # Test valid configuration
        monkeypatch.setenv("MAX_CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "50")
        settings = EmbeddingProviderSettings()
        assert settings.max_chunk_size == 100
        assert settings.chunk_overlap == 50

    def test_chunking_environment_variables(self, monkeypatch):
        """Test that all chunking environment variables work correctly."""
        monkeypatch.setenv("ENABLE_CHUNKING", "0")  # Test falsy value
        monkeypatch.setenv("MAX_CHUNK_SIZE", "256")
        monkeypatch.setenv("CHUNK_OVERLAP", "25")
        monkeypatch.setenv("CHUNK_STRATEGY", "sentence")
        
        settings = EmbeddingProviderSettings()
        assert settings.enable_chunking is False
        assert settings.max_chunk_size == 256
        assert settings.chunk_overlap == 25
        assert settings.chunk_strategy == "sentence"

    def test_research_backed_defaults(self):
        """Test that default values match research-backed recommendations."""
        settings = EmbeddingProviderSettings()
        # Based on requirements: 512 tokens max, 50 token overlap
        assert settings.max_chunk_size == 512
        assert settings.chunk_overlap == 50
        # Semantic chunking for better coherence
        assert settings.chunk_strategy == "semantic"
        # Chunking enabled by default for better retrieval
        assert settings.enable_chunking is True


class TestToolSettings:
    def test_default_values(self):
        """Test that default values are set correctly when no env vars are provided."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_store_description(self, monkeypatch):
        """Test loading custom store description from environment variable."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_find_description(self, monkeypatch):
        """Test loading custom find description from environment variable."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == "Custom find description"

    def test_all_custom_values(self, monkeypatch):
        """Test loading all custom values from environment variables."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"

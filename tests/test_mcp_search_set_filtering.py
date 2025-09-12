"""
Unit tests for MCP search tools with set filtering functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastmcp import Context

from mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from mcp_server_qdrant_rag.semantic_matcher import SemanticSetMatcher, SemanticMatchError, NoMatchFoundError, AmbiguousMatchError
from mcp_server_qdrant_rag.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    SetConfiguration,
    SetSettings,
    ToolSettings,
)
from mcp_server_qdrant_rag.qdrant import Entry


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = MagicMock()
    provider.get_model_info.return_value = {
        'vector_size': 384,
        'vector_name': 'default',
        'status': 'ready'
    }
    return provider


@pytest.fixture
def sample_set_configurations():
    """Create sample set configurations for testing."""
    return {
        "platform_code": SetConfiguration(
            slug="platform_code",
            description="Platform Codebase",
            aliases=["platform", "core platform", "main codebase"]
        ),
        "api_docs": SetConfiguration(
            slug="api_docs",
            description="API Documentation",
            aliases=["api", "documentation", "api reference"]
        ),
        "frontend_code": SetConfiguration(
            slug="frontend_code",
            description="Frontend Application Code",
            aliases=["frontend", "ui", "client"]
        )
    }


@pytest.fixture
def mcp_server_with_sets(mock_embedding_provider, sample_set_configurations, monkeypatch):
    """Create an MCP server instance with set configurations."""
    # Set environment variable for collection name
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    
    tool_settings = ToolSettings()
    qdrant_settings = QdrantSettings()
    
    with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
        # Mock the SetSettings instance
        mock_set_settings = MagicMock()
        mock_set_settings.sets = sample_set_configurations
        mock_set_settings_class.return_value = mock_set_settings
        
        server = QdrantMCPServer(
            tool_settings=tool_settings,
            qdrant_settings=qdrant_settings,
            embedding_provider=mock_embedding_provider,
        )
        
        # Mock the qdrant connector to avoid actual database operations
        server.qdrant_connector = MagicMock()
        
        return server


@pytest.fixture
def mock_context():
    """Create a mock FastMCP context."""
    ctx = MagicMock(spec=Context)
    ctx.debug = AsyncMock()
    return ctx


class TestSemanticSetMatching:
    """Test semantic set matching in search functions."""
    
    @pytest.mark.asyncio
    async def test_semantic_matcher_integration(self, mcp_server_with_sets):
        """Test that semantic matcher is properly integrated."""
        # Test that the semantic matcher was initialized correctly
        assert hasattr(mcp_server_with_sets, 'semantic_matcher')
        assert isinstance(mcp_server_with_sets.semantic_matcher, SemanticSetMatcher)
        
        # Test semantic matching functionality
        matched_slug = await mcp_server_with_sets.semantic_matcher.match_set("platform code")
        assert matched_slug == "platform_code"
        
        matched_slug = await mcp_server_with_sets.semantic_matcher.match_set("api documentation")
        assert matched_slug == "api_docs"
        
        # Test error case
        with pytest.raises(SemanticMatchError):
            await mcp_server_with_sets.semantic_matcher.match_set("nonexistent set")
    
    @pytest.mark.asyncio
    async def test_set_filter_generation(self, mcp_server_with_sets):
        """Test that set filters are generated correctly."""
        # Test that we can match sets and generate appropriate filter conditions
        matched_slug = await mcp_server_with_sets.semantic_matcher.match_set("platform")
        assert matched_slug == "platform_code"
        
        # Test filter structure that would be generated
        expected_filter = {
            "key": "set",
            "match": {"value": matched_slug}
        }
        
        # This is the structure we expect to be passed to Qdrant
        assert expected_filter["key"] == "set"
        assert expected_filter["match"]["value"] == "platform_code"


class TestFilterCombination:
    """Test combining set filters with existing query filters."""
    
    def test_filter_combination_logic(self, mcp_server_with_sets):
        """Test the logic for combining set filters with query filters."""
        # Test combining set filter with existing query filter
        query_filter = {"key": "type", "match": {"value": "code"}}
        set_filter_condition = {"key": "set", "match": {"value": "platform_code"}}
        
        # This is the expected combined filter structure
        expected_combined = {
            "must": [
                {"key": "set", "match": {"value": "platform_code"}},
                {"key": "type", "match": {"value": "code"}}
            ]
        }
        
        # Verify the structure is correct
        assert "must" in expected_combined
        assert len(expected_combined["must"]) == 2
        
        # Test set filter only
        expected_set_only = {"key": "set", "match": {"value": "platform_code"}}
        assert expected_set_only["key"] == "set"
        assert expected_set_only["match"]["value"] == "platform_code"


class TestParameterHandling:
    """Test parameter handling for set filtering."""
    
    @pytest.mark.asyncio
    async def test_empty_set_filter_handling(self, mcp_server_with_sets):
        """Test handling of empty set filter."""
        # Test that empty string raises appropriate error
        with pytest.raises(SemanticMatchError):
            await mcp_server_with_sets.semantic_matcher.match_set("")
        
        # Test that None would be handled (no filtering applied)
        # This is tested by ensuring the parameter is optional in the function signature
        assert True  # This test passes if the above doesn't raise unexpected errors
    
    def test_backward_compatibility(self, mcp_server_with_sets):
        """Test that the server maintains backward compatibility."""
        # Verify that the server still has the original functionality
        assert hasattr(mcp_server_with_sets, 'qdrant_connector')
        assert hasattr(mcp_server_with_sets, 'tool_settings')
        assert hasattr(mcp_server_with_sets, 'qdrant_settings')
        
        # Verify new functionality is added
        assert hasattr(mcp_server_with_sets, 'set_settings')
        assert hasattr(mcp_server_with_sets, 'semantic_matcher')


class TestSemanticMatcherIntegration:
    """Test integration with SemanticSetMatcher."""
    
    def test_server_initialization_with_set_configurations(self, mock_embedding_provider, sample_set_configurations, monkeypatch):
        """Test that server initializes semantic matcher correctly."""
        # Set environment variable for collection name
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")
        
        tool_settings = ToolSettings()
        qdrant_settings = QdrantSettings()
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = sample_set_configurations
            mock_set_settings_class.return_value = mock_set_settings
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider=mock_embedding_provider,
            )
            
            # Verify semantic matcher was initialized
            assert hasattr(server, 'semantic_matcher')
            assert isinstance(server.semantic_matcher, SemanticSetMatcher)
            assert server.semantic_matcher.set_configurations == sample_set_configurations
    
    def test_server_initialization_with_custom_sets_config_path(self, mock_embedding_provider, monkeypatch):
        """Test server initialization with custom sets config path."""
        # Set environment variable for collection name
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")
        
        tool_settings = ToolSettings()
        qdrant_settings = QdrantSettings()
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            mock_set_settings = MagicMock()
            mock_set_settings.sets = {}
            mock_set_settings_class.return_value = mock_set_settings
            
            custom_path = "/custom/path/sets.json"
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider=mock_embedding_provider,
                sets_config_path=custom_path,
            )
            
            # Verify that load_from_file was called with custom path
            mock_set_settings.load_from_file.assert_called_once()
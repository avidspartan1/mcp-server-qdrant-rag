"""
Integration test for set filtering functionality.
"""

import pytest
import os
import uuid
from unittest.mock import patch, MagicMock

from mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from mcp_server_qdrant_rag.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    SetConfiguration,
    ToolSettings,
)


class TestSetFilteringIntegration:
    """Integration tests for set filtering functionality."""

    @pytest.mark.asyncio
    async def test_set_filtering_workflow(self, monkeypatch):
        """Test the complete set filtering workflow."""
        collection_name = f"test_set_filtering_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        # Mock set configurations
        sample_sets = {
            "platform_code": SetConfiguration(
                slug="platform_code",
                description="Platform Codebase",
                aliases=["platform", "core platform", "main codebase"]
            ),
            "api_docs": SetConfiguration(
                slug="api_docs",
                description="API Documentation",
                aliases=["api", "documentation", "api reference"]
            )
        }
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = sample_sets
            mock_set_settings_class.return_value = mock_set_settings
            
            # Create server
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Verify server initialization
            assert hasattr(server, 'semantic_matcher')
            assert server.semantic_matcher.set_configurations == sample_sets
            
            # Test semantic matching
            matched_slug = await server.semantic_matcher.match_set("platform code")
            assert matched_slug == "platform_code"
            
            matched_slug = await server.semantic_matcher.match_set("api documentation")
            assert matched_slug == "api_docs"
            
            # Test that the server has the expected configuration
            assert len(server.set_settings.sets) == 2
            assert "platform_code" in server.set_settings.sets
            assert "api_docs" in server.set_settings.sets

    def test_server_initialization_with_sets_config_path(self, monkeypatch):
        """Test server initialization with custom sets config path."""
        collection_name = f"test_custom_path_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            mock_set_settings = MagicMock()
            mock_set_settings.sets = {}
            mock_set_settings_class.return_value = mock_set_settings
            
            # Create server with custom sets config path
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            custom_path = "/custom/path/sets.json"
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
                sets_config_path=custom_path,
            )
            
            # Verify that load_from_file was called with the custom path
            mock_set_settings.load_from_file.assert_called_once()
            
            # Verify server has semantic matcher
            assert hasattr(server, 'semantic_matcher')

    def test_backward_compatibility(self, monkeypatch):
        """Test that the server maintains backward compatibility."""
        collection_name = f"test_backward_compat_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            mock_set_settings = MagicMock()
            mock_set_settings.sets = {}
            mock_set_settings_class.return_value = mock_set_settings
            
            # Create server without sets_config_path (backward compatibility)
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Verify all original functionality is preserved
            assert hasattr(server, 'tool_settings')
            assert hasattr(server, 'qdrant_settings')
            assert hasattr(server, 'qdrant_connector')
            assert hasattr(server, 'embedding_provider')
            
            # Verify new functionality is added
            assert hasattr(server, 'set_settings')
            assert hasattr(server, 'semantic_matcher')
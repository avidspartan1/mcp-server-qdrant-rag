"""
Test the compatibility analysis MCP tool.
"""

import pytest
import uuid
import os
from unittest.mock import AsyncMock, patch

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


class TestCompatibilityTool:
    """Test the compatibility analysis MCP tool."""

    @pytest.mark.asyncio
    async def test_compatibility_analysis_tool_empty_collection(self):
        """Test compatibility analysis tool with non-existent collection."""
        collection_name = f"test_compat_tool_{uuid.uuid4().hex}"
        
        # Create MCP server with environment variables
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:'
        }):
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Mock context
            ctx = AsyncMock()
            ctx.debug = AsyncMock()
            
            # Get the compatibility analysis tool
            server.setup_tools()
            
            # Test the compatibility analysis directly
            result = await server.qdrant_connector.analyze_collection_compatibility()
            
            assert result["exists"] is False
            assert result["compatible"] is True
            assert "will be created" in result["message"]

    @pytest.mark.asyncio
    async def test_compatibility_analysis_tool_existing_collection(self):
        """Test compatibility analysis tool with existing collection."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_compat_existing_{uuid.uuid4().hex}"
        
        # Create and populate collection
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
            enable_chunking=True,
        )
        
        # Store some content
        await connector.store(Entry(content="Test content for compatibility analysis"))
        
        # Create MCP server with same configuration
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:'
        }):
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Use the same client to simulate same Qdrant instance
            server.qdrant_connector._client = connector._client
            
            # Test the compatibility analysis directly
            result = await server.qdrant_connector.analyze_collection_compatibility()
            
            assert result["exists"] is True
            assert result["compatible"] is True
            assert result["points_count"] > 0

    @pytest.mark.asyncio
    async def test_compatibility_analysis_tool_with_custom_collection(self):
        """Test compatibility analysis tool with custom collection name."""
        collection_name = f"test_compat_custom_{uuid.uuid4().hex}"
        
        # Create MCP server without default collection
        with patch.dict(os.environ, {'QDRANT_URL': ':memory:'}, clear=True):
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()  # No default collection
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Test with custom collection name
            result = await server.qdrant_connector.analyze_collection_compatibility(collection_name)
            
            assert result["collection_name"] == collection_name
            assert result["exists"] is False
            assert result["compatible"] is True
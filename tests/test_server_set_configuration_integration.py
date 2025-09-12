"""
Integration tests for server initialization with set configurations.
Tests the complete integration of set configuration management into the MCP server.
"""

import pytest
import os
import uuid
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from mcp_server_qdrant_rag.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    SetConfiguration,
    SetSettings,
    ToolSettings,
)
from mcp_server_qdrant_rag.semantic_matcher import SemanticSetMatcher


class TestServerSetConfigurationIntegration:
    """Integration tests for server initialization with set configurations."""

    @pytest.fixture
    def sample_sets_config(self):
        """Sample set configurations for testing."""
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
    def temp_config_file(self, sample_sets_config):
        """Create a temporary configuration file for testing."""
        config_data = {
            "version": "1.0",
            "sets": {
                slug: {
                    "slug": config.slug,
                    "description": config.description,
                    "aliases": config.aliases
                }
                for slug, config in sample_sets_config.items()
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    def test_server_initialization_with_default_config_path(self, monkeypatch, sample_sets_config):
        """Test server initialization with default configuration path."""
        collection_name = f"test_default_config_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = sample_sets_config
            mock_set_settings_class.return_value = mock_set_settings
            
            # Create server without specifying sets_config_path
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Verify server initialization
            assert hasattr(server, 'set_settings')
            assert hasattr(server, 'semantic_matcher')
            assert isinstance(server.semantic_matcher, SemanticSetMatcher)
            
            # Verify set configurations are loaded
            assert server.set_settings.sets == sample_sets_config
            assert server.semantic_matcher.set_configurations == sample_sets_config
            
            # Verify load_from_file was called with default path
            mock_set_settings.load_from_file.assert_called_once()

    def test_server_initialization_with_custom_config_path(self, monkeypatch, sample_sets_config):
        """Test server initialization with custom sets config path."""
        collection_name = f"test_custom_config_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = sample_sets_config
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
            
            # Verify server initialization
            assert hasattr(server, 'set_settings')
            assert hasattr(server, 'semantic_matcher')
            
            # Verify load_from_file was called with the custom path
            expected_path = mock_set_settings.get_config_file_path.return_value
            mock_set_settings.get_config_file_path.assert_called_once_with(custom_path)
            mock_set_settings.load_from_file.assert_called_once_with(expected_path)

    def test_server_initialization_with_empty_config(self, monkeypatch):
        """Test server initialization with empty set configuration."""
        collection_name = f"test_empty_config_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance with empty sets
            mock_set_settings = MagicMock()
            mock_set_settings.sets = {}
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
            
            # Verify server initialization with empty configuration
            assert hasattr(server, 'set_settings')
            assert hasattr(server, 'semantic_matcher')
            assert server.set_settings.sets == {}
            assert server.semantic_matcher.set_configurations == {}

    @pytest.mark.asyncio
    async def test_configuration_reload_capability(self, monkeypatch, sample_sets_config):
        """Test configuration reload capability without server restart."""
        collection_name = f"test_reload_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the initial SetSettings instance
            initial_mock_settings = MagicMock()
            initial_mock_settings.sets = {"initial": SetConfiguration(slug="initial", description="Initial Set", aliases=[])}
            
            # Mock the reloaded SetSettings instance
            reloaded_mock_settings = MagicMock()
            reloaded_mock_settings.sets = sample_sets_config
            
            # Configure the mock to return different instances
            mock_set_settings_class.side_effect = [initial_mock_settings, reloaded_mock_settings]
            
            # Create server
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Verify initial configuration
            assert server.set_settings.sets == {"initial": initial_mock_settings.sets["initial"]}
            
            # Test configuration reload
            await server.reload_set_configurations()
            
            # Verify configuration was reloaded
            assert server.set_settings == reloaded_mock_settings
            
            # Verify semantic matcher was updated with new configurations
            assert server.semantic_matcher.set_configurations == sample_sets_config

    @pytest.mark.asyncio
    async def test_configuration_reload_with_custom_path(self, monkeypatch, sample_sets_config):
        """Test configuration reload with custom path."""
        collection_name = f"test_reload_custom_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the initial SetSettings instance
            initial_mock_settings = MagicMock()
            initial_mock_settings.sets = {}
            
            # Mock the reloaded SetSettings instance
            reloaded_mock_settings = MagicMock()
            reloaded_mock_settings.sets = sample_sets_config
            
            # Configure the mock to return different instances
            mock_set_settings_class.side_effect = [initial_mock_settings, reloaded_mock_settings]
            
            # Create server
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Test configuration reload with custom path
            custom_path = "/custom/reload/path.json"
            await server.reload_set_configurations(custom_path)
            
            # Verify new SetSettings instance was created and configured
            assert mock_set_settings_class.call_count == 2
            
            # Verify load_from_file was called with custom path
            expected_path = reloaded_mock_settings.get_config_file_path.return_value
            reloaded_mock_settings.get_config_file_path.assert_called_once_with(custom_path)
            reloaded_mock_settings.load_from_file.assert_called_once_with(expected_path)

    @pytest.mark.asyncio
    async def test_configuration_reload_error_handling(self, monkeypatch):
        """Test error handling during configuration reload."""
        collection_name = f"test_reload_error_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the initial SetSettings instance
            initial_mock_settings = MagicMock()
            initial_mock_settings.sets = {"initial": SetConfiguration(slug="initial", description="Initial Set", aliases=[])}
            
            # Mock the reloaded SetSettings instance to raise an error
            reloaded_mock_settings = MagicMock()
            reloaded_mock_settings.load_from_file.side_effect = Exception("Configuration load failed")
            
            # Configure the mock to return different instances
            mock_set_settings_class.side_effect = [initial_mock_settings, reloaded_mock_settings]
            
            # Create server
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Verify initial configuration
            original_settings = server.set_settings
            
            # Test configuration reload with error
            with pytest.raises(Exception, match="Configuration load failed"):
                await server.reload_set_configurations()
            
            # Verify original configuration is preserved on error
            assert server.set_settings == original_settings

    def test_server_startup_logging(self, monkeypatch, sample_sets_config, caplog):
        """Test that server startup logs set configuration information."""
        collection_name = f"test_logging_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = sample_sets_config
            mock_set_settings_class.return_value = mock_set_settings
            
            # Create server
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            with caplog.at_level("INFO"):
                server = QdrantMCPServer(
                    tool_settings=tool_settings,
                    qdrant_settings=qdrant_settings,
                    embedding_provider_settings=embedding_settings,
                )
            
            # Verify set configuration logging
            log_messages = [record.message for record in caplog.records]
            
            # Check for set configuration logging
            set_config_logs = [msg for msg in log_messages if "Set Configurations" in msg]
            assert len(set_config_logs) > 0
            
            # Check that individual sets are logged
            for slug, config in sample_sets_config.items():
                set_detail_logs = [msg for msg in log_messages if f"{slug}: {config.description}" in msg]
                assert len(set_detail_logs) > 0

    def test_server_tools_include_reload_capability(self, monkeypatch, sample_sets_config):
        """Test that server includes the reload sets config tool."""
        collection_name = f"test_tools_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = sample_sets_config
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
            
            # Verify that the reload tool is registered
            # Note: This is a basic check - in a real scenario, you'd want to test the actual tool registration
            assert hasattr(server, 'reload_set_configurations')

    def test_backward_compatibility_preserved(self, monkeypatch):
        """Test that all existing functionality is preserved with set configuration integration."""
        collection_name = f"test_backward_compat_{uuid.uuid4().hex}"
        
        # Set up environment
        monkeypatch.setenv("COLLECTION_NAME", collection_name)
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        
        with patch('mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
            # Mock the SetSettings instance
            mock_set_settings = MagicMock()
            mock_set_settings.sets = {}
            mock_set_settings_class.return_value = mock_set_settings
            
            # Create server without any set-related parameters (backward compatibility)
            tool_settings = ToolSettings()
            qdrant_settings = QdrantSettings()
            embedding_settings = EmbeddingProviderSettings()
            
            server = QdrantMCPServer(
                tool_settings=tool_settings,
                qdrant_settings=qdrant_settings,
                embedding_provider_settings=embedding_settings,
            )
            
            # Verify all original attributes are preserved
            assert hasattr(server, 'tool_settings')
            assert hasattr(server, 'qdrant_settings')
            assert hasattr(server, 'qdrant_connector')
            assert hasattr(server, 'embedding_provider')
            assert hasattr(server, 'embedding_provider_settings')
            
            # Verify new attributes are added
            assert hasattr(server, 'set_settings')
            assert hasattr(server, 'semantic_matcher')
            
            # Verify original functionality is not broken
            assert server.tool_settings == tool_settings
            assert server.qdrant_settings == qdrant_settings
            assert server.embedding_provider_settings == embedding_settings
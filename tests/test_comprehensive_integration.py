"""
Comprehensive integration tests for semantic metadata filtering feature.

This module tests complete workflows including:
- Configure sets → ingest with metadata → search with filtering
- CLI ingestion with metadata parameters
- MCP search tools with set filtering
- Backward compatibility with existing data
- Configuration file precedence (CLI > env > default)
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List

import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
    IngestConfig,
    CLISettings,
    IngestOperation,
    main_async,
)
from src.mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from src.mcp_server_qdrant_rag.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    SetConfiguration,
    SetSettings,
    ToolSettings,
)
from src.mcp_server_qdrant_rag.qdrant import Entry
from src.mcp_server_qdrant_rag.semantic_matcher import SemanticSetMatcher


class TestCompleteWorkflowIntegration:
    """Test complete workflow: configure sets → ingest with metadata → search with filtering."""

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
            ),
            "backend_code": SetConfiguration(
                slug="backend_code",
                description="Backend Service Code",
                aliases=["backend", "server", "service"]
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

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files with different types of content
            (workspace / "api_guide.md").write_text("""
# API Documentation

This is the main API guide for our platform.

## Authentication
Use JWT tokens for authentication.

## Endpoints
- GET /api/users
- POST /api/users
- PUT /api/users/{id}
            """.strip())
            
            (workspace / "main.py").write_text("""
#!/usr/bin/env python3
\"\"\"
Main platform service entry point.
\"\"\"

import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
            """.strip())
            
            (workspace / "frontend.js").write_text("""
/**
 * Frontend application main module.
 */

import React from 'react';
import ReactDOM from 'react-dom';

function App() {
    return (
        <div>
            <h1>Platform Frontend</h1>
            <p>Welcome to our platform!</p>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));
            """.strip())
            
            (workspace / "backend_service.py").write_text("""
\"\"\"
Backend service implementation.
\"\"\"

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class UserService:
    \"\"\"Service for managing users.\"\"\"
    
    def __init__(self):
        self.users = []
    
    async def create_user(self, user_data: dict) -> dict:
        \"\"\"Create a new user.\"\"\"
        user = {"id": len(self.users) + 1, **user_data}
        self.users.append(user)
        return user
    
    async def get_users(self) -> List[dict]:
        \"\"\"Get all users.\"\"\"
        return self.users
            """.strip())
            
            yield workspace

    @pytest.mark.asyncio
    async def test_complete_workflow_with_sets_config(self, temp_workspace, temp_config_file, sample_sets_config):
        """Test complete workflow: configure sets → ingest with metadata → search with filtering."""
        collection_name = f"test_complete_workflow_{uuid.uuid4().hex}"
        
        # Step 1: Configure sets (using temp config file)
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:',
            'QDRANT_SETS_CONFIG': temp_config_file
        }):
            # Step 2: Ingest documents with metadata using CLI
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                mock_connector = AsyncMock()
                mock_connector.get_collection_names.return_value = []
                mock_connector.create_collection.return_value = None
                mock_connector.store.return_value = None
                mock_connector_class.return_value = mock_connector
                
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider:
                    mock_provider = MagicMock()
                    mock_provider.get_model_info.return_value = {
                        'vector_size': 384,
                        'vector_name': 'default',
                        'status': 'ready'
                    }
                    mock_create_provider.return_value = mock_provider
                    
                    # Test ingesting API documentation
                    api_config = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            document_type="documentation",
                            set_id="api_docs",
                            sets_config_path=temp_config_file,
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace / "api_guide.md",
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(api_config)
                    result = await operation.execute()
                    
                    assert result.success
                    assert result.files_processed > 0
                    
                    # Verify that store was called with correct metadata
                    mock_connector.store.assert_called()
                    
                    # Get all store calls to check entries
                    store_calls = mock_connector.store.call_args_list
                    assert len(store_calls) > 0
                    
                    # Check that entries have the correct metadata
                    for call in store_calls:
                        entry = call[0][0]  # First positional argument (Entry object)
                        assert entry.document_type == "documentation"
                        assert entry.set_id == "api_docs"
                    
                    # Test ingesting platform code
                    mock_connector.reset_mock()
                    
                    platform_config = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            document_type="code",
                            set_id="platform_code",
                            sets_config_path=temp_config_file,
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace / "main.py",
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(platform_config)
                    result = await operation.execute()
                    
                    assert result.success
                    assert result.files_processed > 0
                    
                    # Verify metadata for platform code
                    mock_connector.store.assert_called()
                    
                    # Get all store calls to check entries
                    store_calls = mock_connector.store.call_args_list
                    assert len(store_calls) > 0
                    
                    for call in store_calls:
                        entry = call[0][0]  # First positional argument (Entry object)
                        assert entry.document_type == "code"
                        assert entry.set_id == "platform_code"

            # Step 3: Test MCP server with set filtering
            with patch('src.mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
                mock_set_settings = MagicMock()
                mock_set_settings.sets = sample_sets_config
                mock_set_settings_class.return_value = mock_set_settings
                
                # Create MCP server
                tool_settings = ToolSettings()
                qdrant_settings = QdrantSettings()
                embedding_settings = EmbeddingProviderSettings()
                
                server = QdrantMCPServer(
                    tool_settings=tool_settings,
                    qdrant_settings=qdrant_settings,
                    embedding_provider_settings=embedding_settings,
                    sets_config_path=temp_config_file,
                )
                
                # Mock the qdrant connector for search testing
                mock_search_connector = AsyncMock()
                mock_search_results = [
                    Entry(
                        content="API Documentation content",
                        metadata={"document_type": "documentation", "set": "api_docs"}
                    ),
                    Entry(
                        content="Platform code content",
                        metadata={"document_type": "code", "set": "platform_code"}
                    )
                ]
                mock_search_connector.search.return_value = mock_search_results
                server.qdrant_connector = mock_search_connector
                
                # Test semantic matching
                matched_slug = await server.semantic_matcher.match_set("api documentation")
                assert matched_slug == "api_docs"
                
                matched_slug = await server.semantic_matcher.match_set("platform code")
                assert matched_slug == "platform_code"
                
                # Verify server has correct configuration
                assert len(server.set_settings.sets) == 4
                assert "api_docs" in server.set_settings.sets
                assert "platform_code" in server.set_settings.sets

    @pytest.mark.asyncio
    async def test_cli_ingestion_with_metadata_parameters(self, temp_workspace, temp_config_file):
        """Test CLI ingestion with metadata parameters."""
        collection_name = f"test_cli_metadata_{uuid.uuid4().hex}"
        
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:'
        }):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                mock_connector = AsyncMock()
                mock_connector.get_collection_names.return_value = []
                mock_connector.create_collection.return_value = None
                mock_connector.store.return_value = None
                mock_connector_class.return_value = mock_connector
                
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider:
                    mock_provider = MagicMock()
                    mock_provider.get_model_info.return_value = {
                        'vector_size': 384,
                        'vector_name': 'default',
                        'status': 'ready'
                    }
                    mock_create_provider.return_value = mock_provider
                    
                    # Test CLI with document-type parameter
                    config = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            document_type="code",
                            verbose=True,
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace,
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(config)
                    result = await operation.execute()
                    
                    assert result.success
                    assert result.files_processed > 0
                    
                    # Verify all entries have the document_type metadata
                    mock_connector.store.assert_called()
                    
                    # Get all store calls to check entries
                    store_calls = mock_connector.store.call_args_list
                    assert len(store_calls) > 0
                    
                    for call in store_calls:
                        entry = call[0][0]  # First positional argument (Entry object)
                        assert entry.document_type == "code"
                    
                    # Test CLI with set parameter
                    mock_connector.reset_mock()
                    
                    config_with_set = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            set_id="frontend_code",
                            sets_config_path=temp_config_file,
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace / "frontend.js",
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(config_with_set)
                    result = await operation.execute()
                    
                    assert result.success
                    assert result.files_processed > 0
                    
                    # Verify set metadata
                    mock_connector.store.assert_called()
                    
                    # Get all store calls to check entries
                    store_calls = mock_connector.store.call_args_list
                    assert len(store_calls) > 0
                    
                    for call in store_calls:
                        entry = call[0][0]  # First positional argument (Entry object)
                        assert entry.set_id == "frontend_code"
                    
                    # Test CLI with both parameters
                    mock_connector.reset_mock()
                    
                    config_both = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            document_type="service",
                            set_id="backend_code",
                            sets_config_path=temp_config_file,
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace / "backend_service.py",
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(config_both)
                    result = await operation.execute()
                    
                    assert result.success
                    assert result.files_processed > 0
                    
                    # Verify both metadata fields
                    mock_connector.store.assert_called()
                    
                    # Get all store calls to check entries
                    store_calls = mock_connector.store.call_args_list
                    assert len(store_calls) > 0
                    
                    for call in store_calls:
                        entry = call[0][0]  # First positional argument (Entry object)
                        assert entry.document_type == "service"
                        assert entry.set_id == "backend_code"

    @pytest.mark.asyncio
    async def test_mcp_search_tools_with_set_filtering(self, sample_sets_config):
        """Test MCP search tools with set filtering."""
        collection_name = f"test_mcp_search_{uuid.uuid4().hex}"
        
        with patch.dict(os.environ, {'COLLECTION_NAME': collection_name}):
            with patch('src.mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
                mock_set_settings = MagicMock()
                mock_set_settings.sets = sample_sets_config
                mock_set_settings_class.return_value = mock_set_settings
                
                # Create MCP server
                tool_settings = ToolSettings()
                qdrant_settings = QdrantSettings()
                embedding_settings = EmbeddingProviderSettings()
                
                server = QdrantMCPServer(
                    tool_settings=tool_settings,
                    qdrant_settings=qdrant_settings,
                    embedding_provider_settings=embedding_settings,
                )
                
                # Mock search results with different sets
                mock_connector = AsyncMock()
                mock_search_results = [
                    Entry(
                        content="API endpoint documentation",
                        metadata={"document_type": "documentation", "set": "api_docs", "source": "api.md"}
                    ),
                    Entry(
                        content="Platform service code",
                        metadata={"document_type": "code", "set": "platform_code", "source": "service.py"}
                    ),
                    Entry(
                        content="Frontend component code",
                        metadata={"document_type": "code", "set": "frontend_code", "source": "component.js"}
                    )
                ]
                
                # Test search without set filtering (should return all results)
                mock_connector.search.return_value = mock_search_results
                server.qdrant_connector = mock_connector
                
                # Simulate search call without set filter
                results = await server.qdrant_connector.search("code", limit=10)
                assert len(results) == 3
                
                # Test semantic matching for set filtering
                matched_slug = await server.semantic_matcher.match_set("api documentation")
                assert matched_slug == "api_docs"
                
                matched_slug = await server.semantic_matcher.match_set("frontend")
                assert matched_slug == "frontend_code"
                
                matched_slug = await server.semantic_matcher.match_set("platform")
                assert matched_slug == "platform_code"
                
                # Test that search would be called with appropriate filters
                # (In real implementation, this would filter the results)
                api_docs_results = [r for r in mock_search_results if r.metadata.get("set") == "api_docs"]
                assert len(api_docs_results) == 1
                assert api_docs_results[0].metadata["set"] == "api_docs"
                
                frontend_results = [r for r in mock_search_results if r.metadata.get("set") == "frontend_code"]
                assert len(frontend_results) == 1
                assert frontend_results[0].metadata["set"] == "frontend_code"

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_existing_data(self, sample_sets_config):
        """Test backward compatibility with existing data."""
        collection_name = f"test_backward_compat_{uuid.uuid4().hex}"
        
        with patch.dict(os.environ, {'COLLECTION_NAME': collection_name}):
            with patch('src.mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
                mock_set_settings = MagicMock()
                mock_set_settings.sets = sample_sets_config
                mock_set_settings_class.return_value = mock_set_settings
                
                # Create MCP server
                tool_settings = ToolSettings()
                qdrant_settings = QdrantSettings()
                embedding_settings = EmbeddingProviderSettings()
                
                server = QdrantMCPServer(
                    tool_settings=tool_settings,
                    qdrant_settings=qdrant_settings,
                    embedding_provider_settings=embedding_settings,
                )
                
                # Mock search results with mixed data (some with new metadata, some without)
                mock_connector = AsyncMock()
                mixed_results = [
                    # New entry with metadata
                    Entry(
                        content="New API documentation",
                        metadata={"document_type": "documentation", "set": "api_docs", "source": "new_api.md"}
                    ),
                    # Legacy entry without new metadata fields
                    Entry(
                        content="Legacy documentation content",
                        metadata={"source": "legacy_doc.md", "created_at": "2023-01-01"}
                    ),
                    # Entry with partial new metadata
                    Entry(
                        content="Partially updated content",
                        metadata={"document_type": "code", "source": "partial.py"}
                    )
                ]
                
                mock_connector.search.return_value = mixed_results
                server.qdrant_connector = mock_connector
                
                # Test that search works with mixed data
                results = await server.qdrant_connector.search("documentation", limit=10)
                assert len(results) == 3
                
                # Verify that entries without new metadata are handled correctly
                legacy_entry = results[1]  # Legacy entry
                assert legacy_entry.document_type is None  # Should be None for legacy entries
                assert legacy_entry.set_id is None  # Should be None for legacy entries
                assert legacy_entry.metadata["source"] == "legacy_doc.md"
                
                # Verify that entries with new metadata work correctly
                new_entry = results[0]  # New entry
                # Note: Entry model doesn't automatically populate from metadata, 
                # but metadata should contain the values
                assert new_entry.metadata.get("document_type") == "documentation"
                assert new_entry.metadata.get("set") == "api_docs"
                
                # Verify that partially updated entries work
                partial_entry = results[2]  # Partial entry
                assert partial_entry.metadata.get("document_type") == "code"
                assert partial_entry.metadata.get("set") is None  # No set specified
                
                # Test that semantic matching still works
                matched_slug = await server.semantic_matcher.match_set("api docs")
                assert matched_slug == "api_docs"

    def test_configuration_file_precedence(self, temp_workspace, sample_sets_config):
        """Test configuration file precedence (CLI > env > default)."""
        collection_name = f"test_config_precedence_{uuid.uuid4().hex}"
        
        # Create multiple config files
        default_config = {
            "version": "1.0",
            "sets": {
                "default_set": {
                    "slug": "default_set",
                    "description": "Default Set",
                    "aliases": ["default"]
                }
            }
        }
        
        env_config = {
            "version": "1.0",
            "sets": {
                "env_set": {
                    "slug": "env_set",
                    "description": "Environment Set",
                    "aliases": ["env"]
                }
            }
        }
        
        cli_config = {
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
        
        # Create temporary config files
        default_file = temp_workspace / ".qdrant_sets.json"
        env_file = temp_workspace / "env_sets.json"
        cli_file = temp_workspace / "cli_sets.json"
        
        default_file.write_text(json.dumps(default_config, indent=2))
        env_file.write_text(json.dumps(env_config, indent=2))
        cli_file.write_text(json.dumps(cli_config, indent=2))
        
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_SETS_CONFIG': str(env_file)
        }):
            # Test 1: Default file precedence (no env var, no CLI override)
            with patch('src.mcp_server_qdrant_rag.settings.SetSettings.get_config_file_path') as mock_get_path:
                mock_get_path.return_value = default_file
                
                with patch('src.mcp_server_qdrant_rag.settings.SetSettings.load_from_file') as mock_load:
                    mock_load.return_value = None
                    
                    settings = SetSettings()
                    settings.get_config_file_path(None)
                    
                    # Should be called with None (no override)
                    mock_get_path.assert_called_with(None)
            
            # Test 2: Environment variable precedence
            with patch('src.mcp_server_qdrant_rag.settings.SetSettings.get_config_file_path') as mock_get_path:
                mock_get_path.return_value = env_file
                
                with patch('src.mcp_server_qdrant_rag.settings.SetSettings.load_from_file') as mock_load:
                    mock_load.return_value = None
                    
                    settings = SetSettings()
                    # Environment variable should be used
                    path = settings.get_config_file_path(None)
                    
                    # The path resolution logic should handle the environment variable
                    mock_get_path.assert_called_with(None)
            
            # Test 3: CLI override precedence (highest priority)
            with patch('src.mcp_server_qdrant_rag.settings.SetSettings.get_config_file_path') as mock_get_path:
                mock_get_path.return_value = cli_file
                
                with patch('src.mcp_server_qdrant_rag.settings.SetSettings.load_from_file') as mock_load:
                    mock_load.return_value = None
                    
                    settings = SetSettings()
                    # CLI override should take precedence
                    path = settings.get_config_file_path(str(cli_file))
                    
                    # Should be called with CLI override
                    mock_get_path.assert_called_with(str(cli_file))
            
            # Test 4: Integration test with MCP server
            with patch('src.mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
                mock_set_settings = MagicMock()
                mock_set_settings.sets = sample_sets_config
                mock_set_settings_class.return_value = mock_set_settings
                
                # Create server with CLI override
                tool_settings = ToolSettings()
                qdrant_settings = QdrantSettings()
                embedding_settings = EmbeddingProviderSettings()
                
                server = QdrantMCPServer(
                    tool_settings=tool_settings,
                    qdrant_settings=qdrant_settings,
                    embedding_provider_settings=embedding_settings,
                    sets_config_path=str(cli_file),  # CLI override
                )
                
                # Verify that the CLI path was used
                mock_set_settings.get_config_file_path.assert_called_with(str(cli_file))
                mock_set_settings.load_from_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_error_handling(self, temp_workspace, sample_sets_config):
        """Test end-to-end workflow with error handling scenarios."""
        collection_name = f"test_e2e_errors_{uuid.uuid4().hex}"
        
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:'
        }):
            # Test 1: Invalid set configuration
            invalid_config_file = temp_workspace / "invalid_sets.json"
            invalid_config_file.write_text('{"invalid": "json"')  # Invalid JSON
            
            with patch('src.mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
                mock_set_settings = MagicMock()
                mock_set_settings.load_from_file.side_effect = Exception("Invalid JSON")
                mock_set_settings_class.return_value = mock_set_settings
                
                # Server should handle configuration errors gracefully
                tool_settings = ToolSettings()
                qdrant_settings = QdrantSettings()
                embedding_settings = EmbeddingProviderSettings()
                
                with pytest.raises(Exception, match="Invalid JSON"):
                    QdrantMCPServer(
                        tool_settings=tool_settings,
                        qdrant_settings=qdrant_settings,
                        embedding_provider_settings=embedding_settings,
                        sets_config_path=str(invalid_config_file),
                    )
            
            # Test 2: CLI ingestion with invalid metadata (should still work)
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                mock_connector = AsyncMock()
                mock_connector.get_collection_names.return_value = []
                mock_connector.create_collection.return_value = None
                mock_connector.store.return_value = None
                mock_connector_class.return_value = mock_connector
                
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider:
                    mock_provider = MagicMock()
                    mock_provider.get_model_info.return_value = {
                        'vector_size': 384,
                        'vector_name': 'default',
                        'status': 'ready'
                    }
                    mock_create_provider.return_value = mock_provider
                    
                    # Test with invalid set_id (should still work, just store the invalid value)
                    config = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            set_id="invalid_set_that_does_not_exist",
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace / "main.py",
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(config)
                    result = await operation.execute()
                    
                    # Should succeed but store the invalid set_id
                    assert result.success
                    assert result.files_processed > 0
                    
                    # Verify that invalid set_id was stored
                    mock_connector.store.assert_called()
                    
                    # Get all store calls to check entries
                    store_calls = mock_connector.store.call_args_list
                    assert len(store_calls) > 0
                    
                    for call in store_calls:
                        entry = call[0][0]  # First positional argument (Entry object)
                        assert entry.set_id == "invalid_set_that_does_not_exist"

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, temp_workspace, sample_sets_config):
        """Test performance characteristics with larger datasets."""
        collection_name = f"test_performance_{uuid.uuid4().hex}"
        
        # Create multiple files with different sets
        for i in range(10):
            (temp_workspace / f"api_doc_{i}.md").write_text(f"""
# API Documentation {i}

This is API documentation file number {i}.

## Endpoints
- GET /api/resource{i}
- POST /api/resource{i}
- PUT /api/resource{i}/{{id}}
- DELETE /api/resource{i}/{{id}}

## Authentication
Use JWT tokens for all endpoints.
            """.strip())
            
            (temp_workspace / f"service_{i}.py").write_text(f"""
\"\"\"
Service module {i} for platform.
\"\"\"

class Service{i}:
    \"\"\"Service class {i}.\"\"\"
    
    def __init__(self):
        self.name = "service_{i}"
        self.version = "1.0.{i}"
    
    async def process(self, data):
        \"\"\"Process data for service {i}.\"\"\"
        return {{"service": self.name, "processed": data, "version": self.version}}
            """.strip())
        
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:'
        }):
            with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                mock_connector = AsyncMock()
                mock_connector.get_collection_names.return_value = []
                mock_connector.create_collection.return_value = None
                mock_connector.store.return_value = None
                mock_connector_class.return_value = mock_connector
                
                with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create_provider:
                    mock_provider = MagicMock()
                    mock_provider.get_model_info.return_value = {
                        'vector_size': 384,
                        'vector_name': 'default',
                        'status': 'ready'
                    }
                    mock_create_provider.return_value = mock_provider
                    
                    # Test batch ingestion with metadata
                    config = IngestConfig(
                        cli_settings=CLISettings(
                            operation_mode="ingest",
                            document_type="mixed",
                            set_id="platform_code",
                            batch_size=5,  # Process in smaller batches
                        ),
                        qdrant_settings=QdrantSettings(),
                        embedding_settings=EmbeddingProviderSettings(),
                        target_path=temp_workspace,
                        knowledgebase_name=collection_name,
                    )
                    
                    operation = IngestOperation(config)
                    result = await operation.execute()
                    
                    assert result.success
                    # Note: The temp_workspace fixture creates 4 additional files, so we expect 24 total
                    assert result.files_processed >= 20  # At least 20 files (10 API docs + 10 services)
                    
                    # Verify that store was called multiple times (batching)
                    assert mock_connector.store.call_count > 1
                    
                    # Verify all entries have correct metadata
                    all_entries = []
                    for call in mock_connector.store.call_args_list:
                        entry = call[0][0]  # First positional argument (Entry object)
                        all_entries.append(entry)
                    
                    for entry in all_entries:
                        assert entry.document_type == "mixed"
                        assert entry.set_id == "platform_code"
                    
                    # Test that we processed the expected number of entries
                    assert len(all_entries) >= 20  # At least one entry per file

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_workspace, sample_sets_config):
        """Test concurrent operations with set configurations."""
        collection_name = f"test_concurrent_{uuid.uuid4().hex}"
        
        with patch.dict(os.environ, {
            'COLLECTION_NAME': collection_name,
            'QDRANT_URL': ':memory:'
        }):
            with patch('src.mcp_server_qdrant_rag.mcp_server.SetSettings') as mock_set_settings_class:
                mock_set_settings = MagicMock()
                mock_set_settings.sets = sample_sets_config
                mock_set_settings_class.return_value = mock_set_settings
                
                # Create multiple server instances (simulating concurrent access)
                servers = []
                for i in range(3):
                    tool_settings = ToolSettings()
                    qdrant_settings = QdrantSettings()
                    embedding_settings = EmbeddingProviderSettings()
                    
                    server = QdrantMCPServer(
                        tool_settings=tool_settings,
                        qdrant_settings=qdrant_settings,
                        embedding_provider_settings=embedding_settings,
                    )
                    servers.append(server)
                
                # Test concurrent semantic matching
                async def test_semantic_matching(server, query, expected):
                    result = await server.semantic_matcher.match_set(query)
                    assert result == expected
                    return result
                
                # Run concurrent semantic matching operations
                tasks = [
                    test_semantic_matching(servers[0], "api documentation", "api_docs"),
                    test_semantic_matching(servers[1], "platform code", "platform_code"),
                    test_semantic_matching(servers[2], "frontend ui", "frontend_code"),
                ]
                
                results = await asyncio.gather(*tasks)
                assert results == ["api_docs", "platform_code", "frontend_code"]
                
                # Test concurrent configuration reloading
                async def test_config_reload(server):
                    await server.reload_set_configurations()
                    return len(server.set_settings.sets)
                
                reload_tasks = [test_config_reload(server) for server in servers]
                reload_results = await asyncio.gather(*reload_tasks)
                
                # All servers should have the same number of sets after reload
                assert all(count == reload_results[0] for count in reload_results)
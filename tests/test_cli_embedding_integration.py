"""
Integration tests for CLI embedding model intelligence.

This module tests the integration between the CLI argument parsing,
configuration management, and embedding model intelligence.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import argparse
import tempfile
import os

from src.mcp_server_qdrant_rag.cli_ingest import (
    ConfigurationManager,
    CLIArgumentParser,
    EmbeddingModelInfo,
)


class TestCLIEmbeddingIntegration:
    """Test CLI integration with embedding model intelligence."""
    
    @pytest.mark.asyncio
    async def test_intelligent_config_with_default_model(self):
        """Test intelligent configuration with default model selection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration manager
            config_manager = ConfigurationManager()
            
            # Mock the embedding intelligence methods directly
            mock_model_info = EmbeddingModelInfo(
                model_name='nomic-ai/nomic-embed-text-v1.5-Q',
                vector_size=768,
                vector_name='fast-nomic-embed-text-v1.5-q',
                is_available=True,
                is_compatible=True
            )
            
            with patch.object(config_manager, '_embedding_intelligence') as mock_intelligence:
                mock_intelligence.select_smart_default = AsyncMock(return_value=mock_model_info)
                
                # Create mock arguments for ingest command
                args = argparse.Namespace(
                    command='ingest',
                    path=temp_dir,
                    url='http://localhost:6333',
                    api_key=None,
                    embedding='nomic-ai/nomic-embed-text-v1.5-Q',  # Default model
                    knowledgebase=None,
                    verbose=False,
                    dry_run=False,
                    include_patterns=None,
                    exclude_patterns=None,
                    _embedding_explicitly_set=False  # Simulate default value
                )
                
                # Test intelligent configuration creation
                config = await config_manager.create_intelligent_config_from_args(args)
                
                assert config is not None
                assert config.embedding_settings.model_name == 'nomic-ai/nomic-embed-text-v1.5-Q'
                assert config.knowledgebase_name == os.path.basename(temp_dir)  # Derived from path
                assert config.cli_settings.operation_mode == 'ingest'
                
                # Verify that smart default selection was called
                mock_intelligence.select_smart_default.assert_called_once_with(os.path.basename(temp_dir))
    
    @pytest.mark.asyncio
    async def test_intelligent_config_with_existing_collection(self):
        """Test intelligent configuration with existing collection detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create:
                # Mock successful provider creation
                mock_provider = MagicMock()
                mock_provider.get_vector_size.return_value = 768  # Match the expected model
                mock_provider.get_vector_name.return_value = "fast-custom-model"
                mock_create.return_value = mock_provider
                
                # Mock QdrantConnector with existing collection
                with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                    mock_connector = AsyncMock()
                    mock_connector.get_collection_names.return_value = ['test_collection']
                    mock_connector.check_collection_compatibility.return_value = {
                        "exists": True,
                        "vector_compatible": True,
                        "dimension_compatible": True,
                        "available_vectors": {"fast-custom-model": 768}
                    }
                    mock_connector_class.return_value = mock_connector
                    
                    # Create configuration manager
                    config_manager = ConfigurationManager()
                    
                    # Create mock arguments without specifying embedding model
                    args = argparse.Namespace(
                        command='ingest',
                        path=temp_dir,
                        url='http://localhost:6333',
                        api_key=None,
                        embedding='custom-model',  # This should be validated for compatibility
                        knowledgebase='test_collection',
                        verbose=True,
                        dry_run=False,
                        include_patterns=None,
                        exclude_patterns=None,
                        _embedding_explicitly_set=True  # Simulate user-specified model
                    )
                    
                    # Test intelligent configuration creation
                    config = await config_manager.create_intelligent_config_from_args(args)
                    
                    assert config is not None
                    assert config.knowledgebase_name == 'test_collection'
                    assert config.cli_settings.operation_mode == 'ingest'
                    # The model should be validated for compatibility
                    assert config.embedding_settings.model_name == 'custom-model'
    
    @pytest.mark.asyncio
    async def test_intelligent_config_with_incompatible_model(self):
        """Test intelligent configuration with incompatible model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create:
                # Mock successful provider creation
                mock_provider = MagicMock()
                mock_provider.get_vector_size.return_value = 768  # Different from collection
                mock_provider.get_vector_name.return_value = "fast-nomic-embed-text-v1.5-q"
                mock_create.return_value = mock_provider
                
                # Mock QdrantConnector with incompatible collection
                with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                    mock_connector = AsyncMock()
                    mock_connector.get_collection_names.return_value = ['test_collection']
                    mock_connector.check_collection_compatibility.return_value = {
                        "exists": True,
                        "vector_compatible": True,
                        "dimension_compatible": False,  # Incompatible dimensions
                        "expected_dimensions": 768,
                        "actual_dimensions": 384,
                        "available_vectors": {"fast-all-minilm-l6-v2": 384}
                    }
                    mock_connector_class.return_value = mock_connector
                    
                    # Create configuration manager
                    config_manager = ConfigurationManager()
                    
                    # Create mock arguments with incompatible model
                    args = argparse.Namespace(
                        command='ingest',
                        path=temp_dir,
                        url='http://localhost:6333',
                        api_key=None,
                        embedding='incompatible-model',  # Incompatible with collection
                        knowledgebase='test_collection',
                        verbose=False,
                        dry_run=False,
                        include_patterns=None,
                        exclude_patterns=None,
                        _embedding_explicitly_set=True  # Simulate user-specified model
                    )
                    
                    # Test that configuration creation raises an error
                    with pytest.raises(ValueError, match="incompatible with existing collection"):
                        await config_manager.create_intelligent_config_from_args(args)
    
    @pytest.mark.asyncio
    async def test_intelligent_config_with_model_fallback(self):
        """Test intelligent configuration with model fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            call_count = 0
            
            def mock_create_provider(settings):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    # First call (default model) fails
                    raise Exception("Default model not available")
                else:
                    # Second call (fallback model) succeeds
                    mock_provider = MagicMock()
                    mock_provider.get_vector_size.return_value = 384
                    mock_provider.get_vector_name.return_value = "fast-all-minilm-l6-v2"
                    return mock_provider
            
            with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider', side_effect=mock_create_provider):
                # Mock QdrantConnector with no existing collections
                with patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
                    mock_connector = AsyncMock()
                    mock_connector.get_collection_names.return_value = []
                    mock_connector_class.return_value = mock_connector
                    
                    # Create configuration manager
                    config_manager = ConfigurationManager()
                    
                    # Create mock arguments without specifying embedding model
                    args = argparse.Namespace(
                        command='ingest',
                        path=temp_dir,
                        url='http://localhost:6333',
                        api_key=None,
                        embedding='nomic-ai/nomic-embed-text-v1.5-Q',  # This is the default, but not user-specified
                        knowledgebase=None,
                        verbose=False,
                        dry_run=False,
                        include_patterns=None,
                        exclude_patterns=None,
                        _embedding_explicitly_set=False  # Simulate default value
                    )
                    
                    # Test intelligent configuration creation with fallback
                    config = await config_manager.create_intelligent_config_from_args(args)
                    
                    assert config is not None
                    # Should use fallback model since default failed
                    assert config.embedding_settings.model_name == 'sentence-transformers/all-MiniLM-L6-v2'
                    assert config.knowledgebase_name == os.path.basename(temp_dir)
    
    def test_cli_argument_parsing_with_embedding(self):
        """Test CLI argument parsing with embedding model specification."""
        parser = CLIArgumentParser()
        
        # Test parsing with explicit embedding model
        args = parser.parse_args(['ingest', '/tmp/test', '--embedding', 'custom-model'])
        
        assert args.command == 'ingest'
        assert args.path == '/tmp/test'
        assert args.embedding == 'custom-model'
        assert args.verbose is False
        assert args.dry_run is False
    
    def test_cli_argument_parsing_with_verbose(self):
        """Test CLI argument parsing with verbose flag."""
        parser = CLIArgumentParser()
        
        # Test parsing with verbose flag
        args = parser.parse_args(['ingest', '/tmp/test', '--verbose'])
        
        assert args.command == 'ingest'
        assert args.path == '/tmp/test'
        assert args.verbose is True
        assert args.embedding == 'nomic-ai/nomic-embed-text-v1.5-Q'  # Default
    
    def test_cli_argument_parsing_default_command(self):
        """Test CLI argument parsing with default command."""
        parser = CLIArgumentParser()
        
        # Test parsing without explicit command (should default to ingest)
        args = parser.parse_args(['/tmp/test'])
        
        assert args.command == 'ingest'
        assert args.path == '/tmp/test'
        assert args.embedding == 'nomic-ai/nomic-embed-text-v1.5-Q'  # Default
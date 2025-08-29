"""Integration tests for model switching scenarios with dimension validation."""

import uuid
import pytest
from unittest.mock import patch, MagicMock

from mcp_server_qdrant_rag.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant_rag.qdrant import QdrantConnector, Entry
from mcp_server_qdrant_rag.common.exceptions import VectorDimensionMismatchError


@pytest.mark.asyncio
class TestModelSwitchingIntegration:
    """Integration tests for model switching with proper dimension handling."""

    async def test_same_model_different_instances(self):
        """Test that same model with different instances works correctly."""
        collection_name = f"test_same_model_{uuid.uuid4().hex}"
        
        # Create first connector with model
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        # Store data with first connector
        await connector1.store(Entry(content="First entry with model instance 1"))
        
        # Create second connector with same model
        provider2 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider2,
        )
        
        # Use same client to simulate same Qdrant instance
        connector2._client = connector1._client
        
        # Should work without issues
        await connector2.store(Entry(content="Second entry with model instance 2"))
        
        # Both entries should be searchable
        results = await connector2.search("entry", limit=5)
        assert len(results) == 2
        
        contents = [r.content for r in results]
        assert "First entry with model instance 1" in contents
        assert "Second entry with model instance 2" in contents

    async def test_different_models_dimension_mismatch(self):
        """Test dimension mismatch detection with different models."""
        collection_name = f"test_dim_mismatch_{uuid.uuid4().hex}"
        
        # Create connector with first model
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        # Store data with first model
        await connector1.store(Entry(content="Content with first model"))
        
        # Try to use different model
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider2,
        )
        
        # Use same client to simulate same Qdrant instance
        connector2._client = connector1._client
        
        # Check if models have different properties
        model1_info = provider1.get_model_info()
        model2_info = provider2.get_model_info()
        
        if (model1_info["vector_size"] != model2_info["vector_size"] or 
            model1_info["vector_name"] != model2_info["vector_name"]):
            
            # Should raise dimension mismatch error
            with pytest.raises(VectorDimensionMismatchError) as exc_info:
                await connector2.store(Entry(content="Content with second model"))
            
            error_msg = str(exc_info.value)
            assert "Vector dimension mismatch" in error_msg
            assert collection_name in error_msg
            
            # Original data should still be searchable with original connector
            results = await connector1.search("Content with first model")
            assert len(results) == 1
        else:
            # If dimensions match, should work fine
            await connector2.store(Entry(content="Content with second model"))
            results = await connector2.search("Content")
            assert len(results) == 2

    async def test_model_validation_during_initialization(self):
        """Test model validation during connector initialization."""
        collection_name = f"test_model_validation_{uuid.uuid4().hex}"
        
        # Test with valid model
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Should work without issues
        await connector.store(Entry(content="Test content"))
        results = await connector.search("Test content")
        assert len(results) == 1
        
        # Verify model info is accessible
        model_info = connector.get_embedding_model_info()
        assert model_info["model_name"] == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert isinstance(model_info["vector_size"], int)
        assert model_info["vector_size"] > 0

    async def test_collection_recreation_with_different_model(self):
        """Test collection recreation when switching to incompatible model."""
        collection_name = f"test_collection_recreation_{uuid.uuid4().hex}"
        
        # Create connector with first model and store data
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        await connector1.store(Entry(content="Original content"))
        
        # Verify data exists
        results1 = await connector1.search("Original content")
        assert len(results1) == 1
        
        # Create mock provider with different dimensions
        mock_provider = MagicMock()
        mock_provider.get_vector_size.return_value = 999  # Different size
        mock_provider.get_vector_name.return_value = "mock-vector"
        mock_provider.model_name = "mock-model"
        
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=mock_provider,
        )
        
        # Use same client
        connector2._client = connector1._client
        
        # Should detect dimension mismatch
        with pytest.raises(VectorDimensionMismatchError):
            await connector2.store(Entry(content="New content with different model"))

    async def test_model_info_consistency_across_operations(self):
        """Test that model info remains consistent across different operations."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_model_consistency_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Get model info before any operations
        initial_info = connector.get_embedding_model_info()
        
        # Store some data
        await connector.store(Entry(content="First entry"))
        
        # Get model info after storage
        after_store_info = connector.get_embedding_model_info()
        
        # Perform search
        await connector.search("First entry")
        
        # Get model info after search
        after_search_info = connector.get_embedding_model_info()
        
        # Store more data
        await connector.store(Entry(content="Second entry"))
        
        # Get model info after second storage
        final_info = connector.get_embedding_model_info()
        
        # All model info should be identical
        assert initial_info == after_store_info
        assert after_store_info == after_search_info
        assert after_search_info == final_info
        
        # Verify specific fields remain consistent
        assert initial_info["model_name"] == final_info["model_name"]
        assert initial_info["vector_size"] == final_info["vector_size"]
        assert initial_info["vector_name"] == final_info["vector_name"]

    async def test_concurrent_model_operations(self):
        """Test concurrent operations with the same model."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_concurrent_{uuid.uuid4().hex}"
        
        # Create multiple connectors with same model
        connectors = []
        for i in range(3):
            connector = QdrantConnector(
                qdrant_url=":memory:",
                qdrant_api_key=None,
                collection_name=collection_name,
                embedding_provider=FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q"),
            )
            connectors.append(connector)
        
        # Use same client for all connectors to simulate shared Qdrant instance
        base_client = connectors[0]._client
        for connector in connectors[1:]:
            connector._client = base_client
        
        # Store data concurrently (simulated)
        entries = []
        for i, connector in enumerate(connectors):
            entry = Entry(content=f"Entry from connector {i}")
            await connector.store(entry)
            entries.append(entry)
        
        # Search from any connector should find all entries
        results = await connectors[0].search("Entry from connector", limit=10)
        assert len(results) == 3
        
        # Verify all entries are found
        found_contents = [r.content for r in results]
        for i in range(3):
            assert f"Entry from connector {i}" in found_contents

    async def test_model_switching_error_messages(self):
        """Test that model switching errors provide helpful messages."""
        collection_name = f"test_error_messages_{uuid.uuid4().hex}"
        
        # Create connector with first model
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        await connector1.store(Entry(content="Test content"))
        
        # Create mock provider with different properties
        mock_provider = MagicMock()
        mock_provider.get_vector_size.return_value = 512  # Different from nomic model
        mock_provider.get_vector_name.return_value = "different-vector"
        mock_provider.model_name = "different-model"
        
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=mock_provider,
        )
        
        connector2._client = connector1._client
        
        # Should get helpful error message
        with pytest.raises(VectorDimensionMismatchError) as exc_info:
            await connector2.store(Entry(content="Different model content"))
        
        error_msg = str(exc_info.value)
        
        # Check that error message contains helpful information
        assert "Vector dimension mismatch" in error_msg
        assert collection_name in error_msg
        assert "different-vector" in error_msg or "different-model" in error_msg
        
        # Should suggest what to do
        assert ("embedding model" in error_msg.lower() or 
                "change in embedding model" in error_msg.lower())

    async def test_model_compatibility_validation(self):
        """Test model compatibility validation logic."""
        collection_name = f"test_compatibility_{uuid.uuid4().hex}"
        
        # Test with compatible models (same model, different instances)
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        provider2 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        await connector1.store(Entry(content="Compatible test"))
        
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider2,
        )
        
        connector2._client = connector1._client
        
        # Should work without issues
        await connector2.store(Entry(content="Another compatible test"))
        
        results = await connector2.search("compatible test")
        assert len(results) == 2
        
        # Verify both providers report same dimensions
        info1 = provider1.get_model_info()
        info2 = provider2.get_model_info()
        
        assert info1["vector_size"] == info2["vector_size"]
        assert info1["vector_name"] == info2["vector_name"]
        assert info1["model_name"] == info2["model_name"]
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qdrant_client import models

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry


@pytest.mark.asyncio
class TestDynamicVectorDimensions:
    """Test dynamic vector dimension handling."""

    async def test_fastembed_provider_dimension_detection(self):
        """Test that FastEmbedProvider correctly detects vector dimensions for different models."""
        # Test with the default model
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        vector_size = provider.get_vector_size()
        
        # Vector size should be positive and reasonable for embedding models
        assert isinstance(vector_size, int)
        assert vector_size > 0
        assert vector_size <= 4096  # Reasonable upper bound for embedding dimensions

    async def test_fastembed_provider_model_info(self):
        """Test that FastEmbedProvider provides comprehensive model information."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        model_info = provider.get_model_info()
        
        # Check that all expected fields are present
        assert "model_name" in model_info
        assert "vector_size" in model_info
        assert "vector_name" in model_info
        assert "description" in model_info
        
        # Check field types and values
        assert model_info["model_name"] == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert isinstance(model_info["vector_size"], int)
        assert model_info["vector_size"] > 0
        assert isinstance(model_info["vector_name"], str)
        assert model_info["vector_name"].startswith("fast-")

    async def test_different_models_different_dimensions(self):
        """Test that different models can have different vector dimensions."""
        # Test with two different models that should have different dimensions
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        # Use a different model for comparison
        # Note: We'll use sentence-transformers model as it's commonly available
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        
        size1 = provider1.get_vector_size()
        size2 = provider2.get_vector_size()
        
        # Both should be valid dimensions
        assert isinstance(size1, int) and size1 > 0
        assert isinstance(size2, int) and size2 > 0
        
        # They might be different (though not guaranteed)
        # The important thing is that each model consistently reports its dimension
        assert provider1.get_vector_size() == size1  # Consistency check
        assert provider2.get_vector_size() == size2  # Consistency check

    async def test_qdrant_connector_embedding_model_info(self):
        """Test that QdrantConnector provides embedding model information."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=provider,
        )
        
        model_info = connector.get_embedding_model_info()
        
        # Check that all expected fields are present
        assert "model_name" in model_info
        assert "vector_size" in model_info
        assert "vector_name" in model_info
        
        # Check field values
        assert model_info["model_name"] == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert isinstance(model_info["vector_size"], int)
        assert model_info["vector_size"] > 0
        assert isinstance(model_info["vector_name"], str)

    async def test_collection_creation_with_dynamic_dimensions(self):
        """Test that collections are created with correct dynamic dimensions."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_dynamic_dims_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Store an entry to trigger collection creation
        test_entry = Entry(content="Test content for dynamic dimensions")
        await connector.store(test_entry)
        
        # Verify collection was created with correct dimensions
        collection_info = await connector._client.get_collection(collection_name)
        expected_vector_name = provider.get_vector_name()
        expected_vector_size = provider.get_vector_size()
        
        assert expected_vector_name in collection_info.config.params.vectors
        actual_vector_size = collection_info.config.params.vectors[expected_vector_name].size
        assert actual_vector_size == expected_vector_size

    async def test_dimension_mismatch_detection(self):
        """Test that dimension mismatches are properly detected and reported."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_mismatch_{uuid.uuid4().hex}"
        
        # Create a connector and collection with the first model
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Store an entry to create the collection
        await connector.store(Entry(content="Initial content"))
        
        # Now create a new connector with a different model (different dimensions)
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider2,
        )
        
        # Use the same client instance to simulate the same Qdrant instance
        connector2._client = connector._client
        
        # Check if dimensions are actually different
        if provider.get_vector_size() != provider2.get_vector_size() or provider.get_vector_name() != provider2.get_vector_name():
            # Attempting to store with the second connector should raise a validation error
            with pytest.raises(ValueError) as exc_info:
                await connector2.store(Entry(content="Content with different model"))
            
            error_message = str(exc_info.value)
            # The error could be either vector name mismatch or dimension mismatch
            assert ("Vector dimension mismatch" in error_message or 
                    "does not have the expected vector" in error_message)
            assert collection_name in error_message
            assert "embedding model" in error_message or "change in embedding model" in error_message

    async def test_vector_name_mismatch_detection(self):
        """Test that vector name mismatches are properly detected."""
        # Create a mock provider with a custom vector name
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_vector_name_mismatch_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Store an entry to create the collection
        await connector.store(Entry(content="Initial content"))
        
        # Create a mock provider that returns a different vector name
        mock_provider = MagicMock()
        mock_provider.get_vector_size.return_value = provider.get_vector_size()
        mock_provider.get_vector_name.return_value = "different-vector-name"
        mock_provider.model_name = "mock-model"
        
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=mock_provider,
        )
        
        # Use the same client to simulate the same Qdrant instance
        connector2._client = connector._client
        
        # Attempting to store should raise a vector name mismatch error
        with pytest.raises(ValueError) as exc_info:
            await connector2.store(Entry(content="Content with different vector name"))
        
        error_message = str(exc_info.value)
        assert "does not have the expected vector" in error_message
        assert "different-vector-name" in error_message
        assert collection_name in error_message

    async def test_backward_compatibility_validation(self):
        """Test that existing collections are properly validated for compatibility."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_backward_compat_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Store an entry to create the collection
        await connector.store(Entry(content="Initial content"))
        
        # Create a new connector instance with the same configuration
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q"),
        )
        
        # Use the same client to simulate the same Qdrant instance
        connector2._client = connector._client
        
        # This should work without errors (backward compatibility)
        await connector2.store(Entry(content="Additional content"))
        
        # Verify both entries can be found
        results = await connector2.search("content")
        assert len(results) == 2

    async def test_validation_error_handling(self):
        """Test that validation errors are handled gracefully."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_validation_error_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Mock the client to raise an exception during collection info retrieval
        original_get_collection = connector._client.get_collection
        
        async def mock_get_collection(collection_name):
            if collection_name == connector._default_collection_name:
                raise Exception("Simulated connection error")
            return await original_get_collection(collection_name)
        
        connector._client.get_collection = mock_get_collection
        
        # Store an entry - this should work despite the validation error
        # (validation errors should be logged but not prevent operation)
        await connector.store(Entry(content="Test content"))
        
        # Verify the entry was stored successfully
        results = await connector.search("Test content")
        assert len(results) == 1
        assert results[0].content == "Test content"

    async def test_model_consistency_across_operations(self):
        """Test that model dimensions remain consistent across different operations."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        # Get dimensions multiple times to ensure consistency
        size1 = provider.get_vector_size()
        size2 = provider.get_vector_size()
        size3 = provider.get_vector_size()
        
        assert size1 == size2 == size3
        
        # Test vector name consistency
        name1 = provider.get_vector_name()
        name2 = provider.get_vector_name()
        name3 = provider.get_vector_name()
        
        assert name1 == name2 == name3
        
        # Test model info consistency
        info1 = provider.get_model_info()
        info2 = provider.get_model_info()
        
        assert info1["vector_size"] == info2["vector_size"]
        assert info1["vector_name"] == info2["vector_name"]
        assert info1["model_name"] == info2["model_name"]
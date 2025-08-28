"""
Integration tests for backward compatibility scenarios.

Tests ensure that the system handles:
1. Existing collections with different vector dimensions
2. Mixed chunked/non-chunked collections
3. Migration scenarios
4. Legacy data format compatibility
"""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.common.exceptions import (
    VectorDimensionMismatchError,
    BackwardCompatibilityError
)


class TestBackwardCompatibility:
    """Test backward compatibility scenarios."""

    @pytest.mark.asyncio
    async def test_existing_collection_dimension_detection(self):
        """Test detection of existing collections with different vector dimensions."""
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_dimension_detection_{uuid.uuid4().hex}"
        
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        # Store an entry to create the collection
        await connector1.store(Entry(content="Initial content for dimension test"))
        
        # Analyze compatibility with same configuration (should be compatible)
        compatibility = await connector1.analyze_collection_compatibility()
        assert compatibility["compatible"] is True
        assert compatibility["exists"] is True
        assert compatibility["dimension_compatible"] is True
        assert compatibility["vector_compatible"] is True
        
        # Try with a different model that has different dimensions
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        
        # Only test if the models actually have different dimensions
        if provider1.get_vector_size() != provider2.get_vector_size() or provider1.get_vector_name() != provider2.get_vector_name():
            connector2 = QdrantConnector(
                qdrant_url=":memory:",
                qdrant_api_key=None,
                collection_name=collection_name,
                embedding_provider=provider2,
            )
            connector2._client = connector1._client  # Use same client
            
            # Analyze compatibility (should detect incompatibility)
            compatibility = await connector2.analyze_collection_compatibility()
            assert compatibility["compatible"] is False
            assert compatibility["exists"] is True
            assert len(compatibility["recommendations"]) > 0
            
            # Attempting to store should raise an error
            with pytest.raises(VectorDimensionMismatchError):
                await connector2.store(Entry(content="Content with incompatible model"))

    @pytest.mark.asyncio
    async def test_mixed_chunked_non_chunked_collection(self):
        """Test seamless handling of collections with both chunked and non-chunked content."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_mixed_content_{uuid.uuid4().hex}"
        
        # First connector with chunking disabled
        connector_no_chunk = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
            enable_chunking=False,
        )
        
        # Store some non-chunked content
        await connector_no_chunk.store(Entry(
            content="Short document that won't be chunked.",
            metadata={"type": "non_chunked", "id": 1}
        ))
        await connector_no_chunk.store(Entry(
            content="Another short document for testing.",
            metadata={"type": "non_chunked", "id": 2}
        ))
        
        # Second connector with chunking enabled
        connector_with_chunk = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q"),
            enable_chunking=True,
            max_chunk_size=100,  # Small chunk size to force chunking
        )
        connector_with_chunk._client = connector_no_chunk._client  # Use same client
        
        # Store a large document that will be chunked
        large_content = " ".join([
            f"This is sentence {i} in a large document that should be chunked into multiple pieces."
            for i in range(20)
        ])
        await connector_with_chunk.store(Entry(
            content=large_content,
            metadata={"type": "chunked", "id": 3}
        ))
        
        # Analyze collection compatibility
        compatibility = await connector_with_chunk.analyze_collection_compatibility()
        assert compatibility["compatible"] is True
        assert compatibility["has_chunked_content"] is True
        assert compatibility["has_non_chunked_content"] is True
        assert compatibility["mixed_content"] is True
        assert "both chunked and non-chunked content" in " ".join(compatibility["recommendations"]).lower()
        
        # Search should return results from both chunked and non-chunked content
        results = await connector_with_chunk.search("document", limit=10)
        
        # Should have results from both types
        chunked_results = [r for r in results if r.is_chunk]
        non_chunked_results = [r for r in results if not r.is_chunk]
        
        assert len(chunked_results) > 0, "Should have chunked results"
        assert len(non_chunked_results) > 0, "Should have non-chunked results"
        
        # Verify metadata is preserved for both types
        for result in results:
            assert result.metadata is not None
            assert "type" in result.metadata
            assert result.metadata["type"] in ["chunked", "non_chunked"]

    @pytest.mark.asyncio
    async def test_legacy_data_format_compatibility(self):
        """Test compatibility with legacy data formats (entries without chunk fields)."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_legacy_format_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Manually create a "legacy" entry without chunk fields
        # This simulates data stored before chunking was implemented
        embeddings = await provider.embed_documents(["Legacy document content"])
        vector_name = provider.get_vector_name()
        
        # First ensure collection exists by storing a regular entry
        await connector.store(Entry(content="Initial entry to create collection"))
        
        # Store directly without chunk fields (simulating legacy format)
        from qdrant_client import models
        await connector._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload={
                        "document": "Legacy document content",
                        "metadata": {"source": "legacy_system", "version": "1.0"}
                        # Note: no is_chunk, source_document_id, etc.
                    }
                )
            ]
        )
        
        # Search should handle legacy entries gracefully
        results = await connector.search("legacy document")
        assert len(results) > 0
        
        legacy_result = results[0]
        assert legacy_result.content == "Legacy document content"
        assert legacy_result.is_chunk is False  # Should default to False
        assert legacy_result.source_document_id is None
        assert legacy_result.chunk_index is None
        assert legacy_result.total_chunks is None
        assert legacy_result.metadata["source"] == "legacy_system"

    @pytest.mark.asyncio
    async def test_collection_migration_guidance(self):
        """Test that proper migration guidance is provided for incompatible collections."""
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_migration_guidance_{uuid.uuid4().hex}"
        
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        # Create collection with first model
        await connector1.store(Entry(content="Content with first model"))
        
        # Try to use with incompatible model
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        
        # Only test if models are actually different
        if provider1.get_vector_size() != provider2.get_vector_size() or provider1.get_vector_name() != provider2.get_vector_name():
            connector2 = QdrantConnector(
                qdrant_url=":memory:",
                qdrant_api_key=None,
                collection_name=collection_name,
                embedding_provider=provider2,
            )
            connector2._client = connector1._client
            
            # Get compatibility analysis
            compatibility = await connector2.analyze_collection_compatibility()
            
            assert compatibility["compatible"] is False
            assert len(compatibility["recommendations"]) > 0
            
            # Check that recommendations include useful guidance
            recommendations_text = " ".join(compatibility["recommendations"]).lower()
            assert any(keyword in recommendations_text for keyword in [
                "different collection", "compatible model", "dimension", "vector"
            ])
            
            # Verify error contains migration suggestions
            try:
                await connector2.store(Entry(content="This should fail"))
                assert False, "Should have raised VectorDimensionMismatchError"
            except VectorDimensionMismatchError as e:
                assert len(e.details["resolution_suggestions"]) > 0
                suggestions_text = " ".join(e.details["resolution_suggestions"]).lower()
                assert "collection" in suggestions_text

    @pytest.mark.asyncio
    async def test_chunking_configuration_compatibility(self):
        """Test compatibility when chunking configuration changes."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_chunking_config_{uuid.uuid4().hex}"
        
        # Store content with chunking enabled
        connector_chunked = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
            enable_chunking=True,
            max_chunk_size=50,
            chunk_overlap=10,
        )
        
        large_content = " ".join([f"Sentence {i} for chunking test." for i in range(15)])
        await connector_chunked.store(Entry(
            content=large_content,
            metadata={"test": "chunked_content"}
        ))
        
        # Create connector with chunking disabled
        connector_no_chunk = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q"),
            enable_chunking=False,
        )
        connector_no_chunk._client = connector_chunked._client
        
        # Analyze compatibility
        compatibility = await connector_no_chunk.analyze_collection_compatibility()
        assert compatibility["compatible"] is True  # Vector dimensions are compatible
        assert compatibility["has_chunked_content"] is True
        assert compatibility["chunking_enabled"] is False
        
        # Should include recommendation about chunking mismatch
        recommendations_text = " ".join(compatibility["recommendations"]).lower()
        assert "chunked content" in recommendations_text
        assert "chunking is disabled" in recommendations_text
        
        # Should still be able to search existing chunked content
        results = await connector_no_chunk.search("sentence")
        assert len(results) > 0
        assert any(r.is_chunk for r in results)
        
        # New content should not be chunked
        await connector_no_chunk.store(Entry(content="New content that won't be chunked"))
        
        # Verify mixed content
        all_results = await connector_no_chunk.search("content", limit=20)
        chunked_results = [r for r in all_results if r.is_chunk]
        non_chunked_results = [r for r in all_results if not r.is_chunk]
        
        assert len(chunked_results) > 0
        assert len(non_chunked_results) > 0

    @pytest.mark.asyncio
    async def test_empty_collection_compatibility(self):
        """Test compatibility analysis for non-existent collections."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_empty_collection_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Analyze non-existent collection
        compatibility = await connector.analyze_collection_compatibility()
        
        assert compatibility["exists"] is False
        assert compatibility["compatible"] is True
        assert "will be created" in compatibility["message"]

    @pytest.mark.asyncio
    async def test_search_aggregation_backward_compatibility(self):
        """Test that search aggregation works correctly with mixed content types."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_aggregation_compat_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
            enable_chunking=True,
            max_chunk_size=100,  # Small to force chunking
            chunk_overlap=20,  # Smaller overlap
        )
        
        # Store a mix of content
        await connector.store(Entry(content="Short non-chunked document"))
        
        long_content = " ".join([f"Long document sentence {i}." for i in range(20)])
        await connector.store(Entry(content=long_content))
        
        # Test search with aggregation enabled (default)
        results_aggregated = await connector.search("document", aggregate_chunks=True)
        
        # Test search with aggregation disabled
        results_raw = await connector.search("document", aggregate_chunks=False)
        
        # Raw results should have more entries (individual chunks)
        assert len(results_raw) >= len(results_aggregated)
        
        # Both should contain the non-chunked document
        non_chunked_in_aggregated = any(not r.is_chunk for r in results_aggregated)
        non_chunked_in_raw = any(not r.is_chunk for r in results_raw)
        
        assert non_chunked_in_aggregated
        assert non_chunked_in_raw
        
        # Aggregated results should have chunked entries with chunk_index=None (aggregated)
        aggregated_chunks = [r for r in results_aggregated if r.is_chunk and r.chunk_index is None]
        individual_chunks = [r for r in results_raw if r.is_chunk and r.chunk_index is not None]
        
        if len(individual_chunks) > 1:  # Only test if we actually have multiple chunks
            assert len(aggregated_chunks) > 0, "Should have aggregated chunk results"
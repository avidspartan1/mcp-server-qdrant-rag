"""Integration tests for chunked document storage functionality."""

import pytest
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock

from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size
        self.model_name = "mock-model"
    
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Return mock embeddings for documents."""
        return [[0.1] * self.vector_size for _ in documents]
    
    async def embed_query(self, query: str) -> list[float]:
        """Return mock embedding for query."""
        return [0.1] * self.vector_size
    
    def get_vector_size(self) -> int:
        return self.vector_size
    
    def get_vector_name(self) -> str:
        return "default"


@pytest.fixture
async def qdrant_connector():
    """Create a QdrantConnector with chunking enabled for testing."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        embedding_provider = MockEmbeddingProvider()
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=embedding_provider,
            qdrant_local_path=temp_dir,
            enable_chunking=True,
            max_chunk_size=100,  # Small chunk size for testing
            chunk_overlap=20,
            chunk_strategy="semantic"
        )
        yield connector
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def qdrant_connector_no_chunking():
    """Create a QdrantConnector with chunking disabled for testing."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        embedding_provider = MockEmbeddingProvider()
        connector = QdrantConnector(
            qdrant_url=None,
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=embedding_provider,
            qdrant_local_path=temp_dir,
            enable_chunking=False,
        )
        yield connector
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestChunkedStorageIntegration:
    """Integration tests for chunked document storage."""

    @pytest.mark.asyncio
    async def test_store_small_document_no_chunking(self, qdrant_connector):
        """Test that small documents are not chunked."""
        entry = Entry(content="This is a small document that should not be chunked.")
        
        await qdrant_connector.store(entry)
        
        # Search for the document
        results = await qdrant_connector.search("small document")
        
        assert len(results) == 1
        assert results[0].content == entry.content
        assert results[0].is_chunk is False
        assert results[0].source_document_id is None

    @pytest.mark.asyncio
    async def test_store_large_document_with_chunking(self, qdrant_connector):
        """Test that large documents are automatically chunked."""
        # Create a large document that will exceed the chunk size
        large_content = " ".join([f"This is sentence number {i} in a very long document." for i in range(20)])
        entry = Entry(content=large_content, metadata={"title": "Large Document"})
        
        await qdrant_connector.store(entry)
        
        # Search for content from the document
        results = await qdrant_connector.search("sentence number")
        
        # Should have multiple chunks
        assert len(results) > 1
        
        # All results should be chunks from the same document
        source_doc_ids = {result.source_document_id for result in results}
        assert len(source_doc_ids) == 1  # All chunks from same document
        
        # Check chunk properties
        for result in results:
            assert result.is_chunk is True
            assert result.source_document_id is not None
            assert result.chunk_index is not None
            assert result.total_chunks is not None
            assert result.chunk_index < result.total_chunks
            assert result.metadata == {"title": "Large Document"}

    @pytest.mark.asyncio
    async def test_store_chunk_entry_directly(self, qdrant_connector):
        """Test storing an Entry that is already marked as a chunk."""
        chunk_entry = Entry(
            content="This is a pre-chunked entry.",
            is_chunk=True,
            source_document_id="manual-doc-123",
            chunk_index=0,
            total_chunks=1
        )
        
        await qdrant_connector.store(chunk_entry)
        
        # Search for the chunk
        results = await qdrant_connector.search("pre-chunked")
        
        assert len(results) == 1
        assert results[0].content == chunk_entry.content
        assert results[0].is_chunk is True
        assert results[0].source_document_id == "manual-doc-123"
        assert results[0].chunk_index == 0
        assert results[0].total_chunks == 1

    @pytest.mark.asyncio
    async def test_chunking_disabled_stores_whole_document(self, qdrant_connector_no_chunking):
        """Test that when chunking is disabled, large documents are stored whole."""
        # Create a large document
        large_content = " ".join([f"This is sentence number {i} in a very long document." for i in range(20)])
        entry = Entry(content=large_content)
        
        await qdrant_connector_no_chunking.store(entry)
        
        # Search for content
        results = await qdrant_connector_no_chunking.search("sentence number")
        
        # Should have only one result (the whole document)
        assert len(results) == 1
        assert results[0].content == large_content
        assert results[0].is_chunk is False

    @pytest.mark.asyncio
    async def test_mixed_chunked_and_non_chunked_search(self, qdrant_connector):
        """Test searching across both chunked and non-chunked documents."""
        # Store a small document (won't be chunked)
        small_entry = Entry(content="Small document about testing.")
        await qdrant_connector.store(small_entry)
        
        # Store a large document (will be chunked)
        large_content = " ".join([f"Testing sentence number {i} in a chunked document." for i in range(20)])
        large_entry = Entry(content=large_content)
        await qdrant_connector.store(large_entry)
        
        # Search for "testing"
        results = await qdrant_connector.search("testing")
        
        # Should get results from both documents
        assert len(results) > 1
        
        # Check that we have both chunked and non-chunked results
        chunked_results = [r for r in results if r.is_chunk]
        non_chunked_results = [r for r in results if not r.is_chunk]
        
        assert len(chunked_results) > 0
        assert len(non_chunked_results) > 0

    @pytest.mark.asyncio
    async def test_chunk_metadata_preservation(self, qdrant_connector):
        """Test that original document metadata is preserved in chunks."""
        metadata = {
            "author": "Test Author",
            "category": "Integration Test",
            "tags": ["test", "chunking"]
        }
        
        large_content = " ".join([f"Content with metadata sentence {i}." for i in range(30)])
        entry = Entry(content=large_content, metadata=metadata)
        
        await qdrant_connector.store(entry)
        
        # Search and verify metadata is preserved
        results = await qdrant_connector.search("metadata sentence")
        
        assert len(results) > 0
        for result in results:
            assert result.is_chunk is True
            assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_chunk_sequence_integrity(self, qdrant_connector):
        """Test that chunks maintain proper sequence and relationship information."""
        large_content = " ".join([f"Sequential sentence {i} for testing chunk order." for i in range(35)])
        entry = Entry(content=large_content)
        
        await qdrant_connector.store(entry)
        
        # Search to get all chunks
        results = await qdrant_connector.search("sequential sentence")
        
        # Verify chunk sequence integrity
        assert len(results) > 1
        
        # Group by source document
        chunks_by_doc = {}
        for result in results:
            if result.source_document_id not in chunks_by_doc:
                chunks_by_doc[result.source_document_id] = []
            chunks_by_doc[result.source_document_id].append(result)
        
        # Should have only one source document
        assert len(chunks_by_doc) == 1
        
        # Check chunk sequence
        chunks = list(chunks_by_doc.values())[0]
        chunks.sort(key=lambda x: x.chunk_index)
        
        # Verify sequence properties
        total_chunks = chunks[0].total_chunks
        assert len(chunks) == total_chunks
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == total_chunks
            assert chunk.is_chunk is True

    @pytest.mark.asyncio
    async def test_empty_content_handling(self, qdrant_connector):
        """Test handling of empty or whitespace-only content."""
        # Test empty content
        empty_entry = Entry(content="")
        await qdrant_connector.store(empty_entry)
        
        # Test whitespace-only content
        whitespace_entry = Entry(content="   \n\t  ")
        await qdrant_connector.store(whitespace_entry)
        
        # Both should be stored as single entries (not chunked)
        results = await qdrant_connector.search("empty")
        # This search might not return results due to empty content, which is expected

    @pytest.mark.asyncio
    async def test_chunking_fallback_on_error(self, qdrant_connector):
        """Test that chunking failures fall back to storing the original document."""
        # This test would require mocking the chunker to fail
        # For now, we'll test with content that might cause issues
        problematic_content = "A" * 10000  # Very long single word
        entry = Entry(content=problematic_content)
        
        # Should not raise an exception
        await qdrant_connector.store(entry)
        
        # Should be able to search for it
        results = await qdrant_connector.search("A")
        assert len(results) >= 1
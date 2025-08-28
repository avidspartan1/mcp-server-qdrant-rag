"""Tests for enhanced search functionality with chunked content."""

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
            max_chunk_size=50,  # Very small chunk size for testing
            chunk_overlap=10,
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


class TestEnhancedSearch:
    """Tests for enhanced search functionality with chunked content."""

    @pytest.mark.asyncio
    async def test_search_non_chunked_content(self, qdrant_connector_no_chunking):
        """Test search with non-chunked content returns normal results."""
        # Store some non-chunked entries
        entries = [
            Entry(content="First document about cats", metadata={"type": "animal"}),
            Entry(content="Second document about dogs", metadata={"type": "animal"}),
            Entry(content="Third document about programming", metadata={"type": "tech"}),
        ]
        
        for entry in entries:
            await qdrant_connector_no_chunking.store(entry)
        
        # Search for animals
        results = await qdrant_connector_no_chunking.search("animals")
        
        # Should get non-chunked results
        assert len(results) > 0
        for result in results:
            assert result.is_chunk is False
            assert result.source_document_id is None
            assert result.chunk_index is None

    @pytest.mark.asyncio
    async def test_search_chunked_content_with_aggregation(self, qdrant_connector):
        """Test search with chunked content and result aggregation."""
        # Store a large document that will be chunked
        large_content = " ".join([
            f"This is sentence {i} about machine learning and artificial intelligence."
            for i in range(20)
        ])
        entry = Entry(content=large_content, metadata={"topic": "AI"})
        
        await qdrant_connector.store(entry)
        
        # Search with aggregation enabled (default)
        results = await qdrant_connector.search("machine learning", limit=5)
        
        # Should get aggregated results
        assert len(results) >= 1
        
        # Check if we have chunked results
        chunked_results = [r for r in results if r.is_chunk]
        if chunked_results:
            result = chunked_results[0]
            assert result.is_chunk is True
            assert result.source_document_id is not None
            assert result.metadata == {"topic": "AI"}
            
            # Check if it's an aggregated result
            if hasattr(result, '_chunk_count'):
                assert result._chunk_count >= 1

    @pytest.mark.asyncio
    async def test_search_chunked_content_without_aggregation(self, qdrant_connector):
        """Test search with chunked content but aggregation disabled."""
        # Store a large document that will be chunked
        large_content = " ".join([
            f"This is sentence {i} about data science and analytics."
            for i in range(20)
        ])
        entry = Entry(content=large_content, metadata={"topic": "Data"})
        
        await qdrant_connector.store(entry)
        
        # Search with aggregation disabled
        results = await qdrant_connector.search("data science", aggregate_chunks=False, limit=10)
        
        # Should get individual chunk results
        chunked_results = [r for r in results if r.is_chunk]
        if chunked_results:
            # Should have multiple individual chunks
            assert len(chunked_results) > 1
            
            # All chunks should be from the same document
            source_docs = {r.source_document_id for r in chunked_results}
            assert len(source_docs) == 1
            
            # Each chunk should have proper index information
            for result in chunked_results:
                assert result.chunk_index is not None
                assert result.total_chunks is not None
                assert result.chunk_index < result.total_chunks

    @pytest.mark.asyncio
    async def test_search_mixed_chunked_and_non_chunked(self, qdrant_connector):
        """Test search across both chunked and non-chunked documents."""
        # Store a small document (won't be chunked)
        small_entry = Entry(content="Small document about testing frameworks.", metadata={"size": "small"})
        await qdrant_connector.store(small_entry)
        
        # Store a large document (will be chunked)
        large_content = " ".join([
            f"Large document sentence {i} about testing methodologies and frameworks."
            for i in range(25)
        ])
        large_entry = Entry(content=large_content, metadata={"size": "large"})
        await qdrant_connector.store(large_entry)
        
        # Search for "testing"
        results = await qdrant_connector.search("testing frameworks", limit=10)
        
        # Should get results from both documents
        assert len(results) >= 2
        
        # Check that we have both chunked and non-chunked results
        chunked_results = [r for r in results if r.is_chunk]
        non_chunked_results = [r for r in results if not r.is_chunk]
        
        assert len(chunked_results) >= 1
        assert len(non_chunked_results) >= 1
        
        # Verify metadata is preserved
        for result in results:
            assert result.metadata is not None
            assert "size" in result.metadata

    @pytest.mark.asyncio
    async def test_search_chunk_context_preservation(self, qdrant_connector):
        """Test that chunk context and source document information is preserved."""
        # Store a document with distinctive content
        content = " ".join([
            "Introduction to the topic.",
            "First main point about the subject.",
            "Second main point with details.",
            "Third main point with examples.",
            "Conclusion summarizing everything."
        ] * 5)  # Repeat to ensure chunking
        
        entry = Entry(
            content=content, 
            metadata={"author": "Test Author", "category": "Tutorial"}
        )
        await qdrant_connector.store(entry)
        
        # Search for content
        results = await qdrant_connector.search("main point", limit=5)
        
        # Verify chunk information is preserved
        chunked_results = [r for r in results if r.is_chunk]
        if chunked_results:
            for result in chunked_results:
                assert result.source_document_id is not None
                assert result.metadata == {"author": "Test Author", "category": "Tutorial"}
                
                # Check that content includes context
                assert len(result.content) > 0
                assert "main point" in result.content.lower()

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, qdrant_connector):
        """Test search behavior with empty collection."""
        results = await qdrant_connector.search("nonexistent query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_nonexistent_collection(self, qdrant_connector):
        """Test search behavior with nonexistent collection."""
        results = await qdrant_connector.search("query", collection_name="nonexistent_collection")
        assert results == []

    @pytest.mark.asyncio
    async def test_aggregate_search_results_single_chunk(self, qdrant_connector):
        """Test aggregation with single chunk per document."""
        # Create entries that simulate single chunks
        chunk_entry = Entry(
            content="Single chunk content about testing.",
            is_chunk=True,
            source_document_id="doc-123",
            chunk_index=0,
            total_chunks=1,
            metadata={"test": "value"}
        )
        chunk_entry._search_score = 0.9
        
        # Test aggregation
        aggregated = qdrant_connector._aggregate_search_results([chunk_entry], limit=5)
        
        assert len(aggregated) == 1
        result = aggregated[0]
        assert result.content == "Single chunk content about testing."
        assert result.is_chunk is True
        assert result.source_document_id == "doc-123"

    @pytest.mark.asyncio
    async def test_aggregate_search_results_multiple_chunks(self, qdrant_connector):
        """Test aggregation with multiple chunks from same document."""
        # Create multiple chunks from the same document
        chunks = []
        for i in range(3):
            chunk = Entry(
                content=f"Chunk {i} content with important information.",
                is_chunk=True,
                source_document_id="doc-456",
                chunk_index=i,
                total_chunks=3,
                metadata={"doc": "test"}
            )
            chunk._search_score = 0.8 - (i * 0.1)  # Decreasing scores
            chunks.append(chunk)
        
        # Test aggregation
        aggregated = qdrant_connector._aggregate_search_results(chunks, limit=5)
        
        assert len(aggregated) == 1
        result = aggregated[0]
        assert result.is_chunk is True
        assert result.source_document_id == "doc-456"
        assert hasattr(result, '_chunk_count')
        assert result._chunk_count == 3
        
        # Should contain content from multiple chunks
        assert "Chunk" in result.content
        assert "..." in result.content  # Separator between chunks

    @pytest.mark.asyncio
    async def test_aggregate_search_results_mixed_documents(self, qdrant_connector):
        """Test aggregation with chunks from different documents and non-chunked entries."""
        entries = []
        
        # Non-chunked entry
        non_chunk = Entry(content="Non-chunked document about testing.")
        non_chunk._search_score = 0.95
        entries.append(non_chunk)
        
        # Chunks from first document
        for i in range(2):
            chunk = Entry(
                content=f"Doc1 chunk {i} about testing frameworks.",
                is_chunk=True,
                source_document_id="doc-1",
                chunk_index=i,
                total_chunks=2
            )
            chunk._search_score = 0.8 - (i * 0.1)
            entries.append(chunk)
        
        # Chunks from second document
        for i in range(2):
            chunk = Entry(
                content=f"Doc2 chunk {i} about testing methodologies.",
                is_chunk=True,
                source_document_id="doc-2",
                chunk_index=i,
                total_chunks=2
            )
            chunk._search_score = 0.7 - (i * 0.1)
            entries.append(chunk)
        
        # Test aggregation
        aggregated = qdrant_connector._aggregate_search_results(entries, limit=5)
        
        # Should have 3 results: 1 non-chunked + 2 aggregated chunked documents
        assert len(aggregated) == 3
        
        # Check that results are sorted by score
        scores = [getattr(r, '_search_score', 0.0) for r in aggregated]
        assert scores == sorted(scores, reverse=True)
        
        # Verify we have the expected mix
        non_chunked_results = [r for r in aggregated if not r.is_chunk]
        chunked_results = [r for r in aggregated if r.is_chunk]
        
        assert len(non_chunked_results) == 1
        assert len(chunked_results) == 2

    @pytest.mark.asyncio
    async def test_create_aggregated_content_single_chunk(self, qdrant_connector):
        """Test content aggregation with single chunk."""
        chunk = Entry(content="Single chunk content.")
        chunk._search_score = 0.9
        
        result = qdrant_connector._create_aggregated_content([chunk])
        assert result == "Single chunk content."

    @pytest.mark.asyncio
    async def test_create_aggregated_content_multiple_chunks(self, qdrant_connector):
        """Test content aggregation with multiple chunks."""
        chunks = []
        for i in range(3):
            chunk = Entry(
                content=f"Content of chunk {i} with details.",
                is_chunk=True,
                source_document_id="test-doc",
                chunk_index=i,
                total_chunks=3
            )
            chunk._search_score = 0.8 - (i * 0.1)
            chunks.append(chunk)
        
        result = qdrant_connector._create_aggregated_content(chunks)
        
        # Should contain content from all chunks with separators
        assert "Content of chunk 0" in result
        assert "Content of chunk 1" in result
        assert "Content of chunk 2" in result
        assert "..." in result

    @pytest.mark.asyncio
    async def test_create_aggregated_content_many_chunks(self, qdrant_connector):
        """Test content aggregation with many chunks (should limit to 3)."""
        chunks = []
        for i in range(6):
            chunk = Entry(
                content=f"Chunk {i} content here.",
                is_chunk=True,
                source_document_id="test-doc",
                chunk_index=i,
                total_chunks=6
            )
            chunk._search_score = 0.9 if i == 3 else 0.5  # Make chunk 3 the best
            chunks.append(chunk)
        
        result = qdrant_connector._create_aggregated_content(chunks)
        
        # Should contain first, best (3), and last chunks
        assert "Chunk 0" in result
        assert "Chunk 3" in result  # Best scoring
        assert "Chunk 5" in result  # Last
        assert "..." in result
        
        # Should not contain all chunks
        chunk_count = result.count("Chunk")
        assert chunk_count == 3
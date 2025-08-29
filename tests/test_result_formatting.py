"""Tests for enhanced result formatting with chunk information."""

import pytest
import json
from mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from mcp_server_qdrant_rag.qdrant import Entry
from mcp_server_qdrant_rag.settings import ToolSettings, QdrantSettings, EmbeddingProviderSettings
from mcp_server_qdrant_rag.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self):
        self.model_name = "mock-model"
    
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in documents]
    
    async def embed_query(self, query: str) -> list[float]:
        return [0.1] * 384
    
    def get_vector_size(self) -> int:
        return 384
    
    def get_vector_name(self) -> str:
        return "default"


@pytest.fixture
def mcp_server():
    """Create a QdrantMCPServer for testing."""
    tool_settings = ToolSettings()
    qdrant_settings = QdrantSettings()
    # Set the fields directly since we're not using environment variables
    qdrant_settings.location = ":memory:"
    qdrant_settings.collection_name = "test_collection"
    
    embedding_provider = MockEmbeddingProvider()
    
    return QdrantMCPServer(
        tool_settings=tool_settings,
        qdrant_settings=qdrant_settings,
        embedding_provider=embedding_provider
    )


class TestResultFormatting:
    """Tests for enhanced result formatting with chunk information."""

    def test_format_non_chunked_entry(self, mcp_server):
        """Test formatting of non-chunked entry."""
        entry = Entry(
            content="This is a regular document.",
            metadata={"author": "Test Author", "category": "Test"}
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_metadata = json.dumps({"author": "Test Author", "category": "Test"})
        expected = f"<entry><content>This is a regular document.</content><metadata>{expected_metadata}</metadata></entry>"
        
        assert result == expected

    def test_format_non_chunked_entry_no_metadata(self, mcp_server):
        """Test formatting of non-chunked entry without metadata."""
        entry = Entry(content="Simple document without metadata.")
        
        result = mcp_server.format_entry(entry)
        
        expected = "<entry><content>Simple document without metadata.</content><metadata></metadata></entry>"
        assert result == expected

    def test_format_single_chunk_entry(self, mcp_server):
        """Test formatting of single chunk entry."""
        entry = Entry(
            content="This is chunk content.",
            metadata={"topic": "Testing"},
            is_chunk=True,
            source_document_id="doc-123456789012345",
            chunk_index=2,
            total_chunks=5
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_metadata = json.dumps({"topic": "Testing"})
        # Document ID should be truncated
        expected_chunk_info = " [Chunk 3/5, from document doc-1234...]"
        expected = f"<entry><content>This is chunk content.</content><metadata>{expected_metadata}</metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_chunk_entry_short_doc_id(self, mcp_server):
        """Test formatting of chunk entry with short document ID."""
        entry = Entry(
            content="Chunk with short doc ID.",
            is_chunk=True,
            source_document_id="short-id",
            chunk_index=0,
            total_chunks=3
        )
        
        result = mcp_server.format_entry(entry)
        
        # Short document ID should not be truncated
        expected_chunk_info = " [Chunk 1/3, from document short-id]"
        expected = f"<entry><content>Chunk with short doc ID.</content><metadata></metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_aggregated_chunk_entry(self, mcp_server):
        """Test formatting of aggregated chunk entry."""
        entry = Entry(
            content="Aggregated content from multiple chunks.",
            metadata={"source": "aggregated"},
            is_chunk=True,
            source_document_id="doc-789",
            chunk_index=None,  # No single chunk index for aggregated
            total_chunks=4
        )
        # Add aggregation metadata
        entry._chunk_count = 3
        
        result = mcp_server.format_entry(entry)
        
        expected_metadata = json.dumps({"source": "aggregated"})
        expected_chunk_info = " [Aggregated from 3 chunks, from document doc-789]"
        expected = f"<entry><content>Aggregated content from multiple chunks.</content><metadata>{expected_metadata}</metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_chunk_entry_no_doc_id(self, mcp_server):
        """Test formatting of chunk entry with empty source document ID."""
        entry = Entry(
            content="Chunk with empty source doc ID.",
            is_chunk=True,
            source_document_id="",  # Empty string instead of None
            chunk_index=1,
            total_chunks=2
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_chunk_info = " [Chunk 2/2]"  # No doc ID shown when empty
        expected = f"<entry><content>Chunk with empty source doc ID.</content><metadata></metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_chunk_entry_no_index(self, mcp_server):
        """Test formatting of chunk entry without chunk index."""
        entry = Entry(
            content="Chunk without index info.",
            is_chunk=True,
            source_document_id="doc-456",
            chunk_index=None,
            total_chunks=None
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_chunk_info = " [Multiple chunks, from document doc-456]"
        expected = f"<entry><content>Chunk without index info.</content><metadata></metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_chunk_entry_minimal_info(self, mcp_server):
        """Test formatting of chunk entry with minimal information."""
        entry = Entry(
            content="Minimal chunk info.",
            is_chunk=True,
            source_document_id="minimal-doc",  # Required field
            chunk_index=None,
            total_chunks=None
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_chunk_info = " [Multiple chunks, from document minimal-doc]"
        expected = f"<entry><content>Minimal chunk info.</content><metadata></metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_entry_with_complex_metadata(self, mcp_server):
        """Test formatting with complex metadata structure."""
        complex_metadata = {
            "author": "Test Author",
            "tags": ["test", "formatting", "chunks"],
            "nested": {
                "level": 1,
                "data": {"key": "value"}
            },
            "numbers": [1, 2, 3]
        }
        
        entry = Entry(
            content="Document with complex metadata.",
            metadata=complex_metadata,
            is_chunk=True,
            source_document_id="complex-doc",
            chunk_index=0,
            total_chunks=1
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_metadata = json.dumps(complex_metadata)
        expected_chunk_info = " [Chunk 1/1, from document complex-doc]"
        expected = f"<entry><content>Document with complex metadata.</content><metadata>{expected_metadata}</metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_entry_with_special_characters(self, mcp_server):
        """Test formatting with special characters in content."""
        entry = Entry(
            content="Content with <special> & \"characters\" and 'quotes'.",
            metadata={"type": "special"},
            is_chunk=True,
            source_document_id="special-chars-doc",
            chunk_index=0,
            total_chunks=1
        )
        
        result = mcp_server.format_entry(entry)
        
        # Content should be preserved as-is (XML escaping handled by consumer)
        expected_metadata = json.dumps({"type": "special"})
        expected_chunk_info = " [Chunk 1/1, from document special-...]"
        expected = f"<entry><content>Content with <special> & \"characters\" and 'quotes'.</content><metadata>{expected_metadata}</metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_entry_empty_content(self, mcp_server):
        """Test formatting with empty content."""
        entry = Entry(
            content="",
            is_chunk=True,
            source_document_id="empty-doc",
            chunk_index=0,
            total_chunks=1
        )
        
        result = mcp_server.format_entry(entry)
        
        expected_chunk_info = " [Chunk 1/1, from document empty-doc]"
        expected = f"<entry><content></content><metadata></metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_entry_long_content(self, mcp_server):
        """Test formatting with very long content."""
        long_content = "This is a very long piece of content. " * 100
        
        entry = Entry(
            content=long_content,
            is_chunk=True,
            source_document_id="long-content-doc",
            chunk_index=5,
            total_chunks=10
        )
        
        result = mcp_server.format_entry(entry)
        
        # Content should be preserved in full
        expected_chunk_info = " [Chunk 6/10, from document long-con...]"
        expected = f"<entry><content>{long_content}</content><metadata></metadata><chunk_info>{expected_chunk_info}</chunk_info></entry>"
        
        assert result == expected

    def test_format_multiple_entries_consistency(self, mcp_server):
        """Test that formatting is consistent across multiple entries."""
        entries = [
            Entry(content="First entry", is_chunk=False),
            Entry(
                content="Second entry chunk",
                is_chunk=True,
                source_document_id="doc-1",
                chunk_index=0,
                total_chunks=2
            ),
            Entry(
                content="Third entry aggregated",
                is_chunk=True,
                source_document_id="doc-2"
            )
        ]
        entries[2]._chunk_count = 4  # Mark as aggregated
        
        results = [mcp_server.format_entry(entry) for entry in entries]
        
        # All results should be properly formatted XML-like strings
        for result in results:
            assert result.startswith("<entry>")
            assert result.endswith("</entry>")
            assert "<content>" in result
            assert "<metadata>" in result
        
        # Only chunked entries should have chunk_info
        assert "<chunk_info>" not in results[0]  # Non-chunked
        assert "<chunk_info>" in results[1]      # Single chunk
        assert "<chunk_info>" in results[2]      # Aggregated chunk
        
        # Check specific chunk info content
        assert "Chunk 1/2" in results[1]
        assert "Aggregated from 4 chunks" in results[2]
"""Tests for the enhanced Entry model with chunk support."""

import pytest
from pydantic import ValidationError

from mcp_server_qdrant.qdrant import Entry


class TestEntryModel:
    """Test cases for the Entry model with chunk support."""

    def test_basic_entry_creation(self):
        """Test creating a basic entry without chunk fields."""
        entry = Entry(content="Test content")
        assert entry.content == "Test content"
        assert entry.metadata is None
        assert entry.is_chunk is False
        assert entry.source_document_id is None
        assert entry.chunk_index is None
        assert entry.total_chunks is None

    def test_entry_with_metadata(self):
        """Test creating an entry with metadata."""
        metadata = {"author": "test", "category": "example"}
        entry = Entry(content="Test content", metadata=metadata)
        assert entry.content == "Test content"
        assert entry.metadata == metadata
        assert entry.is_chunk is False

    def test_valid_chunk_entry(self):
        """Test creating a valid chunk entry with all required fields."""
        entry = Entry(
            content="Chunk content",
            is_chunk=True,
            source_document_id="doc-123",
            chunk_index=0,
            total_chunks=3
        )
        assert entry.content == "Chunk content"
        assert entry.is_chunk is True
        assert entry.source_document_id == "doc-123"
        assert entry.chunk_index == 0
        assert entry.total_chunks == 3

    def test_chunk_entry_with_metadata(self):
        """Test creating a chunk entry with metadata."""
        metadata = {"original_title": "Document Title"}
        entry = Entry(
            content="Chunk content",
            metadata=metadata,
            is_chunk=True,
            source_document_id="doc-123",
            chunk_index=1,
            total_chunks=3
        )
        assert entry.metadata == metadata
        assert entry.is_chunk is True
        assert entry.chunk_index == 1

    def test_chunk_missing_source_document_id(self):
        """Test that chunk entry fails validation without source_document_id."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                chunk_index=0,
                total_chunks=3
            )
        assert "source_document_id is required when is_chunk is True" in str(exc_info.value)

    def test_chunk_missing_chunk_index(self):
        """Test that chunk entry fails validation without chunk_index."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                source_document_id="doc-123",
                total_chunks=3
            )
        assert "chunk_index is required when is_chunk is True" in str(exc_info.value)

    def test_chunk_missing_total_chunks(self):
        """Test that chunk entry fails validation without total_chunks."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                source_document_id="doc-123",
                chunk_index=0
            )
        assert "total_chunks is required when is_chunk is True" in str(exc_info.value)

    def test_chunk_index_out_of_range(self):
        """Test that chunk_index must be less than total_chunks."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                source_document_id="doc-123",
                chunk_index=3,
                total_chunks=3
            )
        assert "chunk_index must be less than total_chunks" in str(exc_info.value)

    def test_negative_chunk_index(self):
        """Test that chunk_index cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                source_document_id="doc-123",
                chunk_index=-1,
                total_chunks=3
            )
        assert "chunk_index must be non-negative" in str(exc_info.value)

    def test_zero_total_chunks(self):
        """Test that total_chunks must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                source_document_id="doc-123",
                chunk_index=0,
                total_chunks=0
            )
        assert "total_chunks must be positive" in str(exc_info.value)

    def test_negative_total_chunks(self):
        """Test that total_chunks cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Chunk content",
                is_chunk=True,
                source_document_id="doc-123",
                chunk_index=0,
                total_chunks=-1
            )
        assert "total_chunks must be positive" in str(exc_info.value)

    def test_non_chunk_with_chunk_fields(self):
        """Test that non-chunk entries cannot have chunk-related fields."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Regular content",
                is_chunk=False,
                source_document_id="doc-123"
            )
        assert "chunk-related fields should be None when is_chunk is False" in str(exc_info.value)

    def test_non_chunk_with_chunk_index(self):
        """Test that non-chunk entries cannot have chunk_index."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Regular content",
                is_chunk=False,
                chunk_index=0
            )
        assert "chunk-related fields should be None when is_chunk is False" in str(exc_info.value)

    def test_non_chunk_with_total_chunks(self):
        """Test that non-chunk entries cannot have total_chunks."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Regular content",
                is_chunk=False,
                total_chunks=3
            )
        assert "chunk-related fields should be None when is_chunk is False" in str(exc_info.value)

    def test_valid_chunk_sequence(self):
        """Test creating a sequence of valid chunks."""
        chunks = []
        for i in range(3):
            chunk = Entry(
                content=f"Chunk {i} content",
                is_chunk=True,
                source_document_id="doc-456",
                chunk_index=i,
                total_chunks=3
            )
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert all(chunk.is_chunk for chunk in chunks)
        assert all(chunk.source_document_id == "doc-456" for chunk in chunks)
        assert all(chunk.total_chunks == 3 for chunk in chunks)
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

    def test_chunk_with_complex_metadata(self):
        """Test chunk entry with complex metadata structure."""
        metadata = {
            "original_document": {
                "title": "Complex Document",
                "author": "Test Author",
                "tags": ["tag1", "tag2"]
            },
            "chunk_info": {
                "semantic_boundary": "paragraph",
                "overlap_tokens": 50
            }
        }
        entry = Entry(
            content="Complex chunk content",
            metadata=metadata,
            is_chunk=True,
            source_document_id="doc-complex",
            chunk_index=2,
            total_chunks=5
        )
        assert entry.metadata == metadata
        assert entry.is_chunk is True
        assert entry.chunk_index == 2
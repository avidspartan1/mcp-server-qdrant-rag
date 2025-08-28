"""Unit tests for DocumentChunk model validation and functionality."""

import pytest
from pydantic import ValidationError

from mcp_server_qdrant.chunking.models import DocumentChunk


class TestDocumentChunkModel:
    """Test cases for DocumentChunk model validation and methods."""

    def test_valid_document_chunk_creation(self):
        """Test creating a valid DocumentChunk."""
        chunk = DocumentChunk(
            content="This is a test chunk content.",
            chunk_index=0,
            source_document_id="doc-123",
            total_chunks=3,
            overlap_start=0,
            overlap_end=10,
            chunk_strategy="semantic"
        )
        
        assert chunk.content == "This is a test chunk content."
        assert chunk.chunk_index == 0
        assert chunk.source_document_id == "doc-123"
        assert chunk.total_chunks == 3
        assert chunk.overlap_start == 0
        assert chunk.overlap_end == 10
        assert chunk.chunk_strategy == "semantic"
        assert chunk.metadata is None

    def test_document_chunk_with_metadata(self):
        """Test creating a DocumentChunk with metadata."""
        metadata = {
            "original_title": "Test Document",
            "author": "Test Author",
            "tags": ["test", "chunk"]
        }
        
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=1,
            source_document_id="doc-456",
            total_chunks=2,
            metadata=metadata
        )
        
        assert chunk.metadata == metadata

    def test_document_chunk_defaults(self):
        """Test DocumentChunk default values."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=0,
            source_document_id="doc-789",
            total_chunks=1
        )
        
        assert chunk.overlap_start == 0
        assert chunk.overlap_end == 0
        assert chunk.chunk_strategy == "hybrid"
        assert chunk.metadata is None

    def test_empty_content_validation(self):
        """Test that empty content raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1
            )
        
        assert "at least 1 character" in str(exc_info.value) or "min_length" in str(exc_info.value)

    def test_negative_chunk_index_validation(self):
        """Test that negative chunk_index raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=-1,
                source_document_id="doc-123",
                total_chunks=1
            )
        
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_empty_source_document_id_validation(self):
        """Test that empty source_document_id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="",
                total_chunks=1
            )
        
        assert "at least 1 character" in str(exc_info.value) or "min_length" in str(exc_info.value)

    def test_zero_total_chunks_validation(self):
        """Test that zero total_chunks raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=0
            )
        
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_negative_total_chunks_validation(self):
        """Test that negative total_chunks raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=-1
            )
        
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_chunk_index_greater_than_total_validation(self):
        """Test that chunk_index >= total_chunks raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=3,
                source_document_id="doc-123",
                total_chunks=3
            )
        
        assert "chunk_index (3) must be less than total_chunks (3)" in str(exc_info.value)

    def test_chunk_index_equal_to_total_validation(self):
        """Test that chunk_index == total_chunks raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=2,
                source_document_id="doc-123",
                total_chunks=2
            )
        
        assert "chunk_index (2) must be less than total_chunks (2)" in str(exc_info.value)

    def test_negative_overlap_start_validation(self):
        """Test that negative overlap_start raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1,
                overlap_start=-5
            )
        
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_negative_overlap_end_validation(self):
        """Test that negative overlap_end raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1,
                overlap_end=-3
            )
        
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_chunk_strategy_validation(self):
        """Test that invalid chunk_strategy raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1,
                chunk_strategy="invalid_strategy"
            )
        
        assert "chunk_strategy must be one of" in str(exc_info.value)
        assert "invalid_strategy" in str(exc_info.value)

    def test_valid_chunk_strategies(self):
        """Test that all valid chunk strategies are accepted."""
        valid_strategies = ["hybrid", "semantic", "fixed", "sentence"]
        
        for strategy in valid_strategies:
            chunk = DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1,
                chunk_strategy=strategy
            )
            assert chunk.chunk_strategy == strategy

    def test_metadata_structure_validation(self):
        """Test metadata structure validation."""
        # Valid metadata
        valid_metadata = {
            "title": "Test",
            "original_document_metadata": {"author": "Test Author"}
        }
        
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=0,
            source_document_id="doc-123",
            total_chunks=1,
            metadata=valid_metadata
        )
        
        assert chunk.metadata == valid_metadata

    def test_invalid_metadata_type_validation(self):
        """Test that non-dict metadata raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1,
                metadata="not a dict"
            )
        
        assert "Input should be a valid dictionary" in str(exc_info.value)

    def test_invalid_original_document_metadata_validation(self):
        """Test that invalid original_document_metadata raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-123",
                total_chunks=1,
                metadata={"original_document_metadata": "not a dict"}
            )
        
        assert "original_document_metadata must be a dictionary" in str(exc_info.value)

    def test_get_chunk_metadata_basic(self):
        """Test get_chunk_metadata method with basic chunk."""
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=1,
            source_document_id="doc-456",
            total_chunks=3,
            overlap_start=5,
            overlap_end=10,
            chunk_strategy="semantic"
        )
        
        metadata = chunk.get_chunk_metadata()
        
        expected = {
            "is_chunk": True,
            "source_document_id": "doc-456",
            "chunk_index": 1,
            "total_chunks": 3,
            "chunk_strategy": "semantic",
            "overlap_start": 5,
            "overlap_end": 10,
        }
        
        assert metadata == expected

    def test_get_chunk_metadata_with_original_metadata(self):
        """Test get_chunk_metadata method with original metadata."""
        original_metadata = {
            "title": "Original Document",
            "author": "Test Author"
        }
        
        chunk = DocumentChunk(
            content="Test content",
            chunk_index=0,
            source_document_id="doc-789",
            total_chunks=2,
            metadata=original_metadata
        )
        
        metadata = chunk.get_chunk_metadata()
        
        assert metadata["is_chunk"] is True
        assert metadata["source_document_id"] == "doc-789"
        assert metadata["chunk_index"] == 0
        assert metadata["total_chunks"] == 2
        assert metadata["original_document_metadata"] == original_metadata

    def test_is_first_chunk(self):
        """Test is_first_chunk method."""
        # First chunk
        first_chunk = DocumentChunk(
            content="First chunk",
            chunk_index=0,
            source_document_id="doc-123",
            total_chunks=3
        )
        assert first_chunk.is_first_chunk() is True
        assert first_chunk.is_last_chunk() is False
        
        # Not first chunk
        middle_chunk = DocumentChunk(
            content="Middle chunk",
            chunk_index=1,
            source_document_id="doc-123",
            total_chunks=3
        )
        assert middle_chunk.is_first_chunk() is False

    def test_is_last_chunk(self):
        """Test is_last_chunk method."""
        # Last chunk
        last_chunk = DocumentChunk(
            content="Last chunk",
            chunk_index=2,
            source_document_id="doc-123",
            total_chunks=3
        )
        assert last_chunk.is_last_chunk() is True
        assert last_chunk.is_first_chunk() is False
        
        # Not last chunk
        first_chunk = DocumentChunk(
            content="First chunk",
            chunk_index=0,
            source_document_id="doc-123",
            total_chunks=3
        )
        assert first_chunk.is_last_chunk() is False

    def test_single_chunk_is_both_first_and_last(self):
        """Test that a single chunk is both first and last."""
        single_chunk = DocumentChunk(
            content="Only chunk",
            chunk_index=0,
            source_document_id="doc-123",
            total_chunks=1
        )
        
        assert single_chunk.is_first_chunk() is True
        assert single_chunk.is_last_chunk() is True

    def test_string_representation(self):
        """Test __str__ method."""
        chunk = DocumentChunk(
            content="This is a test chunk with some content that should be truncated in the string representation.",
            chunk_index=1,
            source_document_id="doc-123",
            total_chunks=3
        )
        
        str_repr = str(chunk)
        
        assert "DocumentChunk(2/3)" in str_repr  # 1-based indexing in display
        assert "This is a test chunk with some content that should" in str_repr
        assert "..." in str_repr  # Content should be truncated

    def test_string_representation_short_content(self):
        """Test __str__ method with short content."""
        chunk = DocumentChunk(
            content="Short content",
            chunk_index=0,
            source_document_id="doc-123",
            total_chunks=1
        )
        
        str_repr = str(chunk)
        
        assert "DocumentChunk(1/1)" in str_repr
        assert "Short content" in str_repr
        assert "..." not in str_repr  # No truncation for short content

    def test_chunk_sequence_validation(self):
        """Test creating a valid sequence of chunks."""
        chunks = []
        source_id = "doc-sequence-test"
        total = 3
        
        for i in range(total):
            chunk = DocumentChunk(
                content=f"Chunk {i} content",
                chunk_index=i,
                source_document_id=source_id,
                total_chunks=total,
                overlap_start=5 if i > 0 else 0,
                overlap_end=5 if i < total - 1 else 0
            )
            chunks.append(chunk)
        
        # Verify sequence properties
        assert len(chunks) == 3
        assert all(chunk.source_document_id == source_id for chunk in chunks)
        assert all(chunk.total_chunks == total for chunk in chunks)
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
        
        # Verify first/last chunk properties
        assert chunks[0].is_first_chunk()
        assert not chunks[0].is_last_chunk()
        assert not chunks[1].is_first_chunk()
        assert not chunks[1].is_last_chunk()
        assert not chunks[2].is_first_chunk()
        assert chunks[2].is_last_chunk()

    def test_complex_metadata_structure(self):
        """Test DocumentChunk with complex nested metadata."""
        complex_metadata = {
            "document_info": {
                "title": "Complex Document",
                "author": "Test Author",
                "created_at": "2024-01-01",
                "tags": ["test", "complex", "metadata"]
            },
            "processing_info": {
                "chunking_strategy": "semantic",
                "model_used": "test-model",
                "timestamp": "2024-01-01T12:00:00Z"
            },
            "custom_fields": {
                "priority": 1,
                "category": "test",
                "active": True
            }
        }
        
        chunk = DocumentChunk(
            content="Complex chunk content",
            chunk_index=0,
            source_document_id="doc-complex",
            total_chunks=1,
            metadata=complex_metadata
        )
        
        assert chunk.metadata == complex_metadata
        
        chunk_metadata = chunk.get_chunk_metadata()
        assert chunk_metadata["original_document_metadata"] == complex_metadata
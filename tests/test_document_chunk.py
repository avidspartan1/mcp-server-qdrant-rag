"""Unit tests for DocumentChunk data model."""

import pytest
from pydantic import ValidationError

from mcp_server_qdrant.chunking.models import DocumentChunk


class TestDocumentChunk:
    """Test cases for DocumentChunk model creation and validation."""
    
    def test_create_valid_document_chunk(self):
        """Test creating a valid DocumentChunk with all required fields."""
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
    
    def test_create_chunk_with_metadata(self):
        """Test creating a DocumentChunk with metadata."""
        metadata = {
            "author": "John Doe",
            "title": "Test Document",
            "original_document_metadata": {"category": "test"}
        }
        
        chunk = DocumentChunk(
            content="Test content with metadata.",
            chunk_index=1,
            source_document_id="doc-456",
            total_chunks=2,
            metadata=metadata
        )
        
        assert chunk.metadata == metadata
        assert chunk.metadata["author"] == "John Doe"
    
    def test_chunk_index_validation(self):
        """Test that chunk_index must be less than total_chunks."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=3,  # Invalid: >= total_chunks
                source_document_id="doc-789",
                total_chunks=3
            )
        
        assert "chunk_index (3) must be less than total_chunks (3)" in str(exc_info.value)
    
    def test_negative_chunk_index_validation(self):
        """Test that chunk_index cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=-1,
                source_document_id="doc-789",
                total_chunks=3
            )
        
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
    
    def test_empty_content_validation(self):
        """Test that content cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="",
                chunk_index=0,
                source_document_id="doc-empty",
                total_chunks=1
            )
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_empty_source_document_id_validation(self):
        """Test that source_document_id cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="",
                total_chunks=1
            )
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_invalid_total_chunks_validation(self):
        """Test that total_chunks must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-invalid",
                total_chunks=0
            )
        
        assert "Input should be greater than or equal to 1" in str(exc_info.value)
    
    def test_negative_overlap_validation(self):
        """Test that overlap values cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-overlap",
                total_chunks=1,
                overlap_start=-5
            )
        
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
    
    def test_invalid_chunk_strategy_validation(self):
        """Test that chunk_strategy must be one of allowed values."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-strategy",
                total_chunks=1,
                chunk_strategy="invalid_strategy"
            )
        
        assert "chunk_strategy must be one of" in str(exc_info.value) and "invalid_strategy" in str(exc_info.value)
    
    def test_invalid_metadata_type_validation(self):
        """Test that metadata must be a dictionary."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-meta",
                total_chunks=1,
                metadata="invalid_metadata"  # Should be dict, not string
            )
        
        assert "Input should be a valid dictionary" in str(exc_info.value)
    
    def test_invalid_original_document_metadata_validation(self):
        """Test that original_document_metadata must be a dictionary if present."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                content="Test content",
                chunk_index=0,
                source_document_id="doc-orig-meta",
                total_chunks=1,
                metadata={"original_document_metadata": "not_a_dict"}
            )
        
        assert "original_document_metadata must be a dictionary" in str(exc_info.value)
    
    def test_get_chunk_metadata(self):
        """Test the get_chunk_metadata method."""
        original_metadata = {"title": "Original Doc", "author": "Jane Doe"}
        
        chunk = DocumentChunk(
            content="Test content for metadata",
            chunk_index=1,
            source_document_id="doc-meta-test",
            total_chunks=3,
            overlap_start=5,
            overlap_end=10,
            chunk_strategy="semantic",
            metadata=original_metadata
        )
        
        chunk_metadata = chunk.get_chunk_metadata()
        
        expected_metadata = {
            "is_chunk": True,
            "source_document_id": "doc-meta-test",
            "chunk_index": 1,
            "total_chunks": 3,
            "chunk_strategy": "semantic",
            "overlap_start": 5,
            "overlap_end": 10,
            "original_document_metadata": original_metadata
        }
        
        assert chunk_metadata == expected_metadata
    
    def test_get_chunk_metadata_without_original_metadata(self):
        """Test get_chunk_metadata when no original metadata is present."""
        chunk = DocumentChunk(
            content="Test content without metadata",
            chunk_index=0,
            source_document_id="doc-no-meta",
            total_chunks=1
        )
        
        chunk_metadata = chunk.get_chunk_metadata()
        
        expected_metadata = {
            "is_chunk": True,
            "source_document_id": "doc-no-meta",
            "chunk_index": 0,
            "total_chunks": 1,
            "chunk_strategy": "hybrid",  # default value
            "overlap_start": 0,  # default value
            "overlap_end": 0,  # default value
        }
        
        assert chunk_metadata == expected_metadata
        assert "original_document_metadata" not in chunk_metadata
    
    def test_is_first_chunk(self):
        """Test the is_first_chunk method."""
        first_chunk = DocumentChunk(
            content="First chunk",
            chunk_index=0,
            source_document_id="doc-first",
            total_chunks=3
        )
        
        middle_chunk = DocumentChunk(
            content="Middle chunk",
            chunk_index=1,
            source_document_id="doc-first",
            total_chunks=3
        )
        
        assert first_chunk.is_first_chunk() is True
        assert middle_chunk.is_first_chunk() is False
    
    def test_is_last_chunk(self):
        """Test the is_last_chunk method."""
        middle_chunk = DocumentChunk(
            content="Middle chunk",
            chunk_index=1,
            source_document_id="doc-last",
            total_chunks=3
        )
        
        last_chunk = DocumentChunk(
            content="Last chunk",
            chunk_index=2,
            source_document_id="doc-last",
            total_chunks=3
        )
        
        assert middle_chunk.is_last_chunk() is False
        assert last_chunk.is_last_chunk() is True
    
    def test_string_representation(self):
        """Test the __str__ method."""
        chunk = DocumentChunk(
            content="This is a test chunk with some content to display",
            chunk_index=1,
            source_document_id="doc-str",
            total_chunks=3
        )
        
        str_repr = str(chunk)
        assert "DocumentChunk(2/3):" in str_repr
        assert "This is a test chunk with some content to display" in str_repr
    
    def test_string_representation_long_content(self):
        """Test the __str__ method with long content that gets truncated."""
        long_content = "This is a very long chunk content that should be truncated in the string representation because it exceeds the 50 character limit."
        
        chunk = DocumentChunk(
            content=long_content,
            chunk_index=0,
            source_document_id="doc-long",
            total_chunks=1
        )
        
        str_repr = str(chunk)
        assert "DocumentChunk(1/1):" in str_repr
        assert "This is a very long chunk content that should be t..." in str_repr
        assert len(str_repr) < len(long_content) + 50  # Should be truncated
    
    def test_default_values(self):
        """Test that default values are properly set."""
        chunk = DocumentChunk(
            content="Test with defaults",
            chunk_index=0,
            source_document_id="doc-defaults",
            total_chunks=1
        )
        
        assert chunk.metadata is None
        assert chunk.overlap_start == 0
        assert chunk.overlap_end == 0
        assert chunk.chunk_strategy == "hybrid"
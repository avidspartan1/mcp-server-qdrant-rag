"""
Unit tests for Entry model extensions with document_type and set_id fields.
"""

import pytest
from pydantic import ValidationError

from mcp_server_qdrant_rag.qdrant import Entry


class TestEntryModelExtensions:
    """Test the Entry model with new metadata fields."""

    def test_entry_with_new_metadata_fields(self):
        """Test creating Entry with document_type and set_id fields."""
        entry = Entry(
            content="Test content",
            metadata={"key": "value"},
            document_type="api_docs",
            set_id="platform_code",
        )

        assert entry.content == "Test content"
        assert entry.metadata == {"key": "value"}
        assert entry.document_type == "api_docs"
        assert entry.set_id == "platform_code"
        assert entry.is_chunk is False
        assert entry.source_document_id is None
        assert entry.chunk_index is None
        assert entry.total_chunks is None

    def test_entry_without_new_metadata_fields(self):
        """Test creating Entry without new metadata fields (backward compatibility)."""
        entry = Entry(content="Test content", metadata={"key": "value"})

        assert entry.content == "Test content"
        assert entry.metadata == {"key": "value"}
        assert entry.document_type is None
        assert entry.set_id is None
        assert entry.is_chunk is False

    def test_entry_with_only_document_type(self):
        """Test creating Entry with only document_type field."""
        entry = Entry(content="Test content", document_type="user_guide")

        assert entry.content == "Test content"
        assert entry.document_type == "user_guide"
        assert entry.set_id is None
        assert entry.metadata is None

    def test_entry_with_only_set_id(self):
        """Test creating Entry with only set_id field."""
        entry = Entry(content="Test content", set_id="documentation")

        assert entry.content == "Test content"
        assert entry.set_id == "documentation"
        assert entry.document_type is None
        assert entry.metadata is None

    def test_entry_with_chunk_fields_and_new_metadata(self):
        """Test creating Entry with both chunk fields and new metadata fields."""
        entry = Entry(
            content="Chunk content",
            metadata={"source": "file.txt"},
            document_type="technical_spec",
            set_id="engineering",
            is_chunk=True,
            source_document_id="doc123",
            chunk_index=0,
            total_chunks=5,
        )

        assert entry.content == "Chunk content"
        assert entry.document_type == "technical_spec"
        assert entry.set_id == "engineering"
        assert entry.is_chunk is True
        assert entry.source_document_id == "doc123"
        assert entry.chunk_index == 0
        assert entry.total_chunks == 5

    def test_document_type_validation_empty_string(self):
        """Test that empty document_type string is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(content="Test content", document_type="")

        assert "document_type must be non-empty if provided" in str(exc_info.value)

    def test_document_type_validation_whitespace_only(self):
        """Test that whitespace-only document_type is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(content="Test content", document_type="   ")

        assert "document_type must be non-empty if provided" in str(exc_info.value)

    def test_document_type_validation_strips_whitespace(self):
        """Test that document_type strips leading/trailing whitespace."""
        entry = Entry(content="Test content", document_type="  api_docs  ")

        assert entry.document_type == "api_docs"

    def test_set_id_validation_empty_string(self):
        """Test that empty set_id string is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(content="Test content", set_id="")

        assert "set_id must be non-empty if provided" in str(exc_info.value)

    def test_set_id_validation_whitespace_only(self):
        """Test that whitespace-only set_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Entry(content="Test content", set_id="   ")

        assert "set_id must be non-empty if provided" in str(exc_info.value)

    def test_set_id_validation_strips_whitespace(self):
        """Test that set_id strips leading/trailing whitespace."""
        entry = Entry(content="Test content", set_id="  platform_code  ")

        assert entry.set_id == "platform_code"

    def test_entry_serialization_with_new_fields(self):
        """Test that Entry can be serialized and deserialized with new fields."""
        original_entry = Entry(
            content="Test content",
            metadata={"key": "value"},
            document_type="api_docs",
            set_id="platform_code",
        )

        # Serialize to dict
        entry_dict = original_entry.model_dump()

        # Verify new fields are in the dict
        assert entry_dict["document_type"] == "api_docs"
        assert entry_dict["set_id"] == "platform_code"

        # Deserialize from dict
        reconstructed_entry = Entry(**entry_dict)

        assert reconstructed_entry.content == original_entry.content
        assert reconstructed_entry.metadata == original_entry.metadata
        assert reconstructed_entry.document_type == original_entry.document_type
        assert reconstructed_entry.set_id == original_entry.set_id

    def test_entry_serialization_without_new_fields(self):
        """Test backward compatibility: Entry without new fields can be serialized."""
        original_entry = Entry(content="Test content", metadata={"key": "value"})

        # Serialize to dict
        entry_dict = original_entry.model_dump()

        # Verify new fields are None in the dict
        assert entry_dict["document_type"] is None
        assert entry_dict["set_id"] is None

        # Deserialize from dict
        reconstructed_entry = Entry(**entry_dict)

        assert reconstructed_entry.content == original_entry.content
        assert reconstructed_entry.metadata == original_entry.metadata
        assert reconstructed_entry.document_type is None
        assert reconstructed_entry.set_id is None

    def test_entry_backward_compatibility_with_old_dict(self):
        """Test that Entry can be created from old dict format without new fields."""
        old_entry_dict = {
            "content": "Test content",
            "metadata": {"key": "value"},
            "is_chunk": False,
            "source_document_id": None,
            "chunk_index": None,
            "total_chunks": None,
        }

        # Should not raise an error
        entry = Entry(**old_entry_dict)

        assert entry.content == "Test content"
        assert entry.metadata == {"key": "value"}
        assert entry.document_type is None
        assert entry.set_id is None
        assert entry.is_chunk is False

    def test_entry_with_all_fields(self):
        """Test Entry with all possible fields set."""
        entry = Entry(
            content="Complete test content",
            metadata={"source": "test.txt", "author": "test"},
            document_type="integration_test",
            set_id="test_suite",
            is_chunk=True,
            source_document_id="test_doc_123",
            chunk_index=2,
            total_chunks=10,
        )

        assert entry.content == "Complete test content"
        assert entry.metadata == {"source": "test.txt", "author": "test"}
        assert entry.document_type == "integration_test"
        assert entry.set_id == "test_suite"
        assert entry.is_chunk is True
        assert entry.source_document_id == "test_doc_123"
        assert entry.chunk_index == 2
        assert entry.total_chunks == 10

    def test_entry_model_validation_preserves_existing_chunk_validation(self):
        """Test that existing chunk validation still works with new fields."""
        # Test invalid chunk_index
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Test content",
                document_type="test_doc",
                set_id="test_set",
                chunk_index=-1,
            )

        assert "chunk_index must be non-negative" in str(exc_info.value)

        # Test invalid total_chunks
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Test content",
                document_type="test_doc",
                set_id="test_set",
                total_chunks=0,
            )

        assert "total_chunks must be positive" in str(exc_info.value)

    def test_entry_model_validation_chunk_consistency_with_new_fields(self):
        """Test that chunk consistency validation works with new fields."""
        # Test that is_chunk=True requires source_document_id
        with pytest.raises(ValidationError) as exc_info:
            Entry(
                content="Test content",
                document_type="test_doc",
                set_id="test_set",
                is_chunk=True,
                source_document_id=None,
            )

        assert "source_document_id is required when is_chunk is True" in str(
            exc_info.value
        )

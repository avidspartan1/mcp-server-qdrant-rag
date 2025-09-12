"""
Integration tests for Entry model with new metadata fields in storage and retrieval.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mcp_server_qdrant_rag.qdrant import Entry, QdrantConnector
from mcp_server_qdrant_rag.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self):
        self.model_name = "mock-model"

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        """Return mock query embedding."""
        return [0.1, 0.2, 0.3]

    def get_vector_size(self) -> int:
        """Return mock vector size."""
        return 3

    def get_vector_name(self) -> str:
        """Return mock vector name."""
        return "default"


class TestEntryMetadataIntegration:
    """Integration tests for Entry metadata fields with QdrantConnector."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = AsyncMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = MagicMock()
        client.get_collection.return_value.config.params.vectors = {
            "default": MagicMock(size=3)
        }
        return client

    @pytest.fixture
    def connector(self, mock_qdrant_client):
        """Create a QdrantConnector with mocked client."""
        embedding_provider = MockEmbeddingProvider()
        connector = QdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=embedding_provider,
            enable_chunking=False,  # Disable chunking for simpler tests
        )
        connector._client = mock_qdrant_client
        return connector

    @pytest.mark.asyncio
    async def test_store_entry_with_new_metadata_fields(
        self, connector, mock_qdrant_client
    ):
        """Test storing an entry with document_type and set_id fields."""
        entry = Entry(
            content="Test document content",
            metadata={"author": "test_user"},
            document_type="user_guide",
            set_id="documentation",
        )

        await connector.store(entry)

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()

        # Get the call arguments
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs["points"]

        assert len(points) == 1
        point = points[0]

        # Verify the payload contains new metadata fields
        payload = point.payload
        assert payload["document"] == "Test document content"
        assert payload["metadata"] == {"author": "test_user"}
        assert payload["document_type"] == "user_guide"
        assert payload["set_id"] == "documentation"

    @pytest.mark.asyncio
    async def test_store_entry_without_new_metadata_fields(
        self, connector, mock_qdrant_client
    ):
        """Test storing an entry without new metadata fields (backward compatibility)."""
        entry = Entry(content="Test document content", metadata={"author": "test_user"})

        await connector.store(entry)

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()

        # Get the call arguments
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs["points"]

        assert len(points) == 1
        point = points[0]

        # Verify the payload does not contain new metadata fields
        payload = point.payload
        assert payload["document"] == "Test document content"
        assert payload["metadata"] == {"author": "test_user"}
        assert "document_type" not in payload
        assert "set_id" not in payload

    @pytest.mark.asyncio
    async def test_store_entry_with_partial_new_metadata(
        self, connector, mock_qdrant_client
    ):
        """Test storing an entry with only one of the new metadata fields."""
        entry = Entry(
            content="Test document content",
            document_type="api_reference",
            # set_id is None
        )

        await connector.store(entry)

        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()

        # Get the call arguments
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs["points"]

        assert len(points) == 1
        point = points[0]

        # Verify the payload contains only the provided metadata field
        payload = point.payload
        assert payload["document"] == "Test document content"
        assert payload["document_type"] == "api_reference"
        assert "set_id" not in payload

    @pytest.mark.asyncio
    async def test_search_reconstructs_entry_with_new_metadata_fields(
        self, connector, mock_qdrant_client
    ):
        """Test that search correctly reconstructs Entry objects with new metadata fields."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.payload = {
            "document": "Test document content",
            "metadata": {"author": "test_user"},
            "document_type": "user_guide",
            "set_id": "documentation",
        }
        mock_result.score = 0.95

        mock_qdrant_client.query_points.return_value = MagicMock()
        mock_qdrant_client.query_points.return_value.points = [mock_result]

        results = await connector.search("test query")

        assert len(results) == 1
        entry = results[0]

        assert entry.content == "Test document content"
        assert entry.metadata == {"author": "test_user"}
        assert entry.document_type == "user_guide"
        assert entry.set_id == "documentation"

    @pytest.mark.asyncio
    async def test_search_handles_missing_new_metadata_fields(
        self, connector, mock_qdrant_client
    ):
        """Test that search handles entries without new metadata fields (backward compatibility)."""
        # Mock search results without new metadata fields
        mock_result = MagicMock()
        mock_result.payload = {
            "document": "Test document content",
            "metadata": {"author": "test_user"},
            # No document_type or set_id
        }
        mock_result.score = 0.95

        mock_qdrant_client.query_points.return_value = MagicMock()
        mock_qdrant_client.query_points.return_value.points = [mock_result]

        results = await connector.search("test query")

        assert len(results) == 1
        entry = results[0]

        assert entry.content == "Test document content"
        assert entry.metadata == {"author": "test_user"}
        assert entry.document_type is None
        assert entry.set_id is None

    @pytest.mark.asyncio
    async def test_chunked_storage_propagates_new_metadata_fields(
        self, connector, mock_qdrant_client
    ):
        """Test that chunked storage propagates new metadata fields to all chunks."""
        # Mock the chunker to return multiple chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "First chunk content"
        mock_chunk1.metadata = {"author": "test_user"}
        mock_chunk1.source_document_id = "doc123"
        mock_chunk1.chunk_index = 0
        mock_chunk1.total_chunks = 2

        mock_chunk2 = MagicMock()
        mock_chunk2.content = "Second chunk content"
        mock_chunk2.metadata = {"author": "test_user"}
        mock_chunk2.source_document_id = "doc123"
        mock_chunk2.chunk_index = 1
        mock_chunk2.total_chunks = 2

        # Enable chunking and mock the chunker
        connector._enable_chunking = True
        connector._chunker = AsyncMock()
        connector._chunker.chunk_document.return_value = [mock_chunk1, mock_chunk2]

        # Mock the should_chunk_document method to return True
        connector._should_chunk_document = MagicMock(return_value=True)

        entry = Entry(
            content="This is a long document that will be chunked into multiple pieces",
            metadata={"author": "test_user"},
            document_type="technical_spec",
            set_id="engineering",
        )

        await connector.store(entry)

        # Verify upsert was called twice (once for each chunk)
        assert mock_qdrant_client.upsert.call_count == 2

        # Check that both chunks have the new metadata fields
        for call in mock_qdrant_client.upsert.call_args_list:
            points = call.kwargs["points"]
            assert len(points) == 1
            point = points[0]
            payload = point.payload

            assert payload["document_type"] == "technical_spec"
            assert payload["set_id"] == "engineering"
            assert payload["is_chunk"] is True
            assert payload["source_document_id"] == "doc123"

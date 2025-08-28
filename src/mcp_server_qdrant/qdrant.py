import logging
import uuid
from typing import Any

from pydantic import BaseModel, field_validator, model_validator
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH
from mcp_server_qdrant.chunking.chunker import DocumentChunker
from mcp_server_qdrant.chunking.models import DocumentChunk

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None
    
    # Chunk-related fields for document chunking support
    is_chunk: bool = False
    source_document_id: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None

    @field_validator('chunk_index')
    @classmethod
    def validate_chunk_index(cls, v: int | None) -> int | None:
        """Validate that chunk_index is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError('chunk_index must be non-negative')
        return v

    @field_validator('total_chunks')
    @classmethod
    def validate_total_chunks(cls, v: int | None) -> int | None:
        """Validate that total_chunks is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError('total_chunks must be positive')
        return v

    @model_validator(mode='after')
    def validate_chunk_metadata(self) -> 'Entry':
        """Validate chunk metadata consistency."""
        if self.is_chunk:
            # If this is a chunk, all chunk-related fields should be provided
            if self.source_document_id is None:
                raise ValueError('source_document_id is required when is_chunk is True')
            if self.chunk_index is None:
                raise ValueError('chunk_index is required when is_chunk is True')
            if self.total_chunks is None:
                raise ValueError('total_chunks is required when is_chunk is True')
            
            # Validate chunk_index is within valid range
            if self.chunk_index >= self.total_chunks:
                raise ValueError('chunk_index must be less than total_chunks')
        else:
            # If this is not a chunk, chunk-related fields should be None
            if any([self.source_document_id, self.chunk_index is not None, self.total_chunks is not None]):
                raise ValueError('chunk-related fields should be None when is_chunk is False')
        
        return self


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
        enable_chunking: bool = True,
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunk_strategy: str = "semantic",
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes
        
        # Chunking configuration
        self._enable_chunking = enable_chunking
        self._chunker = None
        if enable_chunking:
            self._chunker = DocumentChunker(
                max_tokens=max_chunk_size,
                overlap_tokens=chunk_overlap,
                sentence_splitter=None,  # Auto-select best available
                tokenizer=None,  # Auto-select best available
            )

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        Automatically chunks large documents when chunking is enabled.
        
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Check if this entry should be chunked
        should_chunk = (
            self._enable_chunking 
            and self._chunker is not None 
            and not entry.is_chunk  # Don't re-chunk already chunked entries
            and self._should_chunk_document(entry.content)
        )

        if should_chunk:
            await self._store_chunked_document(entry, collection_name)
        else:
            await self._store_single_entry(entry, collection_name)

    async def _store_single_entry(self, entry: Entry, collection_name: str):
        """Store a single entry without chunking."""
        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Prepare payload with chunk metadata if this is a chunk
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        
        # Add chunk-related metadata to payload if this is a chunk
        if entry.is_chunk:
            payload.update({
                "is_chunk": entry.is_chunk,
                "source_document_id": entry.source_document_id,
                "chunk_index": entry.chunk_index,
                "total_chunks": entry.total_chunks,
            })

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def _store_chunked_document(self, entry: Entry, collection_name: str):
        """Store a document by chunking it into smaller pieces."""
        # Generate a unique source document ID
        source_document_id = str(uuid.uuid4())
        
        # Create chunks using the DocumentChunker
        document_chunks = await self._chunker.chunk_document(
            content=entry.content,
            metadata=entry.metadata,
            source_document_id=source_document_id
        )
        
        if not document_chunks:
            # If chunking failed or produced no chunks, store as single entry
            logger.warning("Document chunking produced no chunks, storing as single entry")
            await self._store_single_entry(entry, collection_name)
            return
        
        # Convert DocumentChunk objects to Entry objects and store them
        chunk_entries = []
        for doc_chunk in document_chunks:
            chunk_entry = Entry(
                content=doc_chunk.content,
                metadata=doc_chunk.metadata,
                is_chunk=True,
                source_document_id=doc_chunk.source_document_id,
                chunk_index=doc_chunk.chunk_index,
                total_chunks=doc_chunk.total_chunks
            )
            chunk_entries.append(chunk_entry)
        
        # Store all chunks
        for chunk_entry in chunk_entries:
            await self._store_single_entry(chunk_entry, collection_name)
        
        logger.info(f"Stored document as {len(chunk_entries)} chunks in collection '{collection_name}'")

    def _should_chunk_document(self, content: str) -> bool:
        """
        Determine if a document should be chunked based on its size.
        
        :param content: The document content to evaluate
        :return: True if the document should be chunked, False otherwise
        """
        if not self._chunker:
            return False
        
        # Use the chunker's token counting method to determine if chunking is needed
        token_count = self._chunker._count_tokens(content)
        
        # Chunk if the document exceeds the maximum chunk size
        return token_count > self._chunker.max_tokens

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        # Convert search results to Entry objects, handling both chunked and non-chunked content
        entries = []
        for result in search_results.points:
            payload = result.payload
            
            # Extract chunk-related fields from payload if present
            is_chunk = payload.get("is_chunk", False)
            source_document_id = payload.get("source_document_id")
            chunk_index = payload.get("chunk_index")
            total_chunks = payload.get("total_chunks")
            
            entry = Entry(
                content=payload["document"],
                metadata=payload.get("metadata"),
                is_chunk=is_chunk,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            entries.append(entry)
        
        return entries

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        Validates vector dimensions for existing collections.
        :param collection_name: The name of the collection to ensure exists.
        :raises ValueError: If existing collection has incompatible vector dimensions.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        
        if collection_exists:
            # Validate that existing collection has compatible vector dimensions
            await self._validate_collection_dimensions(collection_name)
        else:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            # Create payload indexes if configured
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

    async def _validate_collection_dimensions(self, collection_name: str):
        """
        Validate that the existing collection has compatible vector dimensions.
        :param collection_name: The name of the collection to validate.
        :raises ValueError: If collection has incompatible vector dimensions.
        """
        try:
            collection_info = await self._client.get_collection(collection_name)
            expected_vector_size = self._embedding_provider.get_vector_size()
            expected_vector_name = self._embedding_provider.get_vector_name()
            
            # Check if the expected vector name exists in the collection
            if expected_vector_name not in collection_info.config.params.vectors:
                available_vectors = list(collection_info.config.params.vectors.keys())
                raise ValueError(
                    f"Collection '{collection_name}' does not have the expected vector '{expected_vector_name}'. "
                    f"Available vectors: {available_vectors}. "
                    f"This usually indicates a change in embedding model. "
                    f"Consider using a different collection name or recreating the collection."
                )
            
            # Check vector dimensions
            actual_vector_size = collection_info.config.params.vectors[expected_vector_name].size
            if actual_vector_size != expected_vector_size:
                current_model = getattr(self._embedding_provider, 'model_name', 'unknown')
                raise ValueError(
                    f"Vector dimension mismatch for collection '{collection_name}'. "
                    f"Expected {expected_vector_size} dimensions (model: {current_model}), "
                    f"but collection has {actual_vector_size} dimensions. "
                    f"This usually indicates a change in embedding model. "
                    f"Consider using a different collection name or recreating the collection."
                )
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise our custom validation errors
            else:
                # Log other errors but don't fail - collection might be valid
                logger.warning(f"Could not validate collection dimensions for '{collection_name}': {e}")

    def get_embedding_model_info(self) -> dict[str, Any]:
        """
        Get information about the current embedding model.
        :return: Dictionary containing model information.
        """
        return {
            "model_name": getattr(self._embedding_provider, 'model_name', 'unknown'),
            "vector_size": self._embedding_provider.get_vector_size(),
            "vector_name": self._embedding_provider.get_vector_name(),
        }

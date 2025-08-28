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
            # If this is a chunk, source_document_id is required
            if self.source_document_id is None:
                raise ValueError('source_document_id is required when is_chunk is True')
            
            # For individual chunks, both chunk_index and total_chunks should be provided
            # For aggregated chunks, chunk_index can be None (indicating aggregation)
            if self.chunk_index is not None and self.total_chunks is not None:
                # Validate chunk_index is within valid range
                if self.chunk_index >= self.total_chunks:
                    raise ValueError('chunk_index must be less than total_chunks')
            elif self.chunk_index is not None and self.total_chunks is None:
                raise ValueError('total_chunks is required when chunk_index is provided')
            # Note: chunk_index can be None for aggregated chunks, which is valid
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
        aggregate_chunks: bool = True,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        Handles both chunked and non-chunked results with optional result aggregation.
        
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.
        :param aggregate_chunks: Whether to aggregate chunks from the same source document.

        :return: A list of entries found, potentially aggregated by source document.
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

        # Use a higher limit for raw search to account for potential aggregation
        raw_limit = limit * 3 if aggregate_chunks else limit

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=raw_limit,
            query_filter=query_filter,
        )

        # Convert search results to Entry objects, handling both chunked and non-chunked content
        raw_entries = []
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
            # Store the search score for potential use in aggregation
            entry._search_score = result.score if hasattr(result, 'score') else 0.0
            raw_entries.append(entry)
        
        # Apply aggregation if requested and we have chunked results
        if aggregate_chunks and any(entry.is_chunk for entry in raw_entries):
            return self._aggregate_search_results(raw_entries, limit)
        
        return raw_entries[:limit]

    def _aggregate_search_results(self, entries: list[Entry], limit: int) -> list[Entry]:
        """
        Aggregate search results to group chunks from the same source document.
        
        :param entries: List of Entry objects from search results
        :param limit: Maximum number of results to return after aggregation
        :return: Aggregated list of entries
        """
        # Separate chunked and non-chunked entries
        non_chunked = [entry for entry in entries if not entry.is_chunk]
        chunked = [entry for entry in entries if entry.is_chunk]
        
        # Group chunks by source document
        chunks_by_document = {}
        for chunk in chunked:
            doc_id = chunk.source_document_id
            if doc_id not in chunks_by_document:
                chunks_by_document[doc_id] = []
            chunks_by_document[doc_id].append(chunk)
        
        # Create aggregated entries for each source document
        aggregated_chunks = []
        for doc_id, doc_chunks in chunks_by_document.items():
            # Sort chunks by index to maintain order
            doc_chunks.sort(key=lambda x: x.chunk_index or 0)
            
            # Find the best scoring chunk to use as the representative
            best_chunk = max(doc_chunks, key=lambda x: getattr(x, '_search_score', 0.0))
            
            # Create an aggregated entry with context from multiple chunks
            aggregated_content = self._create_aggregated_content(doc_chunks)
            
            aggregated_entry = Entry(
                content=aggregated_content,
                metadata=best_chunk.metadata,
                is_chunk=True,  # Mark as chunk to indicate this is from chunked content
                source_document_id=doc_id,
                chunk_index=None,  # No single chunk index for aggregated content
                total_chunks=best_chunk.total_chunks,
            )
            # Preserve the best search score
            aggregated_entry._search_score = getattr(best_chunk, '_search_score', 0.0)
            aggregated_entry._chunk_count = len(doc_chunks)  # Track how many chunks were aggregated
            aggregated_chunks.append(aggregated_entry)
        
        # Combine non-chunked and aggregated chunked results
        all_results = non_chunked + aggregated_chunks
        
        # Sort by search score if available
        all_results.sort(key=lambda x: getattr(x, '_search_score', 0.0), reverse=True)
        
        return all_results[:limit]

    def _create_aggregated_content(self, chunks: list[Entry]) -> str:
        """
        Create aggregated content from multiple chunks of the same document.
        
        :param chunks: List of chunks from the same source document
        :return: Aggregated content string
        """
        if len(chunks) == 1:
            return chunks[0].content
        
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index or 0)
        
        # If we have many chunks, limit to the most relevant ones
        if len(sorted_chunks) > 3:
            # Take the first chunk, the best scoring chunk, and the last chunk
            best_chunk = max(sorted_chunks, key=lambda x: getattr(x, '_search_score', 0.0))
            selected_chunks = [sorted_chunks[0]]
            if best_chunk not in selected_chunks:
                selected_chunks.append(best_chunk)
            if sorted_chunks[-1] not in selected_chunks:
                selected_chunks.append(sorted_chunks[-1])
            # Re-sort by index
            selected_chunks.sort(key=lambda x: x.chunk_index or 0)
        else:
            selected_chunks = sorted_chunks
        
        # Join chunks with clear separators
        content_parts = []
        for i, chunk in enumerate(selected_chunks):
            if i > 0:
                content_parts.append("...")  # Indicate potential gap
            content_parts.append(chunk.content.strip())
        
        return " ".join(content_parts)

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

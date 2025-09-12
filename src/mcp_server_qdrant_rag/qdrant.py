import logging
import uuid
from typing import Any

from pydantic import BaseModel, field_validator, model_validator
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant_rag.embeddings.base import EmbeddingProvider
from mcp_server_qdrant_rag.settings import METADATA_PATH
from mcp_server_qdrant_rag.chunking.chunker import DocumentChunker
from mcp_server_qdrant_rag.common.exceptions import (
    VectorDimensionMismatchError,
    ChunkingError,
    CollectionAccessError,
)

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None

    # New metadata fields for semantic filtering (optional for backward compatibility)
    document_type: str | None = None
    set_id: str | None = None

    # Chunk-related fields for document chunking support
    is_chunk: bool = False
    source_document_id: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v: str | None) -> str | None:
        """Validate that document_type is non-empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("document_type must be non-empty if provided")
        return v.strip() if v is not None else None

    @field_validator("set_id")
    @classmethod
    def validate_set_id(cls, v: str | None) -> str | None:
        """Validate that set_id is non-empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("set_id must be non-empty if provided")
        return v.strip() if v is not None else None

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, v: int | None) -> int | None:
        """Validate that chunk_index is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("chunk_index must be non-negative")
        return v

    @field_validator("total_chunks")
    @classmethod
    def validate_total_chunks(cls, v: int | None) -> int | None:
        """Validate that total_chunks is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("total_chunks must be positive")
        return v

    @model_validator(mode="after")
    def validate_chunk_metadata(self) -> "Entry":
        """Validate chunk metadata consistency."""
        if self.is_chunk:
            # If this is a chunk, source_document_id is required
            if self.source_document_id is None:
                raise ValueError("source_document_id is required when is_chunk is True")

            # For individual chunks, both chunk_index and total_chunks should be provided
            # For aggregated chunks, chunk_index can be None (indicating aggregation)
            if self.chunk_index is not None and self.total_chunks is not None:
                # Validate chunk_index is within valid range
                if self.chunk_index >= self.total_chunks:
                    raise ValueError("chunk_index must be less than total_chunks")
            elif self.chunk_index is not None and self.total_chunks is None:
                raise ValueError(
                    "total_chunks is required when chunk_index is provided"
                )
            # Note: chunk_index can be None for aggregated chunks, which is valid
        else:
            # If this is not a chunk, chunk-related fields should be None
            if any(
                [
                    self.source_document_id,
                    self.chunk_index is not None,
                    self.total_chunks is not None,
                ]
            ):
                raise ValueError(
                    "chunk-related fields should be None when is_chunk is False"
                )

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
        :raises CollectionAccessError: If unable to retrieve collection names.
        """
        try:
            response = await self._client.get_collections()
            collection_names = [collection.name for collection in response.collections]
            logger.debug(f"Retrieved {len(collection_names)} collections")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to get collection names: {e}")
            raise CollectionAccessError(
                collection_name="*",
                operation="list",
                original_error=e,
                available_collections=[],
            ) from e

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

        # Log chunking decision
        content_length = len(entry.content)
        if entry.is_chunk:
            logger.debug(
                f"Storing pre-chunked entry (chunk {entry.chunk_index + 1}/{entry.total_chunks}) "
                f"in collection '{collection_name}' - {content_length} chars"
            )
        elif should_chunk:
            logger.debug(
                f"Document will be chunked before storage in collection '{collection_name}' "
                f"- {content_length} chars"
            )
        else:
            if not self._enable_chunking:
                logger.debug(
                    f"Storing single entry in collection '{collection_name}' "
                    f"- {content_length} chars (chunking disabled)"
                )
            elif self._chunker is None:
                logger.debug(
                    f"Storing single entry in collection '{collection_name}' "
                    f"- {content_length} chars (no chunker available)"
                )
            else:
                logger.debug(
                    f"Storing single entry in collection '{collection_name}' "
                    f"- {content_length} chars (below chunk threshold)"
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

        # Prepare payload with basic metadata
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}

        # Add new metadata fields if provided
        if entry.document_type is not None:
            payload["document_type"] = entry.document_type
        if entry.set_id is not None:
            payload["set_id"] = entry.set_id

        # Add chunk-related metadata to payload if this is a chunk
        if entry.is_chunk:
            payload.update(
                {
                    "is_chunk": entry.is_chunk,
                    "source_document_id": entry.source_document_id,
                    "chunk_index": entry.chunk_index,
                    "total_chunks": entry.total_chunks,
                }
            )

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

        logger.debug(
            f"Starting chunked storage for document (source_id: {source_document_id[:8]}...)"
        )

        try:
            # Create chunks using the DocumentChunker
            document_chunks = await self._chunker.chunk_document(
                content=entry.content,
                metadata=entry.metadata,
                source_document_id=source_document_id,
            )

            if not document_chunks:
                # If chunking failed or produced no chunks, store as single entry
                logger.warning(
                    "Document chunking produced no chunks, storing as single entry"
                )
                await self._store_single_entry(entry, collection_name)
                return

            logger.debug(
                f"Chunking produced {len(document_chunks)} chunks, converting to entries"
            )

            # Convert DocumentChunk objects to Entry objects and store them
            chunk_entries = []
            for doc_chunk in document_chunks:
                chunk_entry = Entry(
                    content=doc_chunk.content,
                    metadata=doc_chunk.metadata,
                    document_type=entry.document_type,  # Propagate document_type to chunks
                    set_id=entry.set_id,  # Propagate set_id to chunks
                    is_chunk=True,
                    source_document_id=doc_chunk.source_document_id,
                    chunk_index=doc_chunk.chunk_index,
                    total_chunks=doc_chunk.total_chunks,
                )
                chunk_entries.append(chunk_entry)

            logger.debug(
                f"Storing {len(chunk_entries)} chunk entries to collection '{collection_name}'"
            )

            # Store all chunks
            for i, chunk_entry in enumerate(chunk_entries):
                await self._store_single_entry(chunk_entry, collection_name)
                logger.debug(
                    f"Stored chunk {i + 1}/{len(chunk_entries)} "
                    f"({len(chunk_entry.content)} chars)"
                )

            logger.info(
                f"Successfully stored document as {len(chunk_entries)} chunks in collection '{collection_name}' "
                f"(source_id: {source_document_id[:8]}...)"
            )

        except Exception as e:
            logger.error(
                f"Chunking failed for document (length: {len(entry.content)}): {e}"
            )

            # Create chunking error with fallback
            chunking_config = {
                "max_chunk_size": getattr(self._chunker, "max_tokens", "unknown"),
                "chunk_overlap": getattr(self._chunker, "overlap_tokens", "unknown"),
                "chunk_strategy": "hybrid",
            }

            chunking_error = ChunkingError(
                original_error=e,
                document_length=len(entry.content),
                chunking_config=chunking_config,
                fallback_used=True,
            )

            logger.warning(f"Chunking error: {chunking_error.message}")

            # Fallback to storing as single entry
            try:
                await self._store_single_entry(entry, collection_name)
                logger.info(
                    "Successfully stored document as single entry after chunking failure"
                )
            except Exception as fallback_error:
                logger.error(f"Fallback storage also failed: {fallback_error}")
                raise chunking_error from fallback_error

    def _should_chunk_document(self, content: str) -> bool:
        """
        Determine if a document should be chunked based on its size.

        :param content: The document content to evaluate
        :return: True if the document should be chunked, False otherwise
        """
        if not self._chunker:
            logger.debug("No chunker available, document will not be chunked")
            return False

        # Use the chunker's token counting method to determine if chunking is needed
        token_count = self._chunker._count_tokens(content)
        max_tokens = self._chunker.max_tokens

        should_chunk = token_count > max_tokens

        logger.debug(
            f"Chunking decision: {token_count} tokens vs {max_tokens} max "
            f"-> {'CHUNK' if should_chunk else 'NO CHUNK'}"
        )

        return should_chunk

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
        Provides seamless backward compatibility for mixed content types.

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

        # Validate collection compatibility before searching
        try:
            await self._validate_collection_dimensions(collection_name)
        except VectorDimensionMismatchError as e:
            logger.error(f"Collection compatibility issue during search: {e}")
            # For backward compatibility, we could potentially continue with a warning
            # but for now, we'll raise the error to maintain data integrity
            raise

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

            # Handle backward compatibility: older entries might not have chunk fields
            is_chunk = payload.get("is_chunk", False)
            source_document_id = payload.get("source_document_id")
            chunk_index = payload.get("chunk_index")
            total_chunks = payload.get("total_chunks")

            # Extract metadata with backward compatibility
            metadata = payload.get("metadata") or payload.get(METADATA_PATH)

            # Extract new metadata fields with backward compatibility
            document_type = payload.get("document_type")
            set_id = payload.get("set_id")

            entry = Entry(
                content=payload["document"],
                metadata=metadata,
                document_type=document_type,
                set_id=set_id,
                is_chunk=is_chunk,
                source_document_id=source_document_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            # Store the search score for potential use in aggregation
            entry._search_score = result.score if hasattr(result, "score") else 0.0
            raw_entries.append(entry)

        # Apply aggregation if requested and we have chunked results
        if aggregate_chunks and any(entry.is_chunk for entry in raw_entries):
            aggregated_results = self._aggregate_search_results(raw_entries, limit)
            logger.debug(
                f"Search returned {len(raw_entries)} raw results, aggregated to {len(aggregated_results)} results"
            )
            return aggregated_results

        logger.debug(
            f"Search returned {len(raw_entries)} results (no aggregation applied)"
        )
        return raw_entries[:limit]

    def _aggregate_search_results(
        self, entries: list[Entry], limit: int
    ) -> list[Entry]:
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
            best_chunk = max(doc_chunks, key=lambda x: getattr(x, "_search_score", 0.0))

            # Create an aggregated entry with context from multiple chunks
            aggregated_content = self._create_aggregated_content(doc_chunks)

            aggregated_entry = Entry(
                content=aggregated_content,
                metadata=best_chunk.metadata,
                document_type=best_chunk.document_type,  # Preserve document_type from chunks
                set_id=best_chunk.set_id,  # Preserve set_id from chunks
                is_chunk=True,  # Mark as chunk to indicate this is from chunked content
                source_document_id=doc_id,
                chunk_index=None,  # No single chunk index for aggregated content
                total_chunks=best_chunk.total_chunks,
            )
            # Preserve the best search score
            aggregated_entry._search_score = getattr(best_chunk, "_search_score", 0.0)
            aggregated_entry._chunk_count = len(
                doc_chunks
            )  # Track how many chunks were aggregated
            aggregated_chunks.append(aggregated_entry)

        # Combine non-chunked and aggregated chunked results
        all_results = non_chunked + aggregated_chunks

        # Sort by search score if available
        all_results.sort(key=lambda x: getattr(x, "_search_score", 0.0), reverse=True)

        return all_results[:limit]

    async def find_hybrid(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        fusion_method: str = "rrf",
        dense_limit: int = 20,
        sparse_limit: int = 20,
        final_limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Hybrid search combining dense and sparse vectors using Qdrant's Query API.

        :param query: The text query to search for.
        :param collection_name: The name of the collection to search in.
        :param fusion_method: Fusion method - "rrf" (Reciprocal Rank Fusion) or "dbsf" (Distribution-Based Score Fusion).
        :param dense_limit: Maximum results from dense vector search.
        :param sparse_limit: Maximum results from sparse vector search.
        :param final_limit: Maximum final results after fusion.
        :param query_filter: Optional filter to apply to the search.
        :return: A list of entries found, fused from both dense and sparse search.
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            return []

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Check if collection has both dense and sparse vectors
        # For now, we'll assume dense vector exists and fallback gracefully if sparse doesn't
        vector_name = self._embedding_provider.get_vector_name()

        try:
            # Build prefetch queries for hybrid search
            prefetch_queries = []

            # Dense vector search (semantic similarity)
            query_vector = await self._embedding_provider.embed_query(query)
            prefetch_queries.append(
                models.Prefetch(
                    query=query_vector,
                    using=vector_name,
                    limit=dense_limit,
                )
            )

            # Sparse vector search (keyword matching)
            # Note: This assumes sparse vectors are configured in the collection
            # In practice, you'd want to check collection config first
            try:
                prefetch_queries.append(
                    models.Prefetch(
                        query=models.Document(text=query, model="bm25"),
                        using="sparse",
                        limit=sparse_limit,
                    )
                )
            except Exception:
                # If sparse vectors aren't available, fallback to dense-only search
                logger.warning(
                    f"Sparse vectors not available in collection {collection_name}, using dense-only search"
                )
                return await self.search(
                    query,
                    collection_name=collection_name,
                    limit=final_limit,
                    query_filter=query_filter,
                )

            # Execute hybrid search with fusion
            fusion_type = (
                models.Fusion.RRF
                if fusion_method.lower() == "rrf"
                else models.Fusion.DBSF
            )

            search_results = await self._client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_queries,
                query=models.FusionQuery(fusion=fusion_type),
                limit=final_limit,
                query_filter=query_filter,
            )

            return [
                Entry(
                    content=result.payload["document"] if result.payload else "",
                    metadata=result.payload.get("metadata") if result.payload else None,
                    document_type=result.payload.get("document_type")
                    if result.payload
                    else None,
                    set_id=result.payload.get("set_id") if result.payload else None,
                )
                for result in search_results.points
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to regular dense search
            logger.info(f"Falling back to dense vector search for query: {query}")
            return await self.search(
                query,
                collection_name=collection_name,
                limit=final_limit,
                query_filter=query_filter,
            )

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
            best_chunk = max(
                sorted_chunks, key=lambda x: getattr(x, "_search_score", 0.0)
            )
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
        :raises VectorDimensionMismatchError: If collection has incompatible vector dimensions.
        """
        try:
            collection_info = await self._client.get_collection(collection_name)
            expected_vector_size = self._embedding_provider.get_vector_size()
            expected_vector_name = self._embedding_provider.get_vector_name()
            current_model = getattr(self._embedding_provider, "model_name", "unknown")

            available_vectors = list(collection_info.config.params.vectors.keys())

            # Check if the expected vector name exists in the collection
            if expected_vector_name not in available_vectors:
                raise VectorDimensionMismatchError(
                    collection_name=collection_name,
                    expected_dimensions=expected_vector_size,
                    actual_dimensions=0,  # No matching vector found
                    model_name=current_model,
                    vector_name=expected_vector_name,
                    available_vectors=available_vectors,
                )

            # Check vector dimensions
            actual_vector_size = collection_info.config.params.vectors[
                expected_vector_name
            ].size
            if actual_vector_size != expected_vector_size:
                raise VectorDimensionMismatchError(
                    collection_name=collection_name,
                    expected_dimensions=expected_vector_size,
                    actual_dimensions=actual_vector_size,
                    model_name=current_model,
                    vector_name=expected_vector_name,
                    available_vectors=available_vectors,
                )

            logger.debug(
                f"Collection '{collection_name}' dimensions validated successfully"
            )

        except VectorDimensionMismatchError:
            raise  # Re-raise our custom validation errors
        except Exception as e:
            # Log other errors but don't fail - collection might be valid
            logger.warning(
                f"Could not validate collection dimensions for '{collection_name}': {e}"
            )
            # Don't raise here as the collection might still be usable

    def get_embedding_model_info(self) -> dict[str, Any]:
        """
        Get information about the current embedding model.
        :return: Dictionary containing model information.
        """
        return {
            "model_name": getattr(self._embedding_provider, "model_name", "unknown"),
            "vector_size": self._embedding_provider.get_vector_size(),
            "vector_name": self._embedding_provider.get_vector_name(),
        }

    async def delete_collection(self, collection_name: str | None = None) -> bool:
        """
        Delete a collection from Qdrant.

        :param collection_name: The name of the collection to delete, optional. If not provided,
                                the default collection is used.
        :return: True if collection was deleted, False if it didn't exist
        :raises CollectionAccessError: If deletion fails due to server errors
        """
        collection_name = collection_name or self._default_collection_name
        if not collection_name:
            raise ValueError("Collection name is required for deletion")

        try:
            collection_exists = await self._client.collection_exists(collection_name)
            if not collection_exists:
                logger.debug(
                    f"Collection '{collection_name}' does not exist, nothing to delete"
                )
                return False

            await self._client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise CollectionAccessError(
                collection_name=collection_name,
                operation="delete",
                original_error=e,
                available_collections=await self.get_collection_names(),
            ) from e

    async def clear_collection(self, collection_name: str | None = None) -> int:
        """
        Clear all points from a collection while keeping the collection structure.

        :param collection_name: The name of the collection to clear, optional. If not provided,
                                the default collection is used.
        :return: Number of points that were deleted
        :raises CollectionAccessError: If clearing fails due to server errors
        """
        collection_name = collection_name or self._default_collection_name
        if not collection_name:
            raise ValueError("Collection name is required for clearing")

        try:
            collection_exists = await self._client.collection_exists(collection_name)
            if not collection_exists:
                logger.debug(
                    f"Collection '{collection_name}' does not exist, nothing to clear"
                )
                return 0

            # Get collection info to know how many points we're deleting
            collection_info = await self._client.get_collection(collection_name)
            points_count = collection_info.points_count

            if points_count == 0:
                logger.debug(f"Collection '{collection_name}' is already empty")
                return 0

            # Delete all points by recreating the collection
            # This is more efficient than deleting points individually
            vectors_config = collection_info.config.params.vectors

            # Delete and recreate the collection
            await self._client.delete_collection(collection_name)
            await self._client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
            )

            # Recreate payload indexes if they existed
            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

            logger.info(
                f"Successfully cleared {points_count} points from collection '{collection_name}'"
            )
            return points_count

        except Exception as e:
            logger.error(f"Failed to clear collection '{collection_name}': {e}")
            raise CollectionAccessError(
                collection_name=collection_name,
                operation="clear",
                original_error=e,
                available_collections=await self.get_collection_names(),
            ) from e

    async def analyze_collection_compatibility(
        self, collection_name: str | None = None
    ) -> dict[str, Any]:
        """
        Analyze collection compatibility for backward compatibility assessment.

        :param collection_name: The name of the collection to analyze, optional. If not provided,
                                the default collection is used.
        :return: Dictionary containing compatibility analysis results.
        """
        collection_name = collection_name or self._default_collection_name
        if not collection_name:
            return {
                "error": "No collection name provided and no default collection configured",
                "compatible": False,
            }

        try:
            collection_exists = await self._client.collection_exists(collection_name)
            if not collection_exists:
                return {
                    "collection_name": collection_name,
                    "exists": False,
                    "compatible": True,
                    "message": "Collection does not exist - will be created with current configuration",
                }

            collection_info = await self._client.get_collection(collection_name)
            expected_vector_size = self._embedding_provider.get_vector_size()
            expected_vector_name = self._embedding_provider.get_vector_name()
            current_model = getattr(self._embedding_provider, "model_name", "unknown")

            available_vectors = list(collection_info.config.params.vectors.keys())

            # Check for chunked vs non-chunked content
            points_count = collection_info.points_count
            has_chunked_content = False
            has_non_chunked_content = False

            if points_count > 0:
                # Sample a few points to check for chunked content
                sample_results = await self._client.scroll(
                    collection_name=collection_name,
                    limit=min(10, points_count),
                    with_payload=True,
                )

                points, _ = sample_results  # scroll returns (points, next_page_offset)
                for point in points:
                    if point.payload and point.payload.get("is_chunk", False):
                        has_chunked_content = True
                    else:
                        has_non_chunked_content = True

            # Vector compatibility analysis
            vector_compatible = expected_vector_name in available_vectors
            dimension_compatible = False
            actual_vector_size = 0

            if vector_compatible:
                actual_vector_size = collection_info.config.params.vectors[
                    expected_vector_name
                ].size
                dimension_compatible = actual_vector_size == expected_vector_size

            compatibility_result = {
                "collection_name": collection_name,
                "exists": True,
                "points_count": points_count,
                "current_model": current_model,
                "expected_vector_name": expected_vector_name,
                "expected_dimensions": expected_vector_size,
                "available_vectors": available_vectors,
                "vector_compatible": vector_compatible,
                "dimension_compatible": dimension_compatible,
                "actual_dimensions": actual_vector_size,
                "has_chunked_content": has_chunked_content,
                "has_non_chunked_content": has_non_chunked_content,
                "mixed_content": has_chunked_content and has_non_chunked_content,
                "compatible": vector_compatible and dimension_compatible,
                "chunking_enabled": self._enable_chunking,
            }

            # Add recommendations
            recommendations = []
            if not vector_compatible:
                recommendations.append(
                    f"Vector name mismatch. Collection uses {available_vectors}, but model expects '{expected_vector_name}'"
                )
                recommendations.append(
                    "Consider using a different collection name or switching to a compatible model"
                )
            elif not dimension_compatible:
                recommendations.append(
                    f"Dimension mismatch. Collection has {actual_vector_size} dimensions, but model produces {expected_vector_size}"
                )
                recommendations.append(
                    "Consider using a different collection name or switching to a compatible model"
                )

            if has_chunked_content and not self._enable_chunking:
                recommendations.append(
                    "Collection contains chunked content but chunking is disabled"
                )
                recommendations.append(
                    "Enable chunking or use a different collection for consistency"
                )
            elif not has_chunked_content and self._enable_chunking and points_count > 0:
                recommendations.append(
                    "Collection contains only non-chunked content but chunking is enabled"
                )
                recommendations.append(
                    "New large documents will be chunked while existing content remains unchanged"
                )

            if compatibility_result["mixed_content"]:
                recommendations.append(
                    "Collection contains both chunked and non-chunked content"
                )
                recommendations.append(
                    "Search results will include both types - this is normal and supported"
                )

            compatibility_result["recommendations"] = recommendations

            return compatibility_result

        except Exception as e:
            logger.error(f"Failed to analyze collection compatibility: {e}")
            return {
                "collection_name": collection_name,
                "error": str(e),
                "compatible": False,
                "recommendations": [
                    "Check Qdrant server connectivity",
                    "Verify collection permissions",
                    "Check if collection name is correct",
                ],
            }

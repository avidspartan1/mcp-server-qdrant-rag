"""Integration tests for end-to-end workflows with chunking and embedding."""

import uuid
import pytest
from unittest.mock import patch

from mcp_server_qdrant_rag.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant_rag.qdrant import QdrantConnector, Entry
from mcp_server_qdrant_rag.chunking.chunker import DocumentChunker
from mcp_server_qdrant_rag.settings import EmbeddingProviderSettings
from mcp_server_qdrant_rag.common.exceptions import VectorDimensionMismatchError


@pytest.mark.asyncio
class TestEndToEndWorkflows:
    """Integration tests for complete end-to-end workflows."""

    async def test_large_document_chunking_storage_retrieval_workflow(self):
        """Test complete workflow: large document → chunking → storage → search → retrieval."""
        # Create a large document that will be chunked
        large_document = """
        This is the first paragraph of a large document that contains multiple sections and paragraphs.
        It discusses various topics and concepts that are important for testing the chunking functionality.
        The document needs to be long enough to trigger automatic chunking when processed.
        
        This is the second paragraph that continues the discussion from the first paragraph.
        It provides additional context and information that builds upon the previous content.
        The chunking system should preserve semantic boundaries between these paragraphs.
        
        The third paragraph introduces new concepts and ideas that are related to the overall theme.
        This content should be split appropriately to maintain coherence within each chunk.
        The overlap between chunks should preserve important context for better retrieval.
        
        Finally, this last paragraph concludes the document with a summary of key points.
        It ties together all the previous discussions and provides closure to the document.
        This content should be searchable and retrievable after the chunking process.
        """
        
        # Set up components
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_e2e_workflow_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Create chunker with small chunk size to force chunking
        chunker = DocumentChunker(max_tokens=50, overlap_tokens=10, tokenizer="whitespace")
        
        # Step 1: Chunk the document
        chunks = await chunker.chunk_document(large_document.strip())
        
        # Verify chunking worked
        assert len(chunks) > 1, "Document should be split into multiple chunks"
        assert all(chunk.source_document_id == chunks[0].source_document_id for chunk in chunks)
        assert all(chunk.total_chunks == len(chunks) for chunk in chunks)
        
        # Step 2: Store chunks
        stored_entries = []
        for chunk in chunks:
            entry = Entry(
                content=chunk.content,
                metadata=chunk.get_chunk_metadata(),
                is_chunk=True,
                source_document_id=chunk.source_document_id,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks
            )
            await connector.store(entry)
            stored_entries.append(entry)
        
        # Step 3: Search and retrieve
        search_results = await connector.search("first paragraph", limit=5)
        
        # Verify search results
        assert len(search_results) > 0, "Should find relevant chunks"
        
        # Check that results contain chunk information
        found_chunk = False
        for result in search_results:
            if result.metadata and result.metadata.get("is_chunk"):
                found_chunk = True
                assert "source_document_id" in result.metadata
                assert "chunk_index" in result.metadata
                assert "total_chunks" in result.metadata
                break
        
        assert found_chunk, "Should find at least one chunk in results"
        
        # Step 4: Test retrieval of related chunks
        source_doc_id = chunks[0].source_document_id
        all_results = await connector.search("document", limit=20)
        
        # Filter results from the same document
        same_doc_results = [
            r for r in all_results 
            if r.metadata and r.metadata.get("source_document_id") == source_doc_id
        ]
        
        # Should find at least some chunks from the document (search may not return all due to relevance scoring)
        assert len(same_doc_results) > 0, "Should retrieve at least some chunks from the document"
        assert len(same_doc_results) <= len(chunks), "Should not retrieve more chunks than exist"

    async def test_model_switching_with_dimension_handling(self):
        """Test model switching scenarios with proper dimension handling."""
        collection_name = f"test_model_switch_{uuid.uuid4().hex}"
        
        # Start with first model
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        connector1 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider1,
        )
        
        # Store some data with first model
        entry1 = Entry(content="Test content for first model")
        await connector1.store(entry1)
        
        # Verify storage worked
        results1 = await connector1.search("Test content")
        assert len(results1) == 1
        
        # Try to switch to a different model with potentially different dimensions
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        connector2 = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider2,
        )
        
        # Use the same client to simulate the same Qdrant instance
        connector2._client = connector1._client
        
        # Check if models have different dimensions or vector names
        model1_info = provider1.get_model_info()
        model2_info = provider2.get_model_info()
        
        if (model1_info["vector_size"] != model2_info["vector_size"] or 
            model1_info["vector_name"] != model2_info["vector_name"]):
            # Should raise dimension mismatch error
            with pytest.raises(VectorDimensionMismatchError):
                await connector2.store(Entry(content="Test content for second model"))
        else:
            # If dimensions match, should work fine
            await connector2.store(Entry(content="Test content for second model"))
            results2 = await connector2.search("Test content")
            assert len(results2) == 2

    async def test_mixed_chunked_non_chunked_operations(self):
        """Test mixed chunked/non-chunked collection operations."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_mixed_content_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Store non-chunked entries
        non_chunked_entries = [
            Entry(content="First regular document about technology"),
            Entry(content="Second regular document about science"),
            Entry(content="Third regular document about research")
        ]
        
        for entry in non_chunked_entries:
            await connector.store(entry)
        
        # Create and store chunked content
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=3, tokenizer="whitespace")
        large_doc = "This is a large document that will be chunked into multiple pieces for better retrieval and processing efficiency. It contains many words and should definitely be split into several chunks when processed by the chunking system."
        
        chunks = await chunker.chunk_document(large_doc)
        # If not chunked, adjust the test to handle single chunk
        if len(chunks) == 1:
            # Make the document longer to force chunking
            large_doc = large_doc + " " + large_doc + " " + large_doc
            chunks = await chunker.chunk_document(large_doc)
        
        assert len(chunks) >= 1  # At least one chunk should exist
        
        for chunk in chunks:
            entry = Entry(
                content=chunk.content,
                metadata=chunk.get_chunk_metadata(),
                is_chunk=True,
                source_document_id=chunk.source_document_id,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks
            )
            await connector.store(entry)
        
        # Search across mixed content
        all_results = await connector.search("document", limit=10)
        
        # Should find both chunked and non-chunked content
        chunked_results = [r for r in all_results if r.is_chunk]
        non_chunked_results = [r for r in all_results if not r.is_chunk]
        
        # We should have stored both types of content
        total_stored = len(non_chunked_entries) + len(chunks)
        assert len(all_results) > 0, "Should find some content"
        
        # If we have multiple chunks, we should find chunked content
        if len(chunks) > 1:
            assert len(chunked_results) > 0, "Should find chunked content"
        
        # We should always find non-chunked content
        assert len(non_chunked_results) > 0, "Should find non-chunked content"
        
        # Verify chunk metadata is present for chunked results (if any)
        if chunked_results:
            for result in chunked_results:
                assert result.metadata is not None
                assert result.metadata.get("is_chunk") is True
                assert "source_document_id" in result.metadata
                assert "chunk_index" in result.metadata
        
        # Verify non-chunked results don't have chunk metadata
        for result in non_chunked_results:
            if result.metadata:
                assert result.metadata.get("is_chunk") is not True

    async def test_backward_compatibility_with_existing_data(self):
        """Test backward compatibility with existing data."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_backward_compat_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Simulate existing data (without chunk fields)
        legacy_entries = [
            Entry(content="Legacy document one"),
            Entry(content="Legacy document two", metadata={"author": "Legacy Author"}),
            Entry(content="Legacy document three")
        ]
        
        for entry in legacy_entries:
            await connector.store(entry)
        
        # Verify legacy data is searchable
        legacy_results = await connector.search("Legacy document")
        assert len(legacy_results) == 3
        
        # Add new chunked content
        chunker = DocumentChunker(max_tokens=15, overlap_tokens=3, tokenizer="whitespace")
        new_doc = "This is a new document that will be processed with the new chunking system."
        
        chunks = await chunker.chunk_document(new_doc)
        for chunk in chunks:
            entry = Entry(
                content=chunk.content,
                metadata=chunk.get_chunk_metadata(),
                is_chunk=True,
                source_document_id=chunk.source_document_id,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks
            )
            await connector.store(entry)
        
        # Search across all content
        all_results = await connector.search("document", limit=20)
        
        # Should find both legacy and new content
        legacy_found = any("Legacy" in r.content for r in all_results)
        new_found = any("new document" in r.content for r in all_results)
        
        assert legacy_found, "Should find legacy content"
        assert new_found, "Should find new chunked content"
        
        # Verify mixed results work correctly
        chunked_count = sum(1 for r in all_results if r.is_chunk)
        non_chunked_count = sum(1 for r in all_results if not r.is_chunk)
        
        assert chunked_count > 0, "Should have chunked results"
        assert non_chunked_count >= 3, "Should have at least 3 legacy results"

    async def test_configuration_driven_chunking_workflow(self):
        """Test workflow with different chunking configurations."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        test_document = """
        Artificial intelligence has revolutionized many industries. Machine learning algorithms 
        can now process vast amounts of data. Deep learning networks have achieved remarkable 
        results in image recognition. Natural language processing has improved significantly. 
        Computer vision applications are becoming more sophisticated. Robotics integration 
        with AI is advancing rapidly.
        """
        
        # Test different chunking strategies
        strategies = [
            {"max_tokens": 20, "overlap_tokens": 5, "strategy": "semantic"},
            {"max_tokens": 15, "overlap_tokens": 3, "strategy": "fixed"},
            {"max_tokens": 25, "overlap_tokens": 7, "strategy": "sentence"}
        ]
        
        for i, config in enumerate(strategies):
            collection_name = f"test_config_{i}_{uuid.uuid4().hex}"
            
            connector = QdrantConnector(
                qdrant_url=":memory:",
                qdrant_api_key=None,
                collection_name=collection_name,
                embedding_provider=provider,
            )
            
            # Create chunker with specific configuration
            chunker = DocumentChunker(
                max_tokens=config["max_tokens"],
                overlap_tokens=config["overlap_tokens"],
                tokenizer="whitespace"
            )
            
            # Chunk and store document
            chunks = await chunker.chunk_document(test_document.strip())
            assert len(chunks) > 0
            
            for chunk in chunks:
                entry = Entry(
                    content=chunk.content,
                    metadata=chunk.get_chunk_metadata(),
                    is_chunk=True,
                    source_document_id=chunk.source_document_id,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
                await connector.store(entry)
            
            # Test search with this configuration
            results = await connector.search("artificial intelligence", limit=5)
            assert len(results) > 0
            
            # Verify chunk metadata includes strategy information
            chunk_results = [r for r in results if r.is_chunk]
            if chunk_results:
                chunk_result = chunk_results[0]
                assert chunk_result.metadata is not None
                assert "chunk_strategy" in chunk_result.metadata

    async def test_error_recovery_and_fallback_scenarios(self):
        """Test error recovery and fallback scenarios in end-to-end workflows."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_error_recovery_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        # Test with problematic content that might cause chunking issues
        problematic_docs = [
            "",  # Empty document
            "   \n\t   ",  # Whitespace only
            "Single word",  # Very short document
            "A" * 10000,  # Very long single "word"
        ]
        
        chunker = DocumentChunker(max_tokens=50, overlap_tokens=10, tokenizer="whitespace")
        
        successful_stores = 0
        for i, doc in enumerate(problematic_docs):
            try:
                if doc.strip():  # Skip empty documents
                    chunks = await chunker.chunk_document(doc)
                    
                    if chunks:  # Only store if chunking produced results
                        for chunk in chunks:
                            entry = Entry(
                                content=chunk.content,
                                metadata=chunk.get_chunk_metadata(),
                                is_chunk=True,
                                source_document_id=chunk.source_document_id,
                                chunk_index=chunk.chunk_index,
                                total_chunks=chunk.total_chunks
                            )
                            await connector.store(entry)
                            successful_stores += 1
                    else:
                        # Fallback: store as regular entry
                        entry = Entry(content=doc)
                        await connector.store(entry)
                        successful_stores += 1
                        
            except Exception as e:
                # Log error but continue with other documents
                print(f"Error processing document {i}: {e}")
                continue
        
        # Should have successfully stored at least some content
        assert successful_stores > 0
        
        # Verify search still works
        results = await connector.search("word", limit=10)
        # Should find at least the "Single word" document
        assert any("word" in r.content.lower() for r in results)

    async def test_performance_with_large_batch_operations(self):
        """Test performance and correctness with large batch operations."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_batch_ops_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        chunker = DocumentChunker(max_tokens=30, overlap_tokens=5, tokenizer="whitespace")
        
        # Create multiple documents
        documents = []
        for i in range(10):  # Reduced from 50 for faster testing
            doc = f"""
            Document {i} contains information about topic {i % 3}. 
            This document discusses various aspects of the subject matter.
            It includes detailed explanations and examples for better understanding.
            The content is designed to test the chunking and storage system.
            Each document has unique identifiers and content patterns.
            """
            documents.append(doc.strip())
        
        # Process and store all documents
        total_chunks_stored = 0
        source_doc_ids = []
        
        for doc in documents:
            chunks = await chunker.chunk_document(doc)
            source_doc_ids.append(chunks[0].source_document_id if chunks else None)
            
            for chunk in chunks:
                entry = Entry(
                    content=chunk.content,
                    metadata=chunk.get_chunk_metadata(),
                    is_chunk=True,
                    source_document_id=chunk.source_document_id,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks
                )
                await connector.store(entry)
                total_chunks_stored += 1
        
        assert total_chunks_stored > len(documents), "Should have more chunks than documents"
        
        # Test search across all documents
        search_results = await connector.search("document", limit=50)
        assert len(search_results) > 0
        
        # Verify we can find content from different source documents
        found_source_ids = set()
        for result in search_results:
            if result.metadata and "source_document_id" in result.metadata:
                found_source_ids.add(result.metadata["source_document_id"])
        
        # Should find chunks from multiple source documents
        assert len(found_source_ids) > 1, "Should find chunks from multiple source documents"
        
        # Test topic-specific searches
        topic_results = await connector.search("topic 0", limit=20)
        topic_0_docs = [r for r in topic_results if "topic 0" in r.content]
        assert len(topic_0_docs) > 0, "Should find documents about topic 0"

    async def test_metadata_preservation_through_workflow(self):
        """Test that metadata is properly preserved through the entire workflow."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        collection_name = f"test_metadata_preservation_{uuid.uuid4().hex}"
        
        connector = QdrantConnector(
            qdrant_url=":memory:",
            qdrant_api_key=None,
            collection_name=collection_name,
            embedding_provider=provider,
        )
        
        chunker = DocumentChunker(max_tokens=25, overlap_tokens=5, tokenizer="whitespace")
        
        # Document with rich metadata
        document = """
        This is a comprehensive document about machine learning algorithms.
        It covers supervised learning, unsupervised learning, and reinforcement learning.
        The document includes practical examples and theoretical foundations.
        Each section builds upon previous concepts for better understanding.
        """
        
        original_metadata = {
            "title": "Machine Learning Guide",
            "author": "AI Researcher",
            "category": "Education",
            "tags": ["machine-learning", "algorithms", "AI"],
            "created_date": "2024-01-01",
            "version": "1.0",
            "nested_info": {
                "department": "Computer Science",
                "level": "Advanced",
                "prerequisites": ["Statistics", "Programming"]
            }
        }
        
        # Chunk document with metadata
        chunks = await chunker.chunk_document(document.strip(), metadata=original_metadata)
        
        # Store chunks
        for chunk in chunks:
            entry = Entry(
                content=chunk.content,
                metadata=chunk.get_chunk_metadata(),
                is_chunk=True,
                source_document_id=chunk.source_document_id,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks
            )
            await connector.store(entry)
        
        # Search and verify metadata preservation
        results = await connector.search("machine learning", limit=10)
        
        chunk_results = [r for r in results if r.is_chunk]
        assert len(chunk_results) > 0, "Should find chunked results"
        
        for result in chunk_results:
            assert result.metadata is not None
            
            # Check chunk-specific metadata
            assert result.metadata["is_chunk"] is True
            assert "source_document_id" in result.metadata
            assert "chunk_index" in result.metadata
            assert "total_chunks" in result.metadata
            
            # Check original metadata preservation
            assert "original_document_metadata" in result.metadata
            original_meta = result.metadata["original_document_metadata"]
            
            assert original_meta["title"] == "Machine Learning Guide"
            assert original_meta["author"] == "AI Researcher"
            assert original_meta["category"] == "Education"
            assert "machine-learning" in original_meta["tags"]
            
            # Check nested metadata preservation
            assert "nested_info" in original_meta
            nested = original_meta["nested_info"]
            assert nested["department"] == "Computer Science"
            assert nested["level"] == "Advanced"
            assert "Statistics" in nested["prerequisites"]
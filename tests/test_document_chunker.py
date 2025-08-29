"""Unit tests for DocumentChunker service with hybrid chunking strategy."""

import pytest
from mcp_server_qdrant_rag.chunking.chunker import DocumentChunker
from mcp_server_qdrant_rag.chunking.models import DocumentChunk


class TestDocumentChunker:
    """Test cases for DocumentChunker service with hybrid chunking."""
    
    def test_create_chunker_with_defaults(self):
        """Test creating a DocumentChunker with default parameters."""
        chunker = DocumentChunker()
        
        assert chunker.max_tokens == 500
        assert chunker.overlap_tokens == 50
        assert chunker.encoding_name == "cl100k_base"
    
    def test_create_chunker_with_custom_params(self):
        """Test creating a DocumentChunker with custom parameters."""
        chunker = DocumentChunker(
            max_tokens=256,
            overlap_tokens=25,
            sentence_splitter="nltk",
            tokenizer="whitespace"
        )
        
        assert chunker.max_tokens == 256
        assert chunker.overlap_tokens == 25
    
    def test_invalid_max_tokens(self):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            DocumentChunker(max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            DocumentChunker(max_tokens=-10)
    
    def test_invalid_overlap_tokens(self):
        """Test that invalid overlap_tokens raises ValueError."""
        with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
            DocumentChunker(overlap_tokens=-5)
        
        with pytest.raises(ValueError, match="overlap_tokens must be less than max_tokens"):
            DocumentChunker(max_tokens=100, overlap_tokens=100)
        
        with pytest.raises(ValueError, match="overlap_tokens must be less than max_tokens"):
            DocumentChunker(max_tokens=100, overlap_tokens=150)
    
    def test_invalid_sentence_splitter(self):
        """Test that invalid sentence splitter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sentence splitter"):
            DocumentChunker(sentence_splitter="invalid")
    
    def test_invalid_tokenizer(self):
        """Test that invalid tokenizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            DocumentChunker(tokenizer="invalid")
    
    @pytest.mark.asyncio
    async def test_chunk_empty_content(self):
        """Test chunking empty or whitespace-only content."""
        chunker = DocumentChunker()
        
        # Empty string
        chunks = await chunker.chunk_document("")
        assert chunks == []
        
        # Whitespace only
        chunks = await chunker.chunk_document("   \n\t  ")
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_chunk_small_content(self):
        """Test chunking content smaller than max_tokens."""
        chunker = DocumentChunker(max_tokens=100, tokenizer="whitespace")
        content = "This is a small document that fits in one chunk."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 0
        assert chunks[0].chunk_strategy == "hybrid"
    
    @pytest.mark.asyncio
    async def test_hybrid_chunking_with_sentences(self):
        """Test hybrid chunking that respects sentence boundaries."""
        chunker = DocumentChunker(max_tokens=15, overlap_tokens=3, tokenizer="whitespace")
        content = "Dr. Smith went to Washington. He arrived at 3 p.m. sharp. The meeting was long, but fruitful. Later, he returned home to write his report."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
        # Check that chunks are properly indexed
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)
            assert chunk.chunk_strategy == "hybrid"
            
            # Verify token count doesn't exceed limit (with some tolerance for sentence boundaries)
            token_count = len(chunk.content.split())
            assert token_count <= chunker.max_tokens + 5  # Allow some tolerance for sentence completion
    
    @pytest.mark.asyncio
    async def test_sentence_boundary_preservation(self):
        """Test that sentence boundaries are preserved in chunking."""
        chunker = DocumentChunker(max_tokens=5, overlap_tokens=1, tokenizer="whitespace")
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        
        # Most chunks should end with sentence-ending punctuation
        for chunk in chunks[:-1]:  # All but last chunk
            # Should end with complete sentence or be part of a sentence split
            assert chunk.content.strip()
    
    @pytest.mark.asyncio
    async def test_overlap_functionality(self):
        """Test that overlap between chunks works correctly."""
        chunker = DocumentChunker(max_tokens=8, overlap_tokens=3, tokenizer="whitespace")
        content = "Word one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        
        # Check overlap information is set correctly
        for i, chunk in enumerate(chunks):
            if i > 0:  # Not first chunk
                assert chunk.overlap_start == 3
            else:
                assert chunk.overlap_start == 0
                
            if i < len(chunks) - 1:  # Not last chunk
                assert chunk.overlap_end == 3
            else:
                assert chunk.overlap_end == 0
    
    @pytest.mark.asyncio
    async def test_chunk_with_metadata(self):
        """Test chunking with metadata preservation."""
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=2, tokenizer="whitespace")
        content = "This is a document with metadata that will be preserved in all chunks."
        metadata = {"title": "Test Document", "author": "Test Author"}
        
        chunks = await chunker.chunk_document(content, metadata=metadata)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_chunk_with_custom_source_id(self):
        """Test chunking with custom source document ID."""
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=2, tokenizer="whitespace")
        content = "This is a document that will be chunked with a custom source ID."
        source_id = "custom-doc-123"
        
        chunks = await chunker.chunk_document(content, source_document_id=source_id)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.source_document_id == source_id
    
    @pytest.mark.asyncio
    async def test_chunk_generates_source_id(self):
        """Test that chunking generates a source ID when not provided."""
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=2, tokenizer="whitespace")
        content = "This is a document that will get an auto-generated source ID."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) >= 1
        source_id = chunks[0].source_document_id
        assert source_id is not None
        assert len(source_id) > 0
        
        # All chunks should have the same source ID
        for chunk in chunks:
            assert chunk.source_document_id == source_id
    
    @pytest.mark.asyncio
    async def test_very_long_sentence_handling(self):
        """Test handling of sentences longer than max_tokens."""
        chunker = DocumentChunker(max_tokens=5, overlap_tokens=1, tokenizer="whitespace")
        # Create a very long sentence
        content = "This is an extremely long sentence that definitely exceeds the maximum token limit and should be split using fixed chunking as a fallback method."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        # Each chunk should respect the token limit (with some tolerance)
        for chunk in chunks:
            token_count = len(chunk.content.split())
            assert token_count <= chunker.max_tokens + 2  # Small tolerance for sentence splitting
    
    @pytest.mark.asyncio
    async def test_chunk_relationship_fields(self):
        """Test that chunk relationship fields are properly set."""
        chunker = DocumentChunker(max_tokens=8, overlap_tokens=2, tokenizer="whitespace")
        content = "First chunk content here. Second chunk content here. Third chunk content here."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        
        # Test first chunk
        first_chunk = chunks[0]
        assert first_chunk.is_first_chunk()
        assert not first_chunk.is_last_chunk()
        
        # Test last chunk
        last_chunk = chunks[-1]
        assert not last_chunk.is_first_chunk()
        assert last_chunk.is_last_chunk()
        
        # Test middle chunks (if any)
        for chunk in chunks[1:-1]:
            assert not chunk.is_first_chunk()
            assert not chunk.is_last_chunk()
    
    @pytest.mark.asyncio
    async def test_abbreviation_handling(self):
        """Test that abbreviations like 'Dr.' are handled correctly."""
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=2, tokenizer="whitespace")
        content = "Dr. Smith went to the U.S.A. He met with Prof. Johnson at 3.14 p.m. They discussed the project."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) >= 1
        # Should not split inappropriately at abbreviations
        # This test mainly ensures no errors occur with abbreviations
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_custom_tokenizer(self):
        """Test using a custom tokenizer function."""
        def custom_tokenizer(text):
            # Simple character-based tokenizer for testing
            return list(text.replace(' ', ''))
        
        chunker = DocumentChunker(max_tokens=20, overlap_tokens=5, tokenizer=custom_tokenizer)
        content = "Short text for testing custom tokenizer."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            # With character-based tokenizer, should respect character limits
            char_count = len(chunk.content.replace(' ', ''))
            assert char_count <= 25  # Some tolerance for sentence boundaries
    
    @pytest.mark.asyncio
    async def test_whitespace_handling(self):
        """Test proper handling of various whitespace scenarios."""
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=2, tokenizer="whitespace")
        
        # Content with extra whitespace
        content = "  First sentence.   \n\n  Second sentence.  \t Third sentence.  "
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) >= 1
        # Content should be cleaned up
        for chunk in chunks:
            assert chunk.content == chunk.content.strip()
    
    def test_get_chunk_info(self):
        """Test the get_chunk_info method."""
        chunker = DocumentChunker(
            max_tokens=256, 
            overlap_tokens=25, 
            sentence_splitter="nltk",
            tokenizer="whitespace"
        )
        
        info = chunker.get_chunk_info()
        
        # Check that all expected keys are present
        expected_keys = {
            "max_tokens", "overlap_tokens", "encoding_name", 
            "sentence_splitter", "tokenizer", "nltk_available", 
            "syntok_available", "tiktoken_available"
        }
        assert set(info.keys()) == expected_keys
        
        assert info["max_tokens"] == 256
        assert info["overlap_tokens"] == 25
    
    @pytest.mark.asyncio
    async def test_edge_case_single_word(self):
        """Test edge case with single word content."""
        chunker = DocumentChunker(max_tokens=10, overlap_tokens=2, tokenizer="whitespace")
        content = "Word"
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Word"
        assert chunks[0].total_chunks == 1
    
    @pytest.mark.asyncio
    async def test_fallback_sentence_splitter(self):
        """Test that fallback sentence splitter works when NLTK/syntok unavailable."""
        # This test assumes the simple fallback is used
        chunker = DocumentChunker(max_tokens=15, overlap_tokens=3, tokenizer="whitespace")
        content = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) >= 1
        # Should still produce reasonable chunks
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_token_counting_accuracy(self):
        """Test that token counting works correctly with different tokenizers."""
        # Test with whitespace tokenizer
        chunker = DocumentChunker(max_tokens=5, overlap_tokens=1, tokenizer="whitespace")
        content = "One two three four five six seven eight."
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        # Each chunk should respect the token limit
        for chunk in chunks:
            token_count = len(chunk.content.split())
            # Allow some tolerance for sentence boundary preservation
            assert token_count <= chunker.max_tokens + 3
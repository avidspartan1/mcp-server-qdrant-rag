"""Tests for comprehensive error handling and validation."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pydantic import ValidationError

from mcp_server_qdrant.common.exceptions import (
    ModelValidationError,
    VectorDimensionMismatchError,
    ChunkingError,
    ConfigurationValidationError,
    CollectionAccessError,
    TokenizerError,
    SentenceSplitterError
)
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import QdrantConnector, Entry
from mcp_server_qdrant.chunking.chunker import DocumentChunker
from mcp_server_qdrant.settings import EmbeddingProviderSettings


class TestModelValidationError:
    """Test model validation error handling."""
    
    def test_model_validation_error_basic(self):
        """Test basic model validation error."""
        error = ModelValidationError(
            model_name="invalid-model",
            available_models=["model1", "model2"],
            suggestion="Try model1"
        )
        
        assert "invalid-model" in str(error)
        assert "model1, model2" in str(error)
        assert "Try model1" in str(error)
        assert error.model_name == "invalid-model"
        assert error.available_models == ["model1", "model2"]
        assert error.suggestion == "Try model1"
    
    def test_model_validation_error_many_models(self):
        """Test model validation error with many available models."""
        many_models = [f"model{i}" for i in range(20)]
        error = ModelValidationError(
            model_name="invalid-model",
            available_models=many_models
        )
        
        error_str = str(error)
        assert "showing first 10 of 20" in error_str
        assert "model0" in error_str
        assert "model9" in error_str
        assert "model19" not in error_str  # Should be truncated
    
    def test_model_validation_error_no_models(self):
        """Test model validation error with no available models."""
        error = ModelValidationError(
            model_name="invalid-model",
            available_models=[],
            suggestion="Check documentation"
        )
        
        error_str = str(error)
        assert "invalid-model" in error_str
        assert "Check documentation" in error_str


class TestVectorDimensionMismatchError:
    """Test vector dimension mismatch error handling."""
    
    def test_dimension_mismatch_error(self):
        """Test vector dimension mismatch error."""
        error = VectorDimensionMismatchError(
            collection_name="test_collection",
            expected_dimensions=384,
            actual_dimensions=768,
            model_name="test-model",
            vector_name="test-vector",
            available_vectors=["vector1", "vector2"]
        )
        
        error_str = str(error)
        assert "test_collection" in error_str
        assert "384" in error_str
        assert "768" in error_str
        assert "test-model" in error_str
        assert "test-vector" in error_str
        assert "vector1, vector2" in error_str
        assert "different collection name" in error_str
        
        # Check details
        assert error.details["resolution_suggestions"]
        assert len(error.details["resolution_suggestions"]) > 0
    
    def test_dimension_mismatch_missing_vector(self):
        """Test dimension mismatch when vector doesn't exist."""
        error = VectorDimensionMismatchError(
            collection_name="test_collection",
            expected_dimensions=384,
            actual_dimensions=0,  # No matching vector
            model_name="test-model",
            vector_name="missing-vector",
            available_vectors=["other-vector"]
        )
        
        error_str = str(error)
        assert "missing-vector" in error_str
        assert "other-vector" in error_str


class TestChunkingError:
    """Test chunking error handling."""
    
    def test_chunking_error_with_fallback(self):
        """Test chunking error with fallback used."""
        original_error = ValueError("Chunking failed")
        error = ChunkingError(
            original_error=original_error,
            document_length=1000,
            chunking_config={"max_chunk_size": 512, "chunk_overlap": 50},
            fallback_used=True
        )
        
        error_str = str(error)
        assert "Chunking failed" in error_str
        assert "stored as a single entry" in error_str
        assert error.fallback_used is True
        assert error.document_length == 1000
        assert "suggestions" in error.details
    
    def test_chunking_error_without_fallback(self):
        """Test chunking error without fallback."""
        original_error = RuntimeError("Critical error")
        error = ChunkingError(
            original_error=original_error,
            document_length=2000,
            chunking_config={"max_chunk_size": 256},
            fallback_used=False
        )
        
        error_str = str(error)
        assert "Critical error" in error_str
        assert "adjusting chunking parameters" in error_str
        assert error.fallback_used is False


class TestConfigurationValidationError:
    """Test configuration validation error handling."""
    
    def test_config_error_with_options(self):
        """Test configuration error with valid options."""
        error = ConfigurationValidationError(
            field_name="chunk_strategy",
            invalid_value="invalid",
            validation_error="Must be one of the supported strategies",
            valid_options=["semantic", "fixed", "sentence"],
            suggested_value="semantic"
        )
        
        error_str = str(error)
        assert "chunk_strategy" in error_str
        assert "semantic, fixed, sentence" in error_str
        assert "Suggested value: semantic" in error_str
    
    def test_config_error_without_options(self):
        """Test configuration error without valid options."""
        error = ConfigurationValidationError(
            field_name="max_chunk_size",
            invalid_value=10,
            validation_error="Must be at least 50",
            suggested_value=512
        )
        
        error_str = str(error)
        assert "max_chunk_size" in error_str
        assert "Must be at least 50" in error_str
        assert "Suggested value: 512" in error_str


class TestFastEmbedProviderErrorHandling:
    """Test FastEmbed provider error handling."""
    
    @patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding')
    def test_invalid_model_validation(self, mock_text_embedding_class):
        """Test that invalid models raise ModelValidationError."""
        # Mock the class methods
        mock_text_embedding_class._get_model_description.side_effect = Exception("Model not found")
        mock_text_embedding_class.list_supported_models.return_value = [
            {"model": "model1"}, {"model": "model2"}
        ]
        
        with pytest.raises(ModelValidationError) as exc_info:
            FastEmbedProvider("invalid-model")
        
        error = exc_info.value
        assert error.model_name == "invalid-model"
        assert "model1" in error.available_models
        assert "model2" in error.available_models
    
    @patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding')
    def test_model_suggestion(self, mock_text_embedding):
        """Test model suggestion logic."""
        mock_text_embedding._get_model_description.side_effect = Exception("Model not found")
        mock_text_embedding.list_supported_models.return_value = [
            {"model": "nomic-ai/nomic-embed-text-v1.5-Q"},
            {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        ]
        
        provider = FastEmbedProvider.__new__(FastEmbedProvider)
        
        # Test nomic suggestion
        suggestion = provider._suggest_similar_model(
            "nomic-embed", 
            ["nomic-ai/nomic-embed-text-v1.5-Q", "other-model"]
        )
        assert "nomic-ai/nomic-embed-text-v1.5-Q" in suggestion
        
        # Test sentence-transformers suggestion
        suggestion = provider._suggest_similar_model(
            "sentence-transformers/something", 
            ["sentence-transformers/all-MiniLM-L6-v2", "other-model"]
        )
        assert "sentence-transformers/all-MiniLM-L6-v2" in suggestion
    
    @pytest.mark.asyncio
    async def test_embed_documents_error_handling(self):
        """Test error handling in embed_documents."""
        with patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding') as mock_text_embedding:
            # Setup successful validation
            mock_text_embedding._get_model_description.return_value = Mock()
            
            # Create provider
            provider = FastEmbedProvider("test-model")
            
            # Mock embedding failure
            provider.embedding_model.passage_embed.side_effect = Exception("Embedding failed")
            
            with pytest.raises(RuntimeError) as exc_info:
                await provider.embed_documents(["test document"])
            
            assert "Document embedding failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embed_query_empty_query(self):
        """Test error handling for empty query."""
        with patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding') as mock_text_embedding:
            mock_text_embedding._get_model_description.return_value = Mock()
            
            provider = FastEmbedProvider("test-model")
            
            with pytest.raises(ValueError) as exc_info:
                await provider.embed_query("")
            
            assert "Query cannot be empty" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embed_query_error_handling(self):
        """Test error handling in embed_query."""
        with patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding') as mock_text_embedding:
            mock_text_embedding._get_model_description.return_value = Mock()
            
            provider = FastEmbedProvider("test-model")
            provider.embedding_model.query_embed.side_effect = Exception("Query embedding failed")
            
            with pytest.raises(RuntimeError) as exc_info:
                await provider.embed_query("test query")
            
            assert "Query embedding failed" in str(exc_info.value)


class TestQdrantConnectorErrorHandling:
    """Test QdrantConnector error handling."""
    
    @pytest.mark.asyncio
    async def test_collection_access_error(self):
        """Test collection access error handling."""
        mock_embedding_provider = Mock()
        mock_embedding_provider.get_vector_size.return_value = 384
        mock_embedding_provider.get_vector_name.return_value = "test-vector"
        
        connector = QdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=mock_embedding_provider
        )
        
        # Mock client to raise exception
        connector._client.get_collections = AsyncMock(side_effect=Exception("Connection failed"))
        
        with pytest.raises(CollectionAccessError) as exc_info:
            await connector.get_collection_names()
        
        error = exc_info.value
        assert error.collection_name == "*"
        assert error.operation == "list"
        assert "Connection failed" in str(error)
    
    @pytest.mark.asyncio
    async def test_dimension_mismatch_validation(self):
        """Test dimension mismatch validation."""
        mock_embedding_provider = Mock()
        mock_embedding_provider.get_vector_size.return_value = 384
        mock_embedding_provider.get_vector_name.return_value = "test-vector"
        mock_embedding_provider.model_name = "test-model"
        
        connector = QdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=mock_embedding_provider
        )
        
        # Mock collection info with wrong dimensions
        mock_collection_info = Mock()
        mock_collection_info.config.params.vectors = {
            "test-vector": Mock(size=768)  # Wrong size
        }
        connector._client.get_collection = AsyncMock(return_value=mock_collection_info)
        
        with pytest.raises(VectorDimensionMismatchError) as exc_info:
            await connector._validate_collection_dimensions("test_collection")
        
        error = exc_info.value
        assert error.collection_name == "test_collection"
        assert error.expected_dimensions == 384
        assert error.actual_dimensions == 768
        assert error.model_name == "test-model"
    
    @pytest.mark.asyncio
    async def test_chunking_fallback_on_error(self):
        """Test that chunking failures fall back to storing original document."""
        mock_embedding_provider = Mock()
        mock_embedding_provider.get_vector_size.return_value = 384
        mock_embedding_provider.get_vector_name.return_value = "test-vector"
        mock_embedding_provider.embed_documents = AsyncMock(return_value=[[0.1] * 384])
        
        # Mock chunker that fails
        mock_chunker = Mock()
        mock_chunker.chunk_document = AsyncMock(side_effect=Exception("Chunking failed"))
        mock_chunker.max_tokens = 512
        mock_chunker.overlap_tokens = 50
        mock_chunker._count_tokens.return_value = 1000  # Large document
        
        connector = QdrantConnector(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
            collection_name="test_collection",
            embedding_provider=mock_embedding_provider,
            enable_chunking=True
        )
        connector._chunker = mock_chunker
        
        # Mock successful collection operations
        connector._client.collection_exists = AsyncMock(return_value=True)
        connector._client.get_collection = AsyncMock(return_value=Mock(
            config=Mock(params=Mock(vectors={"test-vector": Mock(size=384)}))
        ))
        connector._client.upsert = AsyncMock()
        
        # This should not raise an error, but should fall back to single entry storage
        entry = Entry(content="A very long document that should be chunked but chunking will fail")
        await connector.store(entry)
        
        # Verify that upsert was called (fallback storage)
        connector._client.upsert.assert_called_once()


class TestDocumentChunkerErrorHandling:
    """Test DocumentChunker error handling."""
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            DocumentChunker(max_tokens=0)
        
        with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
            DocumentChunker(overlap_tokens=-1)
        
        with pytest.raises(ValueError, match="overlap_tokens must be less than max_tokens"):
            DocumentChunker(max_tokens=100, overlap_tokens=100)
    
    def test_invalid_sentence_splitter(self):
        """Test invalid sentence splitter handling."""
        with pytest.raises(ValueError, match="Unknown sentence splitter"):
            DocumentChunker(sentence_splitter="invalid")
    
    def test_invalid_tokenizer(self):
        """Test invalid tokenizer handling."""
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            DocumentChunker(tokenizer="invalid")
    
    @patch('mcp_server_qdrant.chunking.chunker.NLTK_AVAILABLE', False)
    def test_nltk_unavailable_error(self):
        """Test error when NLTK is requested but unavailable."""
        with pytest.raises(SentenceSplitterError) as exc_info:
            DocumentChunker(sentence_splitter="nltk")
        
        error = exc_info.value
        assert error.splitter_name == "nltk"
        assert error.fallback_available is True
    
    @patch('mcp_server_qdrant.chunking.chunker.SYNTOK_AVAILABLE', False)
    def test_syntok_unavailable_error(self):
        """Test error when syntok is requested but unavailable."""
        with pytest.raises(SentenceSplitterError) as exc_info:
            DocumentChunker(sentence_splitter="syntok")
        
        error = exc_info.value
        assert error.splitter_name == "syntok"
        assert error.fallback_available is True  # NLTK or simple fallback available
    
    @patch('mcp_server_qdrant.chunking.chunker.TIKTOKEN_AVAILABLE', False)
    def test_tiktoken_unavailable_error(self):
        """Test error when tiktoken is requested but unavailable."""
        with pytest.raises(TokenizerError) as exc_info:
            DocumentChunker(tokenizer="tiktoken")
        
        error = exc_info.value
        assert error.tokenizer_name == "tiktoken"
        assert error.fallback_available is True
    
    @pytest.mark.asyncio
    async def test_chunking_empty_content(self):
        """Test chunking with empty content."""
        chunker = DocumentChunker()
        
        # Empty string
        chunks = await chunker.chunk_document("")
        assert chunks == []
        
        # Whitespace only
        chunks = await chunker.chunk_document("   \n\t  ")
        assert chunks == []
        
        # None content should be handled gracefully
        chunks = await chunker.chunk_document(None)
        assert chunks == []


class TestSettingsValidationErrors:
    """Test settings validation with enhanced error messages."""
    
    def test_max_chunk_size_too_small(self, monkeypatch):
        """Test max_chunk_size validation with helpful error."""
        monkeypatch.setenv("MAX_CHUNK_SIZE", "10")
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            EmbeddingProviderSettings()
        
        error = exc_info.value
        assert error.field_name == "max_chunk_size"
        assert error.invalid_value == 10
        assert error.suggested_value == 512
    
    def test_max_chunk_size_too_large(self, monkeypatch):
        """Test max_chunk_size validation for large values."""
        monkeypatch.setenv("MAX_CHUNK_SIZE", "10000")
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            EmbeddingProviderSettings()
        
        error = exc_info.value
        assert error.field_name == "max_chunk_size"
        assert error.invalid_value == 10000
        assert error.suggested_value == 2048
    
    def test_chunk_overlap_negative(self, monkeypatch):
        """Test chunk_overlap validation for negative values."""
        monkeypatch.setenv("CHUNK_OVERLAP", "-10")
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            EmbeddingProviderSettings()
        
        error = exc_info.value
        assert error.field_name == "chunk_overlap"
        assert error.invalid_value == -10
        assert error.suggested_value == 50
    
    def test_chunk_overlap_too_large(self, monkeypatch):
        """Test chunk_overlap validation for large values."""
        monkeypatch.setenv("CHUNK_OVERLAP", "2000")
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            EmbeddingProviderSettings()
        
        error = exc_info.value
        assert error.field_name == "chunk_overlap"
        assert error.invalid_value == 2000
        assert error.suggested_value == 100
    
    def test_invalid_chunk_strategy(self, monkeypatch):
        """Test chunk_strategy validation with suggestions."""
        monkeypatch.setenv("CHUNK_STRATEGY", "invalid_strategy")
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            EmbeddingProviderSettings()
        
        error = exc_info.value
        assert error.field_name == "chunk_strategy"
        assert error.invalid_value == "invalid_strategy"
        assert error.suggested_value == "semantic"
        assert "semantic" in error.valid_options
        assert "fixed" in error.valid_options
        assert "sentence" in error.valid_options
    
    def test_chunk_overlap_larger_than_size(self, monkeypatch):
        """Test validation when chunk_overlap >= max_chunk_size."""
        monkeypatch.setenv("MAX_CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        
        with pytest.raises(ConfigurationValidationError) as exc_info:
            EmbeddingProviderSettings()
        
        error = exc_info.value
        assert error.field_name == "chunk_overlap"
        assert error.invalid_value == 100
        assert error.suggested_value == 10  # 10% of chunk size
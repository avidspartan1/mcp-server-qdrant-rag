"""
Tests for embedding model intelligence functionality in CLI ingestion.

This module tests the EmbeddingModelIntelligence class and related functionality
for smart embedding model selection, validation, and compatibility checking.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.mcp_server_qdrant_rag.cli_ingest import (
    EmbeddingModelIntelligence,
    EmbeddingModelInfo,
)
from src.mcp_server_qdrant_rag.settings import QdrantSettings


@pytest.fixture
def qdrant_settings():
    """Create test Qdrant settings."""
    settings = QdrantSettings()
    settings.location = "http://localhost:6333"
    settings.api_key = None
    settings.collection_name = "test_collection"
    return settings


@pytest.fixture
def embedding_intelligence(qdrant_settings):
    """Create EmbeddingModelIntelligence instance for testing."""
    return EmbeddingModelIntelligence(qdrant_settings)


class TestEmbeddingModelInfo:
    """Test EmbeddingModelInfo dataclass."""
    
    def test_embedding_model_info_creation(self):
        """Test creating EmbeddingModelInfo with default values."""
        info = EmbeddingModelInfo(
            model_name="test-model",
            vector_size=768,
            vector_name="fast-test-model"
        )
        
        assert info.model_name == "test-model"
        assert info.vector_size == 768
        assert info.vector_name == "fast-test-model"
        assert info.is_available is True
        assert info.is_compatible is True
        assert info.error_message is None
        assert info.collection_exists is False
    
    def test_embedding_model_info_with_error(self):
        """Test creating EmbeddingModelInfo with error information."""
        info = EmbeddingModelInfo(
            model_name="invalid-model",
            vector_size=0,
            vector_name="unknown",
            is_available=False,
            is_compatible=False,
            error_message="Model not found"
        )
        
        assert info.is_available is False
        assert info.is_compatible is False
        assert info.error_message == "Model not found"


class TestEmbeddingModelIntelligence:
    """Test EmbeddingModelIntelligence class."""
    
    def test_initialization(self, qdrant_settings):
        """Test EmbeddingModelIntelligence initialization."""
        intelligence = EmbeddingModelIntelligence(qdrant_settings)
        
        assert intelligence.qdrant_settings == qdrant_settings
        assert intelligence.DEFAULT_MODEL == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert len(intelligence.FALLBACK_MODELS) > 0
        assert intelligence._connector_cache is None
    
    def test_infer_model_from_collection_info_exact_match(self, embedding_intelligence):
        """Test model inference with exact pattern match."""
        # Test exact match for nomic model
        model = embedding_intelligence._infer_model_from_collection_info(
            "fast-nomic-embed-text-v1.5-q", 768
        )
        assert model == "nomic-ai/nomic-embed-text-v1.5-Q"
        
        # Test exact match for sentence transformers
        model = embedding_intelligence._infer_model_from_collection_info(
            "fast-all-minilm-l6-v2", 384
        )
        assert model == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_infer_model_from_collection_info_partial_match(self, embedding_intelligence):
        """Test model inference with partial pattern match."""
        # Test partial match based on vector size and name components
        model = embedding_intelligence._infer_model_from_collection_info(
            "fast-nomic-embed", 768
        )
        assert model == "nomic-ai/nomic-embed-text-v1.5-Q"
    
    def test_infer_model_from_collection_info_no_match(self, embedding_intelligence):
        """Test model inference when no pattern matches."""
        model = embedding_intelligence._infer_model_from_collection_info(
            "unknown-vector", 999
        )
        assert model is None
        
        # Test with None inputs
        model = embedding_intelligence._infer_model_from_collection_info(None, None)
        assert model is None
    
    @pytest.mark.asyncio
    async def test_validate_model_success(self, embedding_intelligence):
        """Test successful model validation."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create:
            # Mock successful provider creation
            mock_provider = MagicMock()
            mock_provider.get_vector_size.return_value = 768
            mock_provider.get_vector_name.return_value = "fast-test-model"
            mock_create.return_value = mock_provider
            
            result = await embedding_intelligence.validate_model("test-model")
            
            assert result.model_name == "test-model"
            assert result.vector_size == 768
            assert result.vector_name == "fast-test-model"
            assert result.is_available is True
            assert result.is_compatible is True
            assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_validate_model_failure(self, embedding_intelligence):
        """Test model validation failure."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create:
            # Mock provider creation failure
            mock_create.side_effect = Exception("Model not found")
            
            result = await embedding_intelligence.validate_model("invalid-model")
            
            assert result.model_name == "invalid-model"
            assert result.vector_size == 0
            assert result.vector_name == "unknown"
            assert result.is_available is False
            assert result.is_compatible is False
            assert "Model validation failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_detect_collection_model_nonexistent(self, embedding_intelligence):
        """Test detecting model from non-existent collection."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock connector that reports collection doesn't exist
            mock_connector = AsyncMock()
            mock_connector.get_collection_names.return_value = ["other_collection"]
            mock_connector_class.return_value = mock_connector
            
            result = await embedding_intelligence.detect_collection_model("nonexistent_collection")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_detect_collection_model_existing(self, embedding_intelligence):
        """Test detecting model from existing collection."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock connector with existing collection
            mock_connector = AsyncMock()
            mock_connector.get_collection_names.return_value = ["test_collection"]
            mock_connector.check_collection_compatibility.return_value = {
                "exists": True,
                "available_vectors": {"fast-nomic-embed-text-v1.5-q": 768}
            }
            mock_connector_class.return_value = mock_connector
            
            result = await embedding_intelligence.detect_collection_model("test_collection")
            
            assert result is not None
            assert result.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
            assert result.vector_size == 768
            assert result.collection_exists is True
            assert result.is_compatible is True
    
    @pytest.mark.asyncio
    async def test_detect_collection_model_error(self, embedding_intelligence):
        """Test detecting model when an error occurs."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock connector that raises an exception
            mock_connector = AsyncMock()
            mock_connector.get_collection_names.side_effect = Exception("Connection failed")
            mock_connector_class.return_value = mock_connector
            
            result = await embedding_intelligence.detect_collection_model("test_collection")
            
            assert result is not None
            assert result.model_name == "unknown"
            assert result.is_available is False
            assert result.is_compatible is False
            assert "Failed to detect collection model" in result.error_message
    
    @pytest.mark.asyncio
    async def test_select_smart_default_existing_collection(self, embedding_intelligence):
        """Test smart default selection for existing collection."""
        with patch.object(embedding_intelligence, 'detect_collection_model') as mock_detect:
            # Mock existing collection with detected model
            mock_detect.return_value = EmbeddingModelInfo(
                model_name="existing-model",
                vector_size=384,
                vector_name="fast-existing",
                is_available=True,
                collection_exists=True
            )
            
            result = await embedding_intelligence.select_smart_default("test_collection")
            
            assert result.model_name == "existing-model"
            assert result.vector_size == 384
            assert result.is_available is True
    
    @pytest.mark.asyncio
    async def test_select_smart_default_new_collection(self, embedding_intelligence):
        """Test smart default selection for new collection."""
        with patch.object(embedding_intelligence, 'detect_collection_model') as mock_detect, \
             patch.object(embedding_intelligence, 'validate_model') as mock_validate:
            
            # Mock no existing collection
            mock_detect.return_value = None
            
            # Mock successful default model validation
            mock_validate.return_value = EmbeddingModelInfo(
                model_name=embedding_intelligence.DEFAULT_MODEL,
                vector_size=768,
                vector_name="fast-nomic-embed-text-v1.5-q",
                is_available=True
            )
            
            result = await embedding_intelligence.select_smart_default("new_collection")
            
            assert result.model_name == embedding_intelligence.DEFAULT_MODEL
            assert result.is_available is True
    
    @pytest.mark.asyncio
    async def test_select_smart_default_fallback(self, embedding_intelligence):
        """Test smart default selection with fallback models."""
        with patch.object(embedding_intelligence, 'detect_collection_model') as mock_detect, \
             patch.object(embedding_intelligence, 'validate_model') as mock_validate:
            
            # Mock no existing collection
            mock_detect.return_value = None
            
            # Mock default model failure, fallback success
            def validate_side_effect(model_name):
                if model_name == embedding_intelligence.DEFAULT_MODEL:
                    return EmbeddingModelInfo(
                        model_name=model_name,
                        vector_size=0,
                        vector_name="unknown",
                        is_available=False,
                        error_message="Default model failed"
                    )
                else:
                    return EmbeddingModelInfo(
                        model_name=model_name,
                        vector_size=384,
                        vector_name="fast-fallback",
                        is_available=True
                    )
            
            mock_validate.side_effect = validate_side_effect
            
            result = await embedding_intelligence.select_smart_default("new_collection")
            
            assert result.model_name == embedding_intelligence.FALLBACK_MODELS[1]  # Second model (first is default)
            assert result.is_available is True
    
    @pytest.mark.asyncio
    async def test_select_smart_default_all_fail(self, embedding_intelligence):
        """Test smart default selection when all models fail."""
        with patch.object(embedding_intelligence, 'detect_collection_model') as mock_detect, \
             patch.object(embedding_intelligence, 'validate_model') as mock_validate:
            
            # Mock no existing collection
            mock_detect.return_value = None
            
            # Mock all models failing
            mock_validate.return_value = EmbeddingModelInfo(
                model_name="any-model",
                vector_size=0,
                vector_name="unknown",
                is_available=False,
                error_message="Model failed"
            )
            
            result = await embedding_intelligence.select_smart_default("new_collection")
            
            assert result.is_available is False
            assert "No compatible embedding models are available" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_model_compatibility_new_collection(self, embedding_intelligence):
        """Test model compatibility validation for new collection."""
        with patch.object(embedding_intelligence, 'validate_model') as mock_validate, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock successful model validation
            mock_validate.return_value = EmbeddingModelInfo(
                model_name="test-model",
                vector_size=768,
                vector_name="fast-test",
                is_available=True
            )
            
            # Mock connector reporting collection doesn't exist
            mock_connector = AsyncMock()
            mock_connector.check_collection_compatibility.return_value = {"exists": False}
            mock_connector_class.return_value = mock_connector
            
            result = await embedding_intelligence.validate_model_compatibility(
                "test-model", "new_collection"
            )
            
            assert result.is_available is True
            assert result.is_compatible is True
            assert result.collection_exists is False
    
    @pytest.mark.asyncio
    async def test_validate_model_compatibility_compatible_existing(self, embedding_intelligence):
        """Test model compatibility validation for compatible existing collection."""
        with patch.object(embedding_intelligence, 'validate_model') as mock_validate, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock successful model validation
            mock_validate.return_value = EmbeddingModelInfo(
                model_name="test-model",
                vector_size=768,
                vector_name="fast-test",
                is_available=True
            )
            
            # Mock connector reporting compatible collection
            mock_connector = AsyncMock()
            mock_connector.check_collection_compatibility.return_value = {
                "exists": True,
                "vector_compatible": True,
                "dimension_compatible": True,
                "available_vectors": {"fast-test": 768}
            }
            mock_connector_class.return_value = mock_connector
            
            result = await embedding_intelligence.validate_model_compatibility(
                "test-model", "existing_collection"
            )
            
            assert result.is_available is True
            assert result.is_compatible is True
            assert result.collection_exists is True
    
    @pytest.mark.asyncio
    async def test_validate_model_compatibility_incompatible_dimensions(self, embedding_intelligence):
        """Test model compatibility validation for dimension mismatch."""
        with patch.object(embedding_intelligence, 'validate_model') as mock_validate, \
             patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock successful model validation
            mock_validate.return_value = EmbeddingModelInfo(
                model_name="test-model",
                vector_size=768,
                vector_name="fast-test",
                is_available=True
            )
            
            # Mock connector reporting dimension mismatch
            mock_connector = AsyncMock()
            mock_connector.check_collection_compatibility.return_value = {
                "exists": True,
                "vector_compatible": True,
                "dimension_compatible": False,
                "expected_dimensions": 768,
                "actual_dimensions": 384,
                "available_vectors": {"fast-test": 384}
            }
            mock_connector_class.return_value = mock_connector
            
            result = await embedding_intelligence.validate_model_compatibility(
                "test-model", "existing_collection"
            )
            
            assert result.is_available is True
            assert result.is_compatible is False
            assert result.collection_exists is True
            assert "Dimension mismatch" in result.error_message
    
    def test_display_model_info_available(self, embedding_intelligence, capsys):
        """Test displaying information for available model."""
        model_info = EmbeddingModelInfo(
            model_name="test-model",
            vector_size=768,
            vector_name="fast-test",
            is_available=True,
            is_compatible=True
        )
        
        embedding_intelligence.display_model_info(model_info)
        captured = capsys.readouterr()
        
        assert "✅ Model 'test-model' is available for new collection" in captured.out
    
    def test_display_model_info_unavailable(self, embedding_intelligence, capsys):
        """Test displaying information for unavailable model."""
        model_info = EmbeddingModelInfo(
            model_name="invalid-model",
            vector_size=0,
            vector_name="unknown",
            is_available=False,
            error_message="Model not found"
        )
        
        embedding_intelligence.display_model_info(model_info)
        captured = capsys.readouterr()
        
        assert "❌ Model 'invalid-model' is not available" in captured.out
        assert "Error: Model not found" in captured.out
    
    def test_display_model_info_incompatible(self, embedding_intelligence, capsys):
        """Test displaying information for incompatible model."""
        model_info = EmbeddingModelInfo(
            model_name="test-model",
            vector_size=768,
            vector_name="fast-test",
            is_available=True,
            is_compatible=False,
            collection_exists=True,
            error_message="Dimension mismatch"
        )
        
        embedding_intelligence.display_model_info(model_info)
        captured = capsys.readouterr()
        
        assert "❌ Model 'test-model' is incompatible with existing collection" in captured.out
        assert "Error: Dimension mismatch" in captured.out
    
    def test_display_model_mismatch_error(self, embedding_intelligence, capsys):
        """Test displaying detailed mismatch error."""
        model_info = EmbeddingModelInfo(
            model_name="new-model",
            vector_size=768,
            vector_name="fast-new",
            collection_exists=True,
            collection_model="old-model",
            collection_vector_size=384,
            collection_vector_name="fast-old"
        )
        
        embedding_intelligence.display_model_mismatch_error(model_info)
        captured = capsys.readouterr()
        
        assert "🚫 Embedding Model Mismatch Detected" in captured.out
        assert "Collection model: old-model" in captured.out
        assert "Collection dimensions: 384" in captured.out
        assert "Requested model configuration:" in captured.out
        assert "Model: new-model" in captured.out
        assert "Dimensions: 768" in captured.out
        assert "Solutions:" in captured.out
    
    @pytest.mark.asyncio
    async def test_get_available_models_success(self, embedding_intelligence):
        """Test getting available models successfully."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.list_supported_models.return_value = [
                {"model": "model1"},
                {"model": "model2"},
                {"model": "model3"}
            ]
            
            models = await embedding_intelligence.get_available_models()
            
            assert models == ["model1", "model2", "model3"]
    
    @pytest.mark.asyncio
    async def test_get_available_models_failure(self, embedding_intelligence, capsys):
        """Test getting available models when it fails."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.list_supported_models.side_effect = Exception("Import failed")
            
            models = await embedding_intelligence.get_available_models()
            
            assert models == embedding_intelligence.FALLBACK_MODELS
            captured = capsys.readouterr()
            assert "Warning: Could not retrieve available models" in captured.err


class TestEmbeddingModelIntelligenceIntegration:
    """Integration tests for embedding model intelligence."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_new_collection(self, embedding_intelligence):
        """Test complete workflow for new collection."""
        with patch.object(embedding_intelligence, 'detect_collection_model') as mock_detect, \
             patch.object(embedding_intelligence, 'validate_model') as mock_validate:
            
            # Mock no existing collection
            mock_detect.return_value = None
            
            # Mock successful default model validation
            mock_validate.return_value = EmbeddingModelInfo(
                model_name=embedding_intelligence.DEFAULT_MODEL,
                vector_size=768,
                vector_name="fast-nomic-embed-text-v1.5-q",
                is_available=True,
                is_compatible=True
            )
            
            # Test smart default selection
            result = await embedding_intelligence.select_smart_default("new_collection")
            
            assert result.model_name == embedding_intelligence.DEFAULT_MODEL
            assert result.is_available is True
            assert result.is_compatible is True
    
    @pytest.mark.asyncio
    async def test_full_workflow_existing_collection(self, embedding_intelligence):
        """Test complete workflow for existing collection."""
        with patch('src.mcp_server_qdrant_rag.cli_ingest.create_embedding_provider'), \
             patch('src.mcp_server_qdrant_rag.cli_ingest.QdrantConnector') as mock_connector_class:
            
            # Mock connector with existing collection
            mock_connector = AsyncMock()
            mock_connector.get_collection_names.return_value = ["existing_collection"]
            mock_connector.check_collection_compatibility.return_value = {
                "exists": True,
                "available_vectors": {"fast-nomic-embed-text-v1.5-q": 768}
            }
            mock_connector_class.return_value = mock_connector
            
            # Test detection and smart selection
            detected = await embedding_intelligence.detect_collection_model("existing_collection")
            assert detected is not None
            assert detected.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
            
            selected = await embedding_intelligence.select_smart_default("existing_collection")
            assert selected.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
            assert selected.collection_exists is True
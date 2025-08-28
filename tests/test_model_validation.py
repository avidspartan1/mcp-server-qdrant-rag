"""Unit tests for model validation and dimension detection functionality."""

import pytest
from unittest.mock import patch, MagicMock
from fastembed.common.model_description import DenseModelDescription

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.common.exceptions import ModelValidationError


class TestModelValidation:
    """Test model validation functionality."""

    def test_valid_model_initialization(self):
        """Test that valid models initialize successfully."""
        # Test with the default model
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        assert provider.model_name == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert provider.embedding_model is not None

    def test_invalid_model_raises_validation_error(self):
        """Test that invalid model names raise ModelValidationError."""
        with pytest.raises(ModelValidationError) as exc_info:
            FastEmbedProvider("invalid-model-name")
        
        error = exc_info.value
        assert error.model_name == "invalid-model-name"
        assert "not found" in str(error) or "invalid" in str(error).lower()

    @patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding.list_supported_models')
    def test_model_validation_with_suggestions(self, mock_list_models):
        """Test that model validation provides helpful suggestions."""
        # Mock available models
        mock_list_models.return_value = [
            {'model': 'nomic-ai/nomic-embed-text-v1.5-Q'},
            {'model': 'sentence-transformers/all-MiniLM-L6-v2'},
            {'model': 'BAAI/bge-small-en-v1.5'}
        ]
        
        with pytest.raises(ModelValidationError) as exc_info:
            FastEmbedProvider("nomic-invalid")
        
        error = exc_info.value
        assert error.model_name == "nomic-invalid"
        assert len(error.available_models) == 3
        assert "nomic-ai/nomic-embed-text-v1.5-Q" in error.available_models
        assert "nomic-ai/nomic-embed-text-v1.5-Q" in error.suggestion

    def test_suggest_similar_model_patterns(self):
        """Test the model suggestion logic for various patterns."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        available_models = [
            'nomic-ai/nomic-embed-text-v1.5-Q',
            'sentence-transformers/all-MiniLM-L6-v2',
            'BAAI/bge-small-en-v1.5'
        ]
        
        # Test nomic pattern
        suggestion = provider._suggest_similar_model("nomic-invalid", available_models)
        assert "nomic-ai/nomic-embed-text-v1.5-Q" in suggestion
        
        # Test sentence-transformers pattern
        suggestion = provider._suggest_similar_model("sentence-transformers-invalid", available_models)
        assert "sentence-transformers/all-MiniLM-L6-v2" in suggestion
        
        # Test bge pattern
        suggestion = provider._suggest_similar_model("bge-invalid", available_models)
        assert "BAAI/bge-small-en-v1.5" in suggestion
        
        # Test unknown pattern
        suggestion = provider._suggest_similar_model("completely-unknown", available_models)
        assert "nomic-ai/nomic-embed-text-v1.5-Q" in suggestion

    def test_vector_size_detection(self):
        """Test that vector size is correctly detected from model."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        vector_size = provider.get_vector_size()
        
        assert isinstance(vector_size, int)
        assert vector_size > 0
        assert vector_size <= 4096  # Reasonable upper bound

    def test_vector_name_generation(self):
        """Test that vector names are generated consistently."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        vector_name = provider.get_vector_name()
        
        assert isinstance(vector_name, str)
        assert vector_name.startswith("fast-")
        assert len(vector_name) > 5
        
        # Test consistency
        assert provider.get_vector_name() == vector_name

    def test_model_info_structure(self):
        """Test that model info contains all expected fields."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        model_info = provider.get_model_info()
        
        required_fields = ["model_name", "vector_size", "vector_name", "description", "status"]
        for field in required_fields:
            assert field in model_info
        
        assert model_info["model_name"] == "nomic-ai/nomic-embed-text-v1.5-Q"
        assert isinstance(model_info["vector_size"], int)
        assert model_info["vector_size"] > 0
        assert model_info["vector_name"].startswith("fast-")
        assert model_info["status"] == "loaded"

    def test_vector_size_error_handling(self):
        """Test error handling when vector size cannot be determined."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        # Mock the get_vector_size method to raise an exception
        with patch.object(provider.embedding_model, '_get_model_description') as mock_get_description:
            mock_get_description.side_effect = Exception("Model description error")
            
            with pytest.raises(RuntimeError) as exc_info:
                provider.get_vector_size()
            
            assert "Could not determine vector size" in str(exc_info.value)
            assert "nomic-ai/nomic-embed-text-v1.5-Q" in str(exc_info.value)

    def test_model_info_error_handling(self):
        """Test error handling when model info cannot be retrieved."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        # Mock the get_model_info method to raise an exception
        with patch.object(provider.embedding_model, '_get_model_description') as mock_get_description:
            mock_get_description.side_effect = Exception("Model info error")
            
            model_info = provider.get_model_info()
            
            assert model_info["model_name"] == "nomic-ai/nomic-embed-text-v1.5-Q"
            assert model_info["vector_size"] is None
            assert model_info["status"] == "error"
            assert "Error getting model info" in model_info["description"]

    def test_dimension_consistency_across_calls(self):
        """Test that dimension detection is consistent across multiple calls."""
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        
        # Get dimensions multiple times
        size1 = provider.get_vector_size()
        size2 = provider.get_vector_size()
        size3 = provider.get_vector_size()
        
        assert size1 == size2 == size3
        
        # Get vector names multiple times
        name1 = provider.get_vector_name()
        name2 = provider.get_vector_name()
        name3 = provider.get_vector_name()
        
        assert name1 == name2 == name3

    def test_different_models_different_properties(self):
        """Test that different models have different properties."""
        provider1 = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        provider2 = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")
        
        # Vector names should be different
        name1 = provider1.get_vector_name()
        name2 = provider2.get_vector_name()
        assert name1 != name2
        
        # Both should have valid dimensions
        size1 = provider1.get_vector_size()
        size2 = provider2.get_vector_size()
        assert isinstance(size1, int) and size1 > 0
        assert isinstance(size2, int) and size2 > 0

    @patch('mcp_server_qdrant.embeddings.fastembed.TextEmbedding.list_supported_models')
    def test_model_validation_error_without_available_models(self, mock_list_models):
        """Test model validation when available models list fails."""
        # Mock list_supported_models to raise an exception
        mock_list_models.side_effect = Exception("Failed to get models")
        
        with pytest.raises(ModelValidationError) as exc_info:
            FastEmbedProvider("invalid-model")
        
        error = exc_info.value
        assert error.model_name == "invalid-model"
        assert error.available_models == []
        assert "Check FastEmbed documentation" in error.suggestion

    def test_model_validation_logging(self, caplog):
        """Test that model validation logs appropriate messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Test successful validation
        provider = FastEmbedProvider("nomic-ai/nomic-embed-text-v1.5-Q")
        assert "Successfully validated embedding model" in caplog.text
        assert "nomic-ai/nomic-embed-text-v1.5-Q" in caplog.text

    def test_model_validation_error_logging(self, caplog):
        """Test that model validation errors are logged."""
        import logging
        caplog.set_level(logging.ERROR)
        
        # Test failed validation
        with pytest.raises(ModelValidationError):
            FastEmbedProvider("invalid-model-name")
        
        assert "Model validation failed" in caplog.text
        assert "invalid-model-name" in caplog.text
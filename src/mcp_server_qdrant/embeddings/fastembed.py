import asyncio
import logging
from typing import List, Dict, Any

from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.common.exceptions import ModelValidationError

logger = logging.getLogger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._validate_model(model_name)
        self.embedding_model = TextEmbedding(model_name)
    
    def _validate_model(self, model_name: str) -> None:
        """Validate that the specified model is available in FastEmbed."""
        try:
            # Try to get model description to validate model exists
            TextEmbedding._get_model_description(model_name)
            logger.info(f"Successfully validated embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Model validation failed for '{model_name}': {e}")
            
            # Get available models for better error message
            try:
                available_models = [model['model'] for model in TextEmbedding.list_supported_models()]
                
                # Try to suggest a similar model
                suggestion = self._suggest_similar_model(model_name, available_models)
                
                raise ModelValidationError(
                    model_name=model_name,
                    available_models=available_models,
                    suggestion=suggestion
                ) from e
            except ModelValidationError:
                # Re-raise ModelValidationError as-is
                raise
            except Exception as list_error:
                logger.error(f"Failed to get available models: {list_error}")
                raise ModelValidationError(
                    model_name=model_name,
                    available_models=[],
                    suggestion="Check FastEmbed documentation for supported models"
                ) from e
    
    def _suggest_similar_model(self, invalid_model: str, available_models: List[str]) -> str:
        """Suggest a similar model based on the invalid model name."""
        invalid_lower = invalid_model.lower()
        
        # Common model name patterns and their suggestions
        suggestions = {
            'nomic': 'nomic-ai/nomic-embed-text-v1.5-Q',
            'sentence-transformers': 'sentence-transformers/all-MiniLM-L6-v2',
            'all-minilm': 'sentence-transformers/all-MiniLM-L6-v2',
            'bge': 'BAAI/bge-small-en-v1.5',
            'e5': 'intfloat/e5-small-v2',
        }
        
        # Check for partial matches
        for pattern, suggestion in suggestions.items():
            if pattern in invalid_lower and suggestion in available_models:
                return f"Did you mean '{suggestion}'?"
        
        # If no pattern match, suggest the default
        default_model = 'nomic-ai/nomic-embed-text-v1.5-Q'
        if default_model in available_models:
            return f"Try the default model: '{default_model}'"
        
        # Fallback to first available model
        if available_models:
            return f"Try using: '{available_models[0]}'"
        
        return "Check FastEmbed documentation for supported models"

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []
        
        try:
            # Run in a thread pool since FastEmbed is synchronous
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(self.embedding_model.passage_embed(documents))
            )
            result = [embedding.tolist() for embedding in embeddings]
            logger.debug(f"Successfully embedded {len(documents)} documents")
            return result
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise RuntimeError(f"Document embedding failed: {str(e)}") from e

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Run in a thread pool since FastEmbed is synchronous
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: list(self.embedding_model.query_embed([query]))
            )
            result = embeddings[0].tolist()
            logger.debug(f"Successfully embedded query: '{query[:50]}...'")
            return result
        except Exception as e:
            logger.error(f"Failed to embed query '{query[:50]}...': {e}")
            raise RuntimeError(f"Query embedding failed: {str(e)}") from e

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        try:
            model_description: DenseModelDescription = (
                self.embedding_model._get_model_description(self.model_name)
            )
            return model_description.dim
        except Exception as e:
            logger.error(f"Failed to get vector size for model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not determine vector size for model '{self.model_name}': {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the current model."""
        try:
            model_description: DenseModelDescription = (
                self.embedding_model._get_model_description(self.model_name)
            )
            return {
                "model_name": self.model_name,
                "vector_size": model_description.dim,
                "vector_name": self.get_vector_name(),
                "description": getattr(model_description, 'description', 'No description available'),
                "status": "loaded"
            }
        except Exception as e:
            logger.error(f"Failed to get model info for '{self.model_name}': {e}")
            return {
                "model_name": self.model_name,
                "vector_size": None,
                "vector_name": self.get_vector_name(),
                "description": f"Error getting model info: {str(e)}",
                "status": "error"
            }

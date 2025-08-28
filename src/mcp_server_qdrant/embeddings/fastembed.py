import asyncio

from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


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
        except Exception as e:
            available_models = [model['model'] for model in TextEmbedding.list_supported_models()]
            raise ValueError(
                f"Invalid embedding model '{model_name}'. "
                f"Model is not supported by FastEmbed. "
                f"Available models: {', '.join(available_models[:10])}..."
                f" (showing first 10 of {len(available_models)} available models)"
            ) from e

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        model_description: DenseModelDescription = (
            self.embedding_model._get_model_description(self.model_name)
        )
        return model_description.dim

    def get_model_info(self) -> dict[str, any]:
        """Get detailed information about the current model."""
        model_description: DenseModelDescription = (
            self.embedding_model._get_model_description(self.model_name)
        )
        return {
            "model_name": self.model_name,
            "vector_size": model_description.dim,
            "vector_name": self.get_vector_name(),
            "description": getattr(model_description, 'description', 'No description available'),
        }

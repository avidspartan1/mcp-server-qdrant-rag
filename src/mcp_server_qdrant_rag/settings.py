from typing import Literal
import logging

from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic_settings import BaseSettings

from mcp_server_qdrant_rag.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant_rag.common.exceptions import ConfigurationValidationError

logger = logging.getLogger(__name__)

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Store information in the knowledge base for later retrieval and reference."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Search and retrieve information from the knowledge base. Use this tool when you need to: \n"
    " - Find relevant documents or content by semantic similarity \n"
    " - Access stored information for analysis or reference \n"
    " - Query the knowledge base using natural language"
)
DEFAULT_TOOL_HYBRID_FIND_DESCRIPTION = (
    "Advanced hybrid search of the knowledge base combining semantic similarity and keyword matching. "
    "Use this tool when you need: \n"
    " - Best search results by combining meaning and exact word matches \n"
    " - More precise results than semantic search alone \n"
    " - To find content that matches both concepts and specific terms \n"
    " - Superior search quality using RRF or DBSF fusion methods"
)

METADATA_PATH = "metadata"


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_hybrid_find_description: str = Field(
        default=DEFAULT_TOOL_HYBRID_FIND_DESCRIPTION,
        validation_alias="TOOL_HYBRID_FIND_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider and document chunking.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5-Q",
        validation_alias="EMBEDDING_MODEL",
    )
    
    # Chunking configuration
    enable_chunking: bool = Field(
        default=True,
        validation_alias="ENABLE_CHUNKING",
        description="Enable automatic document chunking for large documents",
    )
    max_chunk_size: int = Field(
        default=512,
        validation_alias="MAX_CHUNK_SIZE",
        description="Maximum size of document chunks in tokens/characters",
    )
    chunk_overlap: int = Field(
        default=50,
        validation_alias="CHUNK_OVERLAP",
        description="Number of tokens/characters to overlap between chunks",
    )
    chunk_strategy: str = Field(
        default="semantic",
        validation_alias="CHUNK_STRATEGY",
        description="Chunking strategy: 'semantic', 'fixed', or 'sentence'",
    )

    @field_validator("max_chunk_size")
    @classmethod
    def validate_max_chunk_size(cls, v: int) -> int:
        """Validate that max_chunk_size is within reasonable bounds."""
        logger.debug(f"Validating max_chunk_size: {v}")
        
        if v < 50:
            logger.error(f"max_chunk_size validation failed: {v} is below minimum (50)")
            raise ConfigurationValidationError(
                field_name="max_chunk_size",
                invalid_value=v,
                validation_error="max_chunk_size must be at least 50 tokens/characters",
                valid_options=None,
                suggested_value=512
            )
        if v > 8192:
            logger.error(f"max_chunk_size validation failed: {v} exceeds maximum (8192)")
            raise ConfigurationValidationError(
                field_name="max_chunk_size",
                invalid_value=v,
                validation_error="max_chunk_size must not exceed 8192 tokens/characters",
                valid_options=None,
                suggested_value=2048
            )
        
        logger.debug(f"max_chunk_size validation passed: {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        """Validate that chunk_overlap is non-negative and reasonable."""
        logger.debug(f"Validating chunk_overlap: {v}")
        
        if v < 0:
            logger.error(f"chunk_overlap validation failed: {v} is negative")
            raise ConfigurationValidationError(
                field_name="chunk_overlap",
                invalid_value=v,
                validation_error="chunk_overlap must be non-negative",
                valid_options=None,
                suggested_value=50
            )
        if v > 1000:
            logger.error(f"chunk_overlap validation failed: {v} exceeds maximum (1000)")
            raise ConfigurationValidationError(
                field_name="chunk_overlap",
                invalid_value=v,
                validation_error="chunk_overlap must not exceed 1000 tokens/characters",
                valid_options=None,
                suggested_value=100
            )
        
        logger.debug(f"chunk_overlap validation passed: {v}")
        return v

    @field_validator("chunk_strategy")
    @classmethod
    def validate_chunk_strategy(cls, v: str) -> str:
        """Validate that chunk_strategy is one of the supported strategies."""
        logger.debug(f"Validating chunk_strategy: {v}")
        
        allowed_strategies = {"semantic", "fixed", "sentence"}
        if v not in allowed_strategies:
            logger.error(f"chunk_strategy validation failed: '{v}' not in {allowed_strategies}")
            raise ConfigurationValidationError(
                field_name="chunk_strategy",
                invalid_value=v,
                validation_error=f"chunk_strategy must be one of the supported strategies",
                valid_options=list(allowed_strategies),
                suggested_value="semantic"
            )
        
        logger.debug(f"chunk_strategy validation passed: {v}")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap_vs_size(self) -> "EmbeddingProviderSettings":
        """Validate that chunk_overlap is not larger than max_chunk_size."""
        logger.debug(f"Validating chunk_overlap ({self.chunk_overlap}) vs max_chunk_size ({self.max_chunk_size})")
        
        if self.chunk_overlap >= self.max_chunk_size:
            suggested_overlap = max(10, self.max_chunk_size // 10)  # 10% of chunk size
            logger.error(f"chunk_overlap validation failed: {self.chunk_overlap} >= {self.max_chunk_size}")
            raise ConfigurationValidationError(
                field_name="chunk_overlap",
                invalid_value=self.chunk_overlap,
                validation_error=f"chunk_overlap ({self.chunk_overlap}) must be smaller than max_chunk_size ({self.max_chunk_size})",
                valid_options=None,
                suggested_value=suggested_overlap
            )
        
        overlap_percentage = (self.chunk_overlap / self.max_chunk_size) * 100
        logger.debug(f"chunk_overlap validation passed: {overlap_percentage:.1f}% overlap ratio")
        return self


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self

from typing import Literal, Dict, List, Optional
import logging
import json
from pathlib import Path

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


class SetConfiguration(BaseModel):
    """Configuration for a single set."""
    slug: str = Field(description="Unique identifier for the set")
    description: str = Field(description="Human-readable description")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        """Validate that slug is a valid identifier."""
        # Trim whitespace first
        v = v.strip() if v else ""
        
        if not v:
            raise ValueError("slug cannot be empty")
        
        # Allow alphanumeric, underscore, and hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("slug must contain only alphanumeric characters, underscores, and hyphens")
        
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate that description is not empty."""
        # Trim whitespace first
        v = v.strip() if v else ""
        
        if not v:
            raise ValueError("description cannot be empty")
        return v


class SetSettings(BaseSettings):
    """Settings for set configuration management."""
    config_file_path: str = Field(
        default=".qdrant_sets.json",
        validation_alias="QDRANT_SETS_CONFIG",
        description="Path to sets configuration file"
    )
    sets: Dict[str, SetConfiguration] = Field(default_factory=dict)
    
    def get_config_file_path(self, override_path: Optional[str] = None) -> Path:
        """
        Get the absolute path to the configuration file.
        
        Args:
            override_path: Command-line override path (highest precedence)
            
        Returns:
            Resolved path to configuration file
        """
        if override_path:
            return Path(override_path).resolve()
        
        config_path = Path(self.config_file_path)
        if config_path.is_absolute():
            return config_path
        else:
            return Path.cwd() / config_path
    
    def load_from_file(self, file_path: Optional[Path] = None) -> None:
        """
        Load set configurations from a JSON file.
        
        Args:
            file_path: Path to configuration file. If None, uses get_config_file_path()
        """
        if file_path is None:
            file_path = self.get_config_file_path()
        
        logger.debug(f"Loading set configuration from: {file_path}")
        
        try:
            if not file_path.exists():
                logger.info(f"Configuration file not found at {file_path}, creating default configuration")
                self.create_default_config(file_path)
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Validate the configuration structure
            if not isinstance(config_data, dict):
                raise ValueError("Configuration file must contain a JSON object")
            
            sets_data = config_data.get('sets', {})
            if not isinstance(sets_data, dict):
                raise ValueError("'sets' field must be a JSON object")
            
            # Load and validate each set configuration
            validated_sets = {}
            for slug, set_data in sets_data.items():
                try:
                    # Validate the set data first, then ensure slug matches the key
                    if isinstance(set_data, dict):
                        # Only set slug if it's missing, don't override existing values
                        if 'slug' not in set_data:
                            set_data['slug'] = slug
                    
                    set_config = SetConfiguration.model_validate(set_data)
                    validated_sets[slug] = set_config
                    logger.debug(f"Loaded set configuration: {slug}")
                except Exception as e:
                    logger.warning(f"Skipping invalid set configuration '{slug}': {e}")
                    continue
            
            self.sets = validated_sets
            logger.info(f"Successfully loaded {len(self.sets)} set configurations")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {file_path}: {e}")
            logger.info("Creating default configuration due to JSON error")
            self.create_default_config(file_path)
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            logger.info("Using empty configuration due to load error")
            self.sets = {}
    
    def create_default_config(self, file_path: Path) -> None:
        """
        Create a default configuration file with common examples.
        
        Args:
            file_path: Path where to create the default configuration
        """
        default_sets = {
            "platform_code": SetConfiguration(
                slug="platform_code",
                description="Platform Codebase",
                aliases=["platform", "core platform", "main codebase", "backend"]
            ),
            "api_docs": SetConfiguration(
                slug="api_docs",
                description="API Documentation",
                aliases=["api", "documentation", "api reference", "docs"]
            ),
            "frontend_code": SetConfiguration(
                slug="frontend_code",
                description="Frontend Application Code",
                aliases=["frontend", "ui", "client", "web app"]
            ),
            "database_schema": SetConfiguration(
                slug="database_schema",
                description="Database Schema and Migrations",
                aliases=["database", "schema", "migrations", "db"]
            ),
            "deployment_config": SetConfiguration(
                slug="deployment_config",
                description="Deployment and Infrastructure Configuration",
                aliases=["deployment", "infrastructure", "config", "devops"]
            )
        }
        
        config_data = {
            "version": "1.0",
            "sets": {
                slug: {
                    "slug": set_config.slug,
                    "description": set_config.description,
                    "aliases": set_config.aliases
                }
                for slug, set_config in default_sets.items()
            }
        }
        
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.sets = default_sets
            logger.info(f"Created default set configuration at {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default configuration at {file_path}: {e}")
            # Use the default sets in memory even if we can't write to file
            self.sets = default_sets
    
    def save_to_file(self, file_path: Optional[Path] = None) -> None:
        """
        Save current set configurations to a JSON file.
        
        Args:
            file_path: Path to save configuration file. If None, uses get_config_file_path()
        """
        if file_path is None:
            file_path = self.get_config_file_path()
        
        config_data = {
            "version": "1.0",
            "sets": {
                slug: {
                    "slug": set_config.slug,
                    "description": set_config.description,
                    "aliases": set_config.aliases
                }
                for slug, set_config in self.sets.items()
            }
        }
        
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved set configuration to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise


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

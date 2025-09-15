from typing import Literal, Dict, List, Optional, Any
import logging
import json
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator, field_validator, ValidationError
from pydantic_settings import BaseSettings

from mcp_server_qdrant_rag.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant_rag.common.exceptions import ConfigurationValidationError

logger = logging.getLogger(__name__)


class SetConfigurationError(Exception):
    """Base exception for set configuration errors."""
    pass


class SetConfigurationFileError(SetConfigurationError):
    """Raised when set configuration file operations fail."""
    
    def __init__(self, file_path: Path, operation: str, original_error: Exception):
        self.file_path = file_path
        self.operation = operation
        self.original_error = original_error
        
        message = f"Failed to {operation} set configuration file '{file_path}': {str(original_error)}"
        super().__init__(message)


class SetConfigurationValidationError(SetConfigurationError):
    """Raised when set configuration validation fails."""
    
    def __init__(self, file_path: Path, validation_errors: List[str], valid_sets: int = 0):
        self.file_path = file_path
        self.validation_errors = validation_errors
        self.valid_sets = valid_sets
        
        message = f"Set configuration validation failed for '{file_path}': {'; '.join(validation_errors)}"
        if valid_sets > 0:
            message += f" ({valid_sets} valid sets loaded)"
        super().__init__(message)


class MetadataValidationError(Exception):
    """Raised when metadata field validation fails."""
    
    def __init__(self, field_name: str, field_value: Any, validation_error: str, suggestions: Optional[List[str]] = None):
        self.field_name = field_name
        self.field_value = field_value
        self.validation_error = validation_error
        self.suggestions = suggestions or []
        
        message = f"Invalid {field_name}: {validation_error}"
        if suggestions:
            message += f" Suggestions: {', '.join(suggestions)}"
        super().__init__(message)

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
        Load set configurations from a JSON file with comprehensive error handling.
        
        Args:
            file_path: Path to configuration file. If None, uses get_config_file_path()
            
        Raises:
            SetConfigurationFileError: When file operations fail
            SetConfigurationValidationError: When configuration validation fails
        """
        if file_path is None:
            file_path = self.get_config_file_path()
        
        logger.debug(f"Loading set configuration from: {file_path}")
        
        # Check file accessibility
        try:
            self._validate_file_access(file_path)
        except PermissionError as e:
            logger.error(f"Permission denied accessing configuration file: {file_path}")
            raise SetConfigurationFileError(file_path, "access", e)
        except Exception as e:
            logger.error(f"File access error for configuration file: {file_path}")
            raise SetConfigurationFileError(file_path, "access", e)
        
        # Handle missing file
        if not file_path.exists():
            logger.info(f"Configuration file not found at {file_path}, creating default configuration")
            try:
                self.create_default_config(file_path)
                return
            except Exception as e:
                logger.error(f"Failed to create default configuration: {e}")
                # Fall back to empty configuration
                self.sets = {}
                return
        
        # Load and parse configuration file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {file_path}: {e}")
            error_msg = f"JSON parsing error at line {e.lineno}, column {e.colno}: {e.msg}"
            
            # Try to create backup and default config
            try:
                self._backup_corrupted_config(file_path)
                logger.info("Creating default configuration due to JSON error")
                self.create_default_config(file_path)
                return
            except Exception as backup_error:
                logger.error(f"Failed to create backup or default config: {backup_error}")
                raise SetConfigurationFileError(file_path, "parse", e)
        except Exception as e:
            logger.error(f"Error reading configuration file {file_path}: {e}")
            raise SetConfigurationFileError(file_path, "read", e)
        
        # Validate configuration structure
        validation_errors = []
        try:
            self._validate_config_structure(config_data, validation_errors)
        except Exception as e:
            validation_errors.append(f"Structure validation failed: {str(e)}")
        
        if validation_errors:
            logger.error(f"Configuration structure validation failed: {validation_errors}")
            # Try to use partial configuration if possible
            sets_data = config_data.get('sets', {}) if isinstance(config_data, dict) else {}
        else:
            sets_data = config_data.get('sets', {})
        
        # Load and validate individual set configurations
        validated_sets = {}
        set_validation_errors = []
        
        if isinstance(sets_data, dict):
            for slug, set_data in sets_data.items():
                try:
                    validated_set = self._validate_and_load_set(slug, set_data)
                    validated_sets[slug] = validated_set
                    logger.debug(f"Loaded set configuration: {slug}")
                except Exception as e:
                    error_msg = f"Invalid set '{slug}': {str(e)}"
                    set_validation_errors.append(error_msg)
                    logger.warning(error_msg)
                    continue
        
        # Update sets with validated configurations
        self.sets = validated_sets
        
        # Report results
        total_errors = len(validation_errors) + len(set_validation_errors)
        if total_errors > 0:
            all_errors = validation_errors + set_validation_errors
            logger.warning(f"Configuration loaded with {total_errors} errors, {len(validated_sets)} valid sets")
            
            # Only raise exception if no valid sets were loaded and there were structural errors
            # For structural errors with no valid sets, fall back to default configuration
            if len(validated_sets) == 0 and validation_errors:
                logger.info("Falling back to default configuration due to structural errors")
                try:
                    self.create_default_config(file_path)
                    return
                except Exception as e:
                    logger.error(f"Failed to create default configuration: {e}")
                    # If we can't create default config, raise the validation error
                    raise SetConfigurationValidationError(file_path, all_errors, len(validated_sets))
        else:
            logger.info(f"Successfully loaded {len(validated_sets)} set configurations")
    
    def _validate_file_access(self, file_path: Path) -> None:
        """
        Validate file access permissions.
        
        Args:
            file_path: Path to validate
            
        Raises:
            PermissionError: When file access is denied
            ValueError: When path is invalid
        """
        # Check if parent directory exists and is accessible
        parent_dir = file_path.parent
        if not parent_dir.exists():
            # Try to create parent directory
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise PermissionError(f"Cannot create parent directory: {parent_dir}")
        
        # Check directory permissions
        if not os.access(parent_dir, os.R_OK | os.W_OK):
            raise PermissionError(f"Insufficient permissions for directory: {parent_dir}")
        
        # Check file permissions if it exists
        if file_path.exists():
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Cannot read configuration file: {file_path}")
    
    def _validate_config_structure(self, config_data: Any, validation_errors: List[str]) -> None:
        """
        Validate the overall configuration file structure.
        
        Args:
            config_data: Parsed configuration data
            validation_errors: List to append validation errors to
        """
        if not isinstance(config_data, dict):
            validation_errors.append("Configuration file must contain a JSON object")
            return
        
        # Check for required fields
        if 'sets' not in config_data:
            validation_errors.append("Configuration must contain a 'sets' field")
            return
        
        sets_data = config_data['sets']
        if not isinstance(sets_data, dict):
            validation_errors.append("'sets' field must be a JSON object")
            return
        
        # Validate version if present
        if 'version' in config_data:
            version = config_data['version']
            if not isinstance(version, str):
                validation_errors.append("'version' field must be a string")
            elif version not in ['1.0']:
                validation_errors.append(f"Unsupported configuration version: {version}")
    
    def _validate_and_load_set(self, slug: str, set_data: Any) -> SetConfiguration:
        """
        Validate and load a single set configuration.
        
        Args:
            slug: Set slug from configuration key
            set_data: Set configuration data
            
        Returns:
            Validated SetConfiguration object
            
        Raises:
            ValueError: When set configuration is invalid
        """
        if not isinstance(set_data, dict):
            raise ValueError("Set configuration must be a JSON object")
        
        # Make a copy to avoid modifying original
        set_data_copy = dict(set_data)
        
        # Ensure slug matches the key
        if 'slug' not in set_data_copy:
            set_data_copy['slug'] = slug
        elif set_data_copy['slug'] != slug:
            # If slugs don't match, check if the provided slug is valid
            if not set_data_copy['slug'] or not set_data_copy['slug'].strip():
                logger.warning(f"Set has empty slug, using key '{slug}' instead.")
                set_data_copy['slug'] = slug
            else:
                logger.warning(f"Set slug mismatch: key='{slug}', slug='{set_data_copy['slug']}'. Using key value.")
                set_data_copy['slug'] = slug
        
        # Validate using Pydantic model - this will catch empty slugs and other validation errors
        try:
            return SetConfiguration.model_validate(set_data_copy)
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                error_details.append(f"{field}: {error['msg']}")
            raise ValueError(f"Validation failed: {'; '.join(error_details)}")
    
    def _backup_corrupted_config(self, file_path: Path) -> None:
        """
        Create a backup of corrupted configuration file.
        
        Args:
            file_path: Path to corrupted configuration file
        """
        if not file_path.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f".backup_{timestamp}.json")
        
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup of corrupted configuration: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup of corrupted configuration: {e}")
    
    def create_default_config(self, file_path: Path) -> None:
        """
        Create a default configuration file with common examples and comprehensive error handling.
        
        Args:
            file_path: Path where to create the default configuration
            
        Raises:
            SetConfigurationFileError: When file creation fails
        """
        default_sets = {
            "platform_code": SetConfiguration(
                slug="platform_code",
                description="Platform Codebase",
                aliases=["platform", "core platform", "main codebase", "backend", "server code"]
            ),
            "api_docs": SetConfiguration(
                slug="api_docs",
                description="API Documentation",
                aliases=["api", "documentation", "api reference", "docs", "api guide", "endpoints"]
            ),
            "frontend_code": SetConfiguration(
                slug="frontend_code",
                description="Frontend Application Code",
                aliases=["frontend", "ui", "client", "web app", "user interface", "react", "vue", "angular"]
            ),
            "database_schema": SetConfiguration(
                slug="database_schema",
                description="Database Schema and Migrations",
                aliases=["database", "schema", "migrations", "db", "sql", "tables", "models"]
            ),
            "deployment_config": SetConfiguration(
                slug="deployment_config",
                description="Deployment and Infrastructure Configuration",
                aliases=["deployment", "infrastructure", "config", "devops", "docker", "kubernetes", "ci/cd"]
            ),
            "test_code": SetConfiguration(
                slug="test_code",
                description="Test Code and Test Documentation",
                aliases=["tests", "testing", "unit tests", "integration tests", "test cases", "specs"]
            ),
            "project_docs": SetConfiguration(
                slug="project_docs",
                description="Project Documentation and README Files",
                aliases=["readme", "documentation", "project info", "getting started", "guides", "tutorials"]
            ),
            "configuration": SetConfiguration(
                slug="configuration",
                description="Application Configuration Files",
                aliases=["config", "settings", "environment", "env", "properties", "yaml", "json config"]
            )
        }
        
        config_data = {
            "version": "1.0",
            "description": "Set configurations for semantic filtering in Qdrant RAG server",
            "_documentation": {
                "overview": "This file defines sets for organizing and filtering documents in your knowledge base. Each set represents a logical grouping of documents that can be filtered during search operations.",
                "usage": {
                    "ingestion": "Use --set parameter in CLI or set_id in MCP tools to assign documents to sets",
                    "search": "Use set_filter parameter in search tools with natural language descriptions",
                    "examples": [
                        "Search with: set_filter='platform code' matches 'platform_code' set",
                        "Search with: set_filter='api documentation' matches 'api_docs' set",
                        "Search with: set_filter='frontend' matches 'frontend_code' set"
                    ]
                },
                "configuration": {
                    "slug": "Unique identifier for the set (alphanumeric, underscores, hyphens only)",
                    "description": "Human-readable description used for semantic matching",
                    "aliases": "Alternative names that can be used to reference this set"
                },
                "validation_rules": {
                    "slug": "Must be non-empty, contain only [a-zA-Z0-9_-], max 50 characters",
                    "description": "Must be non-empty after trimming whitespace",
                    "aliases": "Optional list of strings for alternative references"
                },
                "examples": {
                    "minimal_set": {
                        "slug": "my_docs",
                        "description": "My Personal Documents",
                        "aliases": []
                    },
                    "full_set": {
                        "slug": "project_specs",
                        "description": "Project Specifications and Requirements",
                        "aliases": ["specs", "requirements", "project docs", "specifications"]
                    }
                }
            },
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
            # Validate file path security
            self._validate_file_path_security(file_path)
            
            # Ensure parent directory exists with proper permissions
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise SetConfigurationFileError(file_path, "create parent directory", e)
            
            # Write configuration file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            except PermissionError as e:
                raise SetConfigurationFileError(file_path, "write", e)
            except OSError as e:
                raise SetConfigurationFileError(file_path, "write", e)
            
            # Verify file was created successfully
            if not file_path.exists():
                raise SetConfigurationFileError(file_path, "verify creation", 
                                              Exception("File was not created"))
            
            # Verify file is readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
            except Exception as e:
                raise SetConfigurationFileError(file_path, "verify readability", e)
            
            self.sets = default_sets
            logger.info(f"Created default set configuration at {file_path} with {len(default_sets)} sets")
            
        except SetConfigurationFileError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating default configuration at {file_path}: {e}")
            # Use the default sets in memory even if we can't write to file
            self.sets = default_sets
            raise SetConfigurationFileError(file_path, "create", e)
    
    def _validate_file_path_security(self, file_path: Path) -> None:
        """
        Validate file path for security concerns.
        
        Args:
            file_path: Path to validate
            
        Raises:
            ValueError: When path is potentially unsafe
        """
        # Resolve path to check for directory traversal
        resolved_path = file_path.resolve()
        
        # Check if path tries to escape current working directory
        try:
            resolved_path.relative_to(Path.cwd())
        except ValueError:
            # Allow absolute paths that are explicitly set
            if not file_path.is_absolute():
                raise ValueError(f"Path appears to use directory traversal: {file_path}")
        
        # Check for suspicious path components
        suspicious_components = ['..', '.', '~']
        for component in file_path.parts:
            if component in suspicious_components and len(file_path.parts) > 1:
                logger.warning(f"Potentially suspicious path component '{component}' in {file_path}")
        
        # Ensure file extension is appropriate
        if file_path.suffix.lower() not in ['.json', '.jsonc']:
            logger.warning(f"Unusual file extension for configuration: {file_path.suffix}")
    
    def save_to_file(self, file_path: Optional[Path] = None) -> None:
        """
        Save current set configurations to a JSON file.
        
        Args:
            file_path: Path to save configuration file. If None, uses get_config_file_path()
            
        Raises:
            SetConfigurationFileError: When file operations fail
        """
        if file_path is None:
            file_path = self.get_config_file_path()
        
        config_data = {
            "version": "1.0",
            "description": "Set configurations for semantic filtering in Qdrant RAG server",
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
            # Validate file path security
            self._validate_file_path_security(file_path)
            
            # Ensure parent directory exists
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise SetConfigurationFileError(file_path, "create parent directory", e)
            
            # Write configuration file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            except PermissionError as e:
                raise SetConfigurationFileError(file_path, "write", e)
            except OSError as e:
                raise SetConfigurationFileError(file_path, "write", e)
            
            logger.info(f"Saved set configuration to {file_path}")
            
        except SetConfigurationFileError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving configuration to {file_path}: {e}")
            raise SetConfigurationFileError(file_path, "save", e)


def validate_document_type(document_type: Optional[str]) -> Optional[str]:
    """
    Validate document_type metadata field.
    
    Args:
        document_type: Document type value to validate
        
    Returns:
        Validated document type or None
        
    Raises:
        MetadataValidationError: When validation fails
    """
    if document_type is None:
        return None
    
    if not isinstance(document_type, str):
        raise MetadataValidationError(
            "document_type", 
            document_type, 
            "must be a string",
            ["Use a string value like 'code', 'documentation', 'config'"]
        )
    
    # Trim whitespace
    document_type = document_type.strip()
    
    if not document_type:
        raise MetadataValidationError(
            "document_type",
            document_type,
            "cannot be empty or whitespace only",
            ["Use a descriptive type like 'code', 'documentation', 'config'"]
        )
    
    # Check length limits
    if len(document_type) > 100:
        raise MetadataValidationError(
            "document_type",
            document_type,
            "cannot exceed 100 characters",
            ["Use a shorter, more concise document type"]
        )
    
    # Check for potentially problematic characters
    import re
    if not re.match(r'^[a-zA-Z0-9_\-\s\.]+$', document_type):
        raise MetadataValidationError(
            "document_type",
            document_type,
            "contains invalid characters (only alphanumeric, underscore, hyphen, space, and period allowed)",
            ["Use only letters, numbers, underscores, hyphens, spaces, and periods"]
        )
    
    # Normalize common variations
    normalized_type = document_type.lower().replace(' ', '_').replace('-', '_')
    
    # Suggest common document types if the input seems unusual
    common_types = {
        'code', 'documentation', 'config', 'api', 'database', 'frontend', 
        'backend', 'test', 'deployment', 'infrastructure', 'schema'
    }
    
    if normalized_type not in common_types and len(normalized_type.split('_')) == 1:
        suggestions = [t for t in common_types if t.startswith(normalized_type[:3])]
        if suggestions:
            logger.info(f"Document type '{document_type}' is valid but uncommon. "
                       f"Consider using: {', '.join(suggestions)}")
    
    return document_type


def validate_set_id(set_id: Optional[str], available_sets: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Validate set_id metadata field.
    
    Args:
        set_id: Set ID value to validate
        available_sets: Dictionary of available sets for validation (optional)
        
    Returns:
        Validated set ID or None
        
    Raises:
        MetadataValidationError: When validation fails
    """
    if set_id is None:
        return None
    
    if not isinstance(set_id, str):
        raise MetadataValidationError(
            "set_id",
            set_id,
            "must be a string",
            ["Use a string identifier like 'platform_code', 'api_docs'"]
        )
    
    # Trim whitespace
    set_id = set_id.strip()
    
    if not set_id:
        raise MetadataValidationError(
            "set_id",
            set_id,
            "cannot be empty or whitespace only",
            ["Use a valid set identifier"]
        )
    
    # Check length limits
    if len(set_id) > 50:
        raise MetadataValidationError(
            "set_id",
            set_id,
            "cannot exceed 50 characters",
            ["Use a shorter set identifier"]
        )
    
    # Validate format (slug-like)
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]+$', set_id):
        raise MetadataValidationError(
            "set_id",
            set_id,
            "must be a valid slug (only alphanumeric characters, underscores, and hyphens allowed)",
            ["Use format like 'platform_code', 'api-docs', 'frontend_ui'"]
        )
    
    # Check against available sets if provided
    if available_sets is not None:
        if set_id not in available_sets:
            available_list = list(available_sets.keys())
            suggestions = []
            
            # Find similar set IDs
            for available_set in available_list:
                if set_id.lower() in available_set.lower() or available_set.lower() in set_id.lower():
                    suggestions.append(available_set)
            
            if not suggestions and available_list:
                suggestions = available_list[:5]  # Show first 5 as examples
            
            raise MetadataValidationError(
                "set_id",
                set_id,
                f"is not a configured set",
                suggestions if suggestions else ["Configure this set in your sets configuration file"]
            )
    
    return set_id


def validate_metadata_dict(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Validate metadata dictionary for common issues.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        Validated metadata dictionary or None
        
    Raises:
        MetadataValidationError: When validation fails
    """
    if metadata is None:
        return None
    
    if not isinstance(metadata, dict):
        raise MetadataValidationError(
            "metadata",
            metadata,
            "must be a dictionary",
            ["Use a dictionary with string keys and simple values"]
        )
    
    # Check for empty metadata
    if not metadata:
        return None
    
    # Validate keys and values
    validated_metadata = {}
    for key, value in metadata.items():
        # Validate key
        if not isinstance(key, str):
            raise MetadataValidationError(
                f"metadata key",
                key,
                "must be a string",
                ["Use string keys for metadata fields"]
            )
        
        if not key.strip():
            raise MetadataValidationError(
                f"metadata key",
                key,
                "cannot be empty or whitespace only",
                ["Use descriptive string keys"]
            )
        
        # Validate value types
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise MetadataValidationError(
                f"metadata['{key}']",
                value,
                "must be a string, number, boolean, or null",
                ["Use simple data types for metadata values"]
            )
        
        # Check string value length
        if isinstance(value, str) and len(value) > 1000:
            raise MetadataValidationError(
                f"metadata['{key}']",
                value,
                "string value cannot exceed 1000 characters",
                ["Use shorter string values or store large content separately"]
            )
        
        validated_metadata[key.strip()] = value
    
    return validated_metadata
    
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
    
    enable_semantic_set_matching: bool = Field(
        default=False, validation_alias="QDRANT_ENABLE_SEMANTIC_SET_MATCHING"
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

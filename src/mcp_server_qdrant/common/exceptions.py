"""Custom exceptions for the MCP Qdrant server with detailed error messages."""

from typing import List, Optional, Any, Dict


class MCPQdrantError(Exception):
    """Base exception for all MCP Qdrant server errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelValidationError(MCPQdrantError):
    """Raised when embedding model validation fails."""
    
    def __init__(
        self, 
        model_name: str, 
        available_models: Optional[List[str]] = None,
        suggestion: Optional[str] = None
    ):
        self.model_name = model_name
        self.available_models = available_models or []
        self.suggestion = suggestion
        
        message = f"Invalid embedding model '{model_name}'. Model is not supported by FastEmbed."
        
        if available_models:
            if len(available_models) <= 10:
                message += f" Available models: {', '.join(available_models)}"
            else:
                message += f" Available models include: {', '.join(available_models[:10])}... (showing first 10 of {len(available_models)} available models)"
        
        if suggestion:
            message += f" Suggestion: {suggestion}"
        
        details = {
            "invalid_model": model_name,
            "available_models": available_models,
            "suggestion": suggestion
        }
        
        super().__init__(message, details)


class VectorDimensionMismatchError(MCPQdrantError):
    """Raised when vector dimensions don't match between model and collection."""
    
    def __init__(
        self,
        collection_name: str,
        expected_dimensions: int,
        actual_dimensions: int,
        model_name: str,
        vector_name: str,
        available_vectors: Optional[List[str]] = None
    ):
        self.collection_name = collection_name
        self.expected_dimensions = expected_dimensions
        self.actual_dimensions = actual_dimensions
        self.model_name = model_name
        self.vector_name = vector_name
        self.available_vectors = available_vectors or []
        
        message = (
            f"Vector dimension mismatch for collection '{collection_name}'. "
            f"Expected {expected_dimensions} dimensions (model: {model_name}, vector: {vector_name}), "
            f"but collection has {actual_dimensions} dimensions."
        )
        
        if available_vectors and vector_name not in available_vectors:
            message += f" Available vectors in collection: {', '.join(available_vectors)}."
        
        message += (
            " This usually indicates a change in embedding model. "
            "Consider using a different collection name or recreating the collection."
        )
        
        details = {
            "collection_name": collection_name,
            "expected_dimensions": expected_dimensions,
            "actual_dimensions": actual_dimensions,
            "model_name": model_name,
            "vector_name": vector_name,
            "available_vectors": available_vectors,
            "resolution_suggestions": [
                f"Use a different collection name (e.g., '{collection_name}_v2')",
                f"Delete and recreate the collection '{collection_name}'",
                f"Switch back to the original embedding model",
                "Check if the model name is correct"
            ]
        }
        
        super().__init__(message, details)


class ChunkingError(MCPQdrantError):
    """Raised when document chunking fails."""
    
    def __init__(
        self,
        original_error: Exception,
        document_length: int,
        chunking_config: Dict[str, Any],
        fallback_used: bool = False
    ):
        self.original_error = original_error
        self.document_length = document_length
        self.chunking_config = chunking_config
        self.fallback_used = fallback_used
        
        message = f"Document chunking failed: {str(original_error)}"
        
        if fallback_used:
            message += " Document was stored as a single entry instead."
        else:
            message += " Consider adjusting chunking parameters or disabling chunking."
        
        details = {
            "original_error": str(original_error),
            "original_error_type": type(original_error).__name__,
            "document_length": document_length,
            "chunking_config": chunking_config,
            "fallback_used": fallback_used,
            "suggestions": [
                "Try increasing max_chunk_size",
                "Try decreasing chunk_overlap",
                "Try a different chunk_strategy",
                "Disable chunking with ENABLE_CHUNKING=false"
            ]
        }
        
        super().__init__(message, details)


class ConfigurationValidationError(MCPQdrantError):
    """Raised when configuration validation fails with suggestions for correction."""
    
    def __init__(
        self,
        field_name: str,
        invalid_value: Any,
        validation_error: str,
        valid_options: Optional[List[Any]] = None,
        suggested_value: Optional[Any] = None
    ):
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.validation_error = validation_error
        self.valid_options = valid_options
        self.suggested_value = suggested_value
        
        message = f"Configuration error for '{field_name}': {validation_error}"
        
        if valid_options:
            message += f" Valid options: {', '.join(map(str, valid_options))}"
        
        if suggested_value:
            message += f" Suggested value: {suggested_value}"
        
        details = {
            "field_name": field_name,
            "invalid_value": invalid_value,
            "validation_error": validation_error,
            "valid_options": valid_options,
            "suggested_value": suggested_value
        }
        
        super().__init__(message, details)


class CollectionAccessError(MCPQdrantError):
    """Raised when collection access fails with helpful context."""
    
    def __init__(
        self,
        collection_name: str,
        operation: str,
        original_error: Exception,
        available_collections: Optional[List[str]] = None
    ):
        self.collection_name = collection_name
        self.operation = operation
        self.original_error = original_error
        self.available_collections = available_collections or []
        
        message = f"Failed to {operation} collection '{collection_name}': {str(original_error)}"
        
        if available_collections:
            if collection_name not in available_collections:
                message += f" Available collections: {', '.join(available_collections)}"
            else:
                message += " Collection exists but operation failed."
        
        details = {
            "collection_name": collection_name,
            "operation": operation,
            "original_error": str(original_error),
            "original_error_type": type(original_error).__name__,
            "available_collections": available_collections
        }
        
        super().__init__(message, details)


class TokenizerError(MCPQdrantError):
    """Raised when tokenizer initialization or operation fails."""
    
    def __init__(
        self,
        tokenizer_name: str,
        original_error: Exception,
        fallback_available: bool = False
    ):
        self.tokenizer_name = tokenizer_name
        self.original_error = original_error
        self.fallback_available = fallback_available
        
        message = f"Tokenizer '{tokenizer_name}' failed: {str(original_error)}"
        
        if fallback_available:
            message += " Using fallback tokenizer."
        else:
            message += " No fallback available."
        
        details = {
            "tokenizer_name": tokenizer_name,
            "original_error": str(original_error),
            "original_error_type": type(original_error).__name__,
            "fallback_available": fallback_available,
            "suggestions": [
                f"Install required dependencies for {tokenizer_name}",
                "Use a different tokenizer",
                "Use the default whitespace tokenizer"
            ]
        }
        
        super().__init__(message, details)


class SentenceSplitterError(MCPQdrantError):
    """Raised when sentence splitter initialization or operation fails."""
    
    def __init__(
        self,
        splitter_name: str,
        original_error: Exception,
        fallback_available: bool = False
    ):
        self.splitter_name = splitter_name
        self.original_error = original_error
        self.fallback_available = fallback_available
        
        message = f"Sentence splitter '{splitter_name}' failed: {str(original_error)}"
        
        if fallback_available:
            message += " Using fallback splitter."
        else:
            message += " No fallback available."
        
        details = {
            "splitter_name": splitter_name,
            "original_error": str(original_error),
            "original_error_type": type(original_error).__name__,
            "fallback_available": fallback_available,
            "suggestions": [
                f"Install required dependencies for {splitter_name}",
                "Use a different sentence splitter",
                "Use the default simple sentence splitter"
            ]
        }
        
        super().__init__(message, details)
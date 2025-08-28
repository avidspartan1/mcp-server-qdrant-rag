"""Data models for document chunking functionality."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentChunk(BaseModel):
    """
    Represents a chunk of a larger document with metadata and relationship information.
    
    This model stores individual chunks created from larger documents, maintaining
    relationships to the source document and other chunks.
    """
    
    content: str = Field(
        ..., 
        description="The text content of this chunk",
        min_length=1
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for this chunk"
    )
    
    # Chunk relationship fields
    chunk_index: int = Field(
        ...,
        description="The index of this chunk within the source document (0-based)",
        ge=0
    )
    
    source_document_id: str = Field(
        ...,
        description="Unique identifier of the source document this chunk belongs to",
        min_length=1
    )
    
    total_chunks: int = Field(
        ...,
        description="Total number of chunks created from the source document",
        ge=1
    )
    
    # Overlap information for context preservation
    overlap_start: int = Field(
        default=0,
        description="Number of characters/tokens overlapping with the previous chunk",
        ge=0
    )
    
    overlap_end: int = Field(
        default=0,
        description="Number of characters/tokens overlapping with the next chunk",
        ge=0
    )
    
    # Chunking strategy information
    chunk_strategy: str = Field(
        default="hybrid",
        description="The strategy used to create this chunk (hybrid, semantic, fixed, sentence)"
    )
    
    @model_validator(mode='after')
    def validate_chunk_index_against_total(self):
        """Ensure chunk_index is less than total_chunks."""
        if self.chunk_index >= self.total_chunks:
            raise ValueError(f"chunk_index ({self.chunk_index}) must be less than total_chunks ({self.total_chunks})")
        return self
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata_structure(cls, v):
        """Validate that metadata follows expected structure for chunks."""
        if v is None:
            return v
            
        # Ensure metadata is a dictionary
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")
            
        # Validate specific chunk metadata fields if present
        if 'original_document_metadata' in v:
            if not isinstance(v['original_document_metadata'], dict):
                raise ValueError("original_document_metadata must be a dictionary")
                
        return v
    
    @field_validator('chunk_strategy')
    @classmethod
    def validate_chunk_strategy(cls, v):
        """Ensure chunk_strategy is one of the supported values."""
        allowed_strategies = {'hybrid', 'semantic', 'fixed', 'sentence'}
        if v not in allowed_strategies:
            raise ValueError(f"chunk_strategy must be one of {allowed_strategies}, got '{v}'")
        return v
    
    def get_chunk_metadata(self) -> Dict[str, Any]:
        """
        Get chunk-specific metadata that should be stored with the vector.
        
        Returns:
            Dictionary containing chunk relationship and processing metadata
        """
        chunk_metadata = {
            "is_chunk": True,
            "source_document_id": self.source_document_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_strategy": self.chunk_strategy,
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end,
        }
        
        # Include original metadata if present
        if self.metadata:
            chunk_metadata["original_document_metadata"] = self.metadata
            
        return chunk_metadata
    
    def is_first_chunk(self) -> bool:
        """Check if this is the first chunk of the document."""
        return self.chunk_index == 0
    
    def is_last_chunk(self) -> bool:
        """Check if this is the last chunk of the document."""
        return self.chunk_index == self.total_chunks - 1
    
    def __str__(self) -> str:
        """String representation showing chunk position and content preview."""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"DocumentChunk({self.chunk_index + 1}/{self.total_chunks}): {content_preview}"
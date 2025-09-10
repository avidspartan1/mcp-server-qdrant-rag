"""
CLI tool for ingesting files into Qdrant vector database.

This module provides a command-line interface for bulk ingestion of text files
into Qdrant collections, leveraging the existing MCP server components for
consistent behavior and configuration.

Example usage:
    qdrant-ingest /path/to/documents --knowledgebase my-docs
    qdrant-ingest update /path/to/documents --mode add-only
    qdrant-ingest remove my-docs --force
    qdrant-ingest list
"""

import argparse
import asyncio
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings

from .settings import QdrantSettings, EmbeddingProviderSettings
from .qdrant import Entry, QdrantConnector
from .embeddings.factory import create_embedding_provider
from .embeddings.base import EmbeddingProvider
from .common.exceptions import (
    MCPQdrantError,
    ModelValidationError,
    VectorDimensionMismatchError,
    ChunkingError,
    ConfigurationValidationError,
    CollectionAccessError,
    TokenizerError,
    SentenceSplitterError,
    BackwardCompatibilityError,
)


@dataclass
class FileInfo:
    """Information about a discovered file for processing."""

    path: Path
    size: int
    modified_time: datetime
    encoding: str = "utf-8"
    is_binary: bool = False
    estimated_tokens: int = 0

    def __post_init__(self):
        """Validate file information after initialization."""
        if not self.path.exists():
            raise ValueError(f"File does not exist: {self.path}")
        if self.size < 0:
            raise ValueError(f"Invalid file size: {self.size}")


@dataclass
class OperationResult:
    """Results from a CLI operation execution."""

    success: bool
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    @property
    def total_files(self) -> int:
        """Total number of files encountered."""
        return self.files_processed + self.files_skipped + self.files_failed

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.files_processed / self.total_files) * 100


class ProgressReporter:
    """
    Handles progress reporting and user feedback for CLI operations.
    
    This class provides functionality for:
    - Progress indicators during file processing
    - Verbose logging and error reporting
    - Batch processing with progress updates
    - Summary reporting with success/failure counts
    """
    
    def __init__(self, show_progress: bool = True, verbose: bool = False, batch_size: int = 10):
        """
        Initialize progress reporter.
        
        Args:
            show_progress: Whether to show progress indicators
            verbose: Whether to show verbose logging
            batch_size: Number of items to process before showing progress update
        """
        self.show_progress = show_progress
        self.verbose = verbose
        self.batch_size = batch_size
        self._current_operation: Optional[str] = None
        self._total_items: int = 0
        self._processed_items: int = 0
        self._start_time: Optional[datetime] = None
        self._last_progress_update: int = 0
    
    def start_operation(self, operation_name: str, total_items: int = 0) -> None:
        """
        Start tracking a new operation.
        
        Args:
            operation_name: Name of the operation being performed
            total_items: Total number of items to process (0 if unknown)
        """
        self._current_operation = operation_name
        self._total_items = total_items
        self._processed_items = 0
        self._start_time = datetime.now()
        self._last_progress_update = 0
        
        if self.show_progress:
            if total_items > 0:
                print(f"🚀 Starting {operation_name} ({total_items} items)")
            else:
                print(f"🚀 Starting {operation_name}")
    
    def update_progress(self, items_processed: int = 1, item_name: Optional[str] = None) -> None:
        """
        Update progress for the current operation.
        
        Args:
            items_processed: Number of items processed in this update
            item_name: Name of the current item being processed (for verbose mode)
        """
        self._processed_items += items_processed
        
        # Show verbose item-level progress
        if self.verbose and item_name:
            self._log_verbose(f"Processing: {item_name}")
        
        # Show batch progress updates
        if self.show_progress and self._should_show_progress_update():
            self._show_progress_update()
            self._last_progress_update = self._processed_items
    
    def _should_show_progress_update(self) -> bool:
        """
        Determine if a progress update should be shown.
        
        Returns:
            True if progress update should be shown
        """
        if self._total_items == 0:
            # For unknown totals, show every batch_size items
            return (self._processed_items - self._last_progress_update) >= self.batch_size
        else:
            # For known totals, show at percentage milestones or batch intervals
            progress_percent = (self._processed_items / self._total_items) * 100
            last_percent = (self._last_progress_update / self._total_items) * 100
            
            # Show at 10% intervals or every batch_size items, whichever comes first
            return (
                (progress_percent - last_percent) >= 10 or
                (self._processed_items - self._last_progress_update) >= self.batch_size
            )
    
    def _show_progress_update(self) -> None:
        """Show a progress update to the user."""
        if self._total_items > 0:
            progress_percent = (self._processed_items / self._total_items) * 100
            elapsed_time = (datetime.now() - self._start_time).total_seconds()
            
            # Estimate remaining time
            if self._processed_items > 0:
                items_per_second = self._processed_items / elapsed_time
                remaining_items = self._total_items - self._processed_items
                estimated_remaining = remaining_items / items_per_second if items_per_second > 0 else 0
                
                print(f"📊 Progress: {self._processed_items}/{self._total_items} "
                      f"({progress_percent:.1f}%) - "
                      f"ETA: {self._format_duration(estimated_remaining)}")
            else:
                print(f"📊 Progress: {self._processed_items}/{self._total_items} ({progress_percent:.1f}%)")
        else:
            elapsed_time = (datetime.now() - self._start_time).total_seconds()
            rate = self._processed_items / elapsed_time if elapsed_time > 0 else 0
            print(f"📊 Processed: {self._processed_items} items "
                  f"({rate:.1f} items/sec)")
    
    def finish_operation(self, result: OperationResult) -> None:
        """
        Finish the current operation and show final summary.
        
        Args:
            result: Final operation result with statistics
        """
        if not self._start_time:
            return
        
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        result.execution_time = elapsed_time
        
        # Show final progress if we were showing progress
        if self.show_progress and self._total_items > 0:
            print(f"📊 Progress: {self._processed_items}/{self._total_items} (100%)")
        
        # Show operation summary
        self._show_operation_summary(result)
        
        # Reset state
        self._current_operation = None
        self._total_items = 0
        self._processed_items = 0
        self._start_time = None
        self._last_progress_update = 0
    
    def _show_operation_summary(self, result: OperationResult) -> None:
        """
        Show a summary of the operation results.
        
        Args:
            result: Operation result to summarize
        """
        print("\n" + "="*60)
        print(f"📋 {self._current_operation or 'Operation'} Summary")
        print("="*60)
        
        # Overall status
        if result.success:
            print("✅ Status: Completed successfully")
        else:
            print("❌ Status: Completed with errors")
        
        # File statistics
        if result.total_files > 0:
            print(f"📁 Files processed: {result.files_processed}")
            print(f"⏭️  Files skipped: {result.files_skipped}")
            if result.files_failed > 0:
                print(f"❌ Files failed: {result.files_failed}")
            print(f"📊 Success rate: {result.success_rate:.1f}%")
        
        # Chunk statistics
        if result.chunks_created > 0:
            print(f"🧩 Chunks created: {result.chunks_created}")
        
        # Timing information
        print(f"⏱️  Execution time: {self._format_duration(result.execution_time)}")
        
        if result.total_files > 0 and result.execution_time > 0:
            rate = result.files_processed / result.execution_time
            print(f"🚀 Processing rate: {rate:.2f} files/second")
        
        # Warnings and errors
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings[:5]:  # Show first 5 warnings
                print(f"   • {warning}")
            if len(result.warnings) > 5:
                print(f"   ... and {len(result.warnings) - 5} more warnings")
        
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(result.errors) > 5:
                print(f"   ... and {len(result.errors) - 5} more errors")
        
        print("="*60)
    
    def report_file_processed(self, file_path: Path, success: bool, error_message: Optional[str] = None) -> None:
        """
        Report that a file has been processed.
        
        Args:
            file_path: Path of the processed file
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        if success:
            if self.verbose:
                self._log_verbose(f"✅ Processed: {file_path}")
        else:
            if self.verbose:
                self._log_verbose(f"❌ Failed: {file_path}")
            if error_message:
                self._log_error(f"Error processing {file_path}: {error_message}")
        
        self.update_progress(1, str(file_path))
    
    def report_file_skipped(self, file_path: Path, reason: str) -> None:
        """
        Report that a file was skipped.
        
        Args:
            file_path: Path of the skipped file
            reason: Reason why the file was skipped
        """
        if self.verbose:
            self._log_verbose(f"⏭️  Skipped: {file_path} ({reason})")
        
        self.update_progress(1, str(file_path))
    
    def report_batch_processed(self, batch_size: int, success_count: int, error_count: int) -> None:
        """
        Report that a batch of items has been processed.
        
        Args:
            batch_size: Size of the processed batch
            success_count: Number of successful items in the batch
            error_count: Number of failed items in the batch
        """
        if self.show_progress:
            if error_count > 0:
                print(f"📦 Batch processed: {success_count}/{batch_size} successful, {error_count} failed")
            else:
                print(f"📦 Batch processed: {batch_size} items successfully")
        
        self.update_progress(batch_size)
    
    def log_info(self, message: str) -> None:
        """
        Log an informational message.
        
        Args:
            message: Message to log
        """
        print(f"📝 {message}")
    
    def log_success(self, message: str) -> None:
        """
        Log a success message.
        
        Args:
            message: Message to log
        """
        print(f"✅ {message}")
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
        """
        print(f"⚠️  {message}")
    
    def log_error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
        """
        print(f"❌ {message}", file=sys.stderr)
    
    def _log_verbose(self, message: str) -> None:
        """
        Log a verbose message if verbose mode is enabled.
        
        Args:
            message: Message to log
        """
        if self.verbose:
            print(f"ℹ️  {message}")
    
    def _log_error(self, message: str) -> None:
        """
        Log an error message to stderr.
        
        Args:
            message: Message to log
        """
        print(f"❌ {message}", file=sys.stderr)
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.0f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"
    
    def show_discovery_stats(self, stats: dict) -> None:
        """
        Show file discovery statistics.
        
        Args:
            stats: Dictionary with discovery statistics
        """
        if stats['total_files'] == 0:
            self.log_warning("No files discovered")
            return
        
        self.log_info(f"Discovered {stats['total_files']} files")
        
        if self.verbose:
            # Show size information
            total_size_mb = stats['total_size'] / (1024 * 1024)
            self._log_verbose(f"Total size: {total_size_mb:.2f} MB")
            self._log_verbose(f"Estimated tokens: {stats['estimated_tokens']:,}")
            
            # Show file type breakdown
            if stats['extensions']:
                self._log_verbose("File types:")
                for ext, count in sorted(stats['extensions'].items()):
                    self._log_verbose(f"  {ext}: {count} files")
            
            # Show encoding breakdown
            if stats['encodings']:
                self._log_verbose("Encodings:")
                for enc, count in sorted(stats['encodings'].items()):
                    self._log_verbose(f"  {enc}: {count} files")
    
    def show_processing_stats(self, stats: dict) -> None:
        """
        Show content processing statistics.
        
        Args:
            stats: Dictionary with processing statistics
        """
        if stats['total_entries'] == 0:
            return
        
        if self.verbose:
            avg_length = stats['average_content_length']
            self._log_verbose(f"Average content length: {avg_length:.0f} characters")
            
            # Show file type breakdown
            if stats['file_types']:
                self._log_verbose("Processed file types:")
                for file_type, count in sorted(stats['file_types'].items()):
                    self._log_verbose(f"  {file_type}: {count} files")
    
    def confirm_operation(self, message: str, force: bool = False) -> bool:
        """
        Ask user for confirmation unless force mode is enabled.
        
        Args:
            message: Confirmation message to display
            force: Whether to skip confirmation (force mode)
            
        Returns:
            True if user confirms or force mode is enabled, False otherwise
        """
        if force:
            self.log_info(f"Force mode enabled: {message}")
            return True
        
        try:
            response = input(f"❓ {message} (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 Operation cancelled by user")
            return False


class CLISettings(BaseSettings):
    """CLI-specific settings that extend existing settings classes."""

    # File processing settings
    include_patterns: List[str] = Field(default_factory=list)
    exclude_patterns: List[str] = Field(default_factory=list)
    supported_extensions: List[str] = Field(
        default=[
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".yaml",
            ".yml",
            ".rst",
            ".tf",
            ".tftpl",
            ".tpl",
            ".java",
            ".sh",
            ".go",
            ".rb",
            ".ts",
            ".conf",
            ".ini",
            ".cfg",
            ".toml",
            ".xml",
            ".html",
            ".css",
            ".sql",
        ]
    )

    # Operation settings
    operation_mode: Literal["ingest", "update", "remove", "list"] = "ingest"
    update_mode: Literal["add-only", "replace"] = "add-only"
    force_operation: bool = False
    dry_run: bool = False

    # Progress settings
    show_progress: bool = True
    batch_size: int = 10
    verbose: bool = False

    def derive_knowledgebase_name(self, path: Path) -> str:
        """
        Derive knowledgebase name from the provided path.
        
        Args:
            path: Path to derive name from
            
        Returns:
            Derived knowledgebase name
            
        Raises:
            ValueError: If path is invalid or name cannot be derived
        """
        if not path or not isinstance(path, Path):
            raise ValueError("Path must be a valid Path object")
            
        if path.is_dir():
            name = path.name
        else:
            name = path.stem  # filename without extension
            
        # Validate the derived name
        if not name or not name.strip():
            raise ValueError(f"Cannot derive valid knowledgebase name from path: {path}")
            
        # Sanitize the name to ensure it's valid for Qdrant collections
        sanitized_name = self._sanitize_knowledgebase_name(name)
        if not sanitized_name:
            raise ValueError(f"Derived name '{name}' cannot be sanitized to valid knowledgebase name")
            
        return sanitized_name
    
    def _sanitize_knowledgebase_name(self, name: str) -> str:
        """
        Sanitize a name to be valid for Qdrant collections.
        
        Args:
            name: Raw name to sanitize
            
        Returns:
            Sanitized name containing only valid characters
        """
        import re
        
        # Replace invalid characters with hyphens
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', name)
        
        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        
        return sanitized


@dataclass
class IngestConfig:
    """
    Complete configuration for CLI operations.
    
    This class combines CLI-specific settings with Qdrant and embedding settings
    to provide a unified configuration interface for all CLI operations.
    """

    cli_settings: CLISettings
    qdrant_settings: QdrantSettings
    embedding_settings: EmbeddingProviderSettings
    target_path: Optional[Path] = None
    knowledgebase_name: Optional[str] = None

    def __post_init__(self):
        """Validate and complete configuration after initialization."""
        self._validate_configuration()
        self._complete_configuration()
    
    def _validate_configuration(self):
        """Validate the configuration for consistency and completeness."""
        # Validate target path for commands that need it
        if self.cli_settings.operation_mode in ["ingest", "update"]:
            if not self.target_path:
                raise ValueError(f"Target path is required for '{self.cli_settings.operation_mode}' operation")
            
            if not self.target_path.exists():
                raise ValueError(f"Target path does not exist: {self.target_path}")
            
            if not (self.target_path.is_file() or self.target_path.is_dir()):
                raise ValueError(f"Target path must be a file or directory: {self.target_path}")
        
        # Validate knowledgebase name for remove operation
        if self.cli_settings.operation_mode == "remove":
            if not self.knowledgebase_name:
                raise ValueError("Knowledgebase name is required for 'remove' operation")
        
        # Validate Qdrant settings
        if not self.qdrant_settings.location:
            raise ValueError("Qdrant URL is required")
        
        # Validate embedding settings
        if not self.embedding_settings.model_name:
            raise ValueError("Embedding model name is required")
    
    def _complete_configuration(self):
        """Complete the configuration by deriving missing values."""
        # Derive knowledgebase name if not provided and needed
        if not self.knowledgebase_name and self.cli_settings.operation_mode in ["ingest", "update"]:
            if self.target_path:
                self.knowledgebase_name = self.cli_settings.derive_knowledgebase_name(
                    self.target_path
                )
        
        # Set collection name in Qdrant settings to match knowledgebase name
        if self.knowledgebase_name:
            self.qdrant_settings.collection_name = self.knowledgebase_name
    
    def get_effective_settings_summary(self) -> dict:
        """
        Get a summary of effective settings for logging/debugging.
        
        Returns:
            Dictionary containing key configuration values
        """
        return {
            "operation_mode": self.cli_settings.operation_mode,
            "target_path": str(self.target_path) if self.target_path else None,
            "knowledgebase_name": self.knowledgebase_name,
            "qdrant_url": self.qdrant_settings.location,
            "embedding_model": self.embedding_settings.model_name,
            "update_mode": self.cli_settings.update_mode,
            "dry_run": self.cli_settings.dry_run,
            "verbose": self.cli_settings.verbose,
            "include_patterns": self.cli_settings.include_patterns,
            "exclude_patterns": self.cli_settings.exclude_patterns,
            "chunking_enabled": self.embedding_settings.enable_chunking,
            "max_chunk_size": self.embedding_settings.max_chunk_size,
        }
    
    def validate_for_operation(self) -> List[str]:
        """
        Validate configuration for the specific operation mode.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            self._validate_configuration()
        except ValueError as e:
            errors.append(str(e))
        
        # Additional operation-specific validations
        if self.cli_settings.operation_mode in ["ingest", "update"]:
            if self.cli_settings.include_patterns:
                for pattern in self.cli_settings.include_patterns:
                    try:
                        import re
                        re.compile(pattern)
                    except re.error as e:
                        errors.append(f"Invalid include pattern '{pattern}': {e}")
            
            if self.cli_settings.exclude_patterns:
                for pattern in self.cli_settings.exclude_patterns:
                    try:
                        import re
                        re.compile(pattern)
                    except re.error as e:
                        errors.append(f"Invalid exclude pattern '{pattern}': {e}")
        
        return errors


class FileDiscovery:
    """
    Handles file system scanning and filtering for CLI operations.
    
    This class provides functionality to recursively discover files in directories,
    apply include/exclude regex patterns, and filter by supported file extensions.
    """
    
    def __init__(self, supported_extensions: List[str]):
        """
        Initialize file discovery with supported extensions.
        
        Args:
            supported_extensions: List of file extensions to consider (e.g., ['.txt', '.md'])
        """
        self.supported_extensions = [ext.lower() for ext in supported_extensions]
    
    async def discover_files(
        self,
        path: Path,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[FileInfo]:
        """
        Discover files matching the specified criteria.
        
        Args:
            path: Root path to scan (file or directory)
            include_patterns: Regex patterns for files to include
            exclude_patterns: Regex patterns for files to exclude
            recursive: Whether to scan directories recursively
            
        Returns:
            List of FileInfo objects for discovered files
            
        Raises:
            ValueError: If path doesn't exist or patterns are invalid
        """
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        # Compile regex patterns
        include_regexes = self._compile_patterns(include_patterns or [])
        exclude_regexes = self._compile_patterns(exclude_patterns or [])
        
        discovered_files = []
        
        if path.is_file():
            # Single file processing
            if self._should_include_file(path, include_regexes, exclude_regexes):
                file_info = await self._create_file_info(path)
                if file_info:
                    discovered_files.append(file_info)
        else:
            # Directory processing
            discovered_files = await self._scan_directory(
                path, include_regexes, exclude_regexes, recursive
            )
        
        return discovered_files
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """
        Compile regex patterns with validation.
        
        Args:
            patterns: List of regex pattern strings
            
        Returns:
            List of compiled regex patterns
            
        Raises:
            ValueError: If any pattern is invalid
        """
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return compiled_patterns
    
    async def _scan_directory(
        self,
        directory: Path,
        include_regexes: List[re.Pattern],
        exclude_regexes: List[re.Pattern],
        recursive: bool
    ) -> List[FileInfo]:
        """
        Recursively scan directory for matching files.
        
        Args:
            directory: Directory to scan
            include_regexes: Compiled include patterns
            exclude_regexes: Compiled exclude patterns
            recursive: Whether to scan subdirectories
            
        Returns:
            List of discovered FileInfo objects
        """
        discovered_files = []
        
        try:
            # Use os.walk for efficient recursive directory traversal
            if recursive:
                for root, dirs, files in os.walk(directory):
                    root_path = Path(root)
                    
                    # Skip hidden directories (starting with .)
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    for file_name in files:
                        file_path = root_path / file_name
                        if self._should_include_file(file_path, include_regexes, exclude_regexes):
                            file_info = await self._create_file_info(file_path)
                            if file_info:
                                discovered_files.append(file_info)
            else:
                # Non-recursive: only scan immediate directory
                for item in directory.iterdir():
                    if item.is_file() and self._should_include_file(item, include_regexes, exclude_regexes):
                        file_info = await self._create_file_info(item)
                        if file_info:
                            discovered_files.append(file_info)
        
        except PermissionError as e:
            # Log permission errors but continue processing
            print(f"Warning: Permission denied accessing {directory}: {e}", file=sys.stderr)
        except Exception as e:
            # Log other errors but continue processing
            print(f"Warning: Error scanning directory {directory}: {e}", file=sys.stderr)
        
        return discovered_files
    
    def _should_include_file(
        self,
        file_path: Path,
        include_regexes: List[re.Pattern],
        exclude_regexes: List[re.Pattern]
    ) -> bool:
        """
        Determine if a file should be included based on patterns and extensions.
        
        Args:
            file_path: Path to the file
            include_regexes: Compiled include patterns
            exclude_regexes: Compiled exclude patterns
            
        Returns:
            True if file should be included, False otherwise
        """
        # Skip hidden files (starting with .)
        if file_path.name.startswith('.'):
            return False
        
        # Check supported extensions
        if not self._is_supported_extension(file_path):
            return False
        
        # Convert path to string for pattern matching
        path_str = str(file_path)
        
        # Apply include patterns (if any)
        if include_regexes:
            if not any(pattern.search(path_str) for pattern in include_regexes):
                return False
        
        # Apply exclude patterns (if any)
        if exclude_regexes:
            if any(pattern.search(path_str) for pattern in exclude_regexes):
                return False
        
        return True
    
    def _is_supported_extension(self, file_path: Path) -> bool:
        """
        Check if file has a supported extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if extension is supported, False otherwise
        """
        if not self.supported_extensions:
            # If no extensions specified, allow all files
            return True
        
        file_extension = file_path.suffix.lower()
        return file_extension in self.supported_extensions
    
    async def _create_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """
        Create FileInfo object for a file with metadata extraction.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo object or None if file cannot be processed
        """
        try:
            # Get file stats
            stat = file_path.stat()
            file_size = stat.st_size
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Detect encoding and check if binary
            encoding, is_binary = await self._detect_file_encoding(file_path)
            
            # Skip binary files
            if is_binary:
                return None
            
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = max(1, file_size // 4)
            
            return FileInfo(
                path=file_path,
                size=file_size,
                modified_time=modified_time,
                encoding=encoding,
                is_binary=is_binary,
                estimated_tokens=estimated_tokens
            )
            
        except (OSError, PermissionError) as e:
            print(f"Warning: Cannot access file {file_path}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Warning: Error processing file {file_path}: {e}", file=sys.stderr)
            return None
    
    async def _detect_file_encoding(self, file_path: Path) -> tuple[str, bool]:
        """
        Detect file encoding and determine if file is binary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (encoding, is_binary)
        """
        try:
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                sample = f.read(8192)  # Read first 8KB
            
            # Check for null bytes (indicates binary file)
            if b'\x00' in sample:
                return 'binary', True
            
            # Try to decode as UTF-8 first
            try:
                sample.decode('utf-8')
                return 'utf-8', False
            except UnicodeDecodeError:
                pass
            
            # Try other common encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    sample.decode(encoding)
                    return encoding, False
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, treat as binary
            return 'binary', True
            
        except Exception:
            # If we can't read the file, assume it's binary
            return 'binary', True
    
    def get_discovery_stats(self, discovered_files: List[FileInfo]) -> dict:
        """
        Get statistics about discovered files.
        
        Args:
            discovered_files: List of discovered FileInfo objects
            
        Returns:
            Dictionary with discovery statistics
        """
        if not discovered_files:
            return {
                'total_files': 0,
                'total_size': 0,
                'estimated_tokens': 0,
                'extensions': {},
                'encodings': {}
            }
        
        total_size = sum(f.size for f in discovered_files)
        estimated_tokens = sum(f.estimated_tokens for f in discovered_files)
        
        # Count by extension
        extensions = {}
        for file_info in discovered_files:
            ext = file_info.path.suffix.lower() or 'no_extension'
            extensions[ext] = extensions.get(ext, 0) + 1
        
        # Count by encoding
        encodings = {}
        for file_info in discovered_files:
            enc = file_info.encoding
            encodings[enc] = encodings.get(enc, 0) + 1
        
        return {
            'total_files': len(discovered_files),
            'total_size': total_size,
            'estimated_tokens': estimated_tokens,
            'extensions': extensions,
            'encodings': encodings
        }


class ContentProcessor:
    """
    Processes file content for ingestion into Qdrant.
    
    This class handles file reading, encoding detection, binary file detection,
    and metadata extraction to convert FileInfo objects into Entry objects
    suitable for storage in Qdrant collections.
    """
    
    def __init__(self, max_file_size: int = 100 * 1024 * 1024):  # 100MB default
        """
        Initialize content processor.
        
        Args:
            max_file_size: Maximum file size to process in bytes
        """
        self.max_file_size = max_file_size
    
    async def process_file(self, file_info: FileInfo) -> Optional[Entry]:
        """
        Process a single file into an Entry object.
        
        Args:
            file_info: Information about the file to process
            
        Returns:
            Entry object ready for storage, or None if file cannot be processed
            
        Raises:
            ValueError: If file processing fails due to invalid data
            OSError: If file cannot be read due to system errors
        """
        try:
            # Skip binary files
            if file_info.is_binary:
                return None
            
            # Check file size limits
            if file_info.size > self.max_file_size:
                raise ValueError(f"File too large: {file_info.size} bytes (max: {self.max_file_size})")
            
            # Skip empty files
            if file_info.size == 0:
                return None
            
            # Read file content with encoding detection
            content = await self._read_file_content(file_info)
            if not content or not content.strip():
                return None
            
            # Extract metadata
            metadata = self._extract_metadata(file_info)
            
            # Create Entry object
            entry = Entry(
                content=content,
                metadata=metadata,
                is_chunk=False,  # Original files are not chunks
                source_document_id=None,
                chunk_index=None,
                total_chunks=None
            )
            
            return entry
            
        except Exception as e:
            # Re-raise with context about which file failed
            raise ValueError(f"Failed to process file {file_info.path}: {e}") from e
    
    async def _read_file_content(self, file_info: FileInfo) -> str:
        """
        Read file content with proper encoding handling.
        
        Args:
            file_info: File information including detected encoding
            
        Returns:
            File content as string
            
        Raises:
            OSError: If file cannot be read
            UnicodeDecodeError: If file cannot be decoded with any supported encoding
        """
        # Try the detected encoding first
        encodings_to_try = [file_info.encoding]
        
        # Add fallback encodings if the detected one isn't UTF-8
        if file_info.encoding != 'utf-8':
            encodings_to_try.extend(['utf-8', 'latin-1', 'cp1252', 'iso-8859-1'])
        else:
            encodings_to_try.extend(['latin-1', 'cp1252', 'iso-8859-1'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_encodings = []
        for enc in encodings_to_try:
            if enc not in seen and enc != 'binary':
                seen.add(enc)
                unique_encodings.append(enc)
        
        last_error = None
        for encoding in unique_encodings:
            try:
                with open(file_info.path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                
                # Validate that we got meaningful content
                if content and not self._is_mostly_control_chars(content):
                    return content
                    
            except (UnicodeDecodeError, OSError) as e:
                last_error = e
                continue
        
        # If all encodings failed, raise the last error
        if last_error:
            raise OSError(
                f"Could not decode file with any supported encoding. Last error: {last_error}"
            ) from last_error
        
        return ""
    
    def _is_mostly_control_chars(self, content: str, threshold: float = 0.3) -> bool:
        """
        Check if content consists mostly of control characters (likely binary).
        
        Args:
            content: Content to check
            threshold: Fraction of control characters that indicates binary content
            
        Returns:
            True if content is likely binary, False otherwise
        """
        if not content:
            return True
        
        control_chars = sum(1 for c in content if ord(c) < 32 and c not in '\t\n\r')
        return (control_chars / len(content)) > threshold
    
    def _extract_metadata(self, file_info: FileInfo) -> dict[str, Any]:
        """
        Extract metadata from file information.
        
        Args:
            file_info: File information to extract metadata from
            
        Returns:
            Dictionary containing file metadata
        """
        metadata = {
            # File identification
            "file_path": str(file_info.path),
            "file_name": file_info.path.name,
            "file_extension": file_info.path.suffix.lower(),
            "file_stem": file_info.path.stem,
            
            # File properties
            "file_size": file_info.size,
            "encoding": file_info.encoding,
            "estimated_tokens": file_info.estimated_tokens,
            
            # Timestamps
            "modified_time": file_info.modified_time.isoformat(),
            "ingestion_time": datetime.now().isoformat(),
            
            # Source information
            "source_type": "file_ingestion",
            "ingestion_method": "cli_tool",
            
            # Directory information
            "parent_directory": str(file_info.path.parent),
            "relative_path": str(file_info.path),
        }
        
        # Add file type classification
        metadata["file_type"] = self._classify_file_type(file_info.path)
        
        # Add size category
        metadata["size_category"] = self._categorize_file_size(file_info.size)
        
        return metadata
    
    def _classify_file_type(self, file_path: Path) -> str:
        """
        Classify file type based on extension.
        
        Args:
            file_path: Path to classify
            
        Returns:
            File type category
        """
        extension = file_path.suffix.lower()
        
        # Define file type mappings
        type_mappings = {
            # Documentation
            '.md': 'markdown',
            '.txt': 'text',
            '.rst': 'restructuredtext',
            '.rtf': 'rich_text',
            
            # Code files
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'header',
            '.hpp': 'cpp_header',
            '.sh': 'shell',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.fish': 'fish',
            
            # Configuration
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'config',
            '.conf': 'config',
            '.xml': 'xml',
            
            # Infrastructure
            '.tf': 'terraform',
            '.tftpl': 'terraform_template',
            '.tpl': 'template',
            '.dockerfile': 'dockerfile',
            
            # Web
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            
            # Data
            '.sql': 'sql',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.log': 'log',
        }
        
        return type_mappings.get(extension, 'unknown')
    
    def _categorize_file_size(self, size: int) -> str:
        """
        Categorize file size for metadata.
        
        Args:
            size: File size in bytes
            
        Returns:
            Size category string
        """
        if size < 1024:  # < 1KB
            return 'tiny'
        elif size < 10 * 1024:  # < 10KB
            return 'small'
        elif size < 100 * 1024:  # < 100KB
            return 'medium'
        elif size < 1024 * 1024:  # < 1MB
            return 'large'
        else:  # >= 1MB
            return 'very_large'
    
    def can_process_file(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file can be processed, False otherwise
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return False
            
            # Check if it's a hidden file (skip by default)
            if file_path.name.startswith('.'):
                return False
            
            return True
            
        except (OSError, PermissionError):
            return False
    
    def get_processing_stats(self, processed_entries: list[Entry]) -> dict[str, Any]:
        """
        Get statistics about processed entries.
        
        Args:
            processed_entries: List of processed Entry objects
            
        Returns:
            Dictionary with processing statistics
        """
        if not processed_entries:
            return {
                'total_entries': 0,
                'total_content_length': 0,
                'average_content_length': 0,
                'file_types': {},
                'size_categories': {},
                'encodings': {}
            }
        
        total_content_length = sum(len(entry.content) for entry in processed_entries)
        average_content_length = total_content_length / len(processed_entries)
        
        # Analyze metadata
        file_types = {}
        size_categories = {}
        encodings = {}
        
        for entry in processed_entries:
            if entry.metadata:
                # Count file types
                file_type = entry.metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
                
                # Count size categories
                size_cat = entry.metadata.get('size_category', 'unknown')
                size_categories[size_cat] = size_categories.get(size_cat, 0) + 1
                
                # Count encodings
                encoding = entry.metadata.get('encoding', 'unknown')
                encodings[encoding] = encodings.get(encoding, 0) + 1
        
        return {
            'total_entries': len(processed_entries),
            'total_content_length': total_content_length,
            'average_content_length': average_content_length,
            'file_types': file_types,
            'size_categories': size_categories,
            'encodings': encodings
        }


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model and its compatibility."""
    
    model_name: str
    vector_size: int
    vector_name: str
    is_available: bool = True
    is_compatible: bool = True
    error_message: Optional[str] = None
    collection_exists: bool = False
    collection_model: Optional[str] = None
    collection_vector_size: Optional[int] = None
    collection_vector_name: Optional[str] = None


class EmbeddingModelIntelligence:
    """
    Handles intelligent embedding model selection and validation for CLI operations.
    
    This class provides functionality to:
    - Detect existing embedding models from collections
    - Select smart defaults for new collections
    - Validate embedding model compatibility
    - Display embedding model information and errors
    """
    
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"
    FALLBACK_MODELS = [
        "nomic-ai/nomic-embed-text-v1.5-Q",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2"
    ]
    
    def __init__(self, qdrant_settings: QdrantSettings):
        """
        Initialize embedding model intelligence.
        
        Args:
            qdrant_settings: Qdrant connection settings
        """
        self.qdrant_settings = qdrant_settings
        self._connector_cache: Optional[QdrantConnector] = None
    
    async def detect_collection_model(self, collection_name: str) -> Optional[EmbeddingModelInfo]:
        """
        Detect the embedding model used by an existing collection.
        
        Args:
            collection_name: Name of the collection to analyze
            
        Returns:
            EmbeddingModelInfo if collection exists and model can be detected, None otherwise
        """
        try:
            # Create a temporary connector to check collection
            temp_embedding_settings = EmbeddingProviderSettings()
            temp_embedding_settings.model_name = self.DEFAULT_MODEL
            temp_provider = create_embedding_provider(temp_embedding_settings)
            
            connector = QdrantConnector(
                qdrant_url=self.qdrant_settings.location,
                qdrant_api_key=self.qdrant_settings.api_key,
                collection_name=collection_name,
                embedding_provider=temp_provider,
                qdrant_local_path=getattr(self.qdrant_settings, 'local_path', None)
            )
            
            # Check if collection exists
            collection_names = await connector.get_collection_names()
            if collection_name not in collection_names:
                return None
            
            # Get collection compatibility info
            compatibility_info = await connector.check_collection_compatibility(collection_name)
            
            if not compatibility_info.get("exists", False):
                return None
            
            # Extract model information from collection
            collection_vector_name = None
            collection_vector_size = None
            
            available_vectors = compatibility_info.get("available_vectors", {})
            if available_vectors:
                # Use the first available vector as the primary one
                collection_vector_name = list(available_vectors.keys())[0]
                collection_vector_size = available_vectors[collection_vector_name]
            
            # Try to infer model name from vector name and size
            inferred_model = self._infer_model_from_collection_info(
                collection_vector_name, collection_vector_size
            )
            
            return EmbeddingModelInfo(
                model_name=inferred_model or "unknown",
                vector_size=collection_vector_size or 0,
                vector_name=collection_vector_name or "unknown",
                is_available=inferred_model is not None,
                is_compatible=True,  # If we can detect it, it should be compatible
                collection_exists=True,
                collection_model=inferred_model,
                collection_vector_size=collection_vector_size,
                collection_vector_name=collection_vector_name
            )
            
        except Exception as e:
            return EmbeddingModelInfo(
                model_name="unknown",
                vector_size=0,
                vector_name="unknown",
                is_available=False,
                is_compatible=False,
                error_message=f"Failed to detect collection model: {str(e)}",
                collection_exists=True  # Assume it exists if we got an error during detection
            )
    
    def _infer_model_from_collection_info(
        self, 
        vector_name: Optional[str], 
        vector_size: Optional[int]
    ) -> Optional[str]:
        """
        Infer the embedding model from collection vector information.
        
        Args:
            vector_name: Name of the vector in the collection
            vector_size: Size of the vector dimensions
            
        Returns:
            Inferred model name or None if cannot be determined
        """
        if not vector_name or not vector_size:
            return None
        
        # Common model patterns based on vector name and size
        model_patterns = {
            # Nomic models
            ("fast-nomic-embed-text-v1.5-q", 768): "nomic-ai/nomic-embed-text-v1.5-Q",
            ("fast-nomic-embed-text-v1", 768): "nomic-ai/nomic-embed-text-v1",
            
            # Sentence transformers
            ("fast-all-minilm-l6-v2", 384): "sentence-transformers/all-MiniLM-L6-v2",
            ("fast-all-minilm-l12-v2", 384): "sentence-transformers/all-MiniLM-L12-v2",
            
            # BGE models
            ("fast-bge-small-en-v1.5", 384): "BAAI/bge-small-en-v1.5",
            ("fast-bge-base-en-v1.5", 768): "BAAI/bge-base-en-v1.5",
            
            # E5 models
            ("fast-e5-small-v2", 384): "intfloat/e5-small-v2",
            ("fast-e5-base-v2", 768): "intfloat/e5-base-v2",
        }
        
        # Try exact match first
        key = (vector_name.lower(), vector_size)
        if key in model_patterns:
            return model_patterns[key]
        
        # Try partial matches based on vector name patterns
        vector_name_lower = vector_name.lower()
        for (pattern_name, pattern_size), model in model_patterns.items():
            if pattern_size == vector_size:
                # Check if vector name contains key parts of the pattern
                pattern_parts = pattern_name.replace("fast-", "").split("-")
                if all(part in vector_name_lower for part in pattern_parts[:2]):  # Match first 2 parts
                    return model
        
        # If no pattern matches, return None
        return None
    
    async def select_smart_default(self, collection_name: str) -> EmbeddingModelInfo:
        """
        Select a smart default embedding model for a collection.
        
        For existing collections, detects and uses the existing model.
        For new collections, uses the configured default or falls back to a working model.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            EmbeddingModelInfo with the selected model
        """
        # First, try to detect existing collection model
        existing_model_info = await self.detect_collection_model(collection_name)
        if existing_model_info and existing_model_info.is_available:
            return existing_model_info
        
        # For new collections, try the default model
        default_model_info = await self.validate_model(self.DEFAULT_MODEL)
        if default_model_info.is_available:
            return default_model_info
        
        # If default doesn't work, try fallback models
        for fallback_model in self.FALLBACK_MODELS:
            if fallback_model == self.DEFAULT_MODEL:
                continue  # Already tried
            
            fallback_info = await self.validate_model(fallback_model)
            if fallback_info.is_available:
                return fallback_info
        
        # If all models fail, return error info
        return EmbeddingModelInfo(
            model_name=self.DEFAULT_MODEL,
            vector_size=0,
            vector_name="unknown",
            is_available=False,
            is_compatible=False,
            error_message="No compatible embedding models are available. Please check your FastEmbed installation."
        )
    
    async def validate_model(self, model_name: str) -> EmbeddingModelInfo:
        """
        Validate that an embedding model is available and working.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            EmbeddingModelInfo with validation results
        """
        try:
            # Create embedding settings and provider
            embedding_settings = EmbeddingProviderSettings()
            embedding_settings.model_name = model_name
            provider = create_embedding_provider(embedding_settings)
            
            # Get model information
            vector_size = provider.get_vector_size()
            vector_name = provider.get_vector_name()
            
            return EmbeddingModelInfo(
                model_name=model_name,
                vector_size=vector_size,
                vector_name=vector_name,
                is_available=True,
                is_compatible=True
            )
            
        except Exception as e:
            return EmbeddingModelInfo(
                model_name=model_name,
                vector_size=0,
                vector_name="unknown",
                is_available=False,
                is_compatible=False,
                error_message=f"Model validation failed: {str(e)}"
            )
    
    async def validate_model_compatibility(
        self, 
        model_name: str, 
        collection_name: str
    ) -> EmbeddingModelInfo:
        """
        Validate that a model is compatible with an existing collection.
        
        Args:
            model_name: Name of the model to validate
            collection_name: Name of the collection to check compatibility with
            
        Returns:
            EmbeddingModelInfo with compatibility results
        """
        # First validate the model itself
        model_info = await self.validate_model(model_name)
        if not model_info.is_available:
            return model_info
        
        # Check collection compatibility
        try:
            embedding_settings = EmbeddingProviderSettings()
            embedding_settings.model_name = model_name
            provider = create_embedding_provider(embedding_settings)
            
            connector = QdrantConnector(
                qdrant_url=self.qdrant_settings.location,
                qdrant_api_key=self.qdrant_settings.api_key,
                collection_name=collection_name,
                embedding_provider=provider,
                qdrant_local_path=getattr(self.qdrant_settings, 'local_path', None)
            )
            
            # Check collection compatibility
            compatibility_info = await connector.check_collection_compatibility(collection_name)
            
            if not compatibility_info.get("exists", False):
                # Collection doesn't exist, so model is compatible
                model_info.is_compatible = True
                model_info.collection_exists = False
                return model_info
            
            # Collection exists, check compatibility
            vector_compatible = compatibility_info.get("vector_compatible", False)
            dimension_compatible = compatibility_info.get("dimension_compatible", False)
            
            model_info.collection_exists = True
            model_info.is_compatible = vector_compatible and dimension_compatible
            
            if not model_info.is_compatible:
                expected_dims = compatibility_info.get("expected_dimensions", 0)
                actual_dims = compatibility_info.get("actual_dimensions", 0)
                
                if not vector_compatible:
                    model_info.error_message = (
                        f"Vector name mismatch. Collection uses different vector configuration. "
                        f"Expected: {model_info.vector_name}"
                    )
                elif not dimension_compatible:
                    model_info.error_message = (
                        f"Dimension mismatch. Collection has {actual_dims} dimensions, "
                        f"but model '{model_name}' produces {expected_dims} dimensions."
                    )
            
            # Store collection information
            available_vectors = compatibility_info.get("available_vectors", {})
            if available_vectors:
                collection_vector_name = list(available_vectors.keys())[0]
                model_info.collection_vector_name = collection_vector_name
                model_info.collection_vector_size = available_vectors[collection_vector_name]
                model_info.collection_model = self._infer_model_from_collection_info(
                    collection_vector_name, model_info.collection_vector_size
                )
            
            return model_info
            
        except Exception as e:
            model_info.is_compatible = False
            model_info.error_message = f"Compatibility check failed: {str(e)}"
            return model_info
    
    def display_model_info(self, model_info: EmbeddingModelInfo, verbose: bool = False) -> None:
        """
        Display embedding model information to the user.
        
        Args:
            model_info: Model information to display
            verbose: Whether to show detailed information
        """
        if not model_info.is_available:
            print(f"❌ Model '{model_info.model_name}' is not available")
            if model_info.error_message:
                print(f"   Error: {model_info.error_message}")
            return
        
        if model_info.collection_exists:
            if model_info.is_compatible:
                print(f"✅ Model '{model_info.model_name}' is compatible with existing collection")
                if verbose and model_info.collection_model:
                    print(f"   Collection model: {model_info.collection_model}")
                    print(f"   Vector dimensions: {model_info.collection_vector_size}")
            else:
                print(f"❌ Model '{model_info.model_name}' is incompatible with existing collection")
                if model_info.error_message:
                    print(f"   Error: {model_info.error_message}")
                if verbose and model_info.collection_model:
                    print(f"   Collection model: {model_info.collection_model}")
                    print(f"   Collection dimensions: {model_info.collection_vector_size}")
                    print(f"   Requested dimensions: {model_info.vector_size}")
        else:
            print(f"✅ Model '{model_info.model_name}' is available for new collection")
            if verbose:
                print(f"   Vector dimensions: {model_info.vector_size}")
                print(f"   Vector name: {model_info.vector_name}")
    
    def display_model_mismatch_error(self, model_info: EmbeddingModelInfo) -> None:
        """
        Display a detailed error message for model mismatches.
        
        Args:
            model_info: Model information with mismatch details
        """
        print("🚫 Embedding Model Mismatch Detected")
        print()
        print(f"The collection already exists with a different embedding model configuration:")
        print(f"  Collection model: {model_info.collection_model or 'Unknown'}")
        print(f"  Collection dimensions: {model_info.collection_vector_size}")
        print(f"  Collection vector name: {model_info.collection_vector_name}")
        print()
        print(f"Requested model configuration:")
        print(f"  Model: {model_info.model_name}")
        print(f"  Dimensions: {model_info.vector_size}")
        print(f"  Vector name: {model_info.vector_name}")
        print()
        print("Solutions:")
        if model_info.collection_model:
            print(f"  1. Use the existing model: --embedding {model_info.collection_model}")
        print(f"  2. Use a different collection name: --knowledgebase different-name")
        print(f"  3. Remove the existing collection first: qdrant-ingest remove {model_info.collection_model}")
        print()
    
    async def get_available_models(self) -> List[str]:
        """
        Get a list of available embedding models.
        
        Returns:
            List of available model names
        """
        try:
            from fastembed import TextEmbedding
            models = TextEmbedding.list_supported_models()
            return [model['model'] for model in models]
        except Exception as e:
            print(f"Warning: Could not retrieve available models: {e}", file=sys.stderr)
            return self.FALLBACK_MODELS


class BaseOperation(ABC):
    """
    Abstract base class for all CLI operations.
    
    This class provides shared functionality for all knowledge base operations
    including configuration management, Qdrant connection setup, and common
    validation patterns.
    """

    def __init__(self, config: IngestConfig):
        """
        Initialize operation with configuration.
        
        Args:
            config: Complete configuration for the operation
        """
        self.config = config
        self._connector: Optional[QdrantConnector] = None
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self.progress_reporter = ProgressReporter(
            show_progress=config.cli_settings.show_progress,
            verbose=config.cli_settings.verbose,
            batch_size=config.cli_settings.batch_size
        )
        self.error_handler = CLIErrorHandler(self.progress_reporter)

    @abstractmethod
    async def execute(self) -> OperationResult:
        """
        Execute the operation and return results.
        
        Returns:
            OperationResult with execution details and statistics
        """
        pass

    @abstractmethod
    def validate_preconditions(self) -> List[str]:
        """
        Validate operation preconditions and return any error messages.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    async def get_connector(self) -> QdrantConnector:
        """
        Get or create a QdrantConnector instance with comprehensive error handling.
        
        Returns:
            Configured QdrantConnector instance
            
        Raises:
            ValueError: If connector cannot be created after retries
        """
        if self._connector is None:
            try:
                # Create embedding provider with error handling
                if self._embedding_provider is None:
                    try:
                        self._embedding_provider = create_embedding_provider(self.config.embedding_settings)
                    except Exception as e:
                        if not self.error_handler.handle_embedding_error(e, self.config.embedding_settings.model_name):
                            raise ValueError(f"Failed to create embedding provider: {e}") from e
                
                # Create connector with retry logic for connection errors
                max_attempts = 3
                last_error = None
                
                for attempt in range(max_attempts):
                    try:
                        self._connector = QdrantConnector(
                            qdrant_url=self.config.qdrant_settings.location,
                            qdrant_api_key=self.config.qdrant_settings.api_key,
                            collection_name=self.config.knowledgebase_name or "default",
                            embedding_provider=self._embedding_provider,
                            qdrant_local_path=getattr(self.config.qdrant_settings, 'local_path', None),
                            enable_chunking=self.config.embedding_settings.enable_chunking,
                            max_chunk_size=self.config.embedding_settings.max_chunk_size,
                            chunk_overlap=self.config.embedding_settings.chunk_overlap,
                            chunk_strategy=self.config.embedding_settings.chunk_strategy
                        )
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        last_error = e
                        
                        # Handle connection errors with retry logic
                        if attempt < max_attempts - 1:  # Not the last attempt
                            if self.error_handler.handle_connection_error(
                                e, self.config.qdrant_settings.location, f"creating connector (attempt {attempt + 1})"
                            ):
                                continue  # Retry
                            else:
                                break  # Error handler says don't retry
                        else:
                            # Last attempt, handle error without retry
                            self.error_handler.handle_connection_error(
                                e, self.config.qdrant_settings.location, "creating connector (final attempt)"
                            )
                
                # If we still don't have a connector, raise the last error
                if self._connector is None and last_error:
                    raise ValueError(f"Failed to create Qdrant connector after {max_attempts} attempts: {last_error}") from last_error
                
            except Exception as e:
                # Handle any other unexpected errors
                if not isinstance(e, ValueError):  # Don't double-wrap ValueError
                    self.error_handler.handle_unknown_error(e, "connector creation")
                    raise ValueError(f"Failed to create Qdrant connector: {e}") from e
                raise
        
        return self._connector

    async def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            connector = await self.get_connector()
            collection_names = await connector.get_collection_names()
            return collection_name in collection_names
        except Exception:
            return False

    def _log_verbose(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled.
        
        Args:
            message: Message to log
        """
        self.progress_reporter._log_verbose(message)

    def _log_info(self, message: str) -> None:
        """
        Log an informational message.
        
        Args:
            message: Message to log
        """
        self.progress_reporter.log_info(message)

    def _log_success(self, message: str) -> None:
        """
        Log a success message.
        
        Args:
            message: Message to log
        """
        self.progress_reporter.log_success(message)

    def _log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
        """
        self.progress_reporter.log_warning(message)

    def _log_error(self, message: str) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
        """
        self.progress_reporter.log_error(message)

    def _confirm_operation(self, message: str) -> bool:
        """
        Ask user for confirmation unless force mode is enabled.
        
        Args:
            message: Confirmation message to display
            
        Returns:
            True if user confirms or force mode is enabled, False otherwise
        """
        return self.progress_reporter.confirm_operation(
            message, 
            force=self.config.cli_settings.force_operation
        )


class IngestOperation(BaseOperation):
    """
    Handles file ingestion into new or existing collections.
    
    This operation discovers files, processes their content, and stores them
    in Qdrant collections with appropriate metadata and chunking.
    """

    async def execute(self) -> OperationResult:
        """
        Execute the ingestion operation.
        
        Returns:
            OperationResult with ingestion statistics
        """
        result = OperationResult(success=False)
        
        try:
            # Validate preconditions
            validation_errors = self.validate_preconditions()
            if validation_errors:
                result.errors.extend(validation_errors)
                self.progress_reporter.finish_operation(result)
                return result
            
            # Start operation tracking
            operation_name = "File Ingestion"
            if self.config.cli_settings.dry_run:
                operation_name += " (Dry Run)"
            
            self.progress_reporter.start_operation(operation_name)
            self._log_info(f"Starting ingestion into collection '{self.config.knowledgebase_name}'")
            
            if self.config.cli_settings.dry_run:
                self._log_info("DRY RUN MODE - No changes will be made")
            
            # Discover files
            file_discovery = FileDiscovery(self.config.cli_settings.supported_extensions)
            discovered_files = await file_discovery.discover_files(
                self.config.target_path,
                self.config.cli_settings.include_patterns,
                self.config.cli_settings.exclude_patterns
            )
            
            if not discovered_files:
                self._log_warning("No files found matching the specified criteria")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            # Show discovery statistics
            discovery_stats = file_discovery.get_discovery_stats(discovered_files)
            self.progress_reporter.show_discovery_stats(discovery_stats)
            
            # Update progress reporter with total file count
            self.progress_reporter._total_items = len(discovered_files)
            
            # Process files with progress reporting
            content_processor = ContentProcessor()
            processed_entries = []
            
            for file_info in discovered_files:
                try:
                    entry = await content_processor.process_file(file_info)
                    if entry:
                        processed_entries.append(entry)
                        result.files_processed += 1
                        self.progress_reporter.report_file_processed(file_info.path, True)
                    else:
                        result.files_skipped += 1
                        self.progress_reporter.report_file_skipped(
                            file_info.path, 
                            "empty or binary file"
                        )
                except Exception as e:
                    result.files_failed += 1
                    error_msg = f"Failed to process {file_info.path}: {e}"
                    result.errors.append(error_msg)
                    self.progress_reporter.report_file_processed(file_info.path, False, str(e))
            
            if not processed_entries:
                self._log_warning("No files could be processed successfully")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            # Show processing statistics
            processing_stats = content_processor.get_processing_stats(processed_entries)
            self.progress_reporter.show_processing_stats(processing_stats)
            
            # Store entries in Qdrant (unless dry run)
            if not self.config.cli_settings.dry_run:
                self._log_info("Storing entries in Qdrant...")
                connector = await self.get_connector()
                
                # Process entries in batches for better progress reporting
                batch_size = self.config.cli_settings.batch_size
                total_batches = (len(processed_entries) + batch_size - 1) // batch_size
                
                for i in range(0, len(processed_entries), batch_size):
                    batch = processed_entries[i:i + batch_size]
                    batch_success = 0
                    batch_errors = 0
                    
                    for entry in batch:
                        try:
                            await connector.store(entry, collection_name=self.config.knowledgebase_name)
                            # Estimate chunks created (actual chunking happens in QdrantConnector)
                            estimated_chunks = max(1, len(entry.content) // 1000)  # Rough estimate
                            result.chunks_created += estimated_chunks
                            batch_success += 1
                        except Exception as e:
                            # Use error handler for storage errors
                            entry_info = f"entry from {getattr(entry, 'metadata', {}).get('file_path', 'unknown file')}"
                            if self.error_handler.handle_storage_error(e, entry_info, self.config.knowledgebase_name):
                                error_msg = f"Failed to store entry: {e}"
                                result.errors.append(error_msg)
                                batch_errors += 1
                            else:
                                # Fatal storage error, stop processing
                                error_msg = f"Fatal storage error: {e}"
                                result.errors.append(error_msg)
                                result.success = False
                                self.progress_reporter.finish_operation(result)
                                return result
                    
                    # Report batch progress
                    if self.progress_reporter.show_progress and len(processed_entries) > batch_size:
                        self.progress_reporter.report_batch_processed(
                            len(batch), batch_success, batch_errors
                        )
            else:
                # In dry run, just estimate chunks
                for entry in processed_entries:
                    estimated_chunks = max(1, len(entry.content) // 1000)
                    result.chunks_created += estimated_chunks
            
            result.success = True
            self.progress_reporter.finish_operation(result)
            return result
            
        except Exception as e:
            # Handle unexpected errors with error handler
            self.error_handler.handle_unknown_error(e, "ingestion operation")
            error_msg = f"Ingestion failed: {e}"
            result.errors.append(error_msg)
            self._log_error(error_msg)
            
            # Add error summary to result
            error_summary = self.error_handler.get_error_summary()
            if error_summary['recommendations']:
                result.warnings.extend([f"Recommendation: {rec}" for rec in error_summary['recommendations']])
            
            self.progress_reporter.finish_operation(result)
            return result

    def validate_preconditions(self) -> List[str]:
        """
        Validate preconditions for ingestion operation.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check target path exists
        if not self.config.target_path or not self.config.target_path.exists():
            errors.append(f"Target path does not exist: {self.config.target_path}")
        
        # Check knowledgebase name is provided
        if not self.config.knowledgebase_name:
            errors.append("Knowledgebase name is required for ingestion")
        
        # Validate Qdrant settings
        if not self.config.qdrant_settings.location:
            errors.append("Qdrant URL is required")
        
        # Validate embedding settings
        if not self.config.embedding_settings.model_name:
            errors.append("Embedding model name is required")
        
        return errors


class UpdateOperation(BaseOperation):
    """
    Handles updating existing collections with new content.
    
    Supports both add-only mode (append new files) and replace mode
    (clear collection and replace with new content).
    """

    async def execute(self) -> OperationResult:
        """
        Execute the update operation.
        
        Returns:
            OperationResult with update statistics
        """
        result = OperationResult(success=False)
        
        try:
            # Validate preconditions
            validation_errors = self.validate_preconditions()
            if validation_errors:
                result.errors.extend(validation_errors)
                self.progress_reporter.finish_operation(result)
                return result
            
            # Start operation tracking
            mode_text = "replace" if self.config.cli_settings.update_mode == "replace" else "add-only"
            operation_name = f"Collection Update ({mode_text} mode)"
            if self.config.cli_settings.dry_run:
                operation_name += " (Dry Run)"
            
            self.progress_reporter.start_operation(operation_name)
            
            collection_exists = await self.check_collection_exists(self.config.knowledgebase_name)
            
            if self.config.cli_settings.update_mode == "replace":
                if collection_exists:
                    self._log_info(f"Replace mode: Will clear existing collection '{self.config.knowledgebase_name}'")
                    if not self._confirm_operation("⚠️  This will delete all existing data in the collection. Continue?"):
                        self._log_info("Update cancelled by user")
                        result.success = True
                        self.progress_reporter.finish_operation(result)
                        return result
                else:
                    self._log_info(f"Collection '{self.config.knowledgebase_name}' does not exist, creating new collection")
            else:
                # Add-only mode
                if collection_exists:
                    self._log_info(f"Add-only mode: Will append to existing collection '{self.config.knowledgebase_name}'")
                else:
                    self._log_info(f"Collection '{self.config.knowledgebase_name}' does not exist, creating new collection")
            
            if self.config.cli_settings.dry_run:
                self._log_info("DRY RUN MODE - No changes will be made")
            
            # Clear collection if replace mode
            if (self.config.cli_settings.update_mode == "replace" and 
                collection_exists and 
                not self.config.cli_settings.dry_run):
                
                connector = await self.get_connector()
                try:
                    cleared_count = await connector.clear_collection(self.config.knowledgebase_name)
                    self._log_info(f"Existing collection cleared ({cleared_count} points removed)")
                except Exception as e:
                    error_msg = f"Failed to clear collection: {e}"
                    result.errors.append(error_msg)
                    self._log_error(error_msg)
                    self.progress_reporter.finish_operation(result)
                    return result
            
            # Discover files
            file_discovery = FileDiscovery(self.config.cli_settings.supported_extensions)
            discovered_files = await file_discovery.discover_files(
                self.config.target_path,
                self.config.cli_settings.include_patterns,
                self.config.cli_settings.exclude_patterns
            )
            
            if not discovered_files:
                self._log_warning("No files found matching the specified criteria")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            # Show discovery statistics
            discovery_stats = file_discovery.get_discovery_stats(discovered_files)
            self.progress_reporter.show_discovery_stats(discovery_stats)
            
            # In add-only mode, filter out files that already exist
            if (self.config.cli_settings.update_mode == "add-only" and 
                collection_exists and 
                not self.config.cli_settings.dry_run):
                
                discovered_files = await self._filter_existing_files(discovered_files)
                if not discovered_files:
                    self._log_info("All files already exist in collection, nothing to update")
                    result.success = True
                    self.progress_reporter.finish_operation(result)
                    return result
                
                self._log_info(f"After filtering existing files: {len(discovered_files)} files to process")
            
            # Update progress reporter with total file count
            self.progress_reporter._total_items = len(discovered_files)
            
            # Process files with progress reporting
            content_processor = ContentProcessor()
            processed_entries = []
            
            for file_info in discovered_files:
                try:
                    entry = await content_processor.process_file(file_info)
                    if entry:
                        processed_entries.append(entry)
                        result.files_processed += 1
                        self.progress_reporter.report_file_processed(file_info.path, True)
                    else:
                        result.files_skipped += 1
                        self.progress_reporter.report_file_skipped(
                            file_info.path, 
                            "empty or binary file"
                        )
                except Exception as e:
                    result.files_failed += 1
                    error_msg = f"Failed to process {file_info.path}: {e}"
                    result.errors.append(error_msg)
                    self.progress_reporter.report_file_processed(file_info.path, False, str(e))
            
            if not processed_entries:
                self._log_warning("No files could be processed successfully")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            # Show processing statistics
            processing_stats = content_processor.get_processing_stats(processed_entries)
            self.progress_reporter.show_processing_stats(processing_stats)
            
            # Store entries in Qdrant (unless dry run)
            if not self.config.cli_settings.dry_run:
                self._log_info("Storing entries in Qdrant...")
                connector = await self.get_connector()
                
                # Process entries in batches for better progress reporting
                batch_size = self.config.cli_settings.batch_size
                
                for i in range(0, len(processed_entries), batch_size):
                    batch = processed_entries[i:i + batch_size]
                    batch_success = 0
                    batch_errors = 0
                    
                    for entry in batch:
                        try:
                            await connector.store(entry, collection_name=self.config.knowledgebase_name)
                            estimated_chunks = max(1, len(entry.content) // 1000)
                            result.chunks_created += estimated_chunks
                            batch_success += 1
                        except Exception as e:
                            error_msg = f"Failed to store entry: {e}"
                            result.errors.append(error_msg)
                            batch_errors += 1
                    
                    # Report batch progress
                    if self.progress_reporter.show_progress and len(processed_entries) > batch_size:
                        self.progress_reporter.report_batch_processed(
                            len(batch), batch_success, batch_errors
                        )
            else:
                # In dry run, just estimate chunks
                for entry in processed_entries:
                    estimated_chunks = max(1, len(entry.content) // 1000)
                    result.chunks_created += estimated_chunks
            
            result.success = True
            self.progress_reporter.finish_operation(result)
            return result
            
        except Exception as e:
            error_msg = f"Update failed: {e}"
            result.errors.append(error_msg)
            self._log_error(error_msg)
            self.progress_reporter.finish_operation(result)
            return result

    def validate_preconditions(self) -> List[str]:
        """
        Validate preconditions for update operation.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check target path exists
        if not self.config.target_path or not self.config.target_path.exists():
            errors.append(f"Target path does not exist: {self.config.target_path}")
        
        # Check knowledgebase name is provided
        if not self.config.knowledgebase_name:
            errors.append("Knowledgebase name is required for update")
        
        # Validate Qdrant settings
        if not self.config.qdrant_settings.location:
            errors.append("Qdrant URL is required")
        
        # Validate embedding settings
        if not self.config.embedding_settings.model_name:
            errors.append("Embedding model name is required")
        
        # Validate update mode
        if self.config.cli_settings.update_mode not in ["add-only", "replace"]:
            errors.append(f"Invalid update mode: {self.config.cli_settings.update_mode}")
        
        return errors

    async def _filter_existing_files(self, discovered_files: List[FileInfo]) -> List[FileInfo]:
        """
        Filter out files that already exist in the collection.
        
        Args:
            discovered_files: List of discovered files
            
        Returns:
            List of files that don't exist in the collection
        """
        # This is a simplified implementation
        # In a real implementation, we would query the collection to check
        # which files already exist based on file_path metadata
        
        # For now, return all files (actual implementation would require
        # querying Qdrant collection metadata)
        self._log_verbose("File existence filtering not yet implemented, processing all files")
        return discovered_files


class RemoveOperation(BaseOperation):
    """
    Handles knowledge base removal with confirmation prompts.
    
    This operation deletes entire Qdrant collections after user confirmation.
    """

    async def execute(self) -> OperationResult:
        """
        Execute the remove operation.
        
        Returns:
            OperationResult with removal statistics
        """
        result = OperationResult(success=False)
        
        try:
            # Validate preconditions
            validation_errors = self.validate_preconditions()
            if validation_errors:
                result.errors.extend(validation_errors)
                self.progress_reporter.finish_operation(result)
                return result
            
            # Start operation tracking
            operation_name = "Collection Removal"
            if self.config.cli_settings.dry_run:
                operation_name += " (Dry Run)"
            
            self.progress_reporter.start_operation(operation_name, 1)
            
            # Check if collection exists
            collection_exists = await self.check_collection_exists(self.config.knowledgebase_name)
            
            if not collection_exists:
                self._log_info(f"Collection '{self.config.knowledgebase_name}' does not exist")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            self._log_info(f"Found collection '{self.config.knowledgebase_name}'")
            
            if self.config.cli_settings.dry_run:
                self._log_info("DRY RUN MODE - Collection would be deleted")
                result.success = True
                self.progress_reporter.update_progress(1, f"Would delete collection '{self.config.knowledgebase_name}'")
                self.progress_reporter.finish_operation(result)
                return result
            
            # Confirm deletion
            if not self._confirm_operation(f"🗑️  Delete collection '{self.config.knowledgebase_name}' and all its data?"):
                self._log_info("Removal cancelled by user")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            # Delete collection
            connector = await self.get_connector()
            try:
                self.progress_reporter.update_progress(0, f"Deleting collection '{self.config.knowledgebase_name}'...")
                deleted = await connector.delete_collection(self.config.knowledgebase_name)
                if deleted:
                    result.files_processed = 1  # Use this to indicate collection was deleted
                    self._log_success(f"Collection '{self.config.knowledgebase_name}' deleted successfully")
                    self.progress_reporter.update_progress(1, f"Deleted collection '{self.config.knowledgebase_name}'")
                else:
                    self._log_info(f"Collection '{self.config.knowledgebase_name}' was already deleted")
                    self.progress_reporter.update_progress(1, f"Collection '{self.config.knowledgebase_name}' was already deleted")
                    result.success = True
            except Exception as e:
                error_msg = f"Failed to delete collection: {e}"
                result.errors.append(error_msg)
                self._log_error(error_msg)
                self.progress_reporter.finish_operation(result)
                return result
            
            result.success = True
            self.progress_reporter.finish_operation(result)
            return result
            
        except Exception as e:
            error_msg = f"Remove operation failed: {e}"
            result.errors.append(error_msg)
            self._log_error(error_msg)
            self.progress_reporter.finish_operation(result)
            return result

    def validate_preconditions(self) -> List[str]:
        """
        Validate preconditions for remove operation.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check knowledgebase name is provided
        if not self.config.knowledgebase_name:
            errors.append("Knowledgebase name is required for removal")
        
        # Validate Qdrant settings
        if not self.config.qdrant_settings.location:
            errors.append("Qdrant URL is required")
        
        return errors


class ListOperation(BaseOperation):
    """
    Handles listing available knowledge bases with collection information.
    
    This operation displays all available collections with metadata like
    size, embedding model, and compatibility information.
    """

    async def execute(self) -> OperationResult:
        """
        Execute the list operation.
        
        Returns:
            OperationResult with listing statistics
        """
        result = OperationResult(success=False)
        
        try:
            # Validate preconditions
            validation_errors = self.validate_preconditions()
            if validation_errors:
                result.errors.extend(validation_errors)
                self.progress_reporter.finish_operation(result)
                return result
            
            # Start operation tracking
            self.progress_reporter.start_operation("Collection Listing")
            self._log_info("Listing available knowledge bases...")
            
            # Get connector (we need a temporary one for listing)
            temp_embedding_settings = EmbeddingProviderSettings()
            temp_embedding_settings.model_name = "nomic-ai/nomic-embed-text-v1.5-Q"
            temp_provider = create_embedding_provider(temp_embedding_settings)
            
            temp_connector = QdrantConnector(
                qdrant_url=self.config.qdrant_settings.location,
                qdrant_api_key=self.config.qdrant_settings.api_key,
                collection_name="temp",  # Temporary name for listing
                embedding_provider=temp_provider,
                qdrant_local_path=getattr(self.config.qdrant_settings, 'local_path', None)
            )
            
            # Get collection names
            try:
                self.progress_reporter.update_progress(0, "Connecting to Qdrant...")
                collection_names = await temp_connector.get_collection_names()
            except Exception as e:
                error_msg = f"Failed to connect to Qdrant: {e}"
                result.errors.append(error_msg)
                self._log_error(error_msg)
                self.progress_reporter.finish_operation(result)
                return result
            
            if not collection_names:
                self._log_info("No knowledge bases found")
                result.success = True
                self.progress_reporter.finish_operation(result)
                return result
            
            # Update progress reporter with total collection count
            self.progress_reporter._total_items = len(collection_names)
            self._log_info(f"Found {len(collection_names)} knowledge base(s):")
            print()
            
            # Display collection information
            for i, collection_name in enumerate(sorted(collection_names), 1):
                print(f"{i}. {collection_name}")
                
                if self.config.cli_settings.verbose:
                    # Get detailed collection information
                    try:
                        # Create a temporary connector for this specific collection
                        collection_connector = QdrantConnector(
                            qdrant_url=self.config.qdrant_settings.location,
                            qdrant_api_key=self.config.qdrant_settings.api_key,
                            collection_name=collection_name,
                            embedding_provider=temp_provider,
                            qdrant_local_path=getattr(self.config.qdrant_settings, 'local_path', None)
                        )
                        
                        # Get collection compatibility info
                        compatibility_info = await collection_connector.analyze_collection_compatibility(collection_name)
                        if compatibility_info.get("exists"):
                            points_count = compatibility_info.get("points_count", 0)
                            print(f"   📊 Points: {points_count:,}")
                            
                            # Show vector information
                            available_vectors = compatibility_info.get("available_vectors", [])
                            if available_vectors:
                                print(f"   🔢 Vector names: {', '.join(available_vectors)}")
                            
                            # Show dimensions with compatibility status
                            expected_dimensions = compatibility_info.get("expected_dimensions", "Unknown")
                            actual_dimensions = compatibility_info.get("actual_dimensions", expected_dimensions)
                            dimension_compatible = compatibility_info.get("dimension_compatible", True)
                            
                            if dimension_compatible:
                                print(f"   📏 Dimensions: {actual_dimensions} ✅")
                            else:
                                print(f"   📏 Dimensions: {actual_dimensions} (expected: {expected_dimensions}) ⚠️")
                            
                            # Show embedding model
                            current_model = compatibility_info.get("current_model", "Unknown")
                            print(f"   🤖 Model: {current_model}")
                            
                            # Show content type information
                            has_chunked = compatibility_info.get("has_chunked_content", False)
                            has_non_chunked = compatibility_info.get("has_non_chunked_content", False)
                            mixed_content = compatibility_info.get("mixed_content", False)
                            
                            if mixed_content:
                                print(f"   📄 Content: Mixed (chunked + non-chunked)")
                            elif has_chunked:
                                print(f"   📄 Content: Chunked documents")
                            elif has_non_chunked:
                                print(f"   📄 Content: Non-chunked documents")
                            elif points_count == 0:
                                print(f"   📄 Content: Empty collection")
                            
                            # Show compatibility status
                            compatible = compatibility_info.get("compatible", True)
                            if compatible:
                                print(f"   ✅ Status: Compatible")
                            else:
                                print(f"   ⚠️  Status: Compatibility issues detected")
                                
                                # Show recommendations if any
                                recommendations = compatibility_info.get("recommendations", [])
                                if recommendations:
                                    print(f"   💡 Recommendations:")
                                    for rec in recommendations[:2]:  # Show first 2 recommendations
                                        print(f"      • {rec}")
                                    if len(recommendations) > 2:
                                        print(f"      • ... and {len(recommendations) - 2} more")
                        else:
                            print(f"   ❌ Collection does not exist")
                        
                    except Exception as e:
                        print(f"   ❌ Error getting details: {e}")
                
                print()  # Empty line between collections
                
                # Update progress
                self.progress_reporter.update_progress(1, f"Listed collection: {collection_name}")
            
            result.files_processed = len(collection_names)  # Use this to indicate collections listed
            result.success = True
            self.progress_reporter.finish_operation(result)
            return result
            
        except Exception as e:
            error_msg = f"List operation failed: {e}"
            result.errors.append(error_msg)
            self._log_error(error_msg)
            self.progress_reporter.finish_operation(result)
            return result

    def validate_preconditions(self) -> List[str]:
        """
        Validate preconditions for list operation.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate Qdrant settings
        if not self.config.qdrant_settings.location:
            errors.append("Qdrant URL is required")
        
        return errors


class CLIErrorHandler:
    """
    Centralized error handling for CLI operations.
    
    This class provides comprehensive error handling with:
    - Graceful error handling for file processing failures
    - Connection error handling with clear user messages
    - Error recovery mechanisms for transient failures
    - Categorized error reporting with actionable suggestions
    """
    
    def __init__(self, progress_reporter: Optional[ProgressReporter] = None):
        """
        Initialize error handler.
        
        Args:
            progress_reporter: Optional progress reporter for error logging
        """
        self.progress_reporter = progress_reporter
        self.error_counts = {
            'configuration': 0,
            'connection': 0,
            'file_processing': 0,
            'storage': 0,
            'validation': 0,
            'transient': 0,
            'unknown': 0
        }
        self.recovery_attempts = {}
        self.max_retry_attempts = 3
        self.retry_delay = 1.0  # seconds
    
    def handle_configuration_error(self, error: Exception, context: Optional[str] = None) -> bool:
        """
        Handle configuration-related errors.
        
        Args:
            error: The configuration error
            context: Additional context about where the error occurred
            
        Returns:
            True if error was handled gracefully, False if fatal
        """
        self.error_counts['configuration'] += 1
        
        if isinstance(error, ConfigurationValidationError):
            self._log_error(f"Configuration Error: {error.message}")
            if error.suggested_value:
                self._log_info(f"💡 Suggestion: Use {error.suggested_value}")
            if error.valid_options:
                self._log_info(f"Valid options: {', '.join(map(str, error.valid_options))}")
            return False  # Configuration errors are usually fatal
        
        elif isinstance(error, ValidationError):
            self._log_error(f"Configuration Validation Error: {str(error)}")
            self._log_info("💡 Check your command line arguments and environment variables")
            return False
        
        elif isinstance(error, ValueError) and "path" in str(error).lower():
            self._log_error(f"Path Error: {str(error)}")
            self._log_info("💡 Ensure the specified path exists and is accessible")
            return False
        
        else:
            self._log_error(f"Configuration Error: {str(error)}")
            if context:
                self._log_info(f"Context: {context}")
            self._log_info("💡 Check your configuration settings and try again")
            return False
    
    def handle_connection_error(self, error: Exception, qdrant_url: str, context: Optional[str] = None) -> bool:
        """
        Handle Qdrant connection errors with retry logic.
        
        Args:
            error: The connection error
            qdrant_url: The Qdrant URL that failed
            context: Additional context about the operation
            
        Returns:
            True if error was handled and retry is possible, False if fatal
        """
        self.error_counts['connection'] += 1
        error_key = f"connection_{qdrant_url}"
        
        # Check if we should retry
        retry_count = self.recovery_attempts.get(error_key, 0)
        if retry_count >= self.max_retry_attempts:
            self._log_error(f"Connection failed after {self.max_retry_attempts} attempts: {str(error)}")
            self._provide_connection_troubleshooting(qdrant_url, error)
            return False
        
        # Log the error with retry information
        if retry_count == 0:
            self._log_error(f"Connection Error: Failed to connect to Qdrant at {qdrant_url}")
            self._log_error(f"Error details: {str(error)}")
        else:
            self._log_warning(f"Connection retry {retry_count + 1}/{self.max_retry_attempts} failed: {str(error)}")
        
        # Increment retry count
        self.recovery_attempts[error_key] = retry_count + 1
        
        # Provide context-specific guidance
        if context:
            self._log_info(f"Context: {context}")
        
        # Check for specific error types
        if "connection refused" in str(error).lower():
            self._log_info("💡 Qdrant server may not be running. Check if the server is started.")
        elif "timeout" in str(error).lower():
            self._log_info("💡 Connection timeout. Check network connectivity and server load.")
        elif "authentication" in str(error).lower() or "unauthorized" in str(error).lower():
            self._log_info("💡 Authentication failed. Check your API key.")
        elif "not found" in str(error).lower():
            self._log_info("💡 Server not found. Check the URL and port number.")
        
        # Wait before retry
        if retry_count < self.max_retry_attempts - 1:
            import time
            self._log_info(f"⏳ Retrying in {self.retry_delay} seconds...")
            time.sleep(self.retry_delay)
            return True  # Indicate retry is possible
        
        return False
    
    def handle_file_processing_error(
        self, 
        error: Exception, 
        file_path: Path, 
        operation: str = "processing"
    ) -> bool:
        """
        Handle file processing errors with recovery strategies.
        
        Args:
            error: The file processing error
            file_path: Path of the file that failed
            operation: The operation that failed (e.g., "reading", "processing")
            
        Returns:
            True if error was handled and processing can continue, False if fatal
        """
        self.error_counts['file_processing'] += 1
        
        # Log the error
        self._log_error(f"File {operation} failed for {file_path}: {str(error)}")
        
        # Handle specific error types
        if isinstance(error, PermissionError):
            self._log_warning(f"⚠️  Permission denied: {file_path}")
            self._log_info("💡 Check file permissions or run with appropriate privileges")
            return True  # Continue with other files
        
        elif isinstance(error, FileNotFoundError):
            self._log_warning(f"⚠️  File not found: {file_path}")
            self._log_info("💡 File may have been moved or deleted during processing")
            return True  # Continue with other files
        
        elif isinstance(error, UnicodeDecodeError):
            self._log_warning(f"⚠️  Encoding error: {file_path}")
            self._log_info("💡 File may be binary or use an unsupported encoding")
            return True  # Continue with other files
        
        elif isinstance(error, OSError) and "too large" in str(error).lower():
            self._log_warning(f"⚠️  File too large: {file_path}")
            self._log_info("💡 Consider increasing max file size or excluding large files")
            return True  # Continue with other files
        
        elif isinstance(error, ValueError) and "empty" in str(error).lower():
            self._log_warning(f"⚠️  Empty file skipped: {file_path}")
            return True  # Continue with other files
        
        else:
            # Unknown file processing error
            self._log_warning(f"⚠️  Unexpected error processing {file_path}: {str(error)}")
            self._log_info("💡 File will be skipped, processing continues with remaining files")
            return True  # Continue with other files
    
    def handle_storage_error(
        self, 
        error: Exception, 
        entry_info: str, 
        collection_name: str,
        retry_possible: bool = True
    ) -> bool:
        """
        Handle storage errors with retry logic and recovery strategies.
        
        Args:
            error: The storage error
            entry_info: Information about the entry that failed to store
            collection_name: Name of the collection
            retry_possible: Whether retry is possible for this error
            
        Returns:
            True if error was handled and retry is possible, False if fatal
        """
        self.error_counts['storage'] += 1
        error_key = f"storage_{collection_name}_{entry_info}"
        
        # Handle specific MCP Qdrant errors
        if isinstance(error, VectorDimensionMismatchError):
            self._log_error(f"Vector Dimension Mismatch: {error.message}")
            self._log_info("💡 Solutions:")
            for suggestion in error.details.get('resolution_suggestions', []):
                self._log_info(f"   • {suggestion}")
            return False  # Dimension mismatches are fatal
        
        elif isinstance(error, CollectionAccessError):
            self._log_error(f"Collection Access Error: {error.message}")
            if error.available_collections:
                self._log_info(f"Available collections: {', '.join(error.available_collections)}")
            return False  # Collection access errors are usually fatal
        
        elif isinstance(error, ChunkingError):
            self._log_warning(f"Chunking Error: {error.message}")
            if error.fallback_used:
                self._log_info("✅ Fallback strategy applied, continuing...")
                return True  # Chunking errors with fallback can continue
            else:
                self._log_info("💡 Consider adjusting chunking parameters")
                return True  # Skip this entry but continue
        
        # Handle retry logic for transient errors
        if retry_possible:
            retry_count = self.recovery_attempts.get(error_key, 0)
            if retry_count >= self.max_retry_attempts:
                self._log_error(f"Storage failed after {self.max_retry_attempts} attempts: {entry_info}")
                return False
            
            # Check for transient error patterns
            error_str = str(error).lower()
            if any(pattern in error_str for pattern in ['timeout', 'connection', 'temporary', 'retry']):
                self.error_counts['transient'] += 1
                self.recovery_attempts[error_key] = retry_count + 1
                
                self._log_warning(f"Transient storage error (attempt {retry_count + 1}/{self.max_retry_attempts}): {str(error)}")
                self._log_info("⏳ Will retry after brief delay...")
                
                import time
                time.sleep(self.retry_delay * (retry_count + 1))  # Exponential backoff
                return True
        
        # Log non-retryable storage error
        self._log_error(f"Storage Error: {str(error)}")
        self._log_info(f"Entry will be skipped: {entry_info}")
        return True  # Continue with other entries
    
    def handle_validation_error(self, error: Exception, context: str) -> bool:
        """
        Handle validation errors with helpful suggestions.
        
        Args:
            error: The validation error
            context: Context where validation failed
            
        Returns:
            True if error was handled gracefully, False if fatal
        """
        self.error_counts['validation'] += 1
        
        if isinstance(error, ModelValidationError):
            self._log_error(f"Model Validation Error: {error.message}")
            if error.available_models:
                self._log_info("Available models:")
                for model in error.available_models[:10]:  # Show first 10
                    self._log_info(f"   • {model}")
                if len(error.available_models) > 10:
                    self._log_info(f"   ... and {len(error.available_models) - 10} more")
            return False  # Model validation errors are fatal
        
        elif isinstance(error, re.error):
            self._log_error(f"Regex Pattern Error in {context}: {str(error)}")
            self._log_info("💡 Check your include/exclude patterns for valid regex syntax")
            return False  # Regex errors are fatal
        
        else:
            self._log_error(f"Validation Error in {context}: {str(error)}")
            return False  # Most validation errors are fatal
    
    def handle_embedding_error(self, error: Exception, model_name: str) -> bool:
        """
        Handle embedding-related errors with model fallback strategies.
        
        Args:
            error: The embedding error
            model_name: Name of the embedding model that failed
            
        Returns:
            True if fallback is possible, False if fatal
        """
        self.error_counts['unknown'] += 1  # Will be recategorized if needed
        
        if isinstance(error, ModelValidationError):
            return self.handle_validation_error(error, f"embedding model '{model_name}'")
        
        elif "not found" in str(error).lower() or "not available" in str(error).lower():
            self._log_error(f"Embedding model '{model_name}' not found or not available")
            self._log_info("💡 Try installing the model or using a different model")
            self._log_info("💡 Use 'qdrant-ingest list-models' to see available models")
            return False
        
        elif "download" in str(error).lower() or "network" in str(error).lower():
            self._log_error(f"Failed to download embedding model '{model_name}': {str(error)}")
            self._log_info("💡 Check your internet connection and try again")
            self._log_info("💡 Model will be downloaded on first use")
            return False
        
        else:
            self._log_error(f"Embedding Error with model '{model_name}': {str(error)}")
            return False
    
    def handle_unknown_error(self, error: Exception, context: str) -> bool:
        """
        Handle unknown/unexpected errors with generic recovery strategies.
        
        Args:
            error: The unknown error
            context: Context where the error occurred
            
        Returns:
            True if processing can continue, False if fatal
        """
        self.error_counts['unknown'] += 1
        
        self._log_error(f"Unexpected Error in {context}: {str(error)}")
        self._log_error(f"Error type: {type(error).__name__}")
        
        # Provide generic troubleshooting advice
        self._log_info("💡 Troubleshooting suggestions:")
        self._log_info("   • Check your configuration settings")
        self._log_info("   • Verify network connectivity")
        self._log_info("   • Try running with --verbose for more details")
        self._log_info("   • Consider using --dry-run to test configuration")
        
        # For unknown errors, be conservative and suggest stopping
        return False
    
    def should_continue_processing(self) -> bool:
        """
        Determine if processing should continue based on error patterns.
        
        Returns:
            True if processing should continue, False if too many errors
        """
        total_errors = sum(self.error_counts.values())
        
        # Stop if too many configuration or connection errors
        if self.error_counts['configuration'] > 0 or self.error_counts['connection'] > 5:
            return False
        
        # Stop if too many unknown errors
        if self.error_counts['unknown'] > 10:
            self._log_error("Too many unknown errors encountered, stopping processing")
            return False
        
        # Continue if mostly file processing errors (these are expected)
        return True
    
    def get_error_summary(self) -> dict:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Dictionary with error statistics and recommendations
        """
        total_errors = sum(self.error_counts.values())
        
        summary = {
            'total_errors': total_errors,
            'error_counts': self.error_counts.copy(),
            'retry_attempts': len(self.recovery_attempts),
            'recommendations': []
        }
        
        # Add recommendations based on error patterns
        if self.error_counts['configuration'] > 0:
            summary['recommendations'].append("Review configuration settings and command line arguments")
        
        if self.error_counts['connection'] > 0:
            summary['recommendations'].append("Check Qdrant server status and network connectivity")
        
        if self.error_counts['file_processing'] > 5:
            summary['recommendations'].append("Consider adjusting file filters or checking file permissions")
        
        if self.error_counts['storage'] > 0:
            summary['recommendations'].append("Verify Qdrant collection configuration and available space")
        
        if self.error_counts['transient'] > 0:
            summary['recommendations'].append("Transient errors occurred - consider retrying the operation")
        
        return summary
    
    def _provide_connection_troubleshooting(self, qdrant_url: str, error: Exception) -> None:
        """
        Provide detailed connection troubleshooting guidance.
        
        Args:
            qdrant_url: The Qdrant URL that failed
            error: The connection error
        """
        self._log_info("🔧 Connection Troubleshooting:")
        self._log_info(f"   • Verify Qdrant server is running at {qdrant_url}")
        self._log_info("   • Check if the port is correct (default: 6333)")
        self._log_info("   • Verify network connectivity and firewall settings")
        
        if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
            self._log_info("   • For local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        else:
            self._log_info("   • For remote Qdrant: check API key and URL format")
        
        self._log_info("   • Test connection: curl http://localhost:6333/health")
    
    def _log_error(self, message: str) -> None:
        """Log an error message."""
        if self.progress_reporter:
            self.progress_reporter.log_error(message)
        else:
            print(f"❌ {message}", file=sys.stderr)
    
    def _log_warning(self, message: str) -> None:
        """Log a warning message."""
        if self.progress_reporter:
            self.progress_reporter.log_warning(message)
        else:
            print(f"⚠️  {message}")
    
    def _log_info(self, message: str) -> None:
        """Log an info message."""
        if self.progress_reporter:
            self.progress_reporter.log_info(message)
        else:
            print(f"ℹ️  {message}")


class FileProcessor(ABC):
    """Abstract base class for file processing operations."""

    @abstractmethod
    async def process_file(self, file_info: FileInfo) -> Optional[Any]:
        """Process a single file and return the result."""
        pass

    @abstractmethod
    def can_process_file(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file."""
        pass


class CLIArgumentParser:
    """Handles CLI argument parsing and validation."""

    def __init__(self):
        """Initialize the argument parser."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            prog="qdrant-ingest",
            description="Ingest files into Qdrant vector database for semantic search",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  qdrant-ingest /path/to/documents
  qdrant-ingest /path/to/documents --knowledgebase my-docs
  qdrant-ingest update /path/to/documents --mode add-only
  qdrant-ingest remove my-docs --force
  qdrant-ingest list
            """,
        )

        # Add subcommands
        subparsers = parser.add_subparsers(
            dest="command", help="Available commands", metavar="COMMAND", required=False
        )

        # Ingest command (default)
        ingest_parser = subparsers.add_parser(
            "ingest", help="Ingest files into knowledge base (default)"
        )
        self._add_common_arguments(ingest_parser)
        self._add_path_argument(ingest_parser)
        self._add_file_filtering_arguments(ingest_parser)

        # Update command
        update_parser = subparsers.add_parser(
            "update", help="Update existing knowledge base"
        )
        self._add_common_arguments(update_parser)
        self._add_path_argument(update_parser)
        self._add_file_filtering_arguments(update_parser)
        update_parser.add_argument(
            "--mode",
            choices=["add-only", "replace"],
            default="add-only",
            help="Update mode: add-only (default) or replace existing content",
        )

        # Remove command
        remove_parser = subparsers.add_parser(
            "remove", help="Remove knowledge base"
        )
        self._add_qdrant_arguments(remove_parser)
        remove_parser.add_argument(
            "knowledgebase", help="Name of the knowledge base to remove"
        )
        remove_parser.add_argument(
            "--force", action="store_true", help="Skip confirmation prompts"
        )

        # List command
        list_parser = subparsers.add_parser(
            "list", help="List available knowledge bases"
        )
        self._add_qdrant_arguments(list_parser)

        # Add default arguments to main parser for when no subcommand is used
        # Note: We don't add path argument to main parser to avoid conflicts

        return parser

    def _add_common_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add common arguments to a parser."""
        self._add_qdrant_arguments(parser)
        self._add_embedding_arguments(parser)
        
        parser.add_argument(
            "--knowledgebase",
            help="Knowledge base name (default: derived from PATH)",
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be processed without making changes",
        )

    def _add_qdrant_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add Qdrant connection arguments."""
        parser.add_argument(
            "--url",
            default="http://localhost:6333",
            help="Qdrant server URL (default: http://localhost:6333)",
        )
        parser.add_argument("--api-key", help="Qdrant API key")

    def _add_embedding_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add embedding model arguments."""
        parser.add_argument(
            "--embedding",
            default="nomic-ai/nomic-embed-text-v1.5-Q",
            help="Embedding model (default: nomic-ai/nomic-embed-text-v1.5-Q)",
        )

    def _add_path_argument(
        self, parser: argparse.ArgumentParser, required: bool = True
    ) -> None:
        """Add path argument."""
        parser.add_argument(
            "path",
            nargs="?" if not required else None,
            help="Path to files or directory to process",
        )

    def _add_file_filtering_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add file filtering arguments."""
        parser.add_argument(
            "--include",
            action="append",
            dest="include_patterns",
            help="Include files matching regex pattern (can be used multiple times)",
        )
        parser.add_argument(
            "--exclude",
            action="append",
            dest="exclude_patterns",
            help="Exclude files matching regex pattern (can be used multiple times)",
        )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        # Handle default command (ingest when no subcommand specified)
        if args is None:
            args = sys.argv[1:]
            
        # Check if first argument looks like a path (not a command)
        if (args and len(args) > 0 and 
            not args[0] in ["ingest", "update", "remove", "list"] and
            not args[0].startswith("-")):
            # If first argument is not a command and not a flag, assume it's a path for ingest
            args = ["ingest"] + args
        
        parsed_args = self.parser.parse_args(args)
        
        # Set default command if none specified
        if parsed_args.command is None:
            parsed_args.command = "ingest"
            
        return parsed_args


class CLIValidator:
    """Validates CLI arguments and configuration."""

    def __init__(self):
        """Initialize the validator."""
        pass

    def validate_args(self, args: argparse.Namespace) -> List[str]:
        """Validate parsed arguments and return list of error messages."""
        errors = []

        # Validate command-specific requirements
        if args.command in ["ingest", "update"]:
            errors.extend(self._validate_path_command(args))
        elif args.command == "remove":
            errors.extend(self._validate_remove_command(args))
        elif args.command == "list":
            errors.extend(self._validate_list_command(args))

        # Validate common arguments
        errors.extend(self._validate_qdrant_url(args.url))
        errors.extend(self._validate_regex_patterns(args))

        return errors

    def _validate_path_command(self, args: argparse.Namespace) -> List[str]:
        """Validate arguments for path-based commands (ingest, update)."""
        errors = []

        # Path is required for ingest and update commands
        if not args.path:
            errors.append(f"PATH argument is required for '{args.command}' command")
            return errors

        # Validate path exists
        path = Path(args.path)
        if not path.exists():
            errors.append(f"Path does not exist: {args.path}")
        elif not (path.is_file() or path.is_dir()):
            errors.append(f"Path must be a file or directory: {args.path}")

        # Validate knowledgebase name derivation
        if path.exists():
            try:
                derived_name = self._derive_knowledgebase_name(path)
                if not derived_name or not derived_name.strip():
                    errors.append(
                        f"Cannot derive knowledgebase name from path: {args.path}. "
                        "Please specify --knowledgebase explicitly."
                    )
            except Exception as e:
                errors.append(f"Error deriving knowledgebase name: {e}")

        return errors

    def _validate_remove_command(self, args: argparse.Namespace) -> List[str]:
        """Validate arguments for remove command."""
        errors = []

        if not args.knowledgebase:
            errors.append("Knowledgebase name is required for 'remove' command")
        elif not self._is_valid_knowledgebase_name(args.knowledgebase):
            errors.append(
                f"Invalid knowledgebase name: {args.knowledgebase}. "
                "Name must contain only alphanumeric characters, hyphens, and underscores."
            )

        return errors

    def _validate_list_command(self, args: argparse.Namespace) -> List[str]:
        """Validate arguments for list command."""
        # List command only needs Qdrant connection, no additional validation
        return []

    def _validate_qdrant_url(self, url: str) -> List[str]:
        """Validate Qdrant URL format."""
        errors = []

        if not url:
            errors.append("Qdrant URL cannot be empty")
            return errors

        # Basic URL validation
        if not (url.startswith("http://") or url.startswith("https://")):
            errors.append(
                f"Invalid Qdrant URL format: {url}. Must start with http:// or https://"
            )

        return errors

    def _validate_regex_patterns(self, args: argparse.Namespace) -> List[str]:
        """Validate regex patterns for include/exclude filters."""
        errors = []

        # Validate include patterns
        if hasattr(args, "include_patterns") and args.include_patterns:
            for pattern in args.include_patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"Invalid include pattern '{pattern}': {e}")

        # Validate exclude patterns
        if hasattr(args, "exclude_patterns") and args.exclude_patterns:
            for pattern in args.exclude_patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    errors.append(f"Invalid exclude pattern '{pattern}': {e}")

        return errors

    def _derive_knowledgebase_name(self, path: Path) -> str:
        """Derive knowledgebase name from path."""
        if path.is_dir():
            return path.name
        else:
            return path.stem

    def _is_valid_knowledgebase_name(self, name: str) -> bool:
        """Check if knowledgebase name is valid."""
        # Allow alphanumeric characters, hyphens, and underscores
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


class CLIConfigBuilder:
    """Builds configuration objects from parsed CLI arguments."""

    def build_config(self, args: argparse.Namespace) -> IngestConfig:
        """
        Build IngestConfig from parsed arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Complete IngestConfig object
            
        Raises:
            ValueError: If configuration cannot be built from arguments
        """
        try:
            # Build CLI settings
            cli_settings = self._build_cli_settings(args)
            
            # Build Qdrant settings
            qdrant_settings = self._build_qdrant_settings(args)
            
            # Build embedding settings
            embedding_settings = self._build_embedding_settings(args)
            
            # Determine target path and knowledgebase name
            target_path, knowledgebase_name = self._determine_path_and_name(args, cli_settings)

            return IngestConfig(
                cli_settings=cli_settings,
                qdrant_settings=qdrant_settings,
                embedding_settings=embedding_settings,
                target_path=target_path,
                knowledgebase_name=knowledgebase_name,
            )
        except Exception as e:
            raise ValueError(f"Failed to build configuration: {e}") from e
    
    def _build_cli_settings(self, args: argparse.Namespace) -> CLISettings:
        """Build CLI settings from arguments."""
        return CLISettings(
            include_patterns=getattr(args, "include_patterns", []) or [],
            exclude_patterns=getattr(args, "exclude_patterns", []) or [],
            operation_mode=args.command,
            update_mode=getattr(args, "mode", "add-only"),
            force_operation=getattr(args, "force", False),
            dry_run=getattr(args, "dry_run", False),
            verbose=getattr(args, "verbose", False),
        )
    
    def _build_qdrant_settings(self, args: argparse.Namespace) -> QdrantSettings:
        """Build Qdrant settings from arguments."""
        # Create settings with explicit field assignment to avoid validation issues
        qdrant_settings = QdrantSettings()
        qdrant_settings.location = args.url
        qdrant_settings.api_key = getattr(args, "api_key", None)
        return qdrant_settings
    
    def _build_embedding_settings(self, args: argparse.Namespace) -> EmbeddingProviderSettings:
        """Build embedding settings from arguments."""
        # Create settings with explicit field assignment to avoid validation issues
        embedding_settings = EmbeddingProviderSettings()
        embedding_settings.model_name = getattr(args, "embedding", "nomic-ai/nomic-embed-text-v1.5-Q")
        return embedding_settings
    
    def _determine_path_and_name(self, args: argparse.Namespace, cli_settings: CLISettings) -> tuple[Optional[Path], Optional[str]]:
        """
        Determine target path and knowledgebase name based on command and arguments.
        
        Args:
            args: Parsed arguments
            cli_settings: CLI settings for name derivation
            
        Returns:
            Tuple of (target_path, knowledgebase_name)
        """
        target_path = None
        knowledgebase_name = getattr(args, "knowledgebase", None)

        if args.command in ["ingest", "update"]:
            if not args.path:
                raise ValueError(f"Path is required for '{args.command}' command")
            target_path = Path(args.path)
            
            # Derive knowledgebase name if not explicitly provided
            if not knowledgebase_name:
                knowledgebase_name = cli_settings.derive_knowledgebase_name(target_path)
                
        elif args.command == "remove":
            if not hasattr(args, "knowledgebase") or not args.knowledgebase:
                raise ValueError("Knowledgebase name is required for 'remove' command")
            knowledgebase_name = args.knowledgebase
            # No target path needed for remove
            
        elif args.command == "list":
            # No target path or knowledgebase name needed for list
            pass

        return target_path, knowledgebase_name


class ConfigurationManager:
    """
    Manages configuration composition and validation for CLI operations.
    
    This class provides a high-level interface for creating and validating
    configurations from various sources (CLI args, environment variables, etc.).
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.builder = CLIConfigBuilder()
        self.validator = CLIValidator()
        self._embedding_intelligence: Optional[EmbeddingModelIntelligence] = None
    
    def create_config_from_args(self, args: argparse.Namespace) -> IngestConfig:
        """
        Create a complete configuration from parsed CLI arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Validated IngestConfig object
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate arguments first
        validation_errors = self.validator.validate_args(args)
        if validation_errors:
            raise ValueError(f"Invalid arguments: {'; '.join(validation_errors)}")
        
        # Build configuration
        config = self.builder.build_config(args)
        
        # Validate the built configuration
        config_errors = config.validate_for_operation()
        if config_errors:
            raise ValueError(f"Invalid configuration: {'; '.join(config_errors)}")
        
        return config
    
    def create_config_with_overrides(
        self, 
        base_config: IngestConfig, 
        **overrides
    ) -> IngestConfig:
        """
        Create a new configuration based on an existing one with overrides.
        
        Args:
            base_config: Base configuration to copy from
            **overrides: Field overrides to apply
            
        Returns:
            New IngestConfig with overrides applied
        """
        # Create a copy of the base configuration
        new_config = IngestConfig(
            cli_settings=base_config.cli_settings,
            qdrant_settings=base_config.qdrant_settings,
            embedding_settings=base_config.embedding_settings,
            target_path=base_config.target_path,
            knowledgebase_name=base_config.knowledgebase_name,
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown configuration field: {key}")
        
        # Re-validate after overrides
        config_errors = new_config.validate_for_operation()
        if config_errors:
            raise ValueError(f"Invalid configuration after overrides: {'; '.join(config_errors)}")
        
        return new_config
    
    async def create_intelligent_config_from_args(self, args: argparse.Namespace) -> IngestConfig:
        """
        Create a configuration with intelligent embedding model selection.
        
        This method uses embedding model intelligence to:
        - Detect existing models from collections
        - Select smart defaults for new collections
        - Validate model compatibility
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Validated IngestConfig with intelligent model selection
            
        Raises:
            ValueError: If configuration is invalid or model is incompatible
        """
        # Validate arguments first
        validation_errors = self.validator.validate_args(args)
        if validation_errors:
            raise ValueError(f"Invalid arguments: {'; '.join(validation_errors)}")
        
        # Build initial configuration
        config = self.builder.build_config(args)
        
        # Initialize embedding intelligence if needed
        if self._embedding_intelligence is None:
            self._embedding_intelligence = EmbeddingModelIntelligence(config.qdrant_settings)
        
        # Apply intelligent embedding model selection
        config = await self._apply_intelligent_embedding_selection(config, args)
        
        # Validate the final configuration
        config_errors = config.validate_for_operation()
        if config_errors:
            raise ValueError(f"Invalid configuration: {'; '.join(config_errors)}")
        
        return config
    
    async def _apply_intelligent_embedding_selection(
        self, 
        config: IngestConfig, 
        args: argparse.Namespace
    ) -> IngestConfig:
        """
        Apply intelligent embedding model selection to the configuration.
        
        Args:
            config: Base configuration
            args: Original command line arguments
            
        Returns:
            Updated configuration with intelligent model selection
            
        Raises:
            ValueError: If model selection fails or model is incompatible
        """
        # Check if user explicitly specified an embedding model
        # We need to check if the embedding was explicitly provided by the user
        # vs being set to the default by the argument parser
        user_specified_model = (
            hasattr(args, 'embedding') and 
            args.embedding != EmbeddingModelIntelligence.DEFAULT_MODEL and
            # Additional check: if it's the default value, assume it wasn't user-specified
            # unless we have a way to track this explicitly
            getattr(args, '_embedding_explicitly_set', False)
        )
        
        if config.cli_settings.operation_mode in ["ingest", "update"] and config.knowledgebase_name:
            if user_specified_model:
                # User specified a model, validate compatibility
                model_info = await self._embedding_intelligence.validate_model_compatibility(
                    args.embedding, config.knowledgebase_name
                )
                
                if not model_info.is_available:
                    if config.cli_settings.verbose:
                        self._embedding_intelligence.display_model_info(model_info, verbose=True)
                    raise ValueError(f"Specified embedding model '{args.embedding}' is not available: {model_info.error_message}")
                
                if not model_info.is_compatible:
                    if config.cli_settings.verbose:
                        self._embedding_intelligence.display_model_mismatch_error(model_info)
                    raise ValueError(f"Specified embedding model '{args.embedding}' is incompatible with existing collection")
                
                # Model is valid and compatible, display info if verbose
                if config.cli_settings.verbose:
                    self._embedding_intelligence.display_model_info(model_info, verbose=True)
                
            else:
                # No model specified, use smart default selection
                model_info = await self._embedding_intelligence.select_smart_default(config.knowledgebase_name)
                
                if not model_info.is_available:
                    if config.cli_settings.verbose:
                        self._embedding_intelligence.display_model_info(model_info, verbose=True)
                    raise ValueError(f"No compatible embedding model available: {model_info.error_message}")
                
                # Update configuration with selected model
                config.embedding_settings.model_name = model_info.model_name
                
                # Display selection info if verbose
                if config.cli_settings.verbose:
                    if model_info.collection_exists:
                        print(f"🔍 Detected existing collection model: {model_info.model_name}")
                    else:
                        print(f"🎯 Selected default model for new collection: {model_info.model_name}")
                    self._embedding_intelligence.display_model_info(model_info, verbose=True)
        
        elif config.cli_settings.operation_mode == "list":
            # For list operations, we don't need model validation
            pass
        
        elif config.cli_settings.operation_mode == "remove":
            # For remove operations, we don't need model validation
            pass
        
        return config


async def parse_and_validate_args_intelligent(args: Optional[List[str]] = None) -> IngestConfig:
    """
    Parse and validate CLI arguments, returning configuration.
    
    Args:
        args: Command line arguments (None to use sys.argv)
        
    Returns:
        Validated IngestConfig object
        
    Raises:
        SystemExit: If arguments are invalid or configuration cannot be built
    """
    parser = CLIArgumentParser()
    config_manager = ConfigurationManager()

    try:
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Create and validate configuration with intelligent model selection
        config = await config_manager.create_intelligent_config_from_args(parsed_args)
        
        return config
        
    except ValueError as e:
        print("Error: Configuration validation failed:", file=sys.stderr)
        print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print("Error: Pydantic validation failed:", file=sys.stderr)
        for error in e.errors():
            print(f"  - {error['msg']}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error during configuration: {e}", file=sys.stderr)
        sys.exit(1)


def parse_and_validate_args(args: Optional[List[str]] = None) -> IngestConfig:
    """
    Synchronous wrapper for parse_and_validate_args_intelligent.
    
    Args:
        args: Command line arguments (None to use sys.argv)
        
    Returns:
        Validated IngestConfig object
        
    Raises:
        SystemExit: If arguments are invalid or configuration cannot be built
    """
    return asyncio.run(parse_and_validate_args_intelligent(args))


async def main_async():
    """
    Async main entry point for the CLI tool.
    
    This function handles:
    - Command routing based on operation mode
    - Operation result aggregation and display
    - Error handling and user feedback
    - Exit code management
    """
    try:
        # Parse and validate configuration
        config = await parse_and_validate_args_intelligent()
        
        # Create progress reporter
        progress_reporter = ProgressReporter(
            show_progress=config.cli_settings.show_progress,
            verbose=config.cli_settings.verbose,
            batch_size=config.cli_settings.batch_size
        )
        
        # Show configuration summary if verbose
        if config.cli_settings.verbose:
            progress_reporter.log_info("Configuration Summary:")
            settings_summary = config.get_effective_settings_summary()
            for key, value in settings_summary.items():
                if value is not None:
                    progress_reporter._log_verbose(f"  {key}: {value}")
        
        # Route to appropriate operation
        operation = await _create_operation(config, progress_reporter)
        
        # Execute operation
        result = await operation.execute()
        
        # Display final results and determine exit code
        exit_code = _handle_operation_result(result, progress_reporter)
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n🚫 Operation cancelled by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
    except ConfigurationValidationError as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        sys.exit(2)  # Configuration error
    except MCPQdrantError as e:
        print(f"❌ Qdrant error: {e}", file=sys.stderr)
        sys.exit(3)  # Qdrant-specific error
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if os.getenv("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)  # General error


async def _create_operation(config: IngestConfig, progress_reporter: ProgressReporter):
    """
    Create the appropriate operation instance based on configuration.
    
    Args:
        config: Complete CLI configuration
        progress_reporter: Progress reporter instance (for main-level reporting)
        
    Returns:
        Operation instance ready for execution
        
    Raises:
        ValueError: If operation mode is not supported
    """
    operation_mode = config.cli_settings.operation_mode
    
    if operation_mode == "ingest":
        return IngestOperation(config)
    elif operation_mode == "update":
        return UpdateOperation(config)
    elif operation_mode == "remove":
        return RemoveOperation(config)
    elif operation_mode == "list":
        return ListOperation(config)
    else:
        raise ValueError(f"Unsupported operation mode: {operation_mode}")


def _handle_operation_result(result: OperationResult, progress_reporter: ProgressReporter) -> int:
    """
    Handle operation result and determine appropriate exit code.
    
    Args:
        result: Operation result with statistics and status
        progress_reporter: Progress reporter for final messages
        
    Returns:
        Exit code (0 for success, non-zero for various error conditions)
    """
    if result.success:
        if result.files_failed == 0 and len(result.errors) == 0:
            # Complete success
            progress_reporter.log_success("Operation completed successfully!")
            return 0
        else:
            # Partial success - some files failed but operation didn't abort
            progress_reporter.log_warning("Operation completed with some failures")
            return 1
    else:
        # Operation failed
        if result.files_processed == 0 and result.total_files > 0:
            # Complete failure - no files processed
            progress_reporter.log_error("Operation failed - no files were processed")
            return 4
        else:
            # Partial failure - some files processed but operation failed overall
            progress_reporter.log_error("Operation failed with partial results")
            return 5


def main():
    """Main entry point for the CLI tool."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

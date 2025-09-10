"""
Unit tests for CLI operation classes.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import List

from src.mcp_server_qdrant_rag.cli_ingest import (
    BaseOperation,
    IngestOperation,
    UpdateOperation,
    RemoveOperation,
    ListOperation,
    OperationResult,
    FileInfo,
)

def test_operation_result_creation():
    """Test OperationResult creation and properties."""
    from src.mcp_server_qdrant_rag.cli_ingest import OperationResult
    
    result = OperationResult(success=True, files_processed=5, files_failed=1)
    
    assert result.success is True
    assert result.files_processed == 5
    assert result.files_failed == 1
    assert result.total_files == 6  # processed + skipped + failed
    assert result.success_rate == (5/6) * 100  # 83.33%


def test_file_info_creation(tmp_path):
    """Test FileInfo creation."""
    from src.mcp_server_qdrant_rag.cli_ingest import FileInfo
    
    # Create a real temporary file for testing
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    test_time = datetime.now()
    
    file_info = FileInfo(
        path=test_file,
        size=1000,
        modified_time=test_time,
        encoding="utf-8",
        is_binary=False,
        estimated_tokens=250
    )
    
    assert file_info.path == test_file
    assert file_info.size == 1000
    assert file_info.modified_time == test_time
    assert file_info.encoding == "utf-8"
    assert file_info.is_binary is False
    assert file_info.estimated_tokens == 250


class TestBaseOperationFunctionality:
    """Test BaseOperation functionality."""
    
    def test_base_operation_initialization(self):
        """Test BaseOperation initialization."""
        from src.mcp_server_qdrant_rag.cli_ingest import BaseOperation, OperationResult
        
        # Create a concrete implementation for testing
        class ConcreteOperation(BaseOperation):
            async def execute(self) -> OperationResult:
                return OperationResult(success=True)
            
            def validate_preconditions(self) -> List[str]:
                return []
        
        config = MagicMock()
        operation = ConcreteOperation(config)
        
        assert operation.config == config
        assert operation._connector is None
        assert operation._embedding_provider is None
    
    def test_confirm_operation_force_mode(self):
        """Test confirmation in force mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import BaseOperation, OperationResult
        
        class ConcreteOperation(BaseOperation):
            async def execute(self) -> OperationResult:
                return OperationResult(success=True)
            
            def validate_preconditions(self) -> List[str]:
                return []
        
        config = MagicMock()
        config.cli_settings.force_operation = True
        operation = ConcreteOperation(config)
        
        result = operation._confirm_operation("Test confirmation")
        assert result is True


class TestIngestOperationFunctionality:
    """Test IngestOperation functionality."""
    
    def test_validate_preconditions_missing_path(self):
        """Test validation with missing target path."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        config = MagicMock()
        config.target_path = None
        config.knowledgebase_name = "test-kb"
        config.qdrant_settings.location = "http://localhost:6333"
        config.embedding_settings.model_name = "test-model"
        
        operation = IngestOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Target path does not exist" in error for error in errors)
    
    def test_validate_preconditions_missing_knowledgebase(self, tmp_path):
        """Test validation with missing knowledgebase name."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        config = MagicMock()
        config.target_path = test_file
        config.knowledgebase_name = None
        config.qdrant_settings.location = "http://localhost:6333"
        config.embedding_settings.model_name = "test-model"
        
        operation = IngestOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Knowledgebase name is required" in error for error in errors)


class TestUpdateOperationFunctionality:
    """Test UpdateOperation functionality."""
    
    def test_validate_preconditions_invalid_mode(self, tmp_path):
        """Test validation with invalid update mode."""
        from src.mcp_server_qdrant_rag.cli_ingest import UpdateOperation
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        config = MagicMock()
        config.target_path = test_file
        config.knowledgebase_name = "test-kb"
        config.qdrant_settings.location = "http://localhost:6333"
        config.embedding_settings.model_name = "test-model"
        config.cli_settings.update_mode = "invalid-mode"
        
        operation = UpdateOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Invalid update mode" in error for error in errors)


class TestRemoveOperationFunctionality:
    """Test RemoveOperation functionality."""
    
    def test_validate_preconditions_success(self):
        """Test successful precondition validation."""
        from src.mcp_server_qdrant_rag.cli_ingest import RemoveOperation
        
        config = MagicMock()
        config.knowledgebase_name = "test-kb"
        config.qdrant_settings.location = "http://localhost:6333"
        
        operation = RemoveOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) == 0
    
    def test_validate_preconditions_missing_knowledgebase(self):
        """Test validation with missing knowledgebase name."""
        from src.mcp_server_qdrant_rag.cli_ingest import RemoveOperation
        
        config = MagicMock()
        config.knowledgebase_name = None
        config.qdrant_settings.location = "http://localhost:6333"
        
        operation = RemoveOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Knowledgebase name is required" in error for error in errors)


class TestListOperationFunctionality:
    """Test ListOperation functionality."""
    
    def test_validate_preconditions_success(self):
        """Test successful precondition validation."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = "http://localhost:6333"
        
        operation = ListOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) == 0
    
    def test_validate_preconditions_missing_url(self):
        """Test validation with missing Qdrant URL."""
        from src.mcp_server_qdrant_rag.cli_ingest import ListOperation
        
        config = MagicMock()
        config.qdrant_settings.location = None
        
        operation = ListOperation(config)
        errors = operation.validate_preconditions()
        
        assert len(errors) >= 1
        assert any("Qdrant URL is required" in error for error in errors)
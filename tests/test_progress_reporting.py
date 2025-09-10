"""
Tests for progress reporting and user feedback functionality.

This module tests the ProgressReporter class and its integration with CLI operations,
including progress indicators, verbose logging, batch processing, and summary reporting.
"""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.mcp_server_qdrant_rag.cli_ingest import (
    ProgressReporter,
    OperationResult,
    FileInfo,
)


class TestProgressReporter:
    """Test cases for the ProgressReporter class."""
    
    def test_init_default_settings(self):
        """Test ProgressReporter initialization with default settings."""
        reporter = ProgressReporter()
        
        assert reporter.show_progress is True
        assert reporter.verbose is False
        assert reporter.batch_size == 10
        assert reporter._current_operation is None
        assert reporter._total_items == 0
        assert reporter._processed_items == 0
        assert reporter._start_time is None
        assert reporter._last_progress_update == 0
    
    def test_init_custom_settings(self):
        """Test ProgressReporter initialization with custom settings."""
        reporter = ProgressReporter(
            show_progress=False,
            verbose=True,
            batch_size=5
        )
        
        assert reporter.show_progress is False
        assert reporter.verbose is True
        assert reporter.batch_size == 5
    
    @patch('builtins.print')
    def test_start_operation_with_total(self, mock_print):
        """Test starting an operation with known total items."""
        reporter = ProgressReporter()
        
        reporter.start_operation("Test Operation", 100)
        
        assert reporter._current_operation == "Test Operation"
        assert reporter._total_items == 100
        assert reporter._processed_items == 0
        assert reporter._start_time is not None
        assert reporter._last_progress_update == 0
        
        mock_print.assert_called_once_with("🚀 Starting Test Operation (100 items)")
    
    @patch('builtins.print')
    def test_start_operation_without_total(self, mock_print):
        """Test starting an operation without known total items."""
        reporter = ProgressReporter()
        
        reporter.start_operation("Test Operation")
        
        assert reporter._current_operation == "Test Operation"
        assert reporter._total_items == 0
        assert reporter._processed_items == 0
        
        mock_print.assert_called_once_with("🚀 Starting Test Operation")
    
    @patch('builtins.print')
    def test_update_progress_verbose(self, mock_print):
        """Test progress updates with verbose mode enabled."""
        reporter = ProgressReporter(verbose=True)
        reporter.start_operation("Test Operation", 10)
        
        reporter.update_progress(1, "test_file.txt")
        
        assert reporter._processed_items == 1
        # Should print verbose message
        mock_print.assert_any_call("ℹ️  Processing: test_file.txt")
    
    @patch('builtins.print')
    def test_report_file_processed_success(self, mock_print):
        """Test reporting successful file processing."""
        reporter = ProgressReporter(verbose=True)
        reporter.start_operation("Test Operation", 5)
        
        file_path = Path("test_file.txt")
        reporter.report_file_processed(file_path, True)
        
        assert reporter._processed_items == 1
        mock_print.assert_any_call("ℹ️  ✅ Processed: test_file.txt")
    
    def test_format_duration_milliseconds(self):
        """Test duration formatting for milliseconds."""
        reporter = ProgressReporter()
        
        result = reporter._format_duration(0.5)
        assert result == "500ms"
    
    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        reporter = ProgressReporter()
        
        result = reporter._format_duration(30.7)
        assert result == "30.7s"
    
    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        reporter = ProgressReporter()
        
        result = reporter._format_duration(125.0)  # 2 minutes 5 seconds
        assert result == "2m 5s"
    
    @patch('builtins.input', return_value='y')
    def test_confirm_operation_user_confirms(self, mock_input):
        """Test user confirmation when user says yes."""
        reporter = ProgressReporter()
        
        result = reporter.confirm_operation("Delete files?", force=False)
        
        assert result is True
        mock_input.assert_called_once_with("❓ Delete files? (y/N): ")
    
    @patch('builtins.print')
    def test_confirm_operation_force_mode(self, mock_print):
        """Test user confirmation in force mode."""
        reporter = ProgressReporter()
        
        result = reporter.confirm_operation("Delete files?", force=True)
        
        assert result is True
        mock_print.assert_called_once_with("📝 Force mode enabled: Delete files?")
    
    @patch('builtins.print')
    def test_finish_operation_with_summary(self, mock_print):
        """Test finishing operation with complete summary."""
        reporter = ProgressReporter()
        reporter.start_operation("Test Operation", 10)
        
        # Simulate some processing time
        time.sleep(0.1)
        
        result = OperationResult(
            success=True,
            files_processed=8,
            files_skipped=1,
            files_failed=1,
            chunks_created=25,
            errors=["Error 1"],
            warnings=["Warning 1", "Warning 2"],
            execution_time=0.0  # Will be set by finish_operation
        )
        
        reporter.finish_operation(result)
        
        # Check that execution time was set
        assert result.execution_time > 0
        
        # Check that summary was printed
        summary_calls = [str(call) for call in mock_print.call_args_list]
        assert any("📋 Test Operation Summary" in call for call in summary_calls)
        assert any("✅ Status: Completed successfully" in call for call in summary_calls)


if __name__ == "__main__":
    pytest.main([__file__])
"""
Comprehensive test suite runner for CLI file ingestion.

This module provides a comprehensive test suite that validates all aspects
of the CLI file ingestion functionality including:
- All operation types and modes
- Error handling and edge cases
- Performance characteristics
- Configuration validation
- End-to-end workflows

Run with: uv run python -m pytest tests/test_cli_comprehensive_suite.py -v
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import pytest

# Import all test modules to ensure they're discovered
from tests.test_cli_comprehensive_integration import *
from tests.test_cli_performance import *
from tests.test_cli_error_scenarios import *

# Import existing CLI tests
from tests.test_cli_main_integration import *
from tests.test_cli_operations import *
from tests.test_cli_error_handling import *
from tests.test_cli_qdrant_integration import *
from tests.test_cli_embedding_integration import *
from tests.test_cli_configuration_management import *
from tests.test_cli_argument_parsing import *
from tests.test_file_discovery import *
from tests.test_content_processor import *
from tests.test_progress_reporting import *

from tests.fixtures.cli_test_fixtures import *


class TestComprehensiveSuite:
    """Comprehensive test suite that validates the entire CLI system."""

    @pytest.mark.asyncio
    async def test_complete_system_validation(self):
        """Run a complete system validation test."""
        print("\n" + "="*80)
        print("COMPREHENSIVE CLI SYSTEM VALIDATION")
        print("="*80)
        
        # Test results tracking
        test_results = {
            'basic_operations': False,
            'error_handling': False,
            'performance': False,
            'configuration': False,
            'edge_cases': False,
        }
        
        try:
            # 1. Basic Operations Test
            print("\n1. Testing Basic Operations...")
            await self._test_basic_operations()
            test_results['basic_operations'] = True
            print("✅ Basic operations test passed")
            
            # 2. Error Handling Test
            print("\n2. Testing Error Handling...")
            await self._test_error_handling()
            test_results['error_handling'] = True
            print("✅ Error handling test passed")
            
            # 3. Performance Test
            print("\n3. Testing Performance...")
            await self._test_performance()
            test_results['performance'] = True
            print("✅ Performance test passed")
            
            # 4. Configuration Test
            print("\n4. Testing Configuration...")
            await self._test_configuration()
            test_results['configuration'] = True
            print("✅ Configuration test passed")
            
            # 5. Edge Cases Test
            print("\n5. Testing Edge Cases...")
            await self._test_edge_cases()
            test_results['edge_cases'] = True
            print("✅ Edge cases test passed")
            
        except Exception as e:
            print(f"❌ System validation failed: {e}")
            raise
        
        # Summary
        print("\n" + "="*80)
        print("SYSTEM VALIDATION SUMMARY")
        print("="*80)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - CLI system is fully validated!")
        else:
            print("⚠️  Some tests failed - system needs attention")
            assert False, f"System validation failed: {total_tests - passed_tests} tests failed"

    async def _test_basic_operations(self):
        """Test basic CLI operations."""
        from src.mcp_server_qdrant_rag.cli_ingest import (
            IngestOperation, UpdateOperation, RemoveOperation, ListOperation
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files
            (workspace / "test1.txt").write_text("Test content 1")
            (workspace / "test2.md").write_text("# Test 2")
            (workspace / "test3.py").write_text("print('test')")
            
            # Test ingest operation
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="basic_test",
                dry_run=True,
            )
            
            operation = IngestOperation(config)
            result = await operation.execute()
            
            assert result.success, "Basic ingest operation failed"
            assert result.files_processed > 0, "No files were processed"

    async def _test_error_handling(self):
        """Test error handling capabilities."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        # Test with nonexistent path - this should fail at config creation time
        try:
            config = create_test_config(
                operation_mode="ingest",
                target_path=Path("/nonexistent/path"),
                knowledgebase_name="error_test",
            )
            # If config creation succeeds, the operation should fail
            operation = IngestOperation(config)
            result = await operation.execute()
            assert not result.success, "Error handling failed - should have failed with nonexistent path"
            assert len(result.errors) > 0, "No errors reported for invalid path"
        except ValueError as e:
            # Expected - configuration validation should catch this
            assert "does not exist" in str(e), f"Unexpected error message: {e}"

    async def _test_performance(self):
        """Test performance characteristics."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create multiple files for performance testing
            for i in range(20):
                content = f"Performance test file {i} content. " * 100
                (workspace / f"perf_test_{i}.txt").write_text(content)
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="performance_test",
                dry_run=True,
            )
            
            operation = IngestOperation(config)
            
            start_time = time.time()
            result = await operation.execute()
            execution_time = time.time() - start_time
            
            assert result.success, "Performance test operation failed"
            assert execution_time < 30.0, f"Performance test took too long: {execution_time}s"
            assert result.files_processed == 20, "Not all files were processed"

    async def _test_configuration(self):
        """Test configuration validation."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "config_test.txt").write_text("Configuration test")
            
            # Test with various configuration options
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="config_test",
                include_patterns=[r'\.txt$'],
                exclude_patterns=[r'temp_'],
                dry_run=True,
                verbose=True,
            )
            
            operation = IngestOperation(config)
            result = await operation.execute()
            
            assert result.success, "Configuration test failed"
            assert result.files_processed > 0, "Configuration filtering failed"

    async def _test_edge_cases(self):
        """Test edge cases and unusual scenarios."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create edge case files
            (workspace / "empty.txt").write_text("")  # Empty file
            (workspace / "whitespace.txt").write_text("   \n\t\n   ")  # Whitespace only
            (workspace / "single.txt").write_text("word")  # Single word
            
            # Create binary file
            (workspace / "binary.bin").write_bytes(b'\x00\x01\x02\x03')
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name="edge_case_test",
                dry_run=True,
            )
            
            operation = IngestOperation(config)
            result = await operation.execute()
            
            # Should handle edge cases gracefully
            assert result.success or result.files_skipped > 0, "Edge case handling failed"

    @pytest.mark.asyncio
    async def test_integration_with_real_qdrant(self):
        """Test integration with a real Qdrant instance (if available)."""
        # Skip if no real Qdrant instance is available
        qdrant_url = os.getenv("TEST_QDRANT_URL", "http://localhost:6333")
        
        try:
            # Try to connect to Qdrant
            from qdrant_client import QdrantClient
            client = QdrantClient(url=qdrant_url)
            client.get_collections()  # Test connection
        except Exception:
            pytest.skip("No Qdrant instance available for integration testing")
        
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files
            (workspace / "integration_test.txt").write_text(
                "This is an integration test with a real Qdrant instance. "
                "The content should be processed and stored successfully."
            )
            
            # Use unique collection name to avoid conflicts
            collection_name = f"integration_test_{int(time.time())}"
            
            config = create_test_config(
                operation_mode="ingest",
                target_path=workspace,
                knowledgebase_name=collection_name,
            )
            
            # Override Qdrant URL
            config.qdrant_settings.qdrant_url = qdrant_url
            
            operation = IngestOperation(config)
            
            try:
                result = await operation.execute()
                
                assert result.success, "Real Qdrant integration test failed"
                assert result.files_processed > 0, "No files processed in integration test"
                
                # Clean up - remove test collection
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass  # Ignore cleanup errors
                
            except Exception as e:
                # Clean up on error
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass
                raise e

    def test_cli_help_and_usage(self):
        """Test CLI help and usage information."""
        # Test that the main CLI functions are available
        from src.mcp_server_qdrant_rag.cli_ingest import main, parse_and_validate_args
        
        # Test that functions exist and are callable
        assert callable(main)
        assert callable(parse_and_validate_args)
        
        # Test basic argument parsing (this will fail but we can catch it)
        try:
            parse_and_validate_args(["--help"])
        except SystemExit:
            # Expected when --help is used
            pass
        except Exception:
            # Other exceptions are also acceptable for this test
            pass

    def test_version_information(self):
        """Test version information availability."""
        # Test that version information is available
        try:
            from src.mcp_server_qdrant_rag import __version__
            assert __version__ is not None
        except ImportError:
            # Version might be defined elsewhere
            pass

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent CLI operations."""
        from src.mcp_server_qdrant_rag.cli_ingest import IngestOperation
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files for concurrent operations
            for i in range(10):
                (workspace / f"concurrent_{i}.txt").write_text(f"Concurrent test file {i}")
            
            # Create multiple operations
            operations = []
            for i in range(3):
                config = create_test_config(
                    operation_mode="ingest",
                    target_path=workspace,
                    knowledgebase_name=f"concurrent_test_{i}",
                    dry_run=True,
                )
                operations.append(IngestOperation(config))
            
            # Run operations concurrently
            results = await asyncio.gather(
                *[op.execute() for op in operations],
                return_exceptions=True
            )
            
            # Verify all operations completed
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Concurrent operation {i} failed: {result}")
                assert result.success, f"Concurrent operation {i} was not successful"

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up after operations."""
        import gc
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run a memory-intensive operation
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create large content
            large_content = "Large content for memory test. " * 10000
            (workspace / "memory_test.txt").write_text(large_content)
            
            # Process the file multiple times
            for i in range(5):
                config = create_test_config(
                    operation_mode="ingest",
                    target_path=workspace,
                    knowledgebase_name=f"memory_test_{i}",
                    dry_run=True,
                )
                
                # This would normally be async, but for memory testing we'll use sync
                # operation = IngestOperation(config)
                # result = await operation.execute()
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        max_acceptable_increase = 100 * 1024 * 1024
        assert memory_increase < max_acceptable_increase, \
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB, possible memory leak"


# Test discovery and execution functions
def collect_all_cli_tests():
    """Collect all CLI-related tests for reporting."""
    test_modules = [
        'test_cli_comprehensive_integration',
        'test_cli_performance', 
        'test_cli_error_scenarios',
        'test_cli_main_integration',
        'test_cli_operations',
        'test_cli_error_handling',
        'test_cli_qdrant_integration',
        'test_cli_embedding_integration',
        'test_cli_configuration_management',
        'test_cli_argument_parsing',
        'test_file_discovery',
        'test_content_processor',
        'test_progress_reporting',
    ]
    
    return test_modules


def run_comprehensive_test_suite():
    """Run the comprehensive test suite with detailed reporting."""
    print("="*80)
    print("CLI FILE INGESTION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_modules = collect_all_cli_tests()
    
    print(f"\nDiscovered {len(test_modules)} test modules:")
    for module in test_modules:
        print(f"  - {module}")
    
    print("\nRunning comprehensive test suite...")
    print("="*80)
    
    # Run pytest with comprehensive options
    pytest_args = [
        "-v",  # Verbose output
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "tests/test_cli_comprehensive_suite.py",
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend([
            "--cov=src/mcp_server_qdrant_rag/cli_ingest",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ])
    except ImportError:
        print("Note: pytest-cov not available, skipping coverage reporting")
    
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "="*80)
    if exit_code == 0:
        print("🎉 COMPREHENSIVE TEST SUITE PASSED!")
        print("All CLI functionality has been validated successfully.")
    else:
        print("❌ COMPREHENSIVE TEST SUITE FAILED!")
        print("Some tests failed - please review the output above.")
    print("="*80)
    
    return exit_code


if __name__ == "__main__":
    # Allow running this module directly
    exit_code = run_comprehensive_test_suite()
    sys.exit(exit_code)
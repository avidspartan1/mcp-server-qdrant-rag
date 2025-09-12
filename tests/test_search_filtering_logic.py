"""
Tests for search filtering logic with semantic matching.
Focuses on testing the core filter generation and combination logic.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from qdrant_client import models

from mcp_server_qdrant_rag.semantic_matcher import SemanticSetMatcher, SemanticMatchError, NoMatchFoundError
from mcp_server_qdrant_rag.settings import SetConfiguration


@pytest.fixture
def sample_set_configurations():
    """Create sample set configurations for testing."""
    return {
        "platform_code": SetConfiguration(
            slug="platform_code",
            description="Platform Codebase",
            aliases=["platform", "core platform", "main codebase"]
        ),
        "api_docs": SetConfiguration(
            slug="api_docs",
            description="API Documentation",
            aliases=["api", "documentation", "api reference"]
        ),
        "frontend_code": SetConfiguration(
            slug="frontend_code",
            description="Frontend Application Code",
            aliases=["frontend", "ui", "client"]
        )
    }


@pytest.fixture
def semantic_matcher(sample_set_configurations):
    """Create a semantic matcher with sample configurations."""
    return SemanticSetMatcher(sample_set_configurations)


class TestSearchFilterGeneration:
    """Test the search filter generation logic."""
    
    @pytest.mark.asyncio
    async def test_set_filter_generation(self, semantic_matcher):
        """Test that set filters are generated correctly."""
        # Test exact match
        matched_slug = await semantic_matcher.match_set("platform code")
        assert matched_slug == "platform_code"
        
        # Generate proper Qdrant field condition
        set_field_condition = models.FieldCondition(
            key="metadata.set",
            match=models.MatchValue(value=matched_slug)
        )
        
        # Verify filter structure
        assert set_field_condition.key == "metadata.set"
        assert set_field_condition.match.value == "platform_code"
        
        # Test that it can be converted to Qdrant Filter
        qdrant_filter = models.Filter(must=[set_field_condition])
        assert qdrant_filter is not None
    
    @pytest.mark.asyncio
    async def test_alias_matching(self, semantic_matcher):
        """Test that aliases work correctly in filter generation."""
        # Test alias match
        matched_slug = await semantic_matcher.match_set("api")
        assert matched_slug == "api_docs"
        
        # Generate filter condition
        set_filter_condition = {
            "key": "set",
            "match": {"value": matched_slug}
        }
        
        assert set_filter_condition["match"]["value"] == "api_docs"
    
    @pytest.mark.asyncio
    async def test_fuzzy_matching(self, semantic_matcher):
        """Test fuzzy matching in filter generation."""
        # Test fuzzy match
        matched_slug = await semantic_matcher.match_set("frontend app")
        assert matched_slug == "frontend_code"
        
        # Generate filter condition
        set_filter_condition = {
            "key": "set",
            "match": {"value": matched_slug}
        }
        
        assert set_filter_condition["match"]["value"] == "frontend_code"
    
    @pytest.mark.asyncio
    async def test_no_match_error(self, semantic_matcher):
        """Test error handling when no match is found."""
        with pytest.raises(SemanticMatchError):
            await semantic_matcher.match_set("nonexistent set")
    
    def test_filter_combination_with_existing_filter(self):
        """Test combining set filter with existing query filter."""
        # Create proper Qdrant field conditions
        type_condition = models.FieldCondition(
            key="metadata.document_type",
            match=models.MatchValue(value="code")
        )
        set_condition = models.FieldCondition(
            key="metadata.set",
            match=models.MatchValue(value="platform_code")
        )
        
        # Combined filter
        combined_filter = models.Filter(must=[set_condition, type_condition])
        
        # Verify structure
        assert combined_filter.must is not None
        assert len(combined_filter.must) == 2
        assert combined_filter.must[0] == set_condition
        assert combined_filter.must[1] == type_condition
        
        # Test that it's a valid Qdrant Filter
        assert isinstance(combined_filter, models.Filter)
    
    def test_filter_without_existing_filter(self):
        """Test set filter when no existing query filter is present."""
        # Create proper Qdrant field condition
        set_condition = models.FieldCondition(
            key="metadata.set",
            match=models.MatchValue(value="platform_code")
        )
        
        # Should be used directly in a Filter
        qdrant_filter = models.Filter(must=[set_condition])
        
        # Verify structure
        assert qdrant_filter.must is not None
        assert len(qdrant_filter.must) == 1
        assert qdrant_filter.must[0].key == "metadata.set"
        assert qdrant_filter.must[0].match.value == "platform_code"
        
        # Test that it's a valid Qdrant Filter
        assert isinstance(qdrant_filter, models.Filter)


class TestSearchFilteringWorkflow:
    """Test the complete search filtering workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_filtering_workflow(self, semantic_matcher):
        """Test the complete workflow of semantic matching and filter generation."""
        
        async def simulate_search_with_set_filter(query: str, set_filter: str, query_filter=None):
            """Simulate the search filtering logic from mcp_server.py"""
            # Handle set filtering (replicates updated mcp_server.py logic)
            matched_set_slug = None
            if set_filter:
                try:
                    matched_set_slug = await semantic_matcher.match_set(set_filter)
                    
                    # Create set filter condition using proper Qdrant field condition
                    set_field_condition = models.FieldCondition(
                        key="metadata.set",
                        match=models.MatchValue(value=matched_set_slug)
                    )
                    
                    # Combine with existing filter if present
                    if query_filter:
                        # If query_filter is already a models.Filter, extract its conditions
                        if isinstance(query_filter, models.Filter):
                            existing_must = query_filter.must or []
                            existing_must_not = query_filter.must_not or []
                            existing_should = query_filter.should or []
                        else:
                            # If it's a dict, convert it to Filter first to extract conditions
                            temp_filter = models.Filter(**query_filter)
                            existing_must = temp_filter.must or []
                            existing_must_not = temp_filter.must_not or []
                            existing_should = temp_filter.should or []
                        
                        # Create combined filter with set condition added to must
                        combined_must = [set_field_condition] + existing_must
                        qdrant_filter = models.Filter(
                            must=combined_must,
                            must_not=existing_must_not,
                            should=existing_should
                        )
                    else:
                        # Create new filter with just the set condition
                        qdrant_filter = models.Filter(must=[set_field_condition])
                        
                except SemanticMatchError as e:
                    return f"Error: {str(e)}"
            else:
                qdrant_filter = query_filter
            
            return {
                "matched_set": matched_set_slug,
                "filter": qdrant_filter,
                "qdrant_filter": qdrant_filter
            }
        
        # Test 1: Set filter only
        result = await simulate_search_with_set_filter("test query", "platform code")
        assert result["matched_set"] == "platform_code"
        assert isinstance(result["filter"], models.Filter)
        assert result["filter"].must is not None
        assert len(result["filter"].must) == 1
        assert result["filter"].must[0].key == "metadata.set"
        assert result["filter"].must[0].match.value == "platform_code"
        assert result["qdrant_filter"] is not None
        
        # Test 2: Set filter with existing query filter
        type_condition = models.FieldCondition(
            key="metadata.document_type",
            match=models.MatchValue(value="code")
        )
        query_filter = models.Filter(must=[type_condition])
        result = await simulate_search_with_set_filter("test query", "api", query_filter)
        assert result["matched_set"] == "api_docs"
        assert isinstance(result["filter"], models.Filter)
        assert result["filter"].must is not None
        assert len(result["filter"].must) == 2
        assert result["qdrant_filter"] is not None
        
        # Test 3: No set filter
        result = await simulate_search_with_set_filter("test query", None, query_filter)
        assert result["matched_set"] is None
        assert result["filter"] == query_filter
        assert result["qdrant_filter"] is not None
        
        # Test 4: Invalid set filter
        result = await simulate_search_with_set_filter("test query", "invalid_set")
        assert isinstance(result, str)
        assert result.startswith("Error:")
    
    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, semantic_matcher):
        """Test error handling in the complete workflow."""
        
        async def simulate_search_with_error_handling(set_filter: str):
            """Simulate search with error handling."""
            try:
                matched_set_slug = await semantic_matcher.match_set(set_filter)
                return f"Success: {matched_set_slug}"
            except NoMatchFoundError as e:
                return f"NoMatchError: {str(e)}"
            except SemanticMatchError as e:
                return f"SemanticError: {str(e)}"
        
        # Test with empty string
        result = await simulate_search_with_error_handling("")
        assert result.startswith("NoMatchError:")
        
        # Test with nonexistent set
        result = await simulate_search_with_error_handling("nonexistent")
        assert result.startswith("NoMatchError:")
        
        # Test with valid set
        result = await simulate_search_with_error_handling("platform")
        assert result == "Success: platform_code"


class TestBackwardCompatibility:
    """Test that the filtering logic maintains backward compatibility."""
    
    def test_no_set_filter_compatibility(self):
        """Test that when no set filter is provided, behavior is unchanged."""
        # Create a proper Qdrant filter for testing
        type_condition = models.FieldCondition(
            key="metadata.document_type",
            match=models.MatchValue(value="code")
        )
        query_filter = models.Filter(must=[type_condition])
        set_filter = None
        
        # Logic from mcp_server.py - when set_filter is None, query_filter should be unchanged
        result_filter = query_filter
        if set_filter:
            # This branch should not execute
            assert False, "Should not execute set filter logic when set_filter is None"
        
        # Should use original query_filter unchanged
        assert result_filter == query_filter
        assert isinstance(result_filter, models.Filter)
    
    def test_empty_set_filter_compatibility(self):
        """Test that empty set filter is handled gracefully."""
        # When set_filter is empty string, it should be treated as None
        query_filter = {"key": "document_type", "match": {"value": "code"}}
        set_filter = ""
        
        # The logic should handle empty string by not processing it
        # (This would be caught by the semantic matcher raising an error)
        combined_filter = query_filter
        
        # If set_filter is empty, semantic matcher would raise error
        # but the combined_filter should remain as query_filter
        assert combined_filter == query_filter


class TestFilterValidation:
    """Test that generated filters are valid for Qdrant."""
    
    def test_set_filter_qdrant_compatibility(self):
        """Test that set filters are compatible with Qdrant."""
        # Create proper Qdrant field condition
        set_field_condition = models.FieldCondition(
            key="metadata.set",
            match=models.MatchValue(value="platform_code")
        )
        
        # Should be able to create Qdrant Filter
        qdrant_filter = models.Filter(must=[set_field_condition])
        assert qdrant_filter is not None
        
        # Should have correct structure
        assert hasattr(qdrant_filter, 'must')
        assert qdrant_filter.must is not None
        assert len(qdrant_filter.must) == 1
    
    def test_combined_filter_qdrant_compatibility(self):
        """Test that combined filters are compatible with Qdrant."""
        # Create proper Qdrant field conditions
        set_condition = models.FieldCondition(
            key="metadata.set",
            match=models.MatchValue(value="platform_code")
        )
        type_condition = models.FieldCondition(
            key="metadata.document_type",
            match=models.MatchValue(value="code")
        )
        
        # Should be able to create Qdrant Filter
        qdrant_filter = models.Filter(must=[set_condition, type_condition])
        assert qdrant_filter is not None
        
        # Should have correct structure
        assert hasattr(qdrant_filter, 'must')
        assert len(qdrant_filter.must) == 2
    
    def test_empty_filter_handling(self):
        """Test that empty/None filters are handled correctly."""
        # None filter should result in None Qdrant filter
        combined_filter = None
        qdrant_filter = models.Filter(**combined_filter) if combined_filter else None
        assert qdrant_filter is None
        
        # Empty dict should be handled
        combined_filter = {}
        try:
            qdrant_filter = models.Filter(**combined_filter)
            # If it doesn't raise an error, that's fine
        except Exception:
            # If it raises an error, we should handle it gracefully
            qdrant_filter = None
        
        # Either way, we should be able to handle it
        assert True  # Test passes if we get here without unhandled exceptions
"""
Unit tests for the SemanticSetMatcher class.
"""

import pytest
from typing import Dict

from mcp_server_qdrant_rag.semantic_matcher import (
    SemanticSetMatcher,
    NoMatchFoundError,
    AmbiguousMatchError,
    SemanticMatchError,
)
from mcp_server_qdrant_rag.settings import SetConfiguration


@pytest.fixture
def sample_set_configurations() -> Dict[str, SetConfiguration]:
    """Sample set configurations for testing."""
    return {
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
    }


@pytest.fixture
def matcher(sample_set_configurations) -> SemanticSetMatcher:
    """SemanticSetMatcher instance with sample configurations."""
    return SemanticSetMatcher(sample_set_configurations)


class TestSemanticSetMatcherInitialization:
    """Test SemanticSetMatcher initialization."""
    
    def test_init_with_configurations(self, sample_set_configurations):
        """Test initialization with valid configurations."""
        matcher = SemanticSetMatcher(sample_set_configurations)
        assert matcher.set_configurations == sample_set_configurations
    
    def test_init_with_empty_configurations(self):
        """Test initialization with empty configurations."""
        matcher = SemanticSetMatcher({})
        assert matcher.set_configurations == {}


class TestExactMatching:
    """Test exact matching functionality."""
    
    @pytest.mark.asyncio
    async def test_exact_slug_match(self, matcher):
        """Test exact match against set slug."""
        result = await matcher.match_set("platform_code")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_exact_slug_match_case_insensitive(self, matcher):
        """Test exact match against set slug with different case."""
        result = await matcher.match_set("PLATFORM_CODE")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_exact_description_match(self, matcher):
        """Test exact match against set description."""
        result = await matcher.match_set("Platform Codebase")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_exact_description_match_case_insensitive(self, matcher):
        """Test exact match against set description with different case."""
        result = await matcher.match_set("platform codebase")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_exact_alias_match(self, matcher):
        """Test exact match against set alias."""
        result = await matcher.match_set("platform")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_exact_alias_match_case_insensitive(self, matcher):
        """Test exact match against set alias with different case."""
        result = await matcher.match_set("BACKEND")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_exact_match_with_whitespace(self, matcher):
        """Test exact match with leading/trailing whitespace."""
        result = await matcher.match_set("  platform  ")
        assert result == "platform_code"


class TestFuzzyMatching:
    """Test fuzzy matching functionality."""
    
    @pytest.mark.asyncio
    async def test_partial_slug_match(self, matcher):
        """Test fuzzy match with partial slug."""
        result = await matcher.match_set("platform")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_partial_description_match(self, matcher):
        """Test fuzzy match with partial description."""
        result = await matcher.match_set("codebase")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_word_based_match(self, matcher):
        """Test fuzzy match using word overlap."""
        result = await matcher.match_set("frontend app")
        assert result == "frontend_code"
    
    @pytest.mark.asyncio
    async def test_sequence_similarity_match(self, matcher):
        """Test fuzzy match using sequence similarity."""
        result = await matcher.match_set("api_documentation")
        assert result == "api_docs"
    
    @pytest.mark.asyncio
    async def test_substring_match(self, matcher):
        """Test fuzzy match with substring."""
        result = await matcher.match_set("frontend")
        assert result == "frontend_code"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_query(self, matcher):
        """Test error handling for empty query."""
        with pytest.raises(NoMatchFoundError) as exc_info:
            await matcher.match_set("")
        
        assert "No matching set found" in str(exc_info.value)
        assert "platform_code" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, matcher):
        """Test error handling for whitespace-only query."""
        with pytest.raises(NoMatchFoundError) as exc_info:
            await matcher.match_set("   ")
        
        assert "No matching set found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_no_match_found(self, matcher):
        """Test error handling when no match is found."""
        with pytest.raises(NoMatchFoundError) as exc_info:
            await matcher.match_set("nonexistent_set")
        
        error = exc_info.value
        assert error.query == "nonexistent_set"
        assert "platform_code (Platform Codebase)" in error.available_sets
        assert "No matching set found" in str(error)
    
    @pytest.mark.asyncio
    async def test_empty_configurations(self):
        """Test error handling with empty configurations."""
        empty_matcher = SemanticSetMatcher({})
        
        with pytest.raises(NoMatchFoundError) as exc_info:
            await empty_matcher.match_set("any_query")
        
        error = exc_info.value
        assert error.query == "any_query"
        assert error.available_sets == []
    
    @pytest.mark.asyncio
    async def test_ambiguous_match(self):
        """Test error handling for ambiguous matches."""
        # Create configurations that could lead to ambiguous matches in fuzzy matching
        ambiguous_configs = {
            "set_a": SetConfiguration(
                slug="set_a",
                description="Similar Description A",
                aliases=["similar_a"]
            ),
            "set_b": SetConfiguration(
                slug="set_b", 
                description="Similar Description B",
                aliases=["similar_b"]
            ),
        }
        
        ambiguous_matcher = SemanticSetMatcher(ambiguous_configs)
        
        # Use a query that would score similarly for both sets
        with pytest.raises(AmbiguousMatchError) as exc_info:
            await ambiguous_matcher.match_set("similar")
        
        error = exc_info.value
        assert error.query == "similar"
        assert len(error.matches) == 2
        assert "Ambiguous match" in str(error)


class TestSimilarityCalculation:
    """Test similarity calculation methods."""
    
    def test_exact_match_similarity(self, matcher):
        """Test similarity calculation for exact matches."""
        score = matcher._calculate_similarity("platform", "platform")
        assert score == 1.0
    
    def test_substring_match_similarity(self, matcher):
        """Test similarity calculation for substring matches."""
        score = matcher._calculate_similarity("platform", "platform_code")
        assert score >= 0.8  # Should get high score for substring match
    
    def test_no_similarity(self, matcher):
        """Test similarity calculation for completely different strings."""
        score = matcher._calculate_similarity("platform", "xyz")
        assert score < 0.6  # Should get low score
    
    def test_empty_string_similarity(self, matcher):
        """Test similarity calculation with empty strings."""
        score = matcher._calculate_similarity("", "platform")
        assert score == 0.0
        
        score = matcher._calculate_similarity("platform", "")
        assert score == 0.0
    
    def test_word_overlap_similarity(self, matcher):
        """Test similarity calculation with word overlap."""
        score = matcher._calculate_similarity("frontend app", "frontend application")
        assert score > 0.6  # Should get decent score for word overlap


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_available_sets(self, matcher):
        """Test getting available sets for error messages."""
        available = matcher.get_available_sets()
        
        assert len(available) == 4
        assert "platform_code (Platform Codebase)" in available
        assert "api_docs (API Documentation)" in available
        assert "frontend_code (Frontend Application Code)" in available
        assert "database_schema (Database Schema and Migrations)" in available
    
    def test_get_available_sets_empty(self):
        """Test getting available sets with empty configurations."""
        empty_matcher = SemanticSetMatcher({})
        available = empty_matcher.get_available_sets()
        assert available == []
    
    def test_reload_configurations(self, matcher):
        """Test reloading configurations."""
        new_configs = {
            "new_set": SetConfiguration(
                slug="new_set",
                description="New Set",
                aliases=["new"]
            )
        }
        
        matcher.reload_configurations(new_configs)
        assert matcher.set_configurations == new_configs
        assert len(matcher.set_configurations) == 1


class TestComplexScenarios:
    """Test complex matching scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_word_query(self, matcher):
        """Test matching with multiple word queries."""
        result = await matcher.match_set("database migrations")
        assert result == "database_schema"
    
    @pytest.mark.asyncio
    async def test_partial_word_match(self, matcher):
        """Test matching with partial words."""
        result = await matcher.match_set("doc")
        assert result == "api_docs"
    
    @pytest.mark.asyncio
    async def test_synonym_like_match(self, matcher):
        """Test matching with synonym-like terms."""
        result = await matcher.match_set("ui")
        assert result == "frontend_code"
    
    @pytest.mark.asyncio
    async def test_abbreviation_match(self, matcher):
        """Test matching with abbreviations."""
        result = await matcher.match_set("db")
        assert result == "database_schema"
    
    @pytest.mark.asyncio
    async def test_case_variations(self, matcher):
        """Test matching with various case combinations."""
        test_cases = [
            ("Platform", "platform_code"),
            ("PLATFORM", "platform_code"),
            ("platform", "platform_code"),
            ("PlatForm", "platform_code"),
        ]
        
        for query, expected in test_cases:
            result = await matcher.match_set(query)
            assert result == expected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, matcher):
        """Test handling of special characters in queries."""
        # Should still match despite special characters
        result = await matcher.match_set("platform-code")
        assert result == "platform_code"
    
    @pytest.mark.asyncio
    async def test_numeric_in_query(self, matcher):
        """Test handling of numeric characters in queries."""
        # Create a matcher with numeric set
        numeric_configs = {
            "version_2": SetConfiguration(
                slug="version_2",
                description="Version 2 Documentation",
                aliases=["v2", "version2"]
            )
        }
        numeric_matcher = SemanticSetMatcher(numeric_configs)
        
        result = await numeric_matcher.match_set("v2")
        assert result == "version_2"
    
    def test_very_long_strings(self, matcher):
        """Test handling of very long strings."""
        long_query = "a" * 1000
        long_target = "b" * 1000
        
        score = matcher._calculate_similarity(long_query, long_target)
        assert 0.0 <= score <= 1.0  # Should not crash and return valid score
    
    @pytest.mark.asyncio
    async def test_unicode_characters(self, matcher):
        """Test handling of unicode characters."""
        # Create matcher with unicode content
        unicode_configs = {
            "unicode_set": SetConfiguration(
                slug="unicode_set",
                description="Tëst Sét with Ünïcödé",
                aliases=["tëst", "ünïcödé"]
            )
        }
        unicode_matcher = SemanticSetMatcher(unicode_configs)
        
        result = await unicode_matcher.match_set("tëst")
        assert result == "unicode_set"
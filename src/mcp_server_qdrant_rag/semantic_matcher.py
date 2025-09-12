"""
Semantic set matching service for natural language to set slug matching.
"""

import logging
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher

from mcp_server_qdrant_rag.settings import SetConfiguration

logger = logging.getLogger(__name__)


class SemanticMatchError(Exception):
    """Base exception for semantic matching errors."""
    pass


class NoMatchFoundError(SemanticMatchError):
    """Raised when no matching set is found for a query."""
    
    def __init__(self, query: str, available_sets: List[str]):
        self.query = query
        self.available_sets = available_sets
        super().__init__(f"No matching set found for query '{query}'. Available sets: {', '.join(available_sets)}")


class AmbiguousMatchError(SemanticMatchError):
    """Raised when multiple sets match equally well."""
    
    def __init__(self, query: str, matches: List[Tuple[str, float]]):
        self.query = query
        self.matches = matches
        match_strs = [f"{slug} (score: {score:.2f})" for slug, score in matches]
        super().__init__(f"Ambiguous match for query '{query}'. Multiple sets matched equally: {', '.join(match_strs)}")


class SemanticSetMatcher:
    """Handles semantic matching of natural language to configured sets."""
    
    def __init__(self, set_configurations: Dict[str, SetConfiguration]):
        """
        Initialize the semantic matcher with set configurations.
        
        Args:
            set_configurations: Dictionary mapping set slugs to SetConfiguration objects
        """
        self.set_configurations = set_configurations
        logger.debug(f"Initialized SemanticSetMatcher with {len(set_configurations)} sets")
    
    async def match_set(self, query: str) -> str:
        """
        Match a natural language query to a configured set slug.
        
        Args:
            query: Natural language description of the set
            
        Returns:
            Matching set slug
            
        Raises:
            NoMatchFoundError: When no matching set is found
            AmbiguousMatchError: When multiple sets match equally well
        """
        if not query or not query.strip():
            raise NoMatchFoundError("", list(self.set_configurations.keys()))
        
        query = query.strip().lower()
        logger.debug(f"Matching query: '{query}'")
        
        if not self.set_configurations:
            raise NoMatchFoundError(query, [])
        
        # Try exact matches first - but check for multiple exact matches
        exact_matches = self._find_exact_matches(query)
        if len(exact_matches) > 1:
            # Multiple exact matches are ambiguous
            exact_match_tuples = [(slug, 1.0) for slug in exact_matches]
            raise AmbiguousMatchError(query, exact_match_tuples)
        elif len(exact_matches) == 1:
            logger.debug(f"Found exact match: {exact_matches[0]}")
            return exact_matches[0]
        
        # Try fuzzy matching
        fuzzy_matches = self._find_fuzzy_matches(query)
        
        if not fuzzy_matches:
            available_sets = [f"{slug} ({config.description})" for slug, config in self.set_configurations.items()]
            raise NoMatchFoundError(query, available_sets)
        
        # Check for ambiguous matches (multiple matches with same high score)
        best_score = fuzzy_matches[0][1]
        best_matches = [(slug, score) for slug, score in fuzzy_matches if score == best_score]
        
        if len(best_matches) > 1:
            raise AmbiguousMatchError(query, best_matches)
        
        matched_slug = fuzzy_matches[0][0]
        logger.debug(f"Found fuzzy match: {matched_slug} (score: {best_score:.2f})")
        return matched_slug
    
    def _find_exact_matches(self, query: str) -> List[str]:
        """
        Find all exact matches against set slugs, descriptions, and aliases.
        
        Args:
            query: Normalized query string (lowercase, stripped)
            
        Returns:
            List of matching set slugs
        """
        matches = []
        
        for slug, config in self.set_configurations.items():
            # Check slug match
            if query == slug.lower():
                matches.append(slug)
                continue  # Don't check other fields for this set
            
            # Check description match
            if query == config.description.lower():
                matches.append(slug)
                continue  # Don't check aliases for this set
            
            # Check alias matches
            for alias in config.aliases:
                if query == alias.lower():
                    matches.append(slug)
                    break  # Don't check other aliases for this set
        
        return matches
    
    def _find_fuzzy_matches(self, query: str, min_score: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find fuzzy matches using string similarity scoring.
        
        Args:
            query: Normalized query string (lowercase, stripped)
            min_score: Minimum similarity score to consider a match
            
        Returns:
            List of (slug, score) tuples sorted by score descending
        """
        matches = []
        
        for slug, config in self.set_configurations.items():
            # Calculate scores for different match targets
            scores = []
            
            # Score against slug
            slug_score = self._calculate_similarity(query, slug.lower())
            scores.append(slug_score)
            
            # Score against description
            desc_score = self._calculate_similarity(query, config.description.lower())
            scores.append(desc_score)
            
            # Score against aliases
            for alias in config.aliases:
                alias_score = self._calculate_similarity(query, alias.lower())
                scores.append(alias_score)
            
            # Use the best score for this set
            best_score = max(scores) if scores else 0.0
            
            if best_score >= min_score:
                matches.append((slug, best_score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _calculate_similarity(self, query: str, target: str) -> float:
        """
        Calculate similarity score between query and target string.
        
        Uses a combination of:
        - Sequence matching for overall similarity
        - Substring matching for partial matches
        - Word-based matching for multi-word queries
        
        Args:
            query: Query string
            target: Target string to compare against
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not query or not target:
            return 0.0
        
        # Exact match gets perfect score
        if query == target:
            return 1.0
        
        # Substring match gets high score
        if query in target or target in query:
            # Favor longer matches
            overlap_ratio = min(len(query), len(target)) / max(len(query), len(target))
            return 0.8 + (0.2 * overlap_ratio)
        
        # Sequence similarity
        seq_score = SequenceMatcher(None, query, target).ratio()
        
        # Word-based matching for multi-word queries
        query_words = set(query.split())
        target_words = set(target.split())
        
        if query_words and target_words:
            word_overlap = len(query_words.intersection(target_words))
            word_union = len(query_words.union(target_words))
            word_score = word_overlap / word_union if word_union > 0 else 0.0
            
            # Combine sequence and word scores
            combined_score = (seq_score * 0.6) + (word_score * 0.4)
            return max(seq_score, combined_score)
        
        return seq_score
    
    def get_available_sets(self) -> List[str]:
        """
        Get list of available set descriptions for error messages.
        
        Returns:
            List of formatted set descriptions
        """
        return [f"{slug} ({config.description})" for slug, config in self.set_configurations.items()]
    
    def reload_configurations(self, set_configurations: Dict[str, SetConfiguration]) -> None:
        """
        Reload set configurations without recreating the matcher.
        
        Args:
            set_configurations: New set configurations dictionary
        """
        self.set_configurations = set_configurations
        logger.debug(f"Reloaded SemanticSetMatcher with {len(set_configurations)} sets")
"""Hybrid document chunking service for intelligent text segmentation.

This module implements a hybrid chunking strategy that combines:
1. Sentence splitting using NLTK's sent_tokenize (with syntok fallback)
2. Recursive chunking with overlap, similar to LangChain's RecursiveCharacterTextSplitter
3. Token-aware chunking using tiktoken or fallback tokenizers

The chunker is designed to be modular, allowing easy swapping of sentence splitters
and tokenizers while maintaining high-quality, embedding-friendly chunks.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any, Callable, Union

# Import sentence splitters with fallbacks
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download punkt tokenizer data if not already present
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to older punkt if punkt_tab fails
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = None

try:
    import syntok.segmenter as syntok_segmenter
    SYNTOK_AVAILABLE = True
except ImportError:
    SYNTOK_AVAILABLE = False
    syntok_segmenter = None

# Import tokenizers with fallbacks
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

from .models import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Hybrid document chunker that combines sentence splitting with recursive chunking.
    
    Features:
    - Sentence-aware splitting using NLTK or syntok
    - Token-based size limits using tiktoken or fallback tokenizers
    - Configurable overlap between chunks
    - Modular design for easy component swapping
    """
    
    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        sentence_splitter: Optional[str] = None,
        tokenizer: Optional[Union[str, Callable[[str], List[str]]]] = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the DocumentChunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            sentence_splitter: Sentence splitter to use ('nltk', 'syntok', or None for auto)
            tokenizer: Tokenizer to use ('tiktoken', 'whitespace', or custom callable)
            encoding_name: Tiktoken encoding name (e.g., 'cl100k_base' for GPT-4)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name
        
        # Validate parameters
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        
        # Initialize sentence splitter
        self._sentence_splitter = self._init_sentence_splitter(sentence_splitter)
        
        # Initialize tokenizer
        self._tokenizer = self._init_tokenizer(tokenizer)
        
        logger.info(f"DocumentChunker initialized with max_tokens={max_tokens}, "
                   f"overlap_tokens={overlap_tokens}, "
                   f"sentence_splitter={sentence_splitter}, "
                   f"tokenizer={tokenizer}")
    
    def _init_sentence_splitter(self, splitter_name: Optional[str]) -> Callable[[str], List[str]]:
        """Initialize the sentence splitter based on availability and preference."""
        if splitter_name == "nltk":
            if not NLTK_AVAILABLE:
                raise ValueError("NLTK is not available. Install with: pip install nltk")
            return sent_tokenize
        elif splitter_name == "syntok":
            if not SYNTOK_AVAILABLE:
                raise ValueError("syntok is not available. Install with: pip install syntok")
            return self._syntok_sentence_split
        elif splitter_name is None:
            # Auto-select best available
            if NLTK_AVAILABLE:
                logger.info("Using NLTK sentence splitter")
                return sent_tokenize
            elif SYNTOK_AVAILABLE:
                logger.info("Using syntok sentence splitter")
                return self._syntok_sentence_split
            else:
                logger.warning("No advanced sentence splitters available, using simple fallback")
                return self._simple_sentence_split
        else:
            raise ValueError(f"Unknown sentence splitter: {splitter_name}")
    
    def _init_tokenizer(self, tokenizer: Optional[Union[str, Callable[[str], List[str]]]]) -> Callable[[str], List[str]]:
        """Initialize the tokenizer based on type and availability."""
        if callable(tokenizer):
            self._is_tiktoken = False
            return tokenizer
        elif tokenizer == "tiktoken":
            if not TIKTOKEN_AVAILABLE:
                raise ValueError("tiktoken is not available. Install with: pip install tiktoken")
            try:
                encoding = tiktoken.get_encoding(self.encoding_name)
                self._is_tiktoken = True
                return lambda text: encoding.encode(text)  # Returns token IDs, len() gives count
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken with {self.encoding_name}: {e}")
                self._is_tiktoken = False
                return self._whitespace_tokenizer
        elif tokenizer == "whitespace" or tokenizer is None:
            self._is_tiktoken = False
            return self._whitespace_tokenizer
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")
    
    def _syntok_sentence_split(self, text: str) -> List[str]:
        """Split text into sentences using syntok."""
        sentences = []
        for paragraph in syntok_segmenter.segment(text):
            for sentence in paragraph:
                sentences.append(str(sentence))
        return sentences
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback using basic punctuation."""
        import re
        # Basic sentence splitting - not as robust as NLTK/syntok
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _whitespace_tokenizer(self, text: str) -> List[str]:
        """Simple whitespace-based tokenizer."""
        return text.split()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured tokenizer."""
        tokens = self._tokenizer(text)
        return len(tokens)
    
    def _get_token_overlap(self, text: str, num_tokens: int) -> str:
        """Get the last num_tokens from text."""
        tokens = self._tokenizer(text)
        if len(tokens) <= num_tokens:
            return text
        
        # For tiktoken (returns token IDs), we need to decode
        if TIKTOKEN_AVAILABLE and hasattr(self, '_is_tiktoken') and self._is_tiktoken:
            try:
                encoding = tiktoken.get_encoding(self.encoding_name)
                overlap_tokens = tokens[-num_tokens:]
                return encoding.decode(overlap_tokens)
            except:
                # Fallback to word-based overlap
                pass
        
        # For word-based tokenizers
        overlap_tokens = tokens[-num_tokens:]
        return ' '.join(overlap_tokens)
    
    async def chunk_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Split a document into optimally-sized chunks using hybrid strategy.
        
        Args:
            content: The text content to chunk
            metadata: Optional metadata to associate with chunks
            source_document_id: Optional ID for the source document
            
        Returns:
            List of DocumentChunk objects
        """
        if not content or not content.strip():
            return []
        
        # Generate source document ID if not provided
        if source_document_id is None:
            source_document_id = str(uuid.uuid4())
        
        # Clean up the content
        content = content.strip()
        
        # Perform hybrid chunking
        chunk_texts = self._hybrid_chunk(content)
        
        # Create DocumentChunk objects
        document_chunks = []
        total_chunks = len(chunk_texts)
        
        for i, chunk_text in enumerate(chunk_texts):
            # Calculate overlap information
            overlap_start = self.overlap_tokens if i > 0 else 0
            overlap_end = self.overlap_tokens if i < total_chunks - 1 else 0
            
            document_chunk = DocumentChunk(
                content=chunk_text,
                metadata=metadata,
                chunk_index=i,
                source_document_id=source_document_id,
                total_chunks=total_chunks,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                chunk_strategy="hybrid"
            )
            document_chunks.append(document_chunk)
        
        return document_chunks
    
    def _hybrid_chunk(self, text: str) -> List[str]:
        """
        Perform hybrid chunking: sentence splitting + recursive chunking with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk texts
        """
        # Step 1: Split into sentences
        sentences = self._sentence_splitter(text)
        
        if not sentences:
            return []
        
        # Step 2: Combine sentences into chunks with size limits and overlap
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self._count_tokens(sentence)
            
            # If this single sentence exceeds max_tokens, split it further
            if sentence_tokens > self.max_tokens:
                # Finalize current chunk if it has content
                if current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_token_count = 0
                
                # Split the long sentence using fixed chunking
                sentence_chunks = self._split_long_sentence(sentence)
                chunks.extend(sentence_chunks)
                continue
            
            # If adding this sentence would exceed the limit, finalize current chunk
            if current_token_count + sentence_tokens > self.max_tokens and current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                
                # Start new chunk with overlap from previous chunk
                if self.overlap_tokens > 0 and chunks:
                    overlap_text = self._get_token_overlap(chunks[-1], self.overlap_tokens)
                    # Don't re-split overlap text into sentences to avoid complexity
                    current_chunk_sentences = [overlap_text, sentence]
                    current_token_count = self._count_tokens(overlap_text) + sentence_tokens
                else:
                    current_chunk_sentences = [sentence]
                    current_token_count = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
        
        # Handle remaining content
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))
        
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a sentence that exceeds max_tokens using fixed chunking.
        
        Args:
            sentence: Long sentence to split
            
        Returns:
            List of sentence chunks
        """
        tokens = self._tokenizer(sentence)
        chunks = []
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.max_tokens, len(tokens))
            
            # For tiktoken, decode token IDs back to text
            if TIKTOKEN_AVAILABLE and hasattr(self, '_is_tiktoken') and self._is_tiktoken:
                try:
                    encoding = tiktoken.get_encoding(self.encoding_name)
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = encoding.decode(chunk_tokens)
                except:
                    # Fallback to word-based chunking
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = ' '.join(chunk_tokens)
            else:
                # For word-based tokenizers
                chunk_tokens = tokens[start_idx:end_idx]
                chunk_text = ' '.join(chunk_tokens)
            
            chunks.append(chunk_text)
            
            # Move start position, accounting for overlap
            next_start = end_idx - self.overlap_tokens
            
            # Ensure we don't get stuck in an infinite loop
            if next_start <= start_idx or end_idx >= len(tokens):
                break
                
            start_idx = next_start
        
        return chunks
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """
        Get information about the chunker configuration.
        
        Returns:
            Dictionary containing chunker settings
        """
        return {
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "encoding_name": self.encoding_name,
            "sentence_splitter": self._sentence_splitter.__name__ if hasattr(self._sentence_splitter, '__name__') else str(self._sentence_splitter),
            "tokenizer": self._tokenizer.__name__ if hasattr(self._tokenizer, '__name__') else str(self._tokenizer),
            "nltk_available": NLTK_AVAILABLE,
            "syntok_available": SYNTOK_AVAILABLE,
            "tiktoken_available": TIKTOKEN_AVAILABLE
        }
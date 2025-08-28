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
from ..common.exceptions import TokenizerError, SentenceSplitterError

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
                raise SentenceSplitterError(
                    splitter_name="nltk",
                    original_error=ImportError("NLTK is not available"),
                    fallback_available=SYNTOK_AVAILABLE or True  # Simple fallback always available
                )
            try:
                # Test NLTK functionality
                sent_tokenize("Test sentence.")
                logger.info("Using NLTK sentence splitter")
                return sent_tokenize
            except Exception as e:
                logger.warning(f"NLTK sentence splitter failed, using fallback: {e}")
                return self._get_fallback_sentence_splitter()
        elif splitter_name == "syntok":
            if not SYNTOK_AVAILABLE:
                raise SentenceSplitterError(
                    splitter_name="syntok",
                    original_error=ImportError("syntok is not available"),
                    fallback_available=NLTK_AVAILABLE or True  # Simple fallback always available
                )
            try:
                # Test syntok functionality
                list(self._syntok_sentence_split("Test sentence."))
                logger.info("Using syntok sentence splitter")
                return self._syntok_sentence_split
            except Exception as e:
                logger.warning(f"syntok sentence splitter failed, using fallback: {e}")
                return self._get_fallback_sentence_splitter()
        elif splitter_name is None:
            # Auto-select best available
            if NLTK_AVAILABLE:
                try:
                    sent_tokenize("Test sentence.")
                    logger.info("Using NLTK sentence splitter")
                    return sent_tokenize
                except Exception as e:
                    logger.warning(f"NLTK failed, trying syntok: {e}")
            
            if SYNTOK_AVAILABLE:
                try:
                    list(self._syntok_sentence_split("Test sentence."))
                    logger.info("Using syntok sentence splitter")
                    return self._syntok_sentence_split
                except Exception as e:
                    logger.warning(f"syntok failed, using simple fallback: {e}")
            
            logger.warning("No advanced sentence splitters available, using simple fallback")
            return self._simple_sentence_split
        else:
            raise ValueError(f"Unknown sentence splitter: {splitter_name}")
    
    def _get_fallback_sentence_splitter(self) -> Callable[[str], List[str]]:
        """Get the best available fallback sentence splitter."""
        if NLTK_AVAILABLE:
            try:
                sent_tokenize("Test sentence.")
                return sent_tokenize
            except:
                pass
        
        if SYNTOK_AVAILABLE:
            try:
                list(self._syntok_sentence_split("Test sentence."))
                return self._syntok_sentence_split
            except:
                pass
        
        return self._simple_sentence_split
    
    def _init_tokenizer(self, tokenizer: Optional[Union[str, Callable[[str], List[str]]]]) -> Callable[[str], List[str]]:
        """Initialize the tokenizer based on type and availability."""
        if callable(tokenizer):
            self._is_tiktoken = False
            return tokenizer
        elif tokenizer == "tiktoken":
            if not TIKTOKEN_AVAILABLE:
                raise TokenizerError(
                    tokenizer_name="tiktoken",
                    original_error=ImportError("tiktoken is not available"),
                    fallback_available=True
                )
            try:
                encoding = tiktoken.get_encoding(self.encoding_name)
                # Test the encoding
                test_tokens = encoding.encode("Test text")
                encoding.decode(test_tokens)
                
                self._is_tiktoken = True
                logger.info(f"Using tiktoken tokenizer with encoding: {self.encoding_name}")
                return lambda text: encoding.encode(text)  # Returns token IDs, len() gives count
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken with {self.encoding_name}: {e}")
                self._is_tiktoken = False
                logger.info("Falling back to whitespace tokenizer")
                return self._whitespace_tokenizer
        elif tokenizer == "whitespace" or tokenizer is None:
            self._is_tiktoken = False
            logger.info("Using whitespace tokenizer")
            return self._whitespace_tokenizer
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")
    
    def _syntok_sentence_split(self, text: str) -> List[str]:
        """Split text into sentences using syntok."""
        try:
            sentences = []
            for paragraph in syntok_segmenter.segment(text):
                for sentence in paragraph:
                    sentences.append(str(sentence))
            return sentences
        except Exception as e:
            logger.warning(f"syntok sentence splitting failed, using simple fallback: {e}")
            return self._simple_sentence_split(text)
    
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
            
        Raises:
            ValueError: If content is empty or invalid
            RuntimeError: If chunking fails due to internal errors
        """
        if not content or not content.strip():
            logger.debug("Empty content provided for chunking")
            return []
        
        # Generate source document ID if not provided
        if source_document_id is None:
            source_document_id = str(uuid.uuid4())
        
        # Clean up the content
        content = content.strip()
        content_length = len(content)
        content_tokens = self._count_tokens(content)
        
        logger.debug(f"Starting document chunking - Length: {content_length} chars, "
                    f"Tokens: {content_tokens}, Max tokens per chunk: {self.max_tokens}")
        
        # Log chunking decision
        if content_tokens <= self.max_tokens:
            logger.debug("Document fits in single chunk, no chunking needed")
        else:
            logger.debug(f"Document exceeds max tokens ({content_tokens} > {self.max_tokens}), "
                        f"will chunk with {self.overlap_tokens} token overlap")
        
        try:
            # Perform hybrid chunking
            chunk_texts = self._hybrid_chunk(content)
            
            if not chunk_texts:
                logger.warning("Hybrid chunking produced no chunks")
                return []
            
            logger.debug(f"Hybrid chunking produced {len(chunk_texts)} chunks")
            
            # Create DocumentChunk objects
            document_chunks = []
            total_chunks = len(chunk_texts)
            
            for i, chunk_text in enumerate(chunk_texts):
                if not chunk_text.strip():
                    logger.warning(f"Empty chunk at index {i}, skipping")
                    continue
                
                chunk_tokens = self._count_tokens(chunk_text)
                logger.debug(f"Chunk {i+1}/{total_chunks}: {len(chunk_text)} chars, {chunk_tokens} tokens")
                
                # Calculate overlap information
                overlap_start = self.overlap_tokens if i > 0 else 0
                overlap_end = self.overlap_tokens if i < total_chunks - 1 else 0
                
                try:
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
                except Exception as e:
                    logger.error(f"Failed to create DocumentChunk at index {i}: {e}")
                    # Continue with other chunks rather than failing completely
                    continue
            
            logger.debug(f"Successfully created {len(document_chunks)} chunks from document "
                        f"(source: {source_document_id[:8]}...)")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Document chunking failed: {e}")
            raise RuntimeError(f"Failed to chunk document: {str(e)}") from e
    
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
            logger.debug("No sentences found after splitting")
            return []
        
        logger.debug(f"Split text into {len(sentences)} sentences")
        
        # Step 2: Combine sentences into chunks with size limits and overlap
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        long_sentences_split = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self._count_tokens(sentence)
            
            # If this single sentence exceeds max_tokens, split it further
            if sentence_tokens > self.max_tokens:
                logger.debug(f"Sentence {i+1} exceeds max tokens ({sentence_tokens} > {self.max_tokens}), "
                           f"splitting further")
                
                # Finalize current chunk if it has content
                if current_chunk_sentences:
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunks.append(chunk_text)
                    logger.debug(f"Finalized chunk {len(chunks)} with {self._count_tokens(chunk_text)} tokens")
                    current_chunk_sentences = []
                    current_token_count = 0
                
                # Split the long sentence using fixed chunking
                sentence_chunks = self._split_long_sentence(sentence)
                chunks.extend(sentence_chunks)
                long_sentences_split += 1
                logger.debug(f"Split long sentence into {len(sentence_chunks)} sub-chunks")
                continue
            
            # If adding this sentence would exceed the limit, finalize current chunk
            if current_token_count + sentence_tokens > self.max_tokens and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                logger.debug(f"Finalized chunk {len(chunks)} with {self._count_tokens(chunk_text)} tokens "
                           f"(would exceed limit with next sentence)")
                
                # Start new chunk with overlap from previous chunk
                if self.overlap_tokens > 0 and chunks:
                    overlap_text = self._get_token_overlap(chunks[-1], self.overlap_tokens)
                    overlap_actual_tokens = self._count_tokens(overlap_text)
                    logger.debug(f"Adding {overlap_actual_tokens} token overlap to new chunk")
                    # Don't re-split overlap text into sentences to avoid complexity
                    current_chunk_sentences = [overlap_text, sentence]
                    current_token_count = overlap_actual_tokens + sentence_tokens
                else:
                    current_chunk_sentences = [sentence]
                    current_token_count = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
        
        # Handle remaining content
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            logger.debug(f"Finalized final chunk {len(chunks)} with {self._count_tokens(chunk_text)} tokens")
        
        if long_sentences_split > 0:
            logger.debug(f"Split {long_sentences_split} long sentences during chunking")
        
        logger.debug(f"Hybrid chunking complete: {len(chunks)} chunks created")
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
#!/usr/bin/env python3
"""
Demonstration script for embedding model intelligence functionality.

This script shows how the embedding model intelligence works in the CLI ingestion tool.
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_server_qdrant_rag.cli_ingest import EmbeddingModelIntelligence, EmbeddingModelInfo
from mcp_server_qdrant_rag.settings import QdrantSettings


async def demo_model_inference():
    """Demonstrate model inference from collection information."""
    print("🔍 Demonstrating Model Inference from Collection Information")
    print("=" * 60)
    
    # Create settings
    settings = QdrantSettings()
    settings.location = "http://localhost:6333"
    
    # Create intelligence instance
    intelligence = EmbeddingModelIntelligence(settings)
    
    # Test exact matches
    print("\n📋 Testing Exact Pattern Matches:")
    test_cases = [
        ("fast-nomic-embed-text-v1.5-q", 768, "nomic-ai/nomic-embed-text-v1.5-Q"),
        ("fast-all-minilm-l6-v2", 384, "sentence-transformers/all-MiniLM-L6-v2"),
        ("fast-bge-small-en-v1.5", 384, "BAAI/bge-small-en-v1.5"),
        ("fast-e5-small-v2", 384, "intfloat/e5-small-v2"),
    ]
    
    for vector_name, vector_size, expected_model in test_cases:
        inferred = intelligence._infer_model_from_collection_info(vector_name, vector_size)
        status = "✅" if inferred == expected_model else "❌"
        print(f"  {status} {vector_name} ({vector_size}d) → {inferred}")
    
    # Test partial matches
    print("\n🔍 Testing Partial Pattern Matches:")
    partial_cases = [
        ("fast-nomic-embed", 768, "nomic-ai/nomic-embed-text-v1.5-Q"),
        ("fast-all-minilm", 384, "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    
    for vector_name, vector_size, expected_model in partial_cases:
        inferred = intelligence._infer_model_from_collection_info(vector_name, vector_size)
        status = "✅" if inferred == expected_model else "❌"
        print(f"  {status} {vector_name} ({vector_size}d) → {inferred}")
    
    # Test no matches
    print("\n❓ Testing Unknown Patterns:")
    unknown_cases = [
        ("unknown-vector", 999),
        ("custom-model", 512),
        (None, None),
    ]
    
    for vector_name, vector_size in unknown_cases:
        inferred = intelligence._infer_model_from_collection_info(vector_name, vector_size)
        status = "✅" if inferred is None else "❌"
        print(f"  {status} {vector_name} ({vector_size}d) → {inferred}")


async def demo_model_validation():
    """Demonstrate model validation functionality."""
    print("\n\n🧪 Demonstrating Model Validation")
    print("=" * 60)
    
    # Create settings
    settings = QdrantSettings()
    settings.location = "http://localhost:6333"
    
    # Create intelligence instance
    intelligence = EmbeddingModelIntelligence(settings)
    
    # Mock successful validation
    with patch('mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create:
        mock_provider = MagicMock()
        mock_provider.get_vector_size.return_value = 768
        mock_provider.get_vector_name.return_value = "fast-test-model"
        mock_create.return_value = mock_provider
        
        print("\n✅ Testing Successful Model Validation:")
        model_info = await intelligence.validate_model("test-model")
        print(f"  Model: {model_info.model_name}")
        print(f"  Available: {model_info.is_available}")
        print(f"  Vector Size: {model_info.vector_size}")
        print(f"  Vector Name: {model_info.vector_name}")
    
    # Mock failed validation
    with patch('mcp_server_qdrant_rag.cli_ingest.create_embedding_provider') as mock_create:
        mock_create.side_effect = Exception("Model not found")
        
        print("\n❌ Testing Failed Model Validation:")
        model_info = await intelligence.validate_model("invalid-model")
        print(f"  Model: {model_info.model_name}")
        print(f"  Available: {model_info.is_available}")
        print(f"  Error: {model_info.error_message}")


async def demo_smart_defaults():
    """Demonstrate smart default selection."""
    print("\n\n🎯 Demonstrating Smart Default Selection")
    print("=" * 60)
    
    # Create settings
    settings = QdrantSettings()
    settings.location = "http://localhost:6333"
    
    # Create intelligence instance
    intelligence = EmbeddingModelIntelligence(settings)
    
    # Mock existing collection detection
    with patch.object(intelligence, 'detect_collection_model') as mock_detect, \
         patch.object(intelligence, 'validate_model') as mock_validate:
        
        # Test 1: Existing collection with detected model
        print("\n📁 Testing Existing Collection with Detected Model:")
        mock_detect.return_value = EmbeddingModelInfo(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            vector_size=384,
            vector_name="fast-all-minilm-l6-v2",
            is_available=True,
            collection_exists=True
        )
        
        result = await intelligence.select_smart_default("existing_collection")
        print(f"  Selected Model: {result.model_name}")
        print(f"  From Collection: {result.collection_exists}")
        print(f"  Vector Size: {result.vector_size}")
        
        # Test 2: New collection with default model
        print("\n🆕 Testing New Collection with Default Model:")
        mock_detect.return_value = None  # No existing collection
        mock_validate.return_value = EmbeddingModelInfo(
            model_name=intelligence.DEFAULT_MODEL,
            vector_size=768,
            vector_name="fast-nomic-embed-text-v1.5-q",
            is_available=True
        )
        
        result = await intelligence.select_smart_default("new_collection")
        print(f"  Selected Model: {result.model_name}")
        print(f"  From Collection: {result.collection_exists}")
        print(f"  Vector Size: {result.vector_size}")


def demo_display_functionality():
    """Demonstrate display functionality."""
    print("\n\n📺 Demonstrating Display Functionality")
    print("=" * 60)
    
    # Create settings
    settings = QdrantSettings()
    settings.location = "http://localhost:6333"
    
    # Create intelligence instance
    intelligence = EmbeddingModelIntelligence(settings)
    
    # Test available model display
    print("\n✅ Available Model Display:")
    available_model = EmbeddingModelInfo(
        model_name="test-model",
        vector_size=768,
        vector_name="fast-test",
        is_available=True,
        is_compatible=True
    )
    intelligence.display_model_info(available_model)
    
    # Test unavailable model display
    print("\n❌ Unavailable Model Display:")
    unavailable_model = EmbeddingModelInfo(
        model_name="invalid-model",
        vector_size=0,
        vector_name="unknown",
        is_available=False,
        error_message="Model not found"
    )
    intelligence.display_model_info(unavailable_model)
    
    # Test incompatible model display
    print("\n⚠️  Incompatible Model Display:")
    incompatible_model = EmbeddingModelInfo(
        model_name="test-model",
        vector_size=768,
        vector_name="fast-test",
        is_available=True,
        is_compatible=False,
        collection_exists=True,
        error_message="Dimension mismatch"
    )
    intelligence.display_model_info(incompatible_model)
    
    # Test mismatch error display
    print("\n🚫 Mismatch Error Display:")
    mismatch_model = EmbeddingModelInfo(
        model_name="new-model",
        vector_size=768,
        vector_name="fast-new",
        collection_exists=True,
        collection_model="old-model",
        collection_vector_size=384,
        collection_vector_name="fast-old"
    )
    intelligence.display_model_mismatch_error(mismatch_model)


async def main():
    """Run all demonstrations."""
    print("🚀 Embedding Model Intelligence Demonstration")
    print("=" * 60)
    print("This demo shows the key features of the embedding model intelligence system:")
    print("• Model inference from collection metadata")
    print("• Model validation and compatibility checking")
    print("• Smart default selection")
    print("• User-friendly display and error reporting")
    
    await demo_model_inference()
    await demo_model_validation()
    await demo_smart_defaults()
    demo_display_functionality()
    
    print("\n\n🎉 Demonstration Complete!")
    print("The embedding model intelligence system provides:")
    print("• Automatic detection of existing collection models")
    print("• Smart fallback to compatible models")
    print("• Clear error messages and suggestions")
    print("• Seamless integration with CLI operations")


if __name__ == "__main__":
    asyncio.run(main())
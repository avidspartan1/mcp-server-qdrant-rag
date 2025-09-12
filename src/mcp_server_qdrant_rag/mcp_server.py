import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant_rag.common.filters import make_indexes
from mcp_server_qdrant_rag.common.func_tools import make_partial_function
from mcp_server_qdrant_rag.common.wrap_filters import wrap_filters
from mcp_server_qdrant_rag.embeddings.base import EmbeddingProvider
from mcp_server_qdrant_rag.embeddings.factory import create_embedding_provider
from mcp_server_qdrant_rag.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant_rag.semantic_matcher import SemanticSetMatcher, SemanticMatchError
from mcp_server_qdrant_rag.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    SetSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        sets_config_path: Optional[str] = None,
        name: str = "mcp-server-qdrant-rag",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        # Initialize set configuration and semantic matcher
        self.sets_config_path = sets_config_path  # Store the original config path
        self.set_settings = SetSettings()
        if sets_config_path:
            self.set_settings.load_from_file(self.set_settings.get_config_file_path(sets_config_path))
        else:
            self.set_settings.load_from_file()
        
        self.semantic_matcher = SemanticSetMatcher(self.set_settings.sets)

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
            enable_chunking=self.embedding_provider_settings.enable_chunking if self.embedding_provider_settings else True,
            max_chunk_size=self.embedding_provider_settings.max_chunk_size if self.embedding_provider_settings else 512,
            chunk_overlap=self.embedding_provider_settings.chunk_overlap if self.embedding_provider_settings else 50,
            chunk_strategy=self.embedding_provider_settings.chunk_strategy if self.embedding_provider_settings else "semantic",
        )

        # Log startup configuration
        self._log_startup_configuration()

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def _log_startup_configuration(self) -> None:
        """Log the active configuration at startup for debugging and transparency."""
        logger.info("=== MCP Server Qdrant Configuration ===")
        
        # Log embedding provider configuration
        if self.embedding_provider_settings:
            logger.info(f"Embedding Provider: {self.embedding_provider_settings.provider_type.value}")
            logger.info(f"Embedding Model: {self.embedding_provider_settings.model_name}")
            
            # Get model information if available
            try:
                model_info = self.embedding_provider.get_model_info()
                logger.info(f"Vector Dimensions: {model_info.get('vector_size', 'unknown')}")
                logger.info(f"Vector Name: {model_info.get('vector_name', 'unknown')}")
                logger.info(f"Model Status: {model_info.get('status', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not retrieve model information: {e}")
            
            # Log chunking configuration
            logger.info(f"Chunking Enabled: {self.embedding_provider_settings.enable_chunking}")
            if self.embedding_provider_settings.enable_chunking:
                logger.info(f"Max Chunk Size: {self.embedding_provider_settings.max_chunk_size} tokens")
                logger.info(f"Chunk Overlap: {self.embedding_provider_settings.chunk_overlap} tokens")
                logger.info(f"Chunk Strategy: {self.embedding_provider_settings.chunk_strategy}")
                
                # Calculate and log overlap percentage for context
                overlap_percentage = (self.embedding_provider_settings.chunk_overlap / 
                                    self.embedding_provider_settings.max_chunk_size) * 100
                logger.info(f"Overlap Percentage: {overlap_percentage:.1f}%")
        else:
            logger.info("Embedding Provider: Custom provider (no settings available)")
        
        # Log Qdrant configuration
        logger.info(f"Qdrant Location: {self.qdrant_settings.location or 'local'}")
        if self.qdrant_settings.local_path:
            logger.info(f"Qdrant Local Path: {self.qdrant_settings.local_path}")
        logger.info(f"Collection Name: {self.qdrant_settings.collection_name or 'default'}")
        logger.info(f"Search Limit: {self.qdrant_settings.search_limit}")
        logger.info(f"Read Only Mode: {self.qdrant_settings.read_only}")
        
        # Log filterable fields configuration
        filterable_fields = self.qdrant_settings.filterable_fields_dict_with_conditions()
        if filterable_fields:
            logger.info(f"Filterable Fields: {list(filterable_fields.keys())}")
        else:
            logger.info("Filterable Fields: None configured")
        
        logger.info(f"Allow Arbitrary Filter: {self.qdrant_settings.allow_arbitrary_filter}")
        
        # Log set configuration
        if self.set_settings.sets:
            logger.info(f"Set Configurations: {len(self.set_settings.sets)} sets loaded")
            for slug, config in self.set_settings.sets.items():
                logger.info(f"  - {slug}: {config.description}")
        else:
            logger.info("Set Configurations: None loaded")
        
        logger.info("=== Configuration Complete ===")

    async def reload_set_configurations(self, sets_config_path: Optional[str] = None) -> None:
        """
        Reload set configurations without server restart.
        
        Args:
            sets_config_path: Optional path to configuration file. If None, uses current path.
        """
        try:
            logger.info("Reloading set configurations...")
            
            # Create new settings instance and load configurations
            new_set_settings = SetSettings()
            if sets_config_path:
                new_set_settings.load_from_file(new_set_settings.get_config_file_path(sets_config_path))
            else:
                # Use the same path as was used during initialization
                if self.sets_config_path:
                    new_set_settings.load_from_file(new_set_settings.get_config_file_path(self.sets_config_path))
                else:
                    new_set_settings.load_from_file()
            
            # Update the current settings and semantic matcher
            self.set_settings = new_set_settings
            self.semantic_matcher.reload_configurations(self.set_settings.sets)
            
            logger.info(f"Successfully reloaded {len(self.set_settings.sets)} set configurations")
            
        except Exception as e:
            logger.error(f"Failed to reload set configurations: {e}")
            raise

    def format_entry(self, entry: Entry) -> str:
        """
        Format an entry for display, with enhanced chunk information.
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        
        # Add chunk information if this is a chunk
        if entry.is_chunk:
            chunk_info_parts = []
            
            # Check if this is an aggregated chunk result
            if hasattr(entry, '_chunk_count') and entry._chunk_count > 1:
                chunk_info_parts.append(f"Aggregated from {entry._chunk_count} chunks")
            elif entry.chunk_index is not None:
                chunk_info_parts.append(f"Chunk {entry.chunk_index + 1}/{entry.total_chunks}")
            else:
                chunk_info_parts.append("Multiple chunks")
            
            # Add source document information
            if entry.source_document_id and entry.source_document_id.strip():
                # Truncate long document IDs for readability
                doc_id = entry.source_document_id
                if len(doc_id) > 12:
                    doc_id = doc_id[:8] + "..."
                chunk_info_parts.append(f"from document {doc_id}")
            
            chunk_info = f" [{', '.join(chunk_info_parts)}]"
            
            return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata><chunk_info>{chunk_info}</chunk_info></entry>"
        else:
            return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
            set_filter: Annotated[
                Optional[str], 
                Field(description="Natural language description of the set to filter by")
            ] = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :param set_filter: Natural language description of the set to filter by.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")
            await ctx.debug(f"Set filter: {set_filter}")

            # Handle set filtering
            if set_filter:
                try:
                    matched_set_slug = await self.semantic_matcher.match_set(set_filter)
                    await ctx.debug(f"Matched set: {matched_set_slug}")
                    
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
                        query_filter = models.Filter(
                            must=combined_must,
                            must_not=existing_must_not,
                            should=existing_should
                        )
                    else:
                        # Create new filter with just the set condition
                        query_filter = models.Filter(must=[set_field_condition])
                        
                except SemanticMatchError as e:
                    await ctx.debug(f"Set matching error: {e}")
                    return [f"Error: {str(e)}"]

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def hybrid_find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            fusion_method: Annotated[
                str, Field(description="Fusion method: 'rrf' or 'dbsf'")
            ] = "rrf",
            dense_limit: Annotated[
                int, Field(description="Max results from semantic search")
            ] = 20,
            sparse_limit: Annotated[
                int, Field(description="Max results from keyword search")
            ] = 20,
            final_limit: Annotated[
                int, Field(description="Final number of results after fusion")
            ] = 10,
            query_filter: ArbitraryFilter | None = None,
            set_filter: Annotated[
                Optional[str], 
                Field(description="Natural language description of the set to filter by")
            ] = None,
        ) -> list[str] | None:
            """
            Hybrid search combining semantic similarity and keyword matching.
            Uses Qdrant's RRF/DBSF fusion for optimal search results.

            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in.
            :param fusion_method: Fusion method - 'rrf' (Reciprocal Rank Fusion) or 'dbsf' (Distribution-Based Score Fusion).
            :param dense_limit: Maximum results from dense vector search.
            :param sparse_limit: Maximum results from sparse vector search.
            :param final_limit: Maximum final results after fusion.
            :param query_filter: The filter to apply to the query.
            :param set_filter: Natural language description of the set to filter by.
            :return: A list of entries found or None.
            """
            await ctx.debug(
                f"Hybrid search for query '{query}' using fusion method '{fusion_method}'"
            )
            await ctx.debug(f"Set filter: {set_filter}")

            # Handle set filtering
            if set_filter:
                try:
                    matched_set_slug = await self.semantic_matcher.match_set(set_filter)
                    await ctx.debug(f"Matched set: {matched_set_slug}")
                    
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
                        parsed_query_filter = models.Filter(
                            must=combined_must,
                            must_not=existing_must_not,
                            should=existing_should
                        )
                    else:
                        # Create new filter with just the set condition
                        parsed_query_filter = models.Filter(must=[set_field_condition])
                        
                except SemanticMatchError as e:
                    await ctx.debug(f"Set matching error: {e}")
                    return [f"Error: {str(e)}"]
            else:
                parsed_query_filter = query_filter

            entries = await self.qdrant_connector.find_hybrid(
                query,
                collection_name=collection_name,
                fusion_method=fusion_method,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
                final_limit=final_limit,
                query_filter=parsed_query_filter,
            )

            if not entries:
                return None

            content = [
                f"Hybrid search results for '{query}' (fusion: {fusion_method})",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store
        hybrid_find_foo = hybrid_find

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            hybrid_find_foo = wrap_filters(hybrid_find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            hybrid_find_foo = make_partial_function(
                hybrid_find_foo, {"query_filter": None}
            )

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            hybrid_find_foo = make_partial_function(
                hybrid_find_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        self.tool(
            hybrid_find_foo,
            name="qdrant-hybrid-find",
            description=self.tool_settings.tool_hybrid_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

        # Add compatibility analysis tool (read-only operation)
        async def analyze_compatibility(
            ctx: Context,
            collection_name: Annotated[
                str | None, 
                Field(description="The collection to analyze for compatibility. If not provided, uses the default collection.")
            ] = None,
        ) -> str:
            """
            Analyze collection compatibility for backward compatibility assessment.
            This tool helps identify potential issues when switching embedding models,
            chunking configurations, or working with existing collections.
            
            :param ctx: The context for the request.
            :param collection_name: The name of the collection to analyze, optional.
            :return: A detailed compatibility analysis report.
            """
            await ctx.debug(f"Analyzing compatibility for collection: {collection_name or 'default'}")
            
            try:
                compatibility = await self.qdrant_connector.analyze_collection_compatibility(collection_name)
                
                # Format the analysis results for human readability
                report_lines = []
                
                if "error" in compatibility:
                    report_lines.append(f"❌ Error: {compatibility['error']}")
                    if "recommendations" in compatibility:
                        report_lines.append("\n📋 Recommendations:")
                        for rec in compatibility["recommendations"]:
                            report_lines.append(f"  • {rec}")
                    return "\n".join(report_lines)
                
                # Collection status
                collection_name_display = compatibility.get("collection_name", "unknown")
                if compatibility.get("exists", False):
                    report_lines.append(f"📊 Collection '{collection_name_display}' exists")
                    report_lines.append(f"   Points: {compatibility.get('points_count', 0)}")
                else:
                    report_lines.append(f"📊 Collection '{collection_name_display}' does not exist")
                    report_lines.append(f"   {compatibility.get('message', 'Will be created with current configuration')}")
                    return "\n".join(report_lines)
                
                # Compatibility status
                if compatibility.get("compatible", False):
                    report_lines.append("✅ Collection is compatible with current configuration")
                else:
                    report_lines.append("⚠️  Collection has compatibility issues")
                
                # Model and vector information
                report_lines.append(f"\n🔧 Current Configuration:")
                report_lines.append(f"   Model: {compatibility.get('current_model', 'unknown')}")
                report_lines.append(f"   Expected vector: {compatibility.get('expected_vector_name', 'unknown')} ({compatibility.get('expected_dimensions', 0)} dimensions)")
                report_lines.append(f"   Chunking enabled: {compatibility.get('chunking_enabled', False)}")
                
                # Collection vector information
                available_vectors = compatibility.get("available_vectors", [])
                if available_vectors:
                    report_lines.append(f"\n📐 Collection Vectors:")
                    for vector in available_vectors:
                        if vector == compatibility.get("expected_vector_name"):
                            status = "✅" if compatibility.get("dimension_compatible", False) else "❌"
                            dims = compatibility.get("actual_dimensions", 0)
                            report_lines.append(f"   {status} {vector} ({dims} dimensions)")
                        else:
                            report_lines.append(f"   ❓ {vector}")
                
                # Content analysis
                has_chunked = compatibility.get("has_chunked_content", False)
                has_non_chunked = compatibility.get("has_non_chunked_content", False)
                
                if has_chunked or has_non_chunked:
                    report_lines.append(f"\n📄 Content Analysis:")
                    if has_chunked:
                        report_lines.append("   ✅ Contains chunked content")
                    if has_non_chunked:
                        report_lines.append("   ✅ Contains non-chunked content")
                    if compatibility.get("mixed_content", False):
                        report_lines.append("   🔄 Mixed content types detected")
                
                # Recommendations
                recommendations = compatibility.get("recommendations", [])
                if recommendations:
                    report_lines.append(f"\n📋 Recommendations:")
                    for rec in recommendations:
                        report_lines.append(f"   • {rec}")
                
                return "\n".join(report_lines)
                
            except Exception as e:
                await ctx.debug(f"Compatibility analysis failed: {e}")
                return f"❌ Failed to analyze compatibility: {str(e)}"

        # Apply collection name default if configured
        analyze_compatibility_foo = analyze_compatibility
        if self.qdrant_settings.collection_name:
            analyze_compatibility_foo = make_partial_function(
                analyze_compatibility_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            analyze_compatibility_foo,
            name="qdrant-analyze-compatibility",
            description="Analyze collection compatibility for backward compatibility assessment. Helps identify issues when switching models or configurations.",
        )

        # Add set configuration reload tool
        async def reload_sets_config(
            ctx: Context,
            config_path: Annotated[
                Optional[str], 
                Field(description="Optional path to sets configuration file. If not provided, reloads from current path.")
            ] = None,
        ) -> str:
            """
            Reload set configurations without restarting the server.
            This allows updating set definitions and making them immediately available for search filtering.
            
            :param ctx: The context for the request.
            :param config_path: Optional path to configuration file to reload from.
            :return: Status message indicating success or failure.
            """
            await ctx.debug(f"Reloading set configurations from: {config_path or 'current path'}")
            
            try:
                await self.reload_set_configurations(config_path)
                
                # Provide feedback about loaded sets
                if self.set_settings.sets:
                    set_list = [f"  • {slug}: {config.description}" for slug, config in self.set_settings.sets.items()]
                    sets_info = "\n".join(set_list)
                    return f"✅ Successfully reloaded {len(self.set_settings.sets)} set configurations:\n{sets_info}"
                else:
                    return "✅ Configuration reloaded successfully, but no sets are currently defined."
                    
            except Exception as e:
                await ctx.debug(f"Set configuration reload failed: {e}")
                return f"❌ Failed to reload set configurations: {str(e)}"

        self.tool(
            reload_sets_config,
            name="qdrant-reload-sets-config",
            description="Reload set configurations without restarting the server. Allows updating set definitions for search filtering.",
        )

# mcp-server-qdrant-rag: Intelligent RAG with Qdrant Vector Database

[![smithery badge](https://smithery.ai/badge/mcp-server-qdrant-rag)](https://smithery.ai/protocol/mcp-server-qdrant-rag)

> The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that enables
> seamless integration between LLM applications and external data sources and tools. Whether you're building an
> AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to
> connect LLMs with the context they need.

This repository provides an enhanced MCP server for [Qdrant](https://qdrant.tech/) that specializes in intelligent document chunking and retrieval-augmented generation (RAG). It's a fork of the upstream `mcp-server-qdrant` project with advanced chunking capabilities for improved vector search performance.

## Overview

A specialized Model Context Protocol server that provides intelligent document chunking and semantic memory capabilities using Qdrant vector database. This enhanced version automatically splits large documents into optimal chunks for better retrieval performance, while maintaining full backward compatibility with the original MCP server.

**Key Enhancements over upstream:**
- **Intelligent Document Chunking**: Automatic splitting of large documents using semantic, sentence, or fixed strategies
- **Configurable Chunking Parameters**: Customizable chunk sizes, overlap, and splitting strategies
- **Enhanced Retrieval Performance**: Optimized for better context retrieval from large documents
- **Backward Compatibility**: Seamlessly works with existing collections and configurations

## Components

### Tools

1. `qdrant-store`
   - Store some information in the Qdrant database
   - Input:
     - `information` (string): Information to store
     - `metadata` (JSON): Optional metadata to store
     - `collection_name` (string): Name of the collection to store the information in. This parameter is only available when no default collection is configured via `COLLECTION_NAME` environment variable. When a default collection is set, this parameter is removed entirely.
   - Returns: Confirmation message
2. `qdrant-find`
   - Retrieve relevant information from the Qdrant database using semantic search
   - Input:
     - `query` (string): Query to use for searching
     - `collection_name` (string): Name of the collection to search in. This parameter is only available when no default collection is configured via `COLLECTION_NAME` environment variable. When a default collection is set, this parameter is removed entirely.
   - Returns: Information stored in the Qdrant database as separate messages
3. `qdrant-hybrid-find`
   - Advanced hybrid search combining semantic similarity and keyword matching using Qdrant's RRF/DBSF fusion
   - Input:
     - `query` (string): Query to use for searching
     - `collection_name` (string): Name of the collection to search in. This field is required if there are no default collection name.
                                   If there is a default collection name, this field is not enabled.
     - `fusion_method` (string, optional): Fusion method - "rrf" (Reciprocal Rank Fusion) or "dbsf" (Distribution-Based Score Fusion). Default: "rrf"
     - `dense_limit` (integer, optional): Maximum results from semantic search. Default: 20
     - `sparse_limit` (integer, optional): Maximum results from keyword search. Default: 20  
     - `final_limit` (integer, optional): Final number of results after fusion. Default: 10
   - Returns: Fused search results combining both semantic understanding and exact keyword matching

## Environment Variables

The configuration of the server is done using environment variables:

| Name                     | Description                                                         | Default Value                                                     |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `QDRANT_URL`             | URL of the Qdrant server                                            | None                                                              |
| `QDRANT_API_KEY`         | API key for the Qdrant server                                       | None                                                              |
| `COLLECTION_NAME`        | Name of the default collection to use.                              | None                                                              |
| `QDRANT_LOCAL_PATH`      | Path to the local Qdrant database (alternative to `QDRANT_URL`)     | None                                                              |
| `EMBEDDING_PROVIDER`     | Embedding provider to use (currently only "fastembed" is supported) | `fastembed`                                                       |
| `EMBEDDING_MODEL`        | Name of the embedding model to use                                  | `nomic-ai/nomic-embed-text-v1.5-Q`                               |
| `TOOL_STORE_DESCRIPTION` | Custom description for the store tool                               | See default in [`settings.py`](src/mcp_server_qdrant_rag/settings.py) |
| `TOOL_FIND_DESCRIPTION`  | Custom description for the find tool                                | See default in [`settings.py`](src/mcp_server_qdrant_rag/settings.py) |
| `TOOL_HYBRID_FIND_DESCRIPTION` | Custom description for the hybrid find tool                   | See default in [`settings.py`](src/mcp_server_qdrant/settings.py) |

### Document Chunking Configuration

The server now supports intelligent document chunking for improved retrieval performance with large documents:

| Name                     | Description                                                         | Default Value                                                     |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `ENABLE_CHUNKING`        | Enable automatic document chunking for large documents              | `true`                                                            |
| `MAX_CHUNK_SIZE`         | Maximum size of document chunks in tokens/characters                | `512`                                                             |
| `CHUNK_OVERLAP`          | Number of tokens/characters to overlap between chunks               | `50`                                                              |
| `CHUNK_STRATEGY`         | Chunking strategy: `semantic`, `fixed`, or `sentence`               | `semantic`                                                        |

Note: You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

> [!IMPORTANT]
> Command-line arguments are not supported anymore! Please use environment variables for all configuration.

### FastMCP Environment Variables

Since `mcp-server-qdrant-rag` is based on FastMCP, it also supports all the FastMCP environment variables. The most
important ones are listed below:

| Environment Variable                  | Description                                               | Default Value |
|---------------------------------------|-----------------------------------------------------------|---------------|
| `FASTMCP_DEBUG`                       | Enable debug mode                                         | `false`       |
| `FASTMCP_LOG_LEVEL`                   | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO`        |
| `FASTMCP_HOST`                        | Host address to bind the server to                        | `127.0.0.1`   |
| `FASTMCP_PORT`                        | Port to run the server on                                 | `8000`        |
| `FASTMCP_WARN_ON_DUPLICATE_RESOURCES` | Show warnings for duplicate resources                     | `true`        |
| `FASTMCP_WARN_ON_DUPLICATE_TOOLS`     | Show warnings for duplicate tools                         | `true`        |
| `FASTMCP_WARN_ON_DUPLICATE_PROMPTS`   | Show warnings for duplicate prompts                       | `true`        |
| `FASTMCP_DEPENDENCIES`                | List of dependencies to install in the server environment | `[]`          |

## Installation

### Using uvx

When using [`uvx`](https://docs.astral.sh/uv/guides/tools/#running-tools) no specific installation is needed to directly run *mcp-server-qdrant-rag*.

```shell
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-Q" \
uvx mcp-server-qdrant-rag
```

#### Transport Protocols

The server supports different transport protocols that can be specified using the `--transport` flag:

```shell
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
uvx mcp-server-qdrant-rag --transport sse
```

Supported transport protocols:

- `stdio` (default): Standard input/output transport, might only be used by local MCP clients
- `sse`: Server-Sent Events transport, perfect for remote clients
- `streamable-http`: Streamable HTTP transport, perfect for remote clients, more recent than SSE

The default transport is `stdio` if not specified.

When SSE transport is used, the server will listen on the specified port and wait for incoming connections. The default
port is 8000, however it can be changed using the `FASTMCP_PORT` environment variable.

```shell
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
FASTMCP_PORT=1234 \
uvx mcp-server-qdrant-rag --transport sse
```

### Using Docker

A Dockerfile is available for building and running the MCP server:

```bash
# Build the container
docker build -t mcp-server-qdrant-rag .

# Run the container
docker run -p 8000:8000 \
  -e FASTMCP_HOST="0.0.0.0" \
  -e QDRANT_URL="http://your-qdrant-server:6333" \
  -e QDRANT_API_KEY="your-api-key" \
  -e COLLECTION_NAME="your-collection" \
  mcp-server-qdrant-rag
```

> [!TIP]
> Please note that we set `FASTMCP_HOST="0.0.0.0"` to make the server listen on all network interfaces. This is
> necessary when running the server in a Docker container.

### Installing via Smithery

To install Qdrant RAG MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/protocol/mcp-server-qdrant-rag):

```bash
npx @smithery/cli install mcp-server-qdrant-rag --client claude
```

### Manual configuration of Claude Desktop

To use this server with the Claude Desktop app, add the following configuration to the "mcpServers" section of your
`claude_desktop_config.json`:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant-rag"],
    "env": {
      "QDRANT_URL": "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
      "QDRANT_API_KEY": "your_api_key",
      "COLLECTION_NAME": "your-collection-name",
      "EMBEDDING_MODEL": "nomic-ai/nomic-embed-text-v1.5-Q"
    }
  }
}
```

For local Qdrant mode:

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant-rag"],
    "env": {
      "QDRANT_LOCAL_PATH": "/path/to/qdrant/database",
      "COLLECTION_NAME": "your-collection-name",
      "EMBEDDING_MODEL": "nomic-ai/nomic-embed-text-v1.5-Q"
    }
  }
}
```

This MCP server will automatically create a collection with the specified name if it doesn't exist.

By default, the server will use the `nomic-ai/nomic-embed-text-v1.5-Q` embedding model to encode memories.
For the time being, only [FastEmbed](https://qdrant.github.io/fastembed/) models are supported.

The server automatically detects vector dimensions based on the selected embedding model and supports intelligent document chunking for improved retrieval performance with large documents.

## CLI Tool: qdrant-ingest

The package includes a powerful CLI tool for bulk ingestion of files into Qdrant collections. This tool is perfect for building knowledge bases from existing documents, code repositories, or documentation.

### Basic Usage

```bash
# Ingest all supported files from a directory
qdrant-ingest /path/to/documents

# Ingest with custom knowledge base name
qdrant-ingest /path/to/documents --knowledgebase my-docs

# Ingest single file
qdrant-ingest /path/to/document.txt --knowledgebase single-doc

# Connect to remote Qdrant instance
qdrant-ingest /path/to/docs --url https://my-qdrant.example.com:6333 --api-key your-key
```

### Advanced Examples

```bash
# Filter files with regex patterns
qdrant-ingest /path/to/code --include ".*\\.py$" --exclude ".*test.*"

# Use specific embedding model
qdrant-ingest /path/to/docs --embedding sentence-transformers/all-MiniLM-L6-v2

# Verbose output with progress tracking
qdrant-ingest /path/to/docs --verbose

# Update existing knowledge base
qdrant-ingest update /path/to/new-docs --knowledgebase my-docs

# List all knowledge bases
qdrant-ingest list

# Remove knowledge base
qdrant-ingest remove my-docs
```

### Supported File Types

The CLI tool automatically processes these file types:
`.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`, `.yml`, `.rst`, `.tf`, `.java`, `.sh`, `.go`, `.rb`, `.ts`, `.conf`, `.ini`, `.cfg`, `.toml`, `.xml`, `.html`, `.css`, `.sql`

### Getting Help

For complete usage information and all available options:

```bash
qdrant-ingest --help
qdrant-ingest <command> --help  # Help for specific commands
```

## Configuration Examples

### Basic Configuration with Chunking

```bash
# Basic setup with default chunking enabled
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="my-documents" \
EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-Q" \
ENABLE_CHUNKING=true \
uvx mcp-server-qdrant-rag
```

### Advanced Chunking Configuration

```bash
# Custom chunking settings for large documents
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="large-docs" \
EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-Q" \
ENABLE_CHUNKING=true \
MAX_CHUNK_SIZE=1024 \
CHUNK_OVERLAP=100 \
CHUNK_STRATEGY=semantic \
uvx mcp-server-qdrant-rag
```

### Different Chunking Strategies

#### Semantic Chunking (Recommended)
```bash
# Splits at sentence/paragraph boundaries for better context
CHUNK_STRATEGY=semantic \
MAX_CHUNK_SIZE=512 \
CHUNK_OVERLAP=50
```

#### Fixed Chunking
```bash
# Splits at fixed character/token boundaries
CHUNK_STRATEGY=fixed \
MAX_CHUNK_SIZE=800 \
CHUNK_OVERLAP=80
```

#### Sentence Chunking
```bash
# Splits at sentence boundaries only
CHUNK_STRATEGY=sentence \
MAX_CHUNK_SIZE=300 \
CHUNK_OVERLAP=30
```

### Claude Desktop Configuration with Chunking

```json
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant-rag"],
    "env": {
      "QDRANT_URL": "https://xyz-example.eu-central.aws.cloud.qdrant.io:6333",
      "QDRANT_API_KEY": "your_api_key",
      "COLLECTION_NAME": "your-collection-name",
      "EMBEDDING_MODEL": "nomic-ai/nomic-embed-text-v1.5-Q",
      "ENABLE_CHUNKING": "true",
      "MAX_CHUNK_SIZE": "512",
      "CHUNK_OVERLAP": "50",
      "CHUNK_STRATEGY": "semantic"
    }
  }
}
```

### Performance-Optimized Configuration

```bash
# Optimized for retrieval performance
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="optimized-docs" \
EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-Q" \
ENABLE_CHUNKING=true \
MAX_CHUNK_SIZE=512 \
CHUNK_OVERLAP=50 \
CHUNK_STRATEGY=semantic \
QDRANT_SEARCH_LIMIT=20 \
uvx mcp-server-qdrant-rag
```

## Migration Guide

### Upgrading from Previous Versions

When upgrading to the new version with chunking support, existing deployments will continue to work without changes. However, to take advantage of the new features:

#### Automatic Migration
- **Existing collections**: Continue to work with non-chunked documents
- **New documents**: Automatically use chunking if enabled
- **Search**: Seamlessly works across both chunked and non-chunked content
- **Backward compatibility**: Fully maintained

#### Configuration Migration
1. **Update embedding model** (optional but recommended):
   ```bash
   # Old default
   EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   
   # New default (better quality)
   EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-Q"
   ```

2. **Enable chunking** for new documents:
   ```bash
   ENABLE_CHUNKING=true
   MAX_CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   CHUNK_STRATEGY=semantic
   ```

#### Handling Vector Dimension Changes

If you change embedding models, you may encounter dimension mismatches:

```bash
# Error example
ConfigurationValidationError: Vector dimension mismatch: existing collection uses 384 dimensions, but model 'nomic-ai/nomic-embed-text-v1.5-Q' produces 768 dimensions
```

**Solutions:**
1. **Use a new collection** (recommended):
   ```bash
   COLLECTION_NAME="my-collection-v2"
   ```

2. **Keep the old model** for existing collections:
   ```bash
   EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   ```

3. **Recreate the collection** (data loss):
   - Delete the existing collection in Qdrant
   - Restart the server with the new model

## Performance Recommendations

### Chunk Size Selection

Choose chunk sizes based on your use case:

| Use Case | Recommended Size | Overlap | Strategy | Reasoning |
|----------|------------------|---------|----------|-----------|
| **General documents** | 512 tokens | 50 tokens | `semantic` | Balanced performance and context |
| **Code snippets** | 300 tokens | 30 tokens | `sentence` | Preserve function/class boundaries |
| **Long articles** | 1024 tokens | 100 tokens | `semantic` | Better context for large content |
| **Short notes** | 256 tokens | 25 tokens | `fixed` | Minimal overhead |
| **Technical docs** | 768 tokens | 75 tokens | `semantic` | Good for structured content |

### Performance Optimization Tips

1. **Chunk Size vs. Retrieval Quality**:
   - **Smaller chunks** (256-512): Better precision, faster search
   - **Larger chunks** (1024+): Better context, may reduce precision

2. **Overlap Considerations**:
   - **10-15% overlap**: Good balance for most use cases
   - **Higher overlap** (20%+): Better for documents with complex references
   - **Lower overlap** (5%): Faster processing, less storage

3. **Strategy Selection**:
   - **Semantic**: Best for natural language documents
   - **Sentence**: Good for structured content (code, lists)
   - **Fixed**: Fastest processing, predictable sizes

4. **Memory and Storage**:
   - Chunking increases the number of vectors stored
   - Estimate: 1000-word document → 4-8 chunks (depending on settings)
   - Monitor Qdrant collection size and adjust accordingly

### Monitoring and Debugging

Enable detailed logging to monitor chunking performance:

```bash
FASTMCP_LOG_LEVEL=DEBUG \
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="debug-collection" \
ENABLE_CHUNKING=true \
uvx mcp-server-qdrant-rag
```

This will log:
- Chunking decisions and statistics
- Vector dimension validation
- Configuration validation results
- Search performance metrics

## Troubleshooting

### NLTK SSL Certificate Issues

If you encounter SSL certificate errors when the system tries to download NLTK data:

```
[nltk_data] Error loading punkt_tab: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
```

The system automatically handles this by disabling SSL verification for NLTK downloads. This is implemented using the standard approach:

```python
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
```

If NLTK data download fails completely, the system will automatically fall back to alternative sentence splitters (syntok or simple regex-based splitting), so functionality is maintained.

### Graceful Shutdown

The server includes proper signal handling for immediate shutdown. When using stdio transport, you can exit with Ctrl+C (SIGINT) or SIGTERM. The server will:

- Catch the interrupt signal
- Print a shutdown message  
- Force immediate exit without waiting for background threads

This uses `os._exit()` to ensure the process terminates immediately, which is necessary because the stdio transport creates background threads that can prevent normal shutdown. This is particularly important when running with `uvx` or in containerized environments.

## Support for other tools

This MCP server can be used with any MCP-compatible client. For example, you can use it with
[Cursor](https://docs.cursor.com/context/model-context-protocol) and [VS Code](https://code.visualstudio.com/docs), which provide built-in support for the Model Context
Protocol.

### Using with Cursor/Windsurf

You can configure this MCP server to work as a code search tool for Cursor or Windsurf by customizing the tool
descriptions:

```bash
QDRANT_URL="http://localhost:6333" \
COLLECTION_NAME="code-snippets" \
TOOL_STORE_DESCRIPTION="Store reusable code snippets for later retrieval. \
The 'information' parameter should contain a natural language description of what the code does, \
while the actual code should be included in the 'metadata' parameter as a 'code' property. \
The value of 'metadata' is a Python dictionary with strings as keys. \
Use this whenever you generate some code snippet." \
TOOL_FIND_DESCRIPTION="Search for relevant code snippets based on natural language descriptions. \
The 'query' parameter should describe what you're looking for, \
and the tool will return the most relevant code snippets. \
Use this when you need to find existing code snippets for reuse or reference." \
uvx mcp-server-qdrant-rag --transport sse # Enable SSE transport
```

In Cursor/Windsurf, you can then configure the MCP server in your settings by pointing to this running server using
SSE transport protocol. The description on how to add an MCP server to Cursor can be found in the [Cursor
documentation](https://docs.cursor.com/context/model-context-protocol#adding-an-mcp-server-to-cursor). If you are
running Cursor/Windsurf locally, you can use the following URL:

```
http://localhost:8000/sse
```

> [!TIP]
> We suggest SSE transport as a preferred way to connect Cursor/Windsurf to the MCP server, as it can support remote
> connections. That makes it easy to share the server with your team or use it in a cloud environment.

This configuration transforms the Qdrant MCP server into a specialized code search tool that can:

1. Store code snippets, documentation, and implementation details
2. Retrieve relevant code examples based on semantic search
3. Help developers find specific implementations or usage patterns

You can populate the database by storing natural language descriptions of code snippets (in the `information` parameter)
along with the actual code (in the `metadata.code` property), and then search for them using natural language queries
that describe what you're looking for.

> [!NOTE]
> The tool descriptions provided above are examples and may need to be customized for your specific use case. Consider
> adjusting the descriptions to better match your team's workflow and the specific types of code snippets you want to
> store and retrieve.

**If you have successfully installed the `mcp-server-qdrant-rag`, but still can't get it to work with Cursor, please
consider creating the [Cursor rules](https://docs.cursor.com/context/rules-for-ai) so the MCP tools are always used when
the agent produces a new code snippet.** You can restrict the rules to only work for certain file types, to avoid using
the MCP server for the documentation or other types of content.

### Using with Claude Code

You can enhance Claude Code's capabilities by connecting it to this MCP server, enabling semantic search over your
existing codebase.

#### Setting up mcp-server-qdrant-rag

1. Add the MCP server to Claude Code:

    ```shell
    # Add mcp-server-qdrant-rag configured for code search
    claude mcp add code-search \
    -e QDRANT_URL="http://localhost:6333" \
    -e COLLECTION_NAME="code-repository" \
    -e EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5-Q" \
    -e TOOL_STORE_DESCRIPTION="Store code snippets with descriptions. The 'information' parameter should contain a natural language description of what the code does, while the actual code should be included in the 'metadata' parameter as a 'code' property." \
    -e TOOL_FIND_DESCRIPTION="Search for relevant code snippets using natural language. The 'query' parameter should describe the functionality you're looking for." \
    -- uvx mcp-server-qdrant-rag
    ```

2. Verify the server was added:

    ```shell
    claude mcp list
    ```

#### Using Semantic Code Search in Claude Code

Tool descriptions, specified in `TOOL_STORE_DESCRIPTION` and `TOOL_FIND_DESCRIPTION`, guide Claude Code on how to use
the MCP server. The ones provided above are examples and may need to be customized for your specific use case. However,
Claude Code should be already able to:

1. Use the `qdrant-store` tool to store code snippets with descriptions.
2. Use the `qdrant-find` tool to search for relevant code snippets using natural language.

### Run MCP server in Development Mode

The MCP server can be run in development mode using the `mcp dev` command. This will start the server and open the MCP
inspector in your browser.

```shell
COLLECTION_NAME=mcp-dev fastmcp dev src/mcp_server_qdrant_rag/server.py
```

### Using with VS Code

For one-click installation, click one of the install buttons below:

[![Install with UVX in VS Code](https://img.shields.io/badge/VS_Code-UVX-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant-rag&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22mcp-server-qdrant-rag%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D) [![Install with UVX in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-UVX-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant-rag&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22mcp-server-qdrant-rag%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D&quality=insiders)

[![Install with Docker in VS Code](https://img.shields.io/badge/VS_Code-Docker-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant-rag&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-p%22%2C%228000%3A8000%22%2C%22-i%22%2C%22--rm%22%2C%22-e%22%2C%22QDRANT_URL%22%2C%22-e%22%2C%22QDRANT_API_KEY%22%2C%22-e%22%2C%22COLLECTION_NAME%22%2C%22mcp-server-qdrant-rag%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D) [![Install with Docker in VS Code Insiders](https://img.shields.io/badge/VS_Code_Insiders-Docker-24bfa5?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=qdrant-rag&config=%7B%22command%22%3A%22docker%22%2C%22args%22%3A%5B%22run%22%2C%22-p%22%2C%228000%3A8000%22%2C%22-i%22%2C%22--rm%22%2C%22-e%22%2C%22QDRANT_URL%22%2C%22-e%22%2C%22QDRANT_API_KEY%22%2C%22-e%22%2C%22COLLECTION_NAME%22%2C%22mcp-server-qdrant-rag%22%5D%2C%22env%22%3A%7B%22QDRANT_URL%22%3A%22%24%7Binput%3AqdrantUrl%7D%22%2C%22QDRANT_API_KEY%22%3A%22%24%7Binput%3AqdrantApiKey%7D%22%2C%22COLLECTION_NAME%22%3A%22%24%7Binput%3AcollectionName%7D%22%7D%7D&inputs=%5B%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantUrl%22%2C%22description%22%3A%22Qdrant+URL%22%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22qdrantApiKey%22%2C%22description%22%3A%22Qdrant+API+Key%22%2C%22password%22%3Atrue%7D%2C%7B%22type%22%3A%22promptString%22%2C%22id%22%3A%22collectionName%22%2C%22description%22%3A%22Collection+Name%22%7D%5D&quality=insiders)

#### Manual Installation

Add the following JSON block to your User Settings (JSON) file in VS Code. You can do this by pressing `Ctrl + Shift + P` and typing `Preferences: Open User Settings (JSON)`.

```json
{
  "mcp": {
    "inputs": [
      {
        "type": "promptString",
        "id": "qdrantUrl",
        "description": "Qdrant URL"
      },
      {
        "type": "promptString",
        "id": "qdrantApiKey",
        "description": "Qdrant API Key",
        "password": true
      },
      {
        "type": "promptString",
        "id": "collectionName",
        "description": "Collection Name"
      }
    ],
    "servers": {
      "qdrant": {
        "command": "uvx",
        "args": ["mcp-server-qdrant-rag"],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}"
        }
      }
    }
  }
}
```

Or if you prefer using Docker, add this configuration instead:

```json
{
  "mcp": {
    "inputs": [
      {
        "type": "promptString",
        "id": "qdrantUrl",
        "description": "Qdrant URL"
      },
      {
        "type": "promptString",
        "id": "qdrantApiKey",
        "description": "Qdrant API Key",
        "password": true
      },
      {
        "type": "promptString",
        "id": "collectionName",
        "description": "Collection Name"
      }
    ],
    "servers": {
      "qdrant": {
        "command": "docker",
        "args": [
          "run",
          "-p", "8000:8000",
          "-i",
          "--rm",
          "-e", "QDRANT_URL",
          "-e", "QDRANT_API_KEY",
          "-e", "COLLECTION_NAME",
          "mcp-server-qdrant-rag"
        ],
        "env": {
          "QDRANT_URL": "${input:qdrantUrl}",
          "QDRANT_API_KEY": "${input:qdrantApiKey}",
          "COLLECTION_NAME": "${input:collectionName}"
        }
      }
    }
  }
}
```

Alternatively, you can create a `.vscode/mcp.json` file in your workspace with the following content:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "qdrantUrl",
      "description": "Qdrant URL"
    },
    {
      "type": "promptString",
      "id": "qdrantApiKey",
      "description": "Qdrant API Key",
      "password": true
    },
    {
      "type": "promptString",
      "id": "collectionName",
      "description": "Collection Name"
    }
  ],
  "servers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant-rag"],
      "env": {
        "QDRANT_URL": "${input:qdrantUrl}",
        "QDRANT_API_KEY": "${input:qdrantApiKey}",
        "COLLECTION_NAME": "${input:collectionName}"
      }
    }
  }
}
```

For workspace configuration with Docker, use this in `.vscode/mcp.json`:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "qdrantUrl",
      "description": "Qdrant URL"
    },
    {
      "type": "promptString",
      "id": "qdrantApiKey",
      "description": "Qdrant API Key",
      "password": true
    },
    {
      "type": "promptString",
      "id": "collectionName",
      "description": "Collection Name"
    }
  ],
  "servers": {
    "qdrant": {
      "command": "docker",
      "args": [
        "run",
        "-p", "8000:8000",
        "-i",
        "--rm",
        "-e", "QDRANT_URL",
        "-e", "QDRANT_API_KEY",
        "-e", "COLLECTION_NAME",
        "mcp-server-qdrant-rag"
      ],
      "env": {
        "QDRANT_URL": "${input:qdrantUrl}",
        "QDRANT_API_KEY": "${input:qdrantApiKey}",
        "COLLECTION_NAME": "${input:collectionName}"
      }
    }
  }
}
```

## Contributing

If you have suggestions for how mcp-server-qdrant-rag could be improved, or want to report a bug, open an issue!
We'd love all and any contributions.

### Testing `mcp-server-qdrant-rag` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
QDRANT_URL=":memory:" COLLECTION_NAME="test" \
fastmcp dev src/mcp_server_qdrant_rag/server.py
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.

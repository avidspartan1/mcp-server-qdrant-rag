from mcp_server_qdrant_rag.mcp_server import QdrantMCPServer
from mcp_server_qdrant_rag.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
)

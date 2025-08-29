# Development Guide

This guide covers local development setup for mcp-server-qdrant using Docker Compose.

## Quick Start

1. **Set up development environment:**
   ```bash
   make dev-setup
   ```

2. **Start the development environment:**
   ```bash
   make dev
   ```

3. **Access services:**
   - MCP Server SSE: http://localhost:8000/sse
   - Qdrant API: http://localhost:6333
   - Qdrant Dashboard: http://localhost:6333/dashboard

## Available Commands

Run `make help` to see all available commands:

```bash
make help
```

### Essential Commands

- `make dev-setup` - Initial development environment setup
- `make dev` - Start development environment with live reload
- `make test` - Run tests locally
- `make logs` - View logs from all services
- `make clean` - Clean up Docker resources

## Development Workflows

### 1. Production Mode (using PyPI package)

```bash
# Start all services in production mode
make up

# View logs
make logs

# Run tests locally
make test

# Stop services
make down
```

### 2. Development Mode (using local source code)

```bash
# Build development images first
make build-dev

# Start with live code reloading
make dev

# In another terminal, run tests
make test

# View MCP server logs only
make logs-mcp
```

### 3. Testing Different Configurations

Edit the `.env` file to test different configurations:

```bash
# Edit configuration
vim .env

# Restart services to apply changes
make down && make up
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# Chunking configuration
ENABLE_CHUNKING=true
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNK_STRATEGY=semantic

# Embedding model
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5-Q

# Debug settings
FASTMCP_DEBUG=true
FASTMCP_LOG_LEVEL=DEBUG
```

### Docker Compose Profiles

- **Default**: MCP server + Qdrant
- **web-ui**: Includes Qdrant web interface

```bash
# Start with web UI
make qdrant-ui
```

## Testing

### Local Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_chunking.py -v

# Run integration tests only
make test-integration
```

### Docker Testing

```bash
# Run tests in Docker container
make test-docker
```

## Debugging

### MCP Inspector

Use the FastMCP inspector for interactive debugging:

```bash
make inspect-mcp
```

Then open http://localhost:5173 in your browser.

### Service Health Checks

```bash
# Check if services are healthy
make health
```

### Container Shells

```bash
# Access MCP server container
make shell-mcp

# Access Qdrant container
make shell-qdrant
```

### Logs

```bash
# All service logs
make logs

# MCP server logs only
make logs-mcp

# Qdrant logs only
make logs-qdrant
```

## Code Quality

### Linting and Formatting

```bash
# Check code quality
make lint

# Format code
make format

# Run pre-commit hooks
make pre-commit
```

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using the ports
   lsof -i :6333
   lsof -i :8000
   
   # Stop conflicting services or change ports in docker-compose.yml
   ```

2. **Permission issues:**
   ```bash
   # Fix Docker permissions
   sudo chown -R $USER:$USER .
   ```

3. **Qdrant connection issues:**
   ```bash
   # Check Qdrant health
   curl http://localhost:6333/health
   
   # View Qdrant logs
   make logs-qdrant
   ```

4. **Embedding model download issues:**
   ```bash
   # Check MCP server logs for download progress
   make logs-mcp
   
   # Models are cached in container, rebuild if needed
   make clean && make build
   ```

### Clean Slate

If you encounter persistent issues:

```bash
# Clean everything and start fresh
make clean-all
make dev-setup
make dev
```

## Development Tips

1. **Use live reload**: The development setup automatically reloads code changes
2. **Monitor logs**: Keep `make logs` running in a separate terminal
3. **Test configurations**: Use different `.env` settings to test various scenarios
4. **Use the inspector**: The MCP inspector is great for testing tool interactions
5. **Check health**: Use `make health` to verify services are running correctly

## Production Testing

To test production-like behavior:

```bash
# Use production compose file
docker-compose -f docker-compose.yml up

# Or build production image
docker build -t mcp-server-qdrant .
docker run -p 8000:8000 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e COLLECTION_NAME=test \
  mcp-server-qdrant
```
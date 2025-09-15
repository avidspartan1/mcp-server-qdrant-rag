# Sets Configuration Guide

This guide explains how to configure document sets for semantic metadata filtering in mcp-server-qdrant-rag.

## Overview

Sets allow you to organize documents into logical groups that can be filtered during search operations. The system uses semantic matching to map natural language queries to configured sets, making it easy to search within specific document collections.

## Configuration File Format

### File Location

The sets configuration file location follows this precedence order:

1. **Command-line flag**: `--sets-config /path/to/sets.json` (CLI) or server parameter
2. **Environment variable**: `QDRANT_SETS_CONFIG=/path/to/sets.json`
3. **Default location**: `.qdrant_sets.json` in current working directory

### JSON Schema

```json
{
  "sets": {
    "set_slug": {
      "slug": "set_slug",
      "description": "Human-readable description for semantic matching",
      "aliases": ["alternative", "names", "for", "this", "set"]
    }
  }
}
```

### Properties

- **`slug`** (string, required): Unique identifier for the set
  - Must contain only alphanumeric characters, underscores, and hyphens
  - Used internally for storage and filtering
  - Should be descriptive but concise

- **`description`** (string, required): Human-readable description
  - Used for semantic matching against user queries
  - Should clearly describe what documents belong to this set
  - Examples: "Platform API Documentation", "Frontend React Components"

- **`aliases`** (array of strings, optional): Alternative names
  - Additional terms that can be used to reference this set
  - Improves semantic matching flexibility
  - Examples: ["api", "documentation", "api reference"] for API docs

## Example Configurations

### Basic Configuration

```json
{
  "sets": {
    "api_docs": {
      "slug": "api_docs",
      "description": "API Documentation",
      "aliases": ["api", "documentation", "api reference"]
    },
    "frontend_code": {
      "slug": "frontend_code",
      "description": "Frontend Application Code",
      "aliases": ["frontend", "ui", "client code"]
    }
  }
}
```

### Comprehensive Project Configuration

```json
{
  "sets": {
    "platform_api": {
      "slug": "platform_api",
      "description": "Platform API Documentation and Specifications",
      "aliases": [
        "platform api",
        "api docs",
        "platform documentation",
        "api specifications",
        "rest api"
      ]
    },
    "frontend_components": {
      "slug": "frontend_components",
      "description": "React Components and Frontend UI Code",
      "aliases": [
        "frontend",
        "react components",
        "ui components",
        "client code",
        "frontend code"
      ]
    },
    "backend_services": {
      "slug": "backend_services",
      "description": "Backend Services and Microservices Implementation",
      "aliases": [
        "backend",
        "services",
        "microservices",
        "server code",
        "backend code"
      ]
    },
    "database_schemas": {
      "slug": "database_schemas",
      "description": "Database Schemas and Migration Scripts",
      "aliases": [
        "database",
        "schemas",
        "migrations",
        "db scripts",
        "sql"
      ]
    },
    "deployment_configs": {
      "slug": "deployment_configs",
      "description": "Deployment Configurations and Infrastructure Code",
      "aliases": [
        "deployment",
        "infrastructure",
        "configs",
        "terraform",
        "kubernetes",
        "docker"
      ]
    }
  }
}
```

### Multi-Project Configuration

```json
{
  "sets": {
    "project_alpha_backend": {
      "slug": "project_alpha_backend",
      "description": "Project Alpha Backend Services",
      "aliases": ["alpha backend", "project alpha server"]
    },
    "project_alpha_frontend": {
      "slug": "project_alpha_frontend", 
      "description": "Project Alpha Frontend Application",
      "aliases": ["alpha frontend", "project alpha ui"]
    },
    "project_beta_mobile": {
      "slug": "project_beta_mobile",
      "description": "Project Beta Mobile Application",
      "aliases": ["beta mobile", "mobile app", "project beta app"]
    },
    "shared_libraries": {
      "slug": "shared_libraries",
      "description": "Shared Libraries and Common Code",
      "aliases": ["shared code", "libraries", "common utilities"]
    }
  }
}
```

## Usage Examples

### CLI Ingestion with Sets

```bash
# Ingest with predefined set
qdrant-ingest /path/to/api-docs --set api_docs

# Ingest with custom configuration file
qdrant-ingest /path/to/docs --sets-config /path/to/custom-sets.json --set custom_set

# Combine with document type
qdrant-ingest /path/to/code --document-type source_code --set frontend_components
```

### MCP Search with Set Filtering

```python
# Search using natural language set descriptions
results = await qdrant_find(
    query="authentication implementation",
    collection_name="my-kb",
    set_filter="backend services"  # Matches "backend_services" set
)
# Response includes: "Results for the query 'authentication implementation' (filtered by set: backend_services)"

# Search using aliases
results = await qdrant_find(
    query="button components",
    collection_name="my-kb", 
    set_filter="react components"  # Matches "frontend_components" via alias
)
# Response includes: "Results for the query 'button components' (filtered by set: frontend_components)"

# Hybrid search with set filtering
results = await qdrant_hybrid_find(
    query="database migration scripts",
    collection_name="my-kb",
    set_filter="database"  # Matches "database_schemas" via alias
)
# Response includes: "Hybrid search results for 'database migration scripts' (fusion: rrf, filtered by set: database_schemas)"
```

**Set Display Feature**: When using `set_filter`, the search results now include the matched set name in the response header, providing transparency about which set was selected by your natural language query.

## Semantic Matching

The system uses fuzzy string matching to map natural language queries to configured sets:

### Matching Process

1. **Exact Description Match**: Direct match against the `description` field
2. **Alias Match**: Match against any item in the `aliases` array
3. **Fuzzy Matching**: Partial matches using string similarity algorithms
4. **Keyword Matching**: Individual word matches within descriptions and aliases

### Matching Examples

Given this configuration:
```json
{
  "frontend_components": {
    "slug": "frontend_components",
    "description": "React Components and Frontend UI Code", 
    "aliases": ["frontend", "react", "ui components", "client code"]
  }
}
```

These queries would match:
- ✅ "frontend" (exact alias match)
- ✅ "react components" (partial description match)
- ✅ "ui components" (exact alias match)
- ✅ "frontend code" (fuzzy match with description)
- ✅ "client" (partial alias match)
- ❌ "backend" (no match)

## Error Handling

### No Match Found
```
Error: No matching set found for 'unknown set'. Available sets: api_docs, frontend_components, backend_services
```

### Ambiguous Match
```
Error: Multiple sets match 'code'. Please be more specific.
```

### Configuration Errors
- **Missing file**: Default configuration created automatically
- **Invalid JSON**: Error logged, system continues with empty configuration
- **Invalid schema**: Invalid entries skipped, valid entries loaded

## Best Practices

### Naming Conventions

1. **Slugs**: Use descriptive, lowercase names with underscores
   - ✅ `api_documentation`, `frontend_components`
   - ❌ `docs`, `fe`, `stuff`

2. **Descriptions**: Be specific and descriptive
   - ✅ "React Components and Frontend UI Code"
   - ❌ "Frontend stuff"

3. **Aliases**: Include common variations and abbreviations
   - Include both full names and abbreviations
   - Consider how users might naturally refer to the set

### Organization Strategies

#### By Technology Stack
```json
{
  "react_frontend": {...},
  "python_backend": {...},
  "postgres_database": {...}
}
```

#### By Feature/Domain
```json
{
  "authentication": {...},
  "payment_processing": {...},
  "user_management": {...}
}
```

#### By Project Phase
```json
{
  "requirements": {...},
  "design_docs": {...},
  "implementation": {...},
  "testing": {...}
}
```

### Maintenance

1. **Regular Review**: Periodically review and update set configurations
2. **User Feedback**: Monitor search queries to identify missing sets or aliases
3. **Consistency**: Maintain consistent naming patterns across sets
4. **Documentation**: Keep this configuration file documented and version controlled

## Troubleshooting

### Common Issues

1. **Set not found**: Check spelling and available aliases
2. **Ambiguous matches**: Make descriptions more specific
3. **Configuration not loading**: Verify file path and JSON syntax
4. **Semantic matching too loose**: Reduce aliases or make descriptions more specific

### Debugging

Enable verbose logging to see semantic matching decisions:
```bash
FASTMCP_LOG_LEVEL=DEBUG uvx mcp-server-qdrant-rag
```

### Validation

Test your configuration with the CLI:
```bash
# This will show available sets if configuration loads successfully
qdrant-ingest list
```

## Pro Tips and Advanced Usage

### Dynamic Set Management

While sets are typically configured statically, you can implement dynamic workflows:

```bash
# Generate sets configuration from project structure
find /project -type d -name "src" | while read dir; do
  component=$(basename $(dirname $dir))
  echo "Ingesting $component..."
  qdrant-ingest "$dir" --set "${component}_code" --document-type code
done
```

### Integration with CI/CD

Automate documentation updates in your deployment pipeline:

```bash
# .github/workflows/docs-update.yml
- name: Update Knowledge Base
  run: |
    # Update API docs
    qdrant-ingest ./docs/api --set api_documentation --document-type specification
    
    # Update code documentation  
    qdrant-ingest ./src --set platform_code --document-type source_code
    
    # Update deployment docs
    qdrant-ingest ./deploy --set deployment_configs --document-type configuration
```

### Multi-Environment Configurations

Use different set configurations for different environments:

```bash
# Development
QDRANT_SETS_CONFIG=".qdrant_sets.dev.json" uvx mcp-server-qdrant-rag

# Production  
QDRANT_SETS_CONFIG=".qdrant_sets.prod.json" uvx mcp-server-qdrant-rag
```

### Performance Optimization

For large-scale deployments:

1. **Limit Set Count**: Keep sets focused and avoid too many granular divisions
2. **Optimize Aliases**: Use specific aliases to reduce ambiguous matches
3. **Monitor Usage**: Track which sets are queried most frequently
4. **Cache Configuration**: The system caches set configurations for performance

### Migration Strategies

When updating existing knowledge bases:

```bash
# Gradual migration approach
# 1. Add new sets to configuration
# 2. Re-ingest documents with new metadata
qdrant-ingest update /path/to/docs --set new_set_name --mode add-only

# 3. Verify new structure
qdrant-ingest list

# 4. Remove old unorganized documents if needed
# (Manual cleanup via Qdrant API or complete re-ingestion)
```

This completes the comprehensive configuration guide. The semantic metadata filtering system is now fully documented with practical examples, troubleshooting guidance, and advanced usage patterns.
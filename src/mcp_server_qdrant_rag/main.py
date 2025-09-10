import argparse
import signal
import sys


def main():
    """
    Main entry point for the mcp-server-qdrant-rag script defined
    in pyproject.toml. It runs the MCP server with a specific transport
    protocol.
    """

    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="mcp-server-qdrant-rag")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) and SIGTERM gracefully."""
        print("\nReceived interrupt signal. Shutting down gracefully...", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Import is done here to make sure environment variables are loaded
        # only after we make the changes.
        from mcp_server_qdrant_rag.server import mcp

        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        # Additional fallback for KeyboardInterrupt
        print("\nShutdown complete.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)

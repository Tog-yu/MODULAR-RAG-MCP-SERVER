"""Streamable-HTTP entrypoint for the RAG MCP server.

互补于 server.py 的 stdio 入口。复用同一个低层 MCP ``Server`` 对象，用
``FastMCP`` 包一层拿到 Starlette ASGI app（在 ``/mcp`` 上走 MCP streamable-http），
再用 uvicorn 起服务。这样 stdio 与 http 共用同一套已注册工具，切换只改启动方式。

启动：
    python -m src.mcp_server.server --transport http [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.mcp_server.server import (
    SERVER_NAME,
    SERVER_VERSION,
    _preload_heavy_imports,
    _redirect_all_loggers_to_stderr,
)

if TYPE_CHECKING:
    pass

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


async def run_http_server_async(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Run the RAG MCP server over streamable-http (async). Returns exit code."""
    import uvicorn
    from mcp.server.fastmcp import FastMCP

    # stdout 只留给 JSON-RPC（stdio 约定）；http 下同样把日志赶到 stderr，避免污染
    _redirect_all_loggers_to_stderr()
    # 预加载重依赖，避免工具处理器在 worker 线程里触发 import 锁死锁
    _preload_heavy_imports()

    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    app = FastMCP(server).streamable_http_app()

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    srv = uvicorn.Server(config)
    await srv.serve()
    return 0


def run_http_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    """Run the RAG MCP server over streamable-http (sync wrapper)."""
    return asyncio.run(run_http_server_async(host, port))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG MCP server (streamable-http)")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the streamable-http RAG MCP server."""
    args = parse_args(argv)
    return run_http_server(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())

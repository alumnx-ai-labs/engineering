"""
MCP Server — BODMA & CODMA
Transport: Streamable HTTP only

BODMA(a, b) = (a^b) / (a*b)
CODMA(a, b) = (a*b) / (a^b)

Run: python server.py
"""

from mcp.server.fastmcp import FastMCP
import uvicorn

# ── Create server ─────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="BODMA-CODMA",
    stateless_http=True,  # each request is independent, no session state needed
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def bodma(a: float, b: float) -> float:
    """BODMA: (a^b) / (a*b). Example: bodma(2,3) = 8/6 = 1.333"""
    if a * b == 0:
        raise ValueError("a*b cannot be 0 (division by zero)")
    return (a ** b) / (a * b)


@mcp.tool()
def codma(a: float, b: float) -> float:
    """CODMA: (a*b) / (a^b). Inverse of BODMA. Example: codma(2,3) = 6/8 = 0.75"""
    if a ** b == 0:
        raise ValueError("a^b cannot be 0 (division by zero)")
    return (a * b) / (a ** b)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        mcp.streamable_http_app(),  # single app, single transport
        host="0.0.0.0",
        port=8000,
    )
#!/usr/bin/env python3
"""
MCP Server for Calculator Tools (BODMA and CODMA)
Compatible with latest MCP SDK
"""
import json
import sys
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def calculate_bodma(a: float, b: float) -> dict:
    """BODMA Calculation: (a^b) / (a*b)"""
    try:
        if a == 0 and b == 0:
            return {
                "status": "error",
                "message": "Cannot calculate 0^0 and division by 0",
                "result": None
            }
        
        numerator = a ** b
        denominator = a * b
        
        if denominator == 0:
            return {
                "status": "error",
                "message": "Cannot divide by 0 (a*b = 0)",
                "result": None
            }
        
        result = numerator / denominator
        
        return {
            "status": "success",
            "formula": f"({a}^{b}) / ({a}*{b})",
            "numerator": numerator,
            "denominator": denominator,
            "result": result,
            "message": f"BODMA({a}, {b}) = {result}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Calculation error: {str(e)}",
            "result": None
        }


def calculate_codma(a: float, b: float) -> dict:
    """CODMA Calculation: (a*b) / (a^b)"""
    try:
        if a == 0 and b <= 0:
            return {
                "status": "error",
                "message": "Cannot calculate 0^b for b <= 0",
                "result": None
            }
        
        numerator = a * b
        denominator = a ** b
        
        if denominator == 0:
            return {
                "status": "error",
                "message": "Cannot divide by 0 (a^b = 0)",
                "result": None
            }
        
        result = numerator / denominator
        
        return {
            "status": "success",
            "formula": f"({a}*{b}) / ({a}^{b})",
            "numerator": numerator,
            "denominator": denominator,
            "result": result,
            "message": f"CODMA({a}, {b}) = {result}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Calculation error: {str(e)}",
            "result": None
        }


# ============================================================================
# MCP SERVER
# ============================================================================

server = Server("bodma-codma-mcp-server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    print("📋 Listing available tools...", file=sys.stderr)
    return [
        types.Tool(
            name="bodma_calculate",
            description="Calculate (a^b) / (a*b) - BODMA calculation. Takes two numbers and returns the result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number (base)"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number (exponent)"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        types.Tool(
            name="codma_calculate",
            description="Calculate (a*b) / (a^b) - CODMA calculation. Takes two numbers and returns the result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number (base)"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number (exponent)"
                    }
                },
                "required": ["a", "b"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution"""
    
    print(f"🔧 Tool called: {name} with arguments: {arguments}", file=sys.stderr)
    
    if not arguments:
        arguments = {}
    
    if name == "bodma_calculate":
        a = float(arguments.get("a", 0))
        b = float(arguments.get("b", 0))
        result = calculate_bodma(a, b)
        print(f"✅ BODMA calculation completed: {result['status']}", file=sys.stderr)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    elif name == "codma_calculate":
        a = float(arguments.get("a", 0))
        b = float(arguments.get("b", 0))
        result = calculate_codma(a, b)
        print(f"✅ CODMA calculation completed: {result['status']}", file=sys.stderr)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    else:
        print(f"❌ Unknown tool: {name}", file=sys.stderr)
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main entry point for the server"""
    print("🚀 Starting BODMA-CODMA MCP Server...", file=sys.stderr)
    print("📡 Server name: bodma-codma-mcp-server", file=sys.stderr)
    print("📌 Version: 1.0.0", file=sys.stderr)
    print("⚙️  Available tools: bodma_calculate, codma_calculate", file=sys.stderr)
    
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print("✓ Stdio streams initialized", file=sys.stderr)
        
        init_options = InitializationOptions(
            server_name="bodma-codma-mcp-server",
            server_version="1.0.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            )
        )
        
        print("✓ Server initialized successfully", file=sys.stderr)
        print("👂 Listening for requests...\n", file=sys.stderr)
        
        await server.run(
            read_stream,
            write_stream,
            init_options
        )


if __name__ == "__main__":
    import asyncio
    
    # Ensure output is not buffered
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Run the async main function
    asyncio.run(main())
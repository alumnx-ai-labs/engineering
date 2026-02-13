#!/usr/bin/env python3
"""
MCP Server for Calculator Tools (BODMA and CODMA)
FastAPI Implementation
"""
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def calculate_bodma(a: float, b: float) -> dict:
    """BODMA Calculation: (a^b) / (a*b)"""
    print(f"🧮 BODMA Calculation called with a={a}, b={b}")
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
    print(f"🧮 CODMA Calculation called with a={a}, b={b}")
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
# PYDANTIC MODELS
# ============================================================================

class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="BODMA-CODMA MCP Server")


@app.post("/tools/list")
async def list_tools():
    """List available tools"""
    print("📋 /tools/list endpoint called")
    tools = {
        "tools": [
            {
                "name": "bodma_calculate",
                "description": "Calculate (a^b) / (a*b) - BODMA calculation. Takes two numbers and returns the result.",
                "inputSchema": {
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
            },
            {
                "name": "codma_calculate",
                "description": "Calculate (a*b) / (a^b) - CODMA calculation. Takes two numbers and returns the result.",
                "inputSchema": {
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
            }
        ]
    }
    print(f"✅ Returning {len(tools['tools'])} tools")
    return tools


@app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Execute a tool"""
    print(f"\n{'='*60}")
    print(f"🔧 /tools/call endpoint called")
    print(f"{'='*60}")
    print(f"📝 Tool name: {request.name}")
    print(f"📝 Arguments: {request.arguments}")
    
    tool_name = request.name
    arguments = request.arguments
    
    if tool_name == "bodma_calculate":
        a = float(arguments.get("a", 0))
        b = float(arguments.get("b", 0))
        print(f"➡️  Calling BODMA with a={a}, b={b}")
        result = calculate_bodma(a, b)
        print(f"✅ BODMA result: {result}")
        return {"result": result}
    
    elif tool_name == "codma_calculate":
        a = float(arguments.get("a", 0))
        b = float(arguments.get("b", 0))
        print(f"➡️  Calling CODMA with a={a}, b={b}")
        result = calculate_codma(a, b)
        print(f"✅ CODMA result: {result}")
        return {"result": result}
    
    else:
        print(f"❌ Unknown tool: {tool_name}")
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")


if __name__ == "__main__":
    print("\n🚀 Starting MCP Server on port 8011...")
    uvicorn.run(app, host="0.0.0.0", port=8011)
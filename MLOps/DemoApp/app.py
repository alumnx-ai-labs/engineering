from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import time
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()  # Add this line

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

app = FastAPI()

# Simple in-memory monitoring
metrics = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_cost": 0.0,
    "requests": []
}

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    latency_ms: int
    cost_inr: float
    timestamp: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    try:
        # Generate response
        response = model.generate_content(request.message)
        
        # Calculate metrics
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Estimate tokens (rough approximation)
        input_tokens = len(request.message.split()) * 1.3
        output_tokens = len(response.text.split()) * 1.3
        total_tokens = int(input_tokens + output_tokens)
        
        # Cost calculation (Gemini Pro pricing)
        # Input: $0.000125/1K tokens, Output: $0.000375/1K tokens
        cost_usd = (input_tokens/1000 * 0.000125) + (output_tokens/1000 * 0.000375)
        cost_inr = cost_usd * 83  # USD to INR conversion
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_tokens"] += total_tokens
        metrics["total_cost"] += cost_inr
        
        # Log request
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "input_length": len(request.message),
            "output_length": len(response.text),
            "tokens": total_tokens,
            "latency_ms": latency_ms,
            "cost_inr": round(cost_inr, 4)
        }
        metrics["requests"].append(log_entry)
        
        return ChatResponse(
            response=response.text,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            cost_inr=round(cost_inr, 4),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """MLOps Dashboard - View system metrics"""
    avg_latency = sum(r["latency_ms"] for r in metrics["requests"]) / len(metrics["requests"]) if metrics["requests"] else 0
    
    return {
        "total_requests": metrics["total_requests"],
        "total_tokens": metrics["total_tokens"],
        "total_cost_inr": round(metrics["total_cost"], 2),
        "average_latency_ms": round(avg_latency, 2),
        "recent_requests": metrics["requests"][-10:]  # Last 10 requests
    }

@app.get("/")
async def root():
    return {"message": "LLM ChatBot with MLOps Monitoring", "endpoints": ["/chat", "/metrics"]}
# Minimal Working MCP Server - Fix 500 Error
# Built for Puch AI Hackathon 2025

import json
import logging
import os
from typing import Dict, List, Any
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory storage
meetings_db = {}
action_items_db = {}

def summarize_meeting_simple(meeting_transcript: str, summary_type: str = "brief") -> Dict[str, Any]:
    """Simple meeting summarization"""
    try:
        sentences = meeting_transcript.split('.')
        summary = ". ".join(sentences[:3]) + "."
        
        return {
            "meeting_id": f"meeting_{datetime.now().timestamp()}",
            "summary": summary,
            "key_points": ["Meeting discussion captured", "Action items identified"],
            "participants": ["Team Member"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

def extract_action_items_simple(transcript: str) -> List[Dict[str, str]]:
    """Simple action item extraction"""
    try:
        # Simple pattern matching
        if "needs to" in transcript.lower() or "will" in transcript.lower():
            return [
                {
                    "task": "Complete identified action items",
                    "assignee": "Team Member",
                    "due_date": "2025-08-17",
                    "priority": "medium",
                    "status": "pending"
                }
            ]
        return []
    except Exception as e:
        return [{"error": str(e)}]

# Create FastAPI app
app = FastAPI(
    title="AI Collaboration Facilitator MCP Server",
    description="Minimal working MCP server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_handler(request: Request):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/")
async def root():
    """Root endpoint for MCP server info"""
    try:
        return {
            "jsonrpc": "2.0",
            "result": {
                "name": "AI-Driven Real-Time Collaboration Facilitator",
                "version": "1.0.0",
                "description": "MCP server with AI collaboration tools",
                "protocol_version": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "server_info": {
                    "name": "ai-collaboration-facilitator",
                    "version": "1.0.0"
                }
            }
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )

@app.post("/")
async def mcp_handler(request: Request):
    """Handle MCP protocol requests"""
    try:
        body = await request.json()
        method = body.get("method", "")
        request_id = body.get("id", 1)
        
        logger.info(f"MCP request: {method}")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "ai-collaboration-facilitator",
                        "version": "1.0.0"
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "summarize_meeting",
                            "description": "Summarize meeting conversations with AI-powered analysis",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "meeting_transcript": {"type": "string"},
                                    "summary_type": {"type": "string", "default": "brief"}
                                },
                                "required": ["meeting_transcript"]
                            }
                        },
                        {
                            "name": "extract_action_items",
                            "description": "Extract action items from meeting transcripts",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "transcript": {"type": "string"}
                                },
                                "required": ["transcript"]
                            }
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            params = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            logger.info(f"Tool call: {tool_name} with args: {arguments}")
            
            if tool_name == "summarize_meeting":
                result = summarize_meeting_simple(
                    arguments.get("meeting_transcript", ""),
                    arguments.get("summary_type", "brief")
                )
            elif tool_name == "extract_action_items":
                result = extract_action_items_simple(
                    arguments.get("transcript", "")
                )
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {tool_name}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            
    except Exception as e:
        logger.error(f"MCP handler error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": body.get("id", 1) if 'body' in locals() else 1,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mcp_protocol": "2024-11-05",
            "tools_available": 2
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Health check failed: {str(e)}"}
        )

if __name__ == "__main__":
    try:
        print("🚀 Starting Minimal MCP Server")
        print("🔧 Protocol: JSON-RPC 2.0 / MCP 2024-11-05")
        print("📝 Built for Puch AI Hackathon 2025")
        
        port = int(os.environ.get("PORT", 8000))
        print(f"🌐 Starting on port {port}")
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        raise

#!/usr/bin/env python3
"""
MCP Health Check Tool
=====================
Check the health of MCP/Brain server and test tool execution.

Usage:
    python mcp_health_check.py [server_url]
    python mcp_health_check.py --test-tool "greet"
    python mcp_health_check.py --health-only
"""

import sys
import time
from pathlib import Path

# Try to import MCPClient (will be available when files are in path)
try:
    from mcp_client import MCPClient
    HAS_MCP_CLIENT = True
except ImportError:
    print("⚠️  MPCClient not found. Install mcp_client.py in your workspace.")
    sys.exit(1)


def check_mcp_health(server_url=None):
    """Check if MCP server is reachable and responding."""
    client = MCPClient(server_url=server_url or "http://localhost:3001/v1")
    
    print(f"🔍 Checking MCP health at {client.server_url}...")
    
    start_time = time.time()
    try:
        if client.connect():
            elapsed = time.time() - start_time
            print(f"✅ Connected in {elapsed:.3f}s")
            
            # Try to get server info if available
            try:
                health = client.health()
                print(f"  Server Status: {health}")
            except Exception as e:
                print(f"  Health check (optional): {e}")
                
            return True, elapsed
        else:
            print("❌ Failed to connect to MCP server")
            return False, None
            
    except Exception as e:
        print(f"❌ Error connecting: {e}")
        return False, None


def test_tool(tool_name=None):
    """Test execution of a specific tool."""
    client = MCPClient(server_url="http://localhost:3001/v1")
    
    if not client.connect():
        print("❌ Cannot connect to MCP server. Check server is running.")
        return
    
    print(f"✅ Connected to MCP server")
    
    # List available tools
    try:
        tools = client.list_tools()
        print(f"\n📋 Available tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool}")
        
        if not tools:
            print("⚠️  No tools registered on server. Register some or use default.")
    except Exception as e:
        print(f"⚠️  Could not list tools: {e}")
    
    # Test specific tool if provided
    if tool_name:
        print(f"\n🧪 Testing tool: {tool_name}")
        try:
            result = client.execute(tool_name)
            print(f"✅ Tool executed successfully")
            print(f"  Result: {result}")
        except Exception as e:
            print(f"❌ Tool execution failed: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Health Check & Tool Tester")
    parser.add_argument("server_url", nargs="?", default="http://localhost:3001/v1")
    parser.add_argument("--test-tool", action="store_true", 
                        help="Test execution of a specific tool")
    parser.add_argument("--health-only", action="store_true",
                        help="Only check health, don't list tools")
    
    args = parser.parse_args()
    
    # Check health
    healthy, time_taken = check_mcp_health(args.server_url)
    
    if not healthy:
        sys.exit(1)
    
    print(f"\n🎉 MCP Health Check Passed in {time_taken:.3f}s")


if __name__ == "__main__":
    main()

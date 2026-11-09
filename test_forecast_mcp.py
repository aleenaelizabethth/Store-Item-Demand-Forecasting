import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["forecast_server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
            out = await s.call_tool("forecast_sales", {"store": 1, "item": 1, "weeks": 2})
            print(out.content[0].text[:600])
            bad = await s.call_tool("forecast_sales", {"store": 99, "item": 1, "weeks": 2})
            print(bad.content[0].text)

asyncio.run(main())

import httpx
import asyncio

async def f():
    async with httpx.AsyncClient() as client:
        ret = await client.get('https://www.baidu.com')
    print(ret.text)

asyncio.run(f())


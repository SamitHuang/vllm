#!/usr/bin/env python3
"""Send batch requests to vLLM server"""
import asyncio
import sys
import aiohttp

SERVER_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

async def send_request(session, prompt, req_id):
    async with session.post(
        f"{SERVER_URL}/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 100}
    ) as resp:
        result = await resp.json()
        print(f"[{req_id}] {result['choices'][0]['text'][:50]}...")

async def main():
    num_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    prompt = "Write a story about artificial intelligence:"
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, prompt, f"Req{i+1}") for i in range(num_prompts)]
        await asyncio.gather(*tasks)
    
    print(f"\n✓ Sent {num_prompts} concurrent requests")

if __name__ == "__main__":
    asyncio.run(main())

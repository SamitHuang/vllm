import time
import asyncio
from openai import AsyncOpenAI  # 官方OpenAI兼容客户端，pip install openai

# 配置服务地址和API Key
SERVICE_URL = "http://localhost:8000/v1"  # vLLM Serve 地址
API_KEY = "EMPTY"  # 如果无API Key，则可删除相关参数
MODEL_NAME = "/home/mindone/yx/models/Qwen/Qwen2.5-0.5B-Instruct"

# 初始化OpenAI兼容客户端
client = AsyncOpenAI(base_url=SERVICE_URL, api_key=API_KEY)

async def test_batch_inference(batch_size):
    prompts = [{"role": "user", "content": "Hello, this is a vLLM online batch test."}] * batch_size
    start_time = time.time()
    # vLLM兼容OpenAI Chat Completions接口，batch通过并发调用模拟，比如循环或异步并发
    # 这里简单用同步调用的for循环示例，真实场景推荐用异步
    # 为演示，用单次调用批处理只发一个请求，实际可根据服务支持扩展
    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompts
    )
    elapsed = time.time() - start_time
    return elapsed, completion

def main():
    batch_sizes = [1, 4]
    loop = asyncio.get_event_loop()
    for bs in batch_sizes:
        elapsed, completion = loop.run_until_complete(test_batch_inference(bs))
        print(f"Batch size: {bs}, inference time: {elapsed:.4f} seconds")
        print("Sample response:", completion.choices[0].message.content)

if __name__ == "__main__":
    main()

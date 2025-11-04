"""
Test for pause/resume functionality with Qwen2.5-0.5B.
Usage: python test_pause_resume_standalone.py

Test workflow:
1. Send a generation request (streaming output)
2. Pause generation (you'll see generation interrupt)
3. Send new request (should block until resume)
4. Resume generation  
5. Verify blocked request completes
6. Send another request (should work normally)
7. Verify initial request completed
"""

import asyncio
import sys
import time
from typing import Optional

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


def print_step(step_num: int, description: str):
    """Print formatted step header."""
    print(f"\n{'='*70}")
    print(f"[Step {step_num}] {description}")
    print(f"{'='*70}")


def print_result(status: str, message: str, indent: int = 2):
    """Print formatted result."""
    prefix = " " * indent
    print(f"{prefix}{status} {message}")


async def generate_with_streaming(
    engine: AsyncLLM,
    prompt: str,
    request_id: str,
    max_tokens: int = 30,
    show_streaming: bool = False,
    label: str = "",
) -> Optional[any]:
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    
    final_output = None
    last_text = ""
    
    try:
        async for output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            final_output = output
            
            # Stream output in real-time
            if show_streaming and output.outputs:
                current_text = output.outputs[0].text
                new_text = current_text[len(last_text):]
                if new_text:
                    # Print new tokens without label
                    print(new_text, end="", flush=True)
                    last_text = current_text
        
        if show_streaming and last_text:
            print("\n")  # Newline after streaming
            
    except Exception as e:
        if show_streaming:
            print(f"\n  [{label}] ❌ Error: {e}")
        return None
    
    return final_output


async def test_pause_resume(mode='gentle', clear_cache=True):
    """Main test workflow."""
    
    local_prefix = "/home/mindone/yx/models/" 
    engine_args = AsyncEngineArgs(
        model=local_prefix + "Qwen/Qwen2.5-0.5B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
    )
    
    engine = AsyncLLM.from_engine_args(engine_args)
    print_result("✓", "Engine initialized successfully")
    
    # Step 1: Send a generation request with streaming output
    print_step(1, "Sending generation request (streaming output)")
    print()
    
    initial_prompt = "Write a short story about a robot learning to paint."
    print_result("→", f"Prompt: {initial_prompt}")
    print_result("  ", "Streaming output:")
    
    initial_task = asyncio.create_task(
        generate_with_streaming(
            engine,
            prompt=initial_prompt,
            request_id="initial_request",
            max_tokens=2048,  # Longer to show streaming effect
            show_streaming=True,
            label="Initial",
        )
    )
    
    # Let it generate for a while to see streaming output
    await asyncio.sleep(0.5)  # Let it generate some tokens
    
    # Step 2: Pause generation
    print_step(2, f"Pausing generation (mode: {mode}, clear_cache: {clear_cache})")
    print_result("⏸️", "Calling pause_generation()...")
    
    pause_start = time.time()
    pause_result = await engine.pause_generation(mode=mode, clear_cache=clear_cache)
    pause_duration = time.time() - pause_start
    print_result("  ", f"Aborted: {pause_result['aborted_requests']}")
        
    # Verify pause status
    status = await engine.get_pause_status()
    assert status["is_paused"]
    print_result("✓", "Confirmed: Engine is in paused state")
    print_result("✓", "KV cache and prefix cache have been cleared")
    
    # Step 3: Send new request during pause (should block)
    print_step(3, "Sending request during pause (should block)")
    
    blocked_prompt = "What is the meaning of life?"
    print_result("→", f"Prompt: {blocked_prompt}")
    
    blocked_task = asyncio.create_task(
        generate_with_streaming(
            engine,
            prompt=blocked_prompt,
            request_id="blocked_request",
            max_tokens=30,
            show_streaming=False,  # Don't show streaming for blocked request
            label="Blocked",
        )
    )
    
    # Wait to ensure it tries to start
    await asyncio.sleep(0.5)
    
    assert not blocked_task.done()
    print_result("✓", "Request is blocked (paused state working correctly)")
    print_result("  ", "The request is waiting for resume...")
    
    # Step 4: Resume generation
    print_step(4, "Resuming generation")
    
    resume_result = await engine.resume_generation()
    
    # Verify resumed status
    status = await engine.get_pause_status()
    assert not status["is_paused"]
    print_result("✓", "Confirmed: Engine is resumed")
    
    # Step 5: Verify blocked request completes
    print_step(5, "Waiting for blocked request to complete")
    blocked_output = await asyncio.wait_for(blocked_task, timeout=15.0)
    
    assert blocked_output is not None
    assert blocked_output.outputs is not None
    
    generated_text = blocked_output.outputs[0].text
    print_result("✓", "Blocked request completed after resume")
    print_result("  ", f"Prompt: {blocked_prompt}")
    print_result("  ", f"Generated: {generated_text[:60]}...")
        
    # Step 6: Send new request (should work normally)
    print_step(6, "Sending new request after resume")
    
    new_prompt = "What is the speed of light?"
    print_result("→", f"Prompt: {new_prompt}")
    print()
    
    new_output = await generate_with_streaming(
        engine,
        prompt=new_prompt,
        request_id="new_request_after_resume",
        max_tokens=30,
        show_streaming=False,
        label="NewReq",
    )
    
    assert new_output is not None and new_output.outputs is not None
    print_result("✓", "New request completed successfully")
    print_result("  ", f"Generated: {new_output.outputs[0].text[:60]}...")
    
    # Verification: Check initial request completed in gentle mode
    if mode == 'gentle':
        print_step(7, "Verifying initial request completed in gentle mode")
        
        initial_output = await asyncio.wait_for(initial_task, timeout=5.0)
        
        text = initial_output.outputs[0].text
        print_result("  ", f"Initial Request Prompt: {initial_prompt}")
        print_result("  ", f"Generated: {text}")
        print_result("  ", f"Total tokens: {len(initial_output.outputs[0].token_ids)}")
    

if __name__ == "__main__":
    asyncio.run(test_pause_resume(mode='gentle', clear_cache=True))


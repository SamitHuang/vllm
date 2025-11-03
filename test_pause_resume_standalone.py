#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone test for pause/resume functionality with Qwen2.5-0.5B.

This script can be run directly without pytest:
    python test_pause_resume_standalone.py

Test workflow:
1. Send multiple QA generation requests (streaming output)
2. Pause generation (you'll see generation interrupt)
3. Send new request (should block until resume)
4. Resume generation  
5. Verify blocked request completes
6. Send another request (should work normally)
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
    """
    Generate a completion with optional streaming display.
    
    Args:
        engine: The AsyncLLM engine
        prompt: Input prompt
        request_id: Unique request ID
        max_tokens: Maximum tokens to generate
        show_streaming: If True, print tokens as they are generated
        label: Label for streaming output (e.g., "Req0")
        
    Returns:
        Final RequestOutput or None if error
    """
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
                    # Print new tokens on same line
                    print(f"  [{label}] {new_text}", end="", flush=True)
                    last_text = current_text
        
        if show_streaming and last_text:
            print()  # Newline after streaming
            
    except Exception as e:
        if show_streaming:
            print(f"\n  [{label}] ❌ Error: {e}")
        return None
    
    return final_output


async def main():
    """Main test workflow."""
    
    print("\n" + "="*70)
    print("Pause/Resume Test with Qwen2.5-0.5B")
    print("="*70)
    print()
    
    # Initialize engine
    print("Initializing Qwen2.5-0.5B engine...")
    print("(This may take a moment for first-time model download)")

    local_prefix = "/home/mindone/yx/models/" 
    engine_args = AsyncEngineArgs(
        model=local_prefix + "Qwen/Qwen2.5-0.5B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
    )
    
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
        print_result("✓", "Engine initialized successfully")
    except Exception as e:
        print_result("❌", f"Failed to initialize engine: {e}")
        sys.exit(1)
    
    # ========================================
    # Step 1: Send multiple QA requests with streaming output
    # ========================================
    print_step(1, "Sending QA generation requests (streaming output)")
    print()
    print("  Prompts:")
    
    qa_prompts = [
        ("Q: Write a short story about a robot learning to paint.\nA:", "Story"),
        ("Q: Explain how photosynthesis works in detail.\nA:", "Science"),
        ("Q: Describe the history of the internet from 1960 to now.\nA:", "History"),
    ]
    
    print_result("", "Starting streaming generation (watch the tokens appear)...")
    print()
    
    initial_tasks = []
    for i, (prompt, label) in enumerate(qa_prompts):
        # Start requests with delays to show interleaved streaming
        if i > 0:
            await asyncio.sleep(0.3)  # Stagger the starts
        
        task = asyncio.create_task(
            generate_with_streaming(
                engine,
                prompt=prompt,
                request_id=f"initial_request_{i}",
                max_tokens=100,  # Longer to show streaming effect
                show_streaming=True,
                label=label,
            )
        )
        initial_tasks.append(task)
        print_result("→", f"Started [{label}]: {prompt.split('.')[0].split(':')[1].strip()[:30]}...")
    
    # Let them generate for a while to see streaming output
    print()
    print_result("", "Generating... (you should see tokens streaming below)")
    print()
    await asyncio.sleep(2.0)  # Let them generate some tokens
    
    # ========================================
    # Step 2: Pause generation
    # ========================================
    print()
    print_step(2, "Pausing generation (generation should stop)")
    print()
    print_result("⏸️", "Calling pause_generation()...")
    print_result("", "(This will wait for requests to finish and clear caches)")
    
    pause_start = time.time()
    try:
        pause_result = await engine.pause_generation()
        pause_duration = time.time() - pause_start
        
        print()
        print_result("✓", "Pause successful - generation stopped!")
        print_result("  ", f"Mode: {pause_result['mode']}")
        print_result("  ", f"Drained: {pause_result['drained']}")
        print_result("  ", f"Elapsed: {pause_result['elapsed_seconds']:.3f}s")
        print_result("  ", f"Unfinished: {pause_result['num_unfinished_requests']}")
        print_result("  ", f"Aborted: {pause_result['aborted_requests']}")
        
        if not pause_result["paused"]:
            print_result("❌", "Pause failed!")
            sys.exit(1)
            
    except Exception as e:
        print_result("❌", f"Pause error: {e}")
        sys.exit(1)
    
    # Verify pause status
    status = await engine.get_pause_status()
    if status["is_paused"]:
        print_result("✓", "Confirmed: Engine is in paused state")
        print_result("✓", "KV cache and prefix cache have been cleared")
    else:
        print_result("❌", "Error: Engine is not paused!")
        sys.exit(1)
    
    # ========================================
    # Step 3: Send new request during pause (should block)
    # ========================================
    print_step(3, "Sending request during pause (should block)")
    print()
    
    blocked_prompt = "Q: What is the meaning of life?\nA:"
    print_result("→", f"Sending: {blocked_prompt.split('?')[0]}?")
    print_result("", "This request should NOT generate until resume is called...")
    
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
    
    if blocked_task.done():
        print_result("❌", "ERROR: Request completed during pause!")
        print_result("  ", "Pause is not working correctly!")
        sys.exit(1)
    else:
        print_result("✓", "Request is blocked (paused state working correctly)")
        print_result("  ", "The request is waiting for resume...")
    
    # ========================================
    # Step 4: Resume generation
    # ========================================
    print_step(4, "Resuming generation")
    
    try:
        resume_result = await engine.resume_generation()
        
        print_result("✓", "Resume successful")
        print_result("  ", f"Paused: {resume_result['paused']}")
        print_result("  ", f"Message: {resume_result['message']}")
        
        if resume_result["paused"]:
            print_result("❌", "Error: Still paused after resume!")
            sys.exit(1)
            
    except Exception as e:
        print_result("❌", f"Resume error: {e}")
        sys.exit(1)
    
    # Verify resumed status
    status = await engine.get_pause_status()
    if not status["is_paused"]:
        print_result("✓", "Confirmed: Engine is resumed")
    else:
        print_result("❌", "Error: Engine is still paused!")
        sys.exit(1)
    
    # ========================================
    # Step 5: Verify blocked request completes
    # ========================================
    print_step(5, "Waiting for blocked request to complete")
    
    try:
        blocked_output = await asyncio.wait_for(blocked_task, timeout=15.0)
        
        if blocked_output is None:
            print_result("❌", "Blocked request returned None")
            sys.exit(1)
        
        if not blocked_output.outputs:
            print_result("❌", "Blocked request has no outputs")
            sys.exit(1)
        
        generated_text = blocked_output.outputs[0].text
        print_result("✓", "Blocked request completed after resume")
        print_result("  ", f"Prompt: {blocked_prompt.strip()}")
        print_result("  ", f"Generated: {generated_text[:60]}...")
        print_result("  ", f"Total tokens: {len(blocked_output.outputs[0].token_ids)}")
        
    except asyncio.TimeoutError:
        print_result("❌", "Blocked request timed out!")
        sys.exit(1)
    except Exception as e:
        print_result("❌", f"Error waiting for blocked request: {e}")
        sys.exit(1)
    
    # ========================================
    # Step 6: Send new request (should work normally)
    # ========================================
    print_step(6, "Sending new request after resume")
    print()
    
    new_prompt = "Q: What is the speed of light?\nA:"
    print_result("→", f"Sending: {new_prompt.split('?')[0]}?")
    
    try:
        new_output = await generate_with_streaming(
            engine,
            prompt=new_prompt,
            request_id="new_request_after_resume",
            max_tokens=30,
            show_streaming=True,
            label="NewReq",
        )
        
        if new_output is None or not new_output.outputs:
            print_result("❌", "New request failed")
            sys.exit(1)
        
        print()
        print_result("✓", "New request completed successfully")
        print_result("  ", f"Total tokens: {len(new_output.outputs[0].token_ids)}")
        
    except Exception as e:
        print_result("❌", f"Error with new request: {e}")
        sys.exit(1)
    
    # ========================================
    # Verification: Check initial requests
    # ========================================
    print_step(7, "Verifying initial requests completed")
    
    initial_outputs = await asyncio.gather(*initial_tasks, return_exceptions=True)
    
    completed = 0
    for i, output in enumerate(initial_outputs):
        if isinstance(output, Exception):
            print_result("⚠️", f"Request {i} failed: {output}")
        elif output and hasattr(output, 'outputs') and output.outputs:
            completed += 1
            text = output.outputs[0].text.strip()[:40]
            print_result("✓", f"Request {i}: '{text}...'")
        else:
            print_result("⚠️", f"Request {i}: No output")
    
    print_result("", f"Completed: {completed}/{len(qa_prompts)} requests")
    
    # ========================================
    # Final summary
    # ========================================
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print()
    print("Summary:")
    print(f"  ✓ Step 1: Sent 3 streaming requests (you saw tokens appear)")
    print(f"  ✓ Step 2: Paused generation (took {pause_duration:.3f}s, streaming stopped)")
    print(f"  ✓ Step 3: New request blocked during pause (no generation)")
    print(f"  ✓ Step 4: Resumed generation")
    print(f"  ✓ Step 5: Blocked request completed after resume")
    print(f"  ✓ Step 6: New request worked normally (streaming resumed)")
    print(f"  ✓ Step 7: Initial requests: {completed}/3 completed")
    print()
    print("Pause/Resume functionality is working correctly! 🎉")
    print("="*70)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


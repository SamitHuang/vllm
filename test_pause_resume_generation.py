#!/usr/bin/env python3
"""
Test for pause/resume functionality with Qwen2.5-0.5B or OLMoE-1B-7B

Usage:
    # Test with Qwen2.5 (no parallelism)
    python test_pause_resume_generation.py

    # Test with OLMoE (DP+EP)
    Edit the main() call at the bottom of the file

Test workflow:
1. Send a generation request (streaming output)
2. Pause generation (in-flight generation continues in gentle mode, or aborted in force mode)
3. Send new request (should block until resume)
4. Resume generation
5. Verify blocked request completes
6. Send another request (should work normally)
"""

import asyncio
import time

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


def print_step(step_num: int, description: str):
    """Print formatted step header."""
    print(f"\n{'=' * 70}")
    print(f"[Step {step_num}] {description}")
    print(f"{'=' * 70}")


async def generate_with_streaming(
    engine: AsyncLLM,
    prompt: str,
    request_id: str,
    max_tokens: int = 128,
    show_streaming: bool = False,
    label: str = "",
) -> any | None:
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
                new_text = current_text[len(last_text) :]
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


async def test_pause_resume(
    mode="force",
    clear_cache=True,
    model_name="qwen",
    dp_size=1,
    tp_size=1,
    enable_ep=False,
    model_prefix="/home/mindone/yx/models/",
):
    """Main test workflow.

    Args:
        mode: Pause mode ('gentle' or 'force')
        clear_cache: Whether to clear KV cache during pause
        model_name: Model to test ('qwen' or 'olmoe')
        dp_size: Data parallel size (for DP+EP)
        tp_size: Tensor parallel size (for DP+EP)
        enable_ep: Enable expert parallel (for MoE models)
                   When enabled, EP size = TP size × DP size
                   For OLMoE (8 experts): EP size should be ≤ 8

    Note:
        EP size = TP size × DP size determines how many devices are used
        for expert parallelism. For OLMoE-1B-7B (8 experts):
        - EP size = 4: Each GPU handles 2 experts
        - EP size = 8: Each GPU handles 1 expert
        - EP size > 8: Some GPUs will be idle (not recommended)
    """

    # Model configuration
    if model_name == "qwen":
        model_path = "Qwen/Qwen2.5-0.5B-Instruct"
        gpu_util = 0.4
    elif model_name == "olmoe":
        model_path = "allenai/OLMoE-1B-7B-0924"
        gpu_util = 0.7
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Print configuration
    print(f"Pause mode:     {mode}")
    print(f"Clear cache:    {clear_cache}")
    print(f"Data parallel:  {dp_size}")
    print(f"Tensor parallel:{tp_size}")
    if enable_ep:
        ep_size = dp_size * tp_size
        print(
            f"Expert parallel:Enabled (EP size = TP×DP = {tp_size}×{dp_size} = {ep_size})"
        )

    # Build engine arguments
    engine_args_dict = {
        "model": model_prefix + model_path,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_util,
        "max_model_len": 2048,
    }

    # Add parallelism configuration if requested
    if enable_ep:
        engine_args_dict.update(
            {
                "tensor_parallel_size": tp_size,
                "data_parallel_size": dp_size,
                "enable_expert_parallel": True,
            }
        )

    engine_args = AsyncEngineArgs(**engine_args_dict)
    engine = AsyncLLM.from_engine_args(engine_args)
    print("✓  Engine initialized successfully")

    # Step 1: Send a generation request with streaming output
    print_step(1, "Sending generation request (streaming output)")
    initial_prompt = "Write a short story about a robot learning to paint."
    print("→  Prompt:", initial_prompt)
    print("   Streaming output:")

    initial_task = asyncio.create_task(
        generate_with_streaming(
            engine,
            prompt=initial_prompt,
            request_id="initial_request",
            max_tokens=512,  # Longer to show streaming effect
            show_streaming=True,
            label="Initial",
        )
    )

    # Let it generate for a while to see streaming output
    await asyncio.sleep(2)  # Let it generate some tokens

    # Step 2: Pause generation
    print_step(2, f"Pausing generation (mode: {mode}, clear_cache: {clear_cache})")

    pause_start = time.time()
    pause_result = await engine.pause_generation(mode=mode, clear_cache=clear_cache)
    pause_duration = time.time() - pause_start
    print("   Pause time cost:", f"{pause_duration:.4f}s")
    print("   Aborted requests:", pause_result["aborted_requests"])
    print("   Cache cleared:", pause_result["cache_cleared"])

    # Verify pause status
    status = await engine.get_pause_status()
    assert status["is_paused"]
    print("✓  Confirmed: Engine is in paused state")
    if clear_cache:
        print("✓  KV cache and prefix cache have been cleared")

    # Step 3: Send new request during pause (should block)
    print_step(3, "Sending request during pause (should block)")

    blocked_prompt = "What is the meaning of life?"
    print("→  Prompt:", blocked_prompt)

    blocked_task = asyncio.create_task(
        generate_with_streaming(
            engine,
            prompt=blocked_prompt,
            request_id="blocked_request",
            max_tokens=128,
            show_streaming=False,  # Don't show streaming for blocked request
            label="Blocked",
        )
    )

    # Wait to ensure it tries to start
    await asyncio.sleep(0.5)

    assert not blocked_task.done()
    print("✓  Request is blocked (paused state working correctly)")
    print("   The request is waiting for resume...")

    # Step 4: Resume generation
    print_step(4, "Resuming generation")

    resume_result = await engine.resume_generation()

    # Verify resumed status
    status = await engine.get_pause_status()
    assert not status["is_paused"]
    print("✓  Confirmed: Engine is resumed")

    # Step 5: Verify blocked request completes
    print_step(5, "Waiting for blocked request to complete")
    blocked_output = await asyncio.wait_for(blocked_task, timeout=15.0)

    assert blocked_output is not None
    assert blocked_output.outputs is not None

    generated_text = blocked_output.outputs[0].text
    print("✓  Blocked request completed after resume")
    print("   Prompt:", blocked_prompt)
    print("   Generated:", generated_text[:60] + "...")

    # Step 6: Send new request (should work normally)
    print_step(6, "Sending new request after resume")

    new_prompt = "What is the speed of light?"
    print("→  Prompt:", new_prompt)

    new_output = await generate_with_streaming(
        engine,
        prompt=new_prompt,
        request_id="new_request_after_resume",
        max_tokens=128,
        show_streaming=False,
        label="NewReq",
    )

    assert new_output is not None and new_output.outputs is not None
    print("   Generated:", new_output.outputs[0].text[:60] + "...")
    print("✓  New request completed successfully")


if __name__ == "__main__":
    # ========================================================================
    # Configuration Examples
    # ========================================================================

    # Example 1: Test with Qwen2.5-0.5B (simple, no parallelism)
    asyncio.run(
        test_pause_resume(
            mode="force",
            clear_cache=True,
            model_name="qwen",
        )
    )

    # Example 2: Test with OLMoE (DP+EP enabled, 4 GPUs)
    # OLMoE has 8 experts, EP size = 2×2 = 4, so each GPU handles 2 experts
    # Uncomment to test with OLMoE
    # asyncio.run(test_pause_resume(
    #     mode='force',
    #     clear_cache=True,
    #     model_name="olmoe",
    #     dp_size=2,          # Data parallel
    #     tp_size=2,          # Tensor parallel
    #     enable_ep=True,     # EP size = 2×2 = 4
    # ))

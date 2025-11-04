# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end test for pause/resume with Qwen2.5-0.5B model.

This test validates the complete pause/resume workflow:
1. Send multiple QA generation requests
2. Pause generation
3. Verify new requests block during pause
4. Resume generation
5. Verify new requests work after resume
"""

import asyncio
import os
import time

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

# Skip if V1 not enabled
pytestmark = pytest.mark.skipif(
    os.getenv("VLLM_USE_V1") != "1",
    reason="Pause/resume only supported in V1 engine"
)

# Test prompts
QA_PROMPTS = [
    "Q: What is the capital of France?\nA:",
    "Q: What is 2 + 2?\nA:",
    "Q: Who wrote Romeo and Juliet?\nA:",
]


@pytest.fixture(scope="module")
async def qwen_engine():
    """Create Qwen2.5-0.5B engine for testing."""
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    yield engine
    # Cleanup
    del engine


@pytest.mark.asyncio
async def test_pause_resume_workflow_qwen(qwen_engine: AsyncLLM):
    """
    Complete pause/resume workflow test with Qwen2.5-0.5B.
    
    Steps:
    1. Send multiple QA generation requests
    2. Pause generation
    3. Send new request during pause (should block)
    4. Resume generation
    5. Verify blocked request completes
    6. Send another request (should work normally)
    """
    
    print("\n" + "="*70)
    print("Test: Pause/Resume Workflow with Qwen2.5-0.5B")
    print("="*70)
    
    # ========================================
    # Step 1: Send multiple QA requests
    # ========================================
    print("\n[Step 1] Sending initial QA generation requests...")
    
    initial_tasks = []
    for i, prompt in enumerate(QA_PROMPTS):
        task = asyncio.create_task(
            generate_and_collect(
                qwen_engine,
                prompt=prompt,
                request_id=f"initial_request_{i}",
                max_tokens=30,
            )
        )
        initial_tasks.append(task)
        print(f"  ✓ Started request {i}: {prompt.split('?')[0]}?")
    
    # Let them start generating
    await asyncio.sleep(0.3)
    
    # ========================================
    # Step 2: Pause generation
    # ========================================
    print("\n[Step 2] Pausing generation...")
    
    pause_start = time.time()
    pause_result = await qwen_engine.pause_generation()
    pause_duration = time.time() - pause_start
    
    print(f"  ✓ Pause successful")
    print(f"    - Mode: {pause_result['mode']}")
    print(f"    - Elapsed: {pause_result['elapsed_seconds']:.2f}s")
    print(f"    - Unfinished: {pause_result['num_unfinished_requests']}")
    print(f"    - Aborted: {pause_result['aborted_requests']}")
    print(f"    - Cache cleared: {pause_result['cache_cleared']}")
    
    assert pause_result["paused"] is True
    assert pause_result["mode"] == "gentle"
    
    # Verify pause status
    status = await qwen_engine.get_pause_status()
    assert status["is_paused"] is True
    print(f"  ✓ Confirmed engine is paused")
    
    # ========================================
    # Step 3: Send new request during pause (should block)
    # ========================================
    print("\n[Step 3] Sending request during pause (should block)...")
    
    blocked_request_prompt = "Q: What is the meaning of life?\nA:"
    blocked_task = asyncio.create_task(
        generate_and_collect(
            qwen_engine,
            prompt=blocked_request_prompt,
            request_id="blocked_request",
            max_tokens=30,
        )
    )
    
    # Wait to ensure it tries to start
    await asyncio.sleep(0.5)
    
    # Request should still be blocked (not completed)
    assert not blocked_task.done(), "Request should be blocked during pause!"
    print(f"  ✓ Request is blocked (paused state working)")
    
    # ========================================
    # Step 4: Resume generation
    # ========================================
    print("\n[Step 4] Resuming generation...")
    
    resume_result = await qwen_engine.resume_generation()
    print(f"  ✓ Resume successful")
    print(f"    - Paused: {resume_result['paused']}")
    print(f"    - Message: {resume_result['message']}")
    
    assert resume_result["paused"] is False
    
    # Verify not paused
    status = await qwen_engine.get_pause_status()
    assert status["is_paused"] is False
    print(f"  ✓ Confirmed engine is resumed")
    
    # ========================================
    # Step 5: Wait for blocked request to complete
    # ========================================
    print("\n[Step 5] Waiting for blocked request to complete...")
    
    blocked_output = await asyncio.wait_for(blocked_task, timeout=15.0)
    
    assert blocked_output is not None
    assert len(blocked_output.outputs) > 0
    
    generated_text = blocked_output.outputs[0].text
    print(f"  ✓ Blocked request completed after resume")
    print(f"    Prompt: {blocked_request_prompt}")
    print(f"    Generated: {generated_text[:80]}...")
    
    # ========================================
    # Step 6: Send new request (should work normally)
    # ========================================
    print("\n[Step 6] Sending new request after resume...")
    
    new_request_prompt = "Q: What is the speed of light?\nA:"
    new_output = await generate_and_collect(
        qwen_engine,
        prompt=new_request_prompt,
        request_id="new_request_after_resume",
        max_tokens=30,
    )
    
    assert new_output is not None
    assert len(new_output.outputs) > 0
    
    new_text = new_output.outputs[0].text
    print(f"  ✓ New request completed successfully")
    print(f"    Prompt: {new_request_prompt}")
    print(f"    Generated: {new_text[:80]}...")
    
    # ========================================
    # Verify initial requests completed
    # ========================================
    print("\n[Verification] Checking initial requests...")
    
    initial_outputs = await asyncio.gather(*initial_tasks, return_exceptions=True)
    
    completed = sum(1 for out in initial_outputs if out is not None and hasattr(out, 'outputs'))
    print(f"  ✓ Initial requests: {completed}/{len(initial_tasks)} completed")
    
    for i, output in enumerate(initial_outputs):
        if output and hasattr(output, 'outputs') and output.outputs:
            text = output.outputs[0].text[:50]
            print(f"    Request {i}: {text}...")
    
    print("\n" + "="*70)
    print("✅ All steps completed successfully!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Pause duration: {pause_duration:.2f}s")
    print(f"  - Requests blocked during pause: ✓")
    print(f"  - Requests resumed after resume: ✓")
    print(f"  - New requests work after resume: ✓")


@pytest.mark.asyncio
async def test_force_pause_with_qwen(qwen_engine: AsyncLLM):
    """
    Test force pause mode with Qwen model.
    
    Force mode should immediately abort running requests.
    """
    
    print("\n" + "="*70)
    print("Test: Force Pause with Qwen2.5-0.5B")
    print("="*70)
    
    # Start long-running request
    print("\n[Step 1] Starting long-running request...")
    long_task = asyncio.create_task(
        generate_and_collect(
            qwen_engine,
            prompt="Q: Write a detailed essay about machine learning:\nA:",
            request_id="long_request",
            max_tokens=200,  # Long generation
        )
    )
    
    # Let it start
    await asyncio.sleep(0.3)
    print("  ✓ Request started generating")
    
    # Force pause - should abort immediately
    print("\n[Step 2] Force pausing (should abort running requests)...")
    result = await qwen_engine.pause_generation(mode="force")
    
    print(f"  ✓ Force pause completed")
    print(f"    - Mode: {result['mode']}")
    print(f"    - Aborted requests: {result['aborted_requests']}")
    print(f"    - Elapsed: {result['elapsed_seconds']:.2f}s")
    
    assert result["paused"] is True
    assert result["mode"] == "force"
    assert result["aborted_requests"] >= 1
    
    # Verify request was aborted
    print("\n[Step 3] Verifying request was aborted...")
    output = await long_task
    
    assert output is not None
    assert output.outputs
    assert output.outputs[0].finish_reason == FinishReason.ABORT
    
    print(f"  ✓ Request finished with ABORT reason")
    print(f"    Generated tokens before abort: {len(output.outputs[0].token_ids)}")
    
    # Resume
    print("\n[Step 4] Resuming generation...")
    await qwen_engine.resume_generation()
    print("  ✓ Generation resumed")
    
    print("\n" + "="*70)
    print("✅ Force pause test completed!")
    print("="*70)


@pytest.mark.asyncio
async def test_pause_blocks_and_queues_requests(qwen_engine: AsyncLLM):
    """
    Test that pause properly queues requests and processes them after resume.
    
    This validates the core use case: pause blocks new work, resume processes queue.
    """
    
    print("\n" + "="*70)
    print("Test: Pause Queues Requests")
    print("="*70)
    
    # Pause first
    print("\n[Step 1] Pausing generation...")
    await qwen_engine.pause_generation()
    print("  ✓ Paused")
    
    # Send multiple requests while paused
    print("\n[Step 2] Sending 5 requests while paused...")
    queued_tasks = []
    for i in range(5):
        task = asyncio.create_task(
            generate_and_collect(
                qwen_engine,
                prompt=f"Q: What is {i} + {i}?\nA:",
                request_id=f"queued_request_{i}",
                max_tokens=10,
            )
        )
        queued_tasks.append(task)
        print(f"  ✓ Queued request {i}")
    
    # Wait a bit
    await asyncio.sleep(0.5)
    
    # None should complete yet
    completed_before = sum(1 for t in queued_tasks if t.done())
    print(f"\n[Step 3] Checking queued requests...")
    print(f"  ✓ Completed before resume: {completed_before}/5 (should be 0)")
    assert completed_before == 0, "No requests should complete while paused!"
    
    # Resume
    print("\n[Step 4] Resuming generation...")
    await qwen_engine.resume_generation()
    print("  ✓ Resumed")
    
    # Now all should complete
    print("\n[Step 5] Waiting for queued requests to complete...")
    outputs = await asyncio.gather(*queued_tasks, return_exceptions=False)
    
    print(f"  ✓ All requests completed: {len(outputs)}/5")
    for i, output in enumerate(outputs):
        assert output is not None
        assert len(output.outputs) > 0
        text = output.outputs[0].text.strip()
        print(f"    Request {i}: '{text}'")
    
    print("\n" + "="*70)
    print("✅ Queue test completed!")
    print("="*70)


@pytest.mark.asyncio
async def test_gentle_vs_force_pause_comparison(qwen_engine: AsyncLLM):
    """
    Compare gentle vs force pause behavior.
    
    Gentle: requests finish naturally
    Force: requests are aborted
    """
    
    print("\n" + "="*70)
    print("Test: Gentle vs Force Pause Comparison")
    print("="*70)
    
    # Test 1: Gentle pause
    print("\n[Test 1: Gentle Pause]")
    print("Starting request...")
    
    gentle_task = asyncio.create_task(
        generate_and_collect(
            qwen_engine,
            prompt="Q: What is AI?\nA:",
            request_id="gentle_test",
            max_tokens=50,
        )
    )
    
    await asyncio.sleep(0.2)
    
    print("Pausing (gentle mode)...")
    result = await qwen_engine.pause_generation(mode="gentle")
    
    print(f"  - Aborted: {result['aborted_requests']}")
    print(f"  - Unfinished: {result['num_unfinished_requests']}")
    
    gentle_output = await gentle_task
    
    # Gentle mode always waits for requests to finish
    print(f"  ✓ Request finished naturally (not aborted)")
    assert gentle_output.outputs[0].finish_reason != FinishReason.ABORT
    
    await qwen_engine.resume_generation()
    
    # Test 2: Force pause
    print("\n[Test 2: Force Pause]")
    print("Starting request...")
    
    force_task = asyncio.create_task(
        generate_and_collect(
            qwen_engine,
            prompt="Q: Explain quantum physics:\nA:",
            request_id="force_test",
            max_tokens=100,  # Longer to ensure it's running
        )
    )
    
    await asyncio.sleep(0.2)
    
    print("Pausing (force mode)...")
    result = await qwen_engine.pause_generation(mode="force")
    
    print(f"  - Aborted: {result['aborted_requests']}")
    assert result["aborted_requests"] >= 1
    
    force_output = await force_task
    
    print(f"  ✓ Request was aborted")
    assert force_output.outputs[0].finish_reason == FinishReason.ABORT
    print(f"    Generated {len(force_output.outputs[0].token_ids)} tokens before abort")
    
    await qwen_engine.resume_generation()
    
    print("\n" + "="*70)
    print("✅ Comparison test completed!")
    print("  Gentle: Waits for natural completion")
    print("  Force: Immediately aborts requests")
    print("="*70)


@pytest.mark.asyncio
async def test_complete_rl_simulation(qwen_engine: AsyncLLM):
    """
    Simulate a complete RL training cycle.
    
    Agent sends requests → Training pauses → Updates weights → Resumes
    """
    
    print("\n" + "="*70)
    print("Test: Complete RL Training Simulation")
    print("="*70)
    
    # Simulated RL training epochs
    num_epochs = 3
    requests_per_epoch = 5
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Phase 1: Agent collects data
        print(f"\n[Phase 1] Agent collecting trajectories...")
        
        agent_tasks = []
        for i in range(requests_per_epoch):
            task = asyncio.create_task(
                generate_and_collect(
                    qwen_engine,
                    prompt=f"Q: Generate policy for state {epoch*10 + i}:\nA:",
                    request_id=f"epoch_{epoch}_request_{i}",
                    max_tokens=20,
                )
            )
            agent_tasks.append(task)
        
        # Let some complete
        await asyncio.sleep(0.3)
        
        # Phase 2: Pause for weight update
        print(f"\n[Phase 2] Pausing for weight update...")
        pause_result = await qwen_engine.pause_generation()
        
        print(f"  ✓ Paused (cache_cleared={pause_result['cache_cleared']})")
        
        # Phase 3: Simulate weight update
        print(f"\n[Phase 3] Updating model weights...")
        await simulate_weight_update(duration=1.0)
        print(f"  ✓ Weights updated")
        
        # Phase 4: Resume
        print(f"\n[Phase 4] Resuming generation...")
        await qwen_engine.resume_generation()
        print(f"  ✓ Resumed")
        
        # Wait for all epoch requests to complete
        outputs = await asyncio.gather(*agent_tasks, return_exceptions=True)
        completed = sum(1 for o in outputs if o and hasattr(o, 'outputs'))
        print(f"\n[Results] Epoch {epoch + 1} completed: {completed}/{requests_per_epoch} requests")
    
    print("\n" + "="*70)
    print("✅ RL simulation completed!")
    print("="*70)


async def generate_and_collect(
    engine: AsyncLLM,
    prompt: str,
    request_id: str,
    max_tokens: int = 30,
):
    """
    Generate completion and collect final output.
    
    Returns the final RequestOutput.
    """
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    
    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        final_output = output
    
    return final_output


async def simulate_weight_update(duration: float = 1.0):
    """
    Simulate model weight update.
    
    In real RL, this would:
    - Compute gradients from collected trajectories
    - Update policy model parameters
    - Broadcast weights to inference engine
    """
    await asyncio.sleep(duration)


if __name__ == "__main__":
    # Run with: VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume_qwen.py -v -s
    pytest.main([__file__, "-v", "-s"])


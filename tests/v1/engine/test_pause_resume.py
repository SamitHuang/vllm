# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for pause and resume generation functionality for async RL training."""

import asyncio
import os

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine import FinishReason
from vllm.v1.engine.async_llm import AsyncLLM

# Skip tests if V1 is not enabled
pytestmark = pytest.mark.skipif(
    os.getenv("VLLM_USE_V1") != "1",
    reason="Pause/resume only supported in V1 engine"
)


@pytest.fixture
async def engine():
    """Create a test engine."""
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    yield engine
    # Cleanup
    del engine


@pytest.mark.asyncio
async def test_pause_resume_basic(engine: AsyncLLM):
    """Test basic pause and resume functionality."""
    
    # Check initial state
    status = await engine.get_pause_status()
    assert status["is_paused"] is False
    
    # Pause generation
    result = await engine.pause_generation()
    assert result["paused"] is True
    assert result["mode"] == "gentle"
    assert result["aborted_requests"] == 0
    assert "message" in result
    assert "num_unfinished_requests" in result
    
    # Check paused state
    status = await engine.get_pause_status()
    assert status["is_paused"] is True
    
    # Resume generation
    result = await engine.resume_generation()
    assert result["paused"] is False
    assert "message" in result
    
    # Check resumed state
    status = await engine.get_pause_status()
    assert status["is_paused"] is False


@pytest.mark.asyncio
async def test_double_pause(engine: AsyncLLM):
    """Test that pausing twice is idempotent."""
    
    # First pause
    result1 = await engine.pause_generation()
    assert result1["paused"] is True
    assert result1["mode"] == "gentle"
    
    # Second pause should succeed with message
    result2 = await engine.pause_generation()
    assert result2["paused"] is True
    assert result2["mode"] == "gentle"
    assert "Already paused" in result2["message"]
    
    # Resume
    await engine.resume_generation()


@pytest.mark.asyncio
async def test_double_resume(engine: AsyncLLM):
    """Test that resuming twice is idempotent."""
    
    # Pause first
    await engine.pause_generation()
    
    # First resume
    result1 = await engine.resume_generation()
    assert result1["paused"] is False

    # Second resume should succeed with message
    result2 = await engine.resume_generation()
    assert result2["paused"] is False
    assert "Not paused" in result2["message"]


@pytest.mark.asyncio
async def test_pause_blocks_new_requests(engine: AsyncLLM):
    """Test that new requests wait when paused."""
    
    # Pause generation
    await engine.pause_generation()
    
    # Start a generation request (should block)
    gen_task = asyncio.create_task(
        collect_outputs(
            engine.generate(
                prompt="Hello, world!",
                sampling_params=SamplingParams(max_tokens=10),
                request_id="test_request_1",
            )
        )
    )
    
    # Give it a moment to try to start
    await asyncio.sleep(0.2)
    
    # Request should still be pending
    assert not gen_task.done()
    
    # Resume generation
    await engine.resume_generation()
    
    # Now request should complete
    output = await asyncio.wait_for(gen_task, timeout=10.0)
    assert output is not None
    assert len(output.outputs) > 0


@pytest.mark.asyncio
async def test_pause_during_generation(engine: AsyncLLM):
    """Test pausing while generation is in progress."""
    
    # Start multiple generation requests
    tasks = []
    for i in range(3):
        task = asyncio.create_task(
            collect_outputs(
                engine.generate(
                    prompt=f"Request {i}: Tell me a story",
                    sampling_params=SamplingParams(max_tokens=50),
                    request_id=f"test_request_{i}",
                )
            )
        )
        tasks.append(task)
    
    # Let them start generating
    await asyncio.sleep(0.5)
    
    # Pause generation
    result = await engine.pause_generation()
    assert result["paused"] is True
    assert result["mode"] == "gentle"
    assert result["drained"] is True
    assert result["aborted_requests"] == 0
    
    # Wait a bit
    await asyncio.sleep(1.0)
    
    # Resume generation
    result = await engine.resume_generation()
    assert result["paused"] is False
    
    # All requests should eventually complete
    outputs = await asyncio.gather(*tasks, return_exceptions=False)
    assert len(outputs) == 3
    for output in outputs:
        assert output is not None
        assert len(output.outputs) > 0


@pytest.mark.asyncio
async def test_pause_resume_multiple_cycles(engine: AsyncLLM):
    """Test multiple pause/resume cycles."""
    
    for cycle in range(3):
        # Pause
        result = await engine.pause_generation()
        assert result["paused"] is True
        assert result["mode"] == "gentle"
        assert result["drained"] is True
        assert result["aborted_requests"] == 0

        # Verify paused
        status = await engine.get_pause_status()
        assert status["is_paused"] is True

        # Resume
        result = await engine.resume_generation()
        assert result["paused"] is False

        # Verify resumed
        status = await engine.get_pause_status()
        assert status["is_paused"] is False


@pytest.mark.asyncio
async def test_concurrent_pause_resume(engine: AsyncLLM):
    """Test that concurrent pause/resume calls are handled safely."""
    
    # Try to pause from multiple tasks simultaneously
    pause_tasks = [
        asyncio.create_task(engine.pause_generation())
        for _ in range(5)
    ]
    
    results = await asyncio.gather(*pause_tasks)
    
    # All should succeed
    for result in results:
        assert result["paused"] is True
        assert result["mode"] == "gentle"
    
    # Should be paused
    status = await engine.get_pause_status()
    assert status["is_paused"] is True
    
    # Resume
    await engine.resume_generation()


@pytest.mark.asyncio
async def test_force_pause_aborts_requests(engine: AsyncLLM):
    """Force pause should abort running requests immediately."""

    gen_task = asyncio.create_task(
        collect_outputs(
            engine.generate(
                prompt="Force pause test",
                sampling_params=SamplingParams(max_tokens=50),
                request_id="force_request",
            )
        )
    )

    await asyncio.sleep(0.2)

    result = await engine.pause_generation(mode="force")
    assert result["paused"] is True
    assert result["mode"] == "force"
    assert result["aborted_requests"] >= 1

    output = await gen_task
    assert output is not None
    assert output.outputs
    assert output.outputs[0].finish_reason == FinishReason.ABORT

    await engine.resume_generation()


@pytest.mark.asyncio
async def test_pause_invalid_mode(engine: AsyncLLM):
    with pytest.raises(ValueError):
        await engine.pause_generation(mode="invalid")


async def collect_outputs(generator):
    """Helper to collect all outputs from an async generator."""
    final_output = None
    async for output in generator:
        final_output = output
    return final_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Pause and Resume Generation for Async RL Training

## Overview

vLLM V1 engine supports **pause and resume** functionality for generation requests, designed specifically for asynchronous Reinforcement Learning (RL) training scenarios. This feature allows you to temporarily halt all generation while updating model weights, then resume seamlessly.

## Motivation

In asynchronous RL training with external agent environments:

1. **Agent environments** continuously send generation requests to the vLLM server
2. **Training process** periodically needs to update model weights
3. **Problem**: Cannot safely update weights while generation is active
4. **Solution**: Pause all generation, update weights, then resume

This implementation is based on [SGLang PR #7419](https://github.com/sgl-project/sglang/pull/7419).

## Requirements

- **V1 Engine**: Must set `VLLM_USE_V1=1` environment variable
- **vLLM Version**: Latest version with pause/resume support

## API Reference

### Python API

The `AsyncLLM` class provides three methods for pause/resume control:

#### `async def pause_generation(*, mode="gentle") -> dict`

Pauses all generation requests. Two modes are available:

- `mode="gentle"` (default): wait for in-flight requests to finish.
- `mode="force"`: immediately abort running requests.

Both modes clear the GPU KV cache so that no cached state survives across
weight updates.

**Returns:**
```python
{
    "paused": True,
    "message": "Generation paused successfully",
    "drained": True,
    "num_unfinished_requests": 0,
    "elapsed_seconds": 1.23,
    "mode": "gentle",
    "aborted_requests": 0
}
```

**Example:**
```python
from vllm.v1.engine.async_llm import AsyncLLM

engine = AsyncLLM.from_engine_args(engine_args)
await engine.pause_generation()              # gentle pause
await engine.pause_generation(mode="force")  # force pause
```

#### `async def resume_generation() -> dict`

Resumes all generation requests after pause.

**Returns:**
```python
{
    "paused": False,
    "num_unfinished_requests": 5,
    "message": "Generation resumed successfully"
}
```

**Example:**
```python
result = await engine.resume_generation()
print(result["message"])
```

#### `async def get_pause_status() -> dict`

Gets the current pause status.

**Returns:**
```python
{
    "is_paused": False,
    "num_unfinished_requests": 5
}
```

**Example:**
```python
status = await engine.get_pause_status()
if status["is_paused"]:
    print("Generation is paused")
```

### REST API

The OpenAI-compatible API server exposes three endpoints:

#### `POST /v1/pause`

Pause all generation requests. Use the optional `mode` query parameter to
choose between gentle and force pause.

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/pause?mode=gentle"
curl -X POST "http://localhost:8000/v1/pause?mode=force"
```

**Response:**
```json
{
    "paused": true,
    "message": "Generation paused successfully",
    "drained": true,
    "num_unfinished_requests": 0,
    "elapsed_seconds": 1.23,
    "mode": "gentle",
    "aborted_requests": 0
}
```

When `mode=force`, `aborted_requests` reports how many requests were
terminated.

#### `POST /v1/resume`

Resume all generation requests.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/resume
```

**Response:**
```json
{
    "paused": false,
    "num_unfinished_requests": 5,
    "message": "Generation resumed successfully"
}
```

#### `GET /v1/pause_status`

Get current pause status.

**Request:**
```bash
curl http://localhost:8000/v1/pause_status
```

**Response:**
```json
{
    "is_paused": false,
    "num_unfinished_requests": 5
}
```

## Usage Examples

### Example 1: Basic Async RL Training Loop

```python
import asyncio
from vllm.v1.engine.async_llm import AsyncLLM
from vllm import SamplingParams

async def rl_training_loop():
    # Initialize engine
    engine = AsyncLLM.from_engine_args(engine_args)
    
    # Training loop
    for epoch in range(num_epochs):
        # Phase 1: Collect data from agent
        print(f"Epoch {epoch}: Collecting data...")
        await collect_data_from_agents(engine, duration=10.0)
        
        # Phase 2: Pause for weight update
        print("Pausing generation...")
        await engine.pause_generation()
        
        # Phase 3: Update weights
        print("Updating model weights...")
        await update_model_weights()
        
        # Phase 4: Resume generation
        print("Resuming generation...")
        await engine.resume_generation()

async def collect_data_from_agents(engine, duration):
    """Agents send requests continuously."""
    start_time = asyncio.get_event_loop().time()
    request_id = 0
    
    while asyncio.get_event_loop().time() - start_time < duration:
        async for output in engine.generate(
            prompt="Generate action:",
            sampling_params=SamplingParams(max_tokens=20),
            request_id=f"request_{request_id}",
        ):
            # Process output for RL training
            process_for_rl(output)
        
        request_id += 1
        await asyncio.sleep(0.1)

async def update_model_weights():
    """Update model weights via collective communication."""
    # Similar to examples/offline_inference/rlhf.py
    # 1. Compute gradients from collected data
    # 2. Update training model
    # 3. Broadcast weights to inference engine
    await asyncio.sleep(2.0)  # Simulate weight update

asyncio.run(rl_training_loop())
```

### Example 2: REST API with Multiple Agents

```python
import asyncio
import aiohttp

async def agent_worker(session, worker_id):
    """Simulates an agent environment."""
    for i in range(100):
        async with session.post(
            "http://localhost:8000/v1/completions",
            json={
                "model": "facebook/opt-125m",
                "prompt": f"State {i}:",
                "max_tokens": 20,
            }
        ) as response:
            result = await response.json()
            print(f"Agent {worker_id}: {result['choices'][0]['text']}")
        await asyncio.sleep(0.1)

async def training_worker(session):
    """Manages weight updates."""
    for epoch in range(5):
        # Let agents collect data
        await asyncio.sleep(5.0)
        
        # Pause
        async with session.post("http://localhost:8000/v1/pause") as resp:
            print(await resp.json())
        
        # Update weights
        await asyncio.sleep(2.0)
        
        # Resume
        async with session.post("http://localhost:8000/v1/resume") as resp:
            print(await resp.json())

async def main():
    async with aiohttp.ClientSession() as session:
        # Run agents and training concurrently
        await asyncio.gather(
            agent_worker(session, 0),
            agent_worker(session, 1),
            agent_worker(session, 2),
            training_worker(session),
        )

asyncio.run(main())
```

### Example 3: With Weight Broadcasting (Like RLHF)

```python
import ray
import torch
from vllm import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from transformers import AutoModelForCausalLM

async def rl_training_with_weight_sync():
    # Load training model on GPU 0
    train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    train_model.to("cuda:0")
    
    # Launch vLLM inference engine on GPUs 1-2
    engine = ray.remote(num_gpus=2)(AsyncLLM).remote(
        model="facebook/opt-125m",
        tensor_parallel_size=2,
    )
    
    for epoch in range(num_epochs):
        # Collect trajectories
        trajectories = await collect_rl_trajectories(engine)
        
        # Pause generation
        await ray.get(engine.pause_generation.remote())
        
        # Compute RL loss and update training model
        loss = compute_ppo_loss(trajectories)
        loss.backward()
        optimizer.step()
        
        # Broadcast updated weights to inference engine
        for name, param in train_model.named_parameters():
            # Use Ray collective communication
            await broadcast_weight_to_engine(engine, name, param)
        
        # Resume generation
        await ray.get(engine.resume_generation.remote())

# See examples/offline_inference/rlhf.py for full weight broadcasting example
```

## How It Works

### Internal Mechanism

1. **Pause Event**: Uses `asyncio.Event` to block new requests
2. **Lock**: `asyncio.Lock` ensures atomic pause/resume operations
3. **Request Queue**: New requests wait in queue when paused
4. **KV Cache**: Preserved during pause for efficient resume

### State Transitions

```
Normal Operation
  ↓ pause_generation()
Paused (requests queued)
  ↓ resume_generation()
Normal Operation (queued requests processed)
```

### Timing

- **Pause Latency**: ~100ms (waits for current operations)
- **Resume Latency**: Immediate
- **Overhead**: Negligible when not paused
- **Memory**: KV cache is cleared during pause to avoid stale state

## Best Practices

### 1. Short Pause Durations

Keep pauses as short as possible to minimize agent wait time:

```python
# Good: Quick weight update
await engine.pause_generation()
await quick_weight_update()  # < 5 seconds
await engine.resume_generation()

# Avoid: Long pauses block all agents
await engine.pause_generation()
await slow_operation()  # > 30 seconds
await engine.resume_generation()
```

### 2. Handle Errors

Always resume even if weight update fails:

```python
try:
    await engine.pause_generation()
    await update_weights()
finally:
    await engine.resume_generation()
```

### 3. Monitor Status

Check pause status before weight updates:

```python
status = await engine.get_pause_status()
if not status["is_paused"]:
    await engine.pause_generation()
```

### 4. Coordinate with Agents

Design agents to handle temporary delays:

```python
# Agent with timeout
try:
    response = await session.post(
        url,
        json=request_data,
        timeout=aiohttp.ClientTimeout(total=30),  # Allow time for pause
    )
except asyncio.TimeoutError:
    # Handle pause gracefully
    print("Request delayed (likely paused)")
```

## Limitations

1. **V1 Engine Only**: Not supported in V0 engine
2. **In-Flight Requests**: Gentle mode waits for running requests to finish
3. **Force Mode**: Force mode aborts running requests; clients must handle abort outputs
4. **Single Server**: Applies to single vLLM server instance

## Troubleshooting

### Error: "Pause/resume not supported"

**Cause**: Using V0 engine

**Solution**: Set `VLLM_USE_V1=1`:
```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
```

### Requests Timing Out

**Cause**: Pause duration too long

**Solution**: Reduce weight update time or increase client timeout

### High Memory Usage

**Cause**: Pause waited too long for requests to drain

**Solution**: Ensure in-flight requests complete promptly or increase the
timeout in your orchestration logic

## See Also

- [RLHF Example](../../examples/offline_inference/rlhf.py) - Weight broadcasting
- [Async RL Example](../../examples/async_rl_pause_resume.py) - Complete demo
- [SGLang PR #7419](https://github.com/sgl-project/sglang/pull/7419) - Original implementation

## References

- **SGLang Implementation**: https://github.com/sgl-project/sglang/pull/7419
- **OpenRLHF Framework**: https://github.com/OpenRLHF/OpenRLHF
- **vLLM V1 Engine**: [Architecture Overview](../design/arch_overview.md)


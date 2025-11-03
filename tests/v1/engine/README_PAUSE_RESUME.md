# Pause/Resume Tests

## Overview

This directory contains comprehensive tests for the pause/resume functionality designed for async RL training.

## Test Files

### 1. `test_pause_resume.py`
Basic unit tests covering core functionality:
- Pause and resume operations
- Idempotent pause/resume
- Request blocking during pause
- Concurrent pause/resume calls
- Force vs gentle pause modes

### 2. `test_pause_resume_qwen.py` 
End-to-end integration tests with Qwen2.5-0.5B model:
- Complete pause/resume workflow
- Request queuing during pause
- Force pause behavior
- RL training simulation

## Running Tests

### Prerequisites

```bash
# Set V1 engine flag (REQUIRED)
export VLLM_USE_V1=1

# Install vLLM (if not already installed)
pip install -e .
```

### Run All Pause/Resume Tests

```bash
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume*.py -v
```

### Run Specific Tests

```bash
# Basic unit tests
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume.py -v

# Qwen end-to-end tests (with output)
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume_qwen.py -v -s

# Specific test
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume_qwen.py::test_pause_resume_workflow_qwen -v -s
```

## Test Descriptions

### `test_pause_resume_workflow_qwen`

Complete workflow test that validates:

1. **Multiple QA requests** - Sends 3 QA generation requests
2. **Pause** - Pauses generation (gentle mode)
3. **Block verification** - Sends new request, verifies it blocks
4. **Resume** - Resumes generation
5. **Completion** - Verifies blocked request completes
6. **New request** - Verifies new requests work after resume

**Expected Output:**
```
======================================================================
Test: Pause/Resume Workflow with Qwen2.5-0.5B
======================================================================

[Step 1] Sending initial QA generation requests...
  ✓ Started request 0: Q: What is the capital of France?
  ✓ Started request 1: Q: What is 2 + 2?
  ✓ Started request 2: Q: Who wrote Romeo and Juliet?

[Step 2] Pausing generation...
  ✓ Pause successful
    - Mode: gentle
    - Drained: True
    - Elapsed: 0.52s
    - Unfinished: 0
    - Aborted: 0
  ✓ Confirmed engine is paused

[Step 3] Sending request during pause (should block)...
  ✓ Request is blocked (paused state working)

[Step 4] Resuming generation...
  ✓ Resume successful
    - Paused: False
    - Message: Generation resumed
  ✓ Confirmed engine is resumed

[Step 5] Waiting for blocked request to complete...
  ✓ Blocked request completed after resume
    Prompt: Q: What is the meaning of life?
    A:
    Generated: The meaning of life is a deeply philosophical question...

[Step 6] Sending new request after resume...
  ✓ New request completed successfully
    Prompt: Q: What is the speed of light?
    A:
    Generated: The speed of light in a vacuum is approximately 299,792,458...

[Verification] Checking initial requests...
  ✓ Initial requests: 3/3 completed
    Request 0: Paris...
    Request 1: 4...
    Request 2: William Shakespeare...

======================================================================
✅ All steps completed successfully!
======================================================================
```

### `test_force_pause_with_qwen`

Tests force pause mode:

1. Start long-running request (200 tokens)
2. Force pause - immediately aborts
3. Verify request finished with ABORT reason
4. Resume generation

**Expected Output:**
```
======================================================================
Test: Force Pause with Qwen2.5-0.5B
======================================================================

[Step 1] Starting long-running request...
  ✓ Request started generating

[Step 2] Force pausing (should abort running requests)...
  ✓ Force pause completed
    - Mode: force
    - Aborted requests: 1
    - Elapsed: 0.15s

[Step 3] Verifying request was aborted...
  ✓ Request finished with ABORT reason
    Generated tokens before abort: 12

[Step 4] Resuming generation...
  ✓ Generation resumed

======================================================================
✅ Force pause test completed!
======================================================================
```

### `test_pause_blocks_and_queues_requests`

Tests request queuing:

1. Pause generation
2. Send 5 requests (should queue)
3. Verify none complete while paused
4. Resume
5. Verify all complete after resume

### `test_complete_rl_simulation`

Simulates 3 epochs of RL training:
- Agent generates requests
- Training pauses periodically
- Weights are "updated"
- Generation resumes
- Cycle repeats

## Troubleshooting

### "Pause/resume only supported in V1 engine"

Make sure `VLLM_USE_V1=1` is set:

```bash
export VLLM_USE_V1=1
pytest tests/v1/engine/test_pause_resume_qwen.py -v
```

### Model Download

First run will download Qwen2.5-0.5B (~1GB). To use cached model:

```bash
# Set HuggingFace cache
export HF_HOME=/path/to/cache

# Or download manually first
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

### GPU Memory

Tests use `gpu_memory_utilization=0.4` for safety. If OOM:

```bash
# Reduce batch size in test or memory utilization
# Edit test_pause_resume_qwen.py:
#   gpu_memory_utilization=0.3
```

### Timeout Errors

If tests timeout, increase wait times:

```python
# In test file, adjust timeouts:
await asyncio.wait_for(blocked_task, timeout=30.0)  # Increase from 15.0
```

## Performance Expectations

### Test Duration

- `test_pause_resume.py`: ~30 seconds (basic tests)
- `test_pause_resume_qwen.py`: ~2-3 minutes (includes model loading)

### Resource Usage

- **GPU Memory**: ~2-3 GB (Qwen2.5-0.5B + KV cache)
- **CPU**: Minimal
- **Network**: ~1 GB (first run for model download)

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Test Pause/Resume
  env:
    VLLM_USE_V1: "1"
  run: |
    pytest tests/v1/engine/test_pause_resume*.py -v
```

### Local Testing Script

```bash
#!/bin/bash
set -e

export VLLM_USE_V1=1

echo "Running basic pause/resume tests..."
pytest tests/v1/engine/test_pause_resume.py -v

echo "Running Qwen end-to-end tests..."
pytest tests/v1/engine/test_pause_resume_qwen.py -v -s

echo "All tests passed!"
```

## Debugging

### Verbose Output

```bash
# See detailed output
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume_qwen.py -v -s --log-cli-level=INFO
```

### Single Test

```bash
# Run just the workflow test
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume_qwen.py::test_pause_resume_workflow_qwen -v -s
```

### With Debugging

```bash
# Add breakpoints and use pdb
VLLM_USE_V1=1 pytest tests/v1/engine/test_pause_resume_qwen.py -v -s --pdb
```

## Coverage

These tests cover:

✅ Basic pause/resume functionality  
✅ Gentle mode (wait for drain)  
✅ Force mode (immediate abort)  
✅ Request blocking during pause  
✅ Request queuing and processing  
✅ Multiple pause/resume cycles  
✅ Concurrent pause calls  
✅ Error handling  
✅ Complete RL workflow simulation

## Related Documentation

- Implementation: `../../IMPLEMENTATION_SUMMARY.md`
- User Guide: `../../docs/features/async_rl_pause_resume.md`
- Quick Start: `../../QUICKSTART_PAUSE_RESUME.md`
- Example: `../../examples/async_rl_pause_resume.py`

## Support

For issues or questions:
- Check documentation in `docs/features/async_rl_pause_resume.md`
- Run tests with `-v -s` flags for detailed output
- Report issues on GitHub with test logs


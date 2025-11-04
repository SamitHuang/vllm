# REST API Pause/Resume 测试指南

## 概述

本测试通过 REST API（HTTP 请求）验证 pause/resume 功能，使用纯 bash + curl 命令，完全模拟真实的客户端场景。

## 快速开始

### 两步测试（推荐）

**Terminal 1** - 启动服务器：

```bash
./start_server.sh
```

**Terminal 2** - 运行测试：

```bash
./test_api_pause_resume.sh
```

### 手动启动服务器

如果你不想用脚本：

```bash
# Terminal 1
VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

# Terminal 2
./test_api_pause_resume.sh
```

## 测试步骤详解

### Step 1: 发送多个流式生成请求

```bash
# 请求 1
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "写一个关于机器人的故事：",
    "max_tokens": 100,
    "stream": true
  }' &

# 请求 2 和 3 类似...
```

**预期**：看到流式输出，token 逐个出现

### Step 2: 暂停生成

```bash
# 暂停并清除缓存（默认）
curl -X POST "http://localhost:8000/v1/pause?mode=gentle&clear_cache=true"

# 或暂停但保留缓存
curl -X POST "http://localhost:8000/v1/pause?mode=gentle&clear_cache=false"
```

**参数说明**：
- `mode`: 暂停模式
  - `gentle` - 等待正在执行的请求完成（默认）
  - `force` - 立即中止正在执行的请求
- `clear_cache`: 是否清除 KV cache 和 prefix cache（默认 true）

**预期响应**：
```json
{
  "paused": true,
  "mode": "gentle",
  "elapsed_seconds": 1.23,
  "num_unfinished_requests": 0,
  "aborted_requests": 0,
  "cache_cleared": true,
  "message": "Generation paused successfully"
}
```

**预期效果**：
- ✅ 流式输出立即停止
- ✅ `paused: true`
- ✅ `cache_cleared: true/false`（取决于参数）
- ✅ 请求完全 drain，`num_unfinished_requests: 0`

### Step 3: 在 pause 期间发送新请求

```bash
# 这个请求会被阻塞
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Q: 生命的意义？\nA:",
    "max_tokens": 30
  }' &

BLOCKED_PID=$!
```

**预期效果**：
- ✅ curl 命令挂起，不返回
- ✅ 没有任何生成输出
- ✅ 进程仍在运行（`ps` 可以看到）

**验证阻塞**：
```bash
# 等待 1 秒
sleep 1

# 检查进程是否还在（说明被阻塞了）
if kill -0 $BLOCKED_PID 2>/dev/null; then
    echo "✓ 请求被阻塞了"
else
    echo "❌ 请求没有被阻塞"
fi
```

### Step 4: 恢复生成

```bash
curl -X POST http://localhost:8000/v1/resume
```

**预期响应**：
```json
{
  "paused": false,
  "message": "Generation resumed",
  "num_unfinished_requests": 1
}
```

**预期效果**：
- ✅ `paused: false`
- ✅ 被阻塞的请求开始执行

### Step 5: 验证被阻塞的请求完成

**预期**：
- ✅ 之前挂起的 curl 命令现在返回
- ✅ 返回正常的生成结果

```json
{
  "choices": [{
    "text": "生命的意义是一个深刻的哲学问题...",
    "finish_reason": "length"
  }]
}
```

### Step 6: 发送新请求（验证正常工作）

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Q: 光速是多少？\nA:",
    "max_tokens": 30
  }'
```

**预期**：
- ✅ 立即开始生成
- ✅ 正常返回结果

## 预期输出示例

### 完整输出示例

```bash
$ ./test_api_pause_resume.sh --with-server

======================================================================
Starting vLLM Server
======================================================================

VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000

Starting server in background...
Server PID: 12345
Server logs: /tmp/vllm_server.log

Waiting for server to be ready...
  Waiting... (2s / 120s)
  Waiting... (4s / 120s)
✓ Server is ready!

======================================================================
API Pause/Resume Test
======================================================================

Server: http://localhost:8000

======================================================================
[Step 0] Checking server health
======================================================================

  ✓ Server is healthy

======================================================================
[Step 1] Sending multiple generation requests (streaming)
======================================================================

  → Request 1: Starting story generation...
  → Request 2: Starting science explanation...
  → Request 3: Starting history description...
  ✓ 3 streaming requests started (PIDs: 12346, 12347, 12348)
  → Letting them generate for 2 seconds...

  (You should see streaming output above)

======================================================================
[Step 2] Pausing generation (streaming should stop)
======================================================================

{
    "paused": true,
    "mode": "gentle",
    "drained": true,
    "elapsed_seconds": 1.23,
    "num_unfinished_requests": 0,
    "aborted_requests": 0,
    "message": "Generation paused successfully"
}
  ✓ Generation paused successfully
  ✓ Streaming output should have stopped

======================================================================
[Step 3] Sending request during pause (should block)
======================================================================

  → Starting blocked request in background...
  → Blocked request PID: 12349
  ✓ Request is blocked (still waiting)
  ✓ Pause is working correctly!

======================================================================
[Step 4] Resuming generation
======================================================================

{
    "paused": false,
    "message": "Generation resumed",
    "num_unfinished_requests": 1
}
  ✓ Generation resumed successfully
  ✓ Blocked request should now complete

======================================================================
[Step 5] Waiting for blocked request to complete
======================================================================

  ✓ Blocked request completed!

  Response:
    生命的意义是一个深刻的哲学问题...
    Duration: 5s (includes pause time)

======================================================================
[Step 6] Sending new request after resume
======================================================================

  → Sending request...
  ✓ New request completed successfully
    Generated: 光速约为每秒30万公里...

======================================================================
[Step 7] Cleanup
======================================================================

  ✓ Cleaned up background processes

======================================================================
✅ ALL TESTS PASSED!
======================================================================

Summary:
  ✓ Step 1: Sent 3 streaming requests
  ✓ Step 2: Paused generation (streaming stopped)
  ✓ Step 3: New request blocked during pause
  ✓ Step 4: Resumed generation
  ✓ Step 5: Blocked request completed
  ✓ Step 6: New request worked normally

Pause/Resume API functionality is working correctly! 🎉
======================================================================

======================================================================
Stopping server (PID: 12345)...
======================================================================
✓ Server stopped
```

## 手动测试命令

如果你想手动测试，可以直接使用这些 curl 命令：

### 1. 检查服务器

```bash
curl http://localhost:8000/health
```

### 2. 发送生成请求（流式）

```bash
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "写一个故事：",
    "max_tokens": 50,
    "stream": true
  }'
```

### 3. Pause

```bash
curl -X POST "http://localhost:8000/v1/pause?mode=gentle"
```

### 4. 发送请求（会被阻塞）

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "测试：",
    "max_tokens": 20
  }'
# 这个命令会挂起！
```

### 5. Resume

在另一个 terminal：
```bash
curl -X POST http://localhost:8000/v1/resume
```

观察第 4 步的 curl 命令应该立即返回结果！

### 6. 查看 pause 状态

```bash
curl http://localhost:8000/v1/pause_status
```

## 故障排查

### 错误: "Connection refused"

**原因**: 服务器未启动

**解决**:
```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

### 错误: "Pause/resume not supported"

**原因**: 未使用 V1 引擎

**解决**: 确保启动服务器时设置了 `VLLM_USE_V1=1`

### 请求在 pause 期间没有被阻塞

**检查**:
1. 确认 pause 响应中 `paused: true`
2. 检查 pause_status: `curl http://localhost:8000/v1/pause_status`
3. 确保使用的是同一个服务器实例

### 流式输出没有停止

**可能原因**:
1. 请求在 pause 前已经完成
2. 增加 `max_tokens` 让生成更长

## 测试变体

### Force Pause 测试

```bash
# 启动长生成
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一篇长文章：",
    "max_tokens": 200,
    "stream": true
  }' &

# 等待一下
sleep 1

# Force pause - 立即中止
curl -X POST "http://localhost:8000/v1/pause?mode=force"

# 应该看到流式输出立即停止
# 响应中 aborted_requests > 0
```

### 多次 Pause/Resume 循环

```bash
for i in {1..3}; do
    echo "Cycle $i"
    curl -X POST http://localhost:8000/v1/pause
    sleep 1
    curl -X POST http://localhost:8000/v1/resume
    sleep 1
done
```

## 服务器日志查看

使用 `./start_server.sh` 时，日志会直接输出到 terminal。

如果想保存日志：

```bash
# 启动并保存日志
./start_server.sh 2>&1 | tee /tmp/vllm_server.log

# 或后台运行并保存日志
VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 \
  > /tmp/vllm_server.log 2>&1 &

# 查看日志
tail -f /tmp/vllm_server.log
```

## 相关文件

- **服务器启动脚本**: `start_server.sh`
- **API 测试脚本**: `test_api_pause_resume.sh`
- **快速开始**: `QUICKSTART_API_TEST.md`
- **直接引擎测试**: `quick_test_pause.py`
- **完整测试**: `test_pause_resume_standalone.py`
- **Pytest 测试**: `tests/v1/engine/test_pause_resume.py`
- **完整文档**: `docs/features/async_rl_pause_resume.md`
- **设计文档**: `PAUSE_RESUME_DESIGN.md`

## 总结

```bash
# Terminal 1: 启动服务器
./start_server.sh

# Terminal 2: 运行测试
./test_api_pause_resume.sh
```

就这么简单！🎉


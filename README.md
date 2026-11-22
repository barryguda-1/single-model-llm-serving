# single-model-llm-serving

A minimal, hand-rolled LLM serving stack (FastAPI + a single model worker
process) built to understand what systems like vLLM do under the hood. A
vLLM-backed endpoint is included for comparison.

## Architecture

```
FastAPI (main.py)
  └── LLMEngine (llm/llm.py)
        ├── WorkloadManager (llm/workload_manager.py)
        │     sequence registry + shared streaming pool (continuous batching)
        ├── ModelExecutor (llm/model_executor.py)
        │     task/result queues to the worker process, serialized round-trips
        │     └── ModelWorker (llm/model_worker.py, separate process)
        │           └── ModelManager (llm/model_manager.py) — HF model loading
        └── vLLM model (direct, for /generate_vllm)
```

- **Streaming** (`/generate_stream`): requests join a shared pool; a
  background thread runs continuous batching — one forward pass per step for
  all active sequences — and pushes tokens to clients over per-request
  asyncio queues (SSE).
- **Batch** (`/generate`, `/basic_generate`): each call builds its own local
  sequences and runs them through `model.generate` in fixed-size batches,
  isolated from the streaming pool.

## Endpoints

| Endpoint | Body | Notes |
|---|---|---|
| `POST /generate_stream` | `{"prompt": "..."}` | SSE, token-by-token |
| `POST /basic_generate` | `{"prompt": "..."}` | single prompt, full text |
| `POST /generate` | `{"prompts": ["...", "..."]}` | batched, full texts |
| `POST /generate_vllm` | `{"prompts": ["...", "..."]}` | vLLM backend, batched |

## Run

```bash
pip install -r requirements.txt      # vLLM is Linux/CUDA only — see requirements.txt
python main.py                       # starts uvicorn on :8000
```

Environment variables:

| Var | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `facebook/opt-125m` | HF model id (used by both the HF worker and vLLM) |
| `MAX_TOKENS` | `20` | max new tokens per request |
| `PORT` | `8000` | server port |

## Known trade-offs (intentional, learning scope)

- **No KV cache**: streaming re-tokenizes the growing prompt every step
  (`use_cache=False`) — O(n²) work per sequence.
  `ModelWorker.stream_states` is the placeholder for a proper
  incremental-decoding implementation.
- **Serialized batches**: one lock over a single queue pair — correctness
  over throughput.
- **vLLM platform support**: vLLM does not install on native Windows; use
  Linux/WSL with CUDA for `/generate_vllm` (the `vllm` import in `llm.py`
  is required regardless, since it lives at module top level).

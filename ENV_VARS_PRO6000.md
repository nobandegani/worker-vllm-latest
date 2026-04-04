# Environment Variables — Huihui-Qwen3.5-35B-A3B-abliterated on RTX Pro 6000 (96GB)

Only variables that differ from defaults are listed.

| Variable | Value | Default | Notes |
|---|---|---|---|
| `MODEL_NAME` | `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` | *(required)* | |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `qwen3.5-35b-a3b` | model path | Cleaner model name in API responses |
| `DTYPE` | `bfloat16` | `auto` | Full precision weights — best quality |
| `KV_CACHE_DTYPE` | `auto` | `auto` | Keep bf16 KV cache — no quality loss on cache values |
| `MAX_MODEL_LEN` | `65536` | `131072` | 64K context; bf16 KV cache + model weights fit well in 96GB |
| `GPU_MEMORY_UTILIZATION` | `0.95` | `0.95` | Use all available VRAM |
| `MAX_NUM_SEQS` | `128` | `256` | Good throughput without over-saturating single GPU |
| `MAX_CONCURRENCY` | `30` | `30` | Match vLLM's internal queue capacity |
| `ENABLE_PREFIX_CACHING` | `true` | `false` | Reuses KV cache — faster repeated/system prompts |
| `ENABLE_CHUNKED_PREFILL` | `true` | `false` | Prevents stalls on long prompts, smoother scheduling |
| `REASONING_PARSER` | `qwen3` | `""` | Enables `/think` and `/no_think` reasoning mode |

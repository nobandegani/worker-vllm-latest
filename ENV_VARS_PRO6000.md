# Environment Variables — Qwen3.5-40B-Deckard-Thinking on RTX Pro 6000 (96GB)

Model: `DavidAU/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking`
Architecture: `Qwen3_5ForConditionalGeneration` (hybrid Mamba/Attention)

Only variables that differ from defaults are listed.

| Variable | Value | Default | Notes |
| --- | --- | --- | --- |
| `MODEL_NAME` | `DavidAU/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking` | *(required)* | |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | `fork` | **Required** — fixes "Cannot re-initialize CUDA in forked subprocess" crash with Mamba models |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `qwen3.5-40b-deckard` | model path | Cleaner model name in API responses |
| `DTYPE` | `bfloat16` | `auto` | Full precision — best quality |
| `KV_CACHE_DTYPE` | `auto` | `auto` | bf16 KV cache — no quality loss |
| `MAX_MODEL_LEN` | `32768` | `131072` | 40B weights ~80GB; only ~11GB left for KV cache. 32K is the safe ceiling. |
| `GPU_MEMORY_UTILIZATION` | `0.95` | `0.95` | Use all available VRAM |
| `MAX_NUM_SEQS` | `16` | `256` | Bigger model = less room for concurrent sequences |
| `MAX_CONCURRENCY` | `16` | `30` | Match `MAX_NUM_SEQS` |
| `ENABLE_PREFIX_CACHING` | `false` | `false` | **Important** — vLLM marks this experimental for Mamba layers; can cause correctness issues |
| `ENABLE_CHUNKED_PREFILL` | `true` | `false` | Smoother scheduling for long prompts |
| `REASONING_PARSER` | `qwen3` | `""` | Enables `/think` and `/no_think` reasoning mode |

## Memory Budget

- 40B params × 2 bytes (bf16) = **~80GB weights**
- 96GB × 0.95 util = **~91GB available**
- Leaves **~11GB** for KV cache + activations + CUDA graphs

## If You Hit OOM

Drop further:
- `MAX_MODEL_LEN=16384`
- `MAX_NUM_SEQS=8`

Or enable fp8 KV cache for ~2x more cache room (small quality hit):
- `KV_CACHE_DTYPE=fp8`

## Notes on This Model

- **Hybrid Mamba/Attention** architecture — different memory profile than pure transformer
- Logs show: `Mamba cache mode is set to 'align'` and `Setting attention block size to 784 tokens to ensure that attention page size is >= mamba page size`
- Native vLLM 0.19.0 support — `TRUST_REMOTE_CODE` not needed
- Thinking model — supports reasoning mode via `qwen3` parser

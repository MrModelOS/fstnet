"""jarvis_engine/runtime — Native Llama Engine (Блок 1).

Запланировано (Фаза 2, см. SPEC_JARVIS.md):
  bit_kernels.cu  — CUDA 1-bit W1A8, XOR/POPCNT BitLinear, Field Synthesizer.
  paged_kv.py     — страничный KV-кэш (PagedAttention-style) под 8k контекста.
  sampler.py      — temperature/top-p/min-p/repetition + Grammar/Schema-mask.
  mcp_server.py   — IPC/Socket сервер с подмешиванием GGM-памяти.
"""
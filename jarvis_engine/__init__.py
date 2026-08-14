"""jarvis_engine — автономная система JARVIS Core (Native Llama Engine + Trainer).

trainer/  — Custom Trainer Engine: многофазный луп S0→S2, STE, MoF-поля,
            Adafactor, gradient checkpointing (Фаза 1, рабочая).
runtime/  — Native Llama Engine: paged KV, sampler, MCP-сервер (Фаза 2, Gap).
"""
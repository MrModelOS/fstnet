# JARVIS Core — автономная система (jarvis_engine/)

Полный отказ от transformers/accelerate/trl. Два монолитных блока.

## Gap-анализ: план jarvis_engine/ vs факт в репозитории

### Блок 2 — Trainer (Фаза 1: ГОТОВО, приоритет)

| Модуль плана                | Факт в репо                                                     | Статус |
|-----------------------------|-----------------------------------------------------------------|--------|
| ste_layers.py (BitLinear+STE)| `brain/model/core_mof.py` — BitLinear(STE), ContinuousField, GQA+RoPE, Hypernet, L_orth | 100% — переносим без переписывания |
| run_trainer.py (S0→S2)      | `jarvis_engine/trainer/run_trainer.py` — многофазный луп, resume, val, npz-кэш, Drive | 100% |
| mof_loss.py (CE + L_orth)   | орт-штраф встроен в модель: `ContinuousField.orth_loss()` + `FSTMoFModel.orth_loss()` | 100% |
| zero_alloc_adam.py (8-bit AdamW/Lion) | `jarvis_engine/trainer/ste_optimizer.py` — **Adafactor** (факторизованные состояния) | 90% — уже решает OOM (не 8-bit AdamW); 8-bit AdamW/Lion — опция ниже |
| Gradient Acc/Checkpointing  | `jarvis_engine/trainer/memory_manager.py` — grad-ckpt + empty_cache | 100% |
| data_pipeline               | `brain/data/jarvis_full.json` (500k convs, 616MB) + jarvis_special (500k), fetch/merge/distill/build | 100% |

Данные: **500 000 convs — подтверждено** в обоих датасетах.

### Блок 1 — Runtime (Фаза 2: GAP 100%, строим пока идёт Colab)

| Модуль плана                | Факт в репо                                                     | Статус |
|-----------------------------|-----------------------------------------------------------------|--------|
| bit_kernels.cu (1-bit W1A8) | —                                                               | GAP — писать (XOR/POPCNT + Field Synthesizer) |
| paged_kv.py (8k paged KV)   | наивный `torch.cat` в `GQAAttention`; conf max_seq_len=4096      | GAP — писать; надо поднять seq до 8192 |
| sampler.py (temp/top-p/min-p/repeat/grammar) | `fstnet_server.py`: только top-k + temperature + multinomial | GAP — добавить min-p, repetition penalty, grammar/schema-mask |
| mcp_server.py (GGM-подмес)  | `engine/llama_runtime/mcp_server.py` + `brain/memory/ggm_mcp.py` | ~60% — адаптировать |

## Известные узкие места (зафиксированы)

1. **ContinuousField.forward** при CUDA делает `indices.cpu().tolist()` → синхронизация CPU↔GPU на каждый токен. Рефакторинг: batched top-k + gather вместо цикла.
2. **sys.path-fix**: после переноса `run_trainer.py` добавляет в path `jarvis_engine/`, корень и `brain/` (config/model/colab_drive живут там).
3. Конфиг 3b_mof: `max_seq_len=4096` (план — 8192), `n_kv_heads` — проверить относительно 16 heads.
4. Датасет 500k s/файл — при 616MB каждый, токенизация в memmap уже вне RAM (npz-кэш).

## Roadmap

- [x] **Фаза 1**: trainer в `jarvis_engine/trainer/`; запуск S0/S1 в Colab — `python jarvis_engine/trainer/run_trainer.py`
- [ ] **Фаза 2** (параллельно с Colab, T4): `runtime/`: paged_kv.py → sampler.py (grammar) → bit_kernels.cu → mcp_server.py
- [ ] **Фаза 3**: JS-инференс 1-bit чекпоинта через runtime; интеграция GGM-памяти.

### Запуск тренировки (Colab)

```bash
# Stage 1 (общая база) -> checkpoints/3b_mof/moF_best.pt
FSTNET_EPOCHS=4 FSTNET_LR=2e-4 python jarvis_engine/trainer/run_trainer.py
# Stage 2 (спец. датасет, W0 frozen, L_orth)
FSTNET_STAGE=2 FSTNET_DATA=data/jarvis_special.json FSTNET_EPOCHS=4 python jarvis_engine/trainer/run_trainer.py
```
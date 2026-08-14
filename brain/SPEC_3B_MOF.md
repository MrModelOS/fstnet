# JARVIS Core — 3B 1-bit MoF FST-Net
**Полная спецификация архитектуры (v1.0, 2026)**

> Заменяет направление «800M dense». Целевая система: локальный ИИ-агент JARVIS
> на ноутбуке с NVIDIA MX450 (2GB VRAM), 3.2B параметров знаний при ~400MB весов.

---

## 1. Архитектура (Model Architecture)

Формат: **MoF (Mixture of Fields / Continuous Parameter Fields) + BNN (BitNet)**.

Ключевое отличие от MoE: нет дискретных экспертов и router'а. Вместо этого —
**непрерывное параметрическое поле знаний**: гиперсеть синтезирует на лету
низкоранговую матричную добавку к базовым весам для каждого токена.

### 1.1 Математика MoF-слоя

```
W_eff(x) = W0 + Σ_{i=1}^{K} α_i(x) · (U_i ⊗ V_i)
```

- `W0` — базовый вес FFN (всегда активен, 1-bit), shape (d, d_ff)
- `U_i` — левый базис поля i, shape (d, r); `V_i` — правый базис, shape (r, d_ff)
- `α_i(x)` — непрерывный коэффициент (softmax выход гиперсети по x)
- Применяется БЕЗ материализации `W_eff`: коррекция считается как
  `y += α_i · ((x U_i) V_i)` — последовательные низкоранговые GEMM-ы
- Гейтинг — **top-α маска с renormalize** (дифференцируемая, STE на маску):
  активными считаются только GS полей, остальные выпадают из вычислений

### 1.2 Бинаризация (1-bit)

- Веса `w ∈ {-1, +1}` через `sign(w)`, пер-канал-скейл `s = mean|w|` (absmax-группа)
- Прямой проход: `w_q = sign(w) * s`; обратный: **STE** (градиент как через identity)
  `w_eff = w + (sign(w) - w).detach()`
- Активации: A8 (per-token int8 QDQ) — для будущих XOR/POPCNT-ядер
- Формат хранения: 1 bit на вес, упаковка 32 веса в uint32 → `sign` + `popcount`

### 1.3 Параметры (конкретные числа)

| Параметр | Значение |
|---|---|
| vocab_size | 32770 (существующий BPE) |
| d_model | 2048 |
| n_layers | 32 |
| n_heads / n_kv_heads | 16 / 4 (GQA) |
| head_dim | 128 |
| d_ff (gate/up/down) | 6144 |
| n_fields (K) | 32 |
| field_rank (r) | 64 |
| gating_top_k (GS) | 8 |
| ctx train / infer | 4096 / 8192 (RoPE-экстенд) |
| rope_base | 10000 |

### 1.4 Сводка параметров (Math Check, из config_3b_mof.py)

- **Total ≈ 3.39B** (включая embedding 67M + logit-head 67M)
- **Хранение 1-bit ≈ 424MB** (+гиперсеть/роутер ~10MB) → ~434MB VRAM — укладывается в лимит MX450
- **KV-кэш (fp16, ctx 8192) ≈ 268MB**; опция fp8 → 134MB

Active/токен (base + top-GS/32 полей):

| GS | BW-active параметров |
|---|---|
| 4 | 1.95B |
| 8 | 2.15B |
| 16 | 2.55B |

> Важно: в MoF «active» не равно «скачанные веса». Base W0 (1-bit, ~1.2B параметров = ~150MB) всегда в кэше и читается линейно. Вычислительная стоимость токена ~эквивалент 600-700M dense (base GEMM + GS×3 низкоранговых коррекций + attention) — диал `gating_top_k` регулирует бюджет. Диалог GS=4 → compute-truthful ≈ 550-650M (близко к исходному требованию 600M).

---

## 2. Пайплайн данных и дистилляция (Data & Teacher)

### 2.1 Учитель
- **Qwen 27B 1-bit (Bonsai)** — llama.cpp server в Colab через `run_teacher_colab.py` (~3.5GB VRAM на T4)
- Источник модели: `--gguf-url`, `--hf-repo+--hf-file` или `--local-gguf` (env `TEACHER_*`)
- OpenAI-совместимый `/v1/chat/completions` на :8001 → его ест `distill_colab.py`
- Только генерация (offline), не фиксируется в артефактах

### 2.2 Датасет
- **150-200K** синтетических траекторий, JSONL
- Формат строки:
  ```
  <think> пошаговый CoT </think>\n<tool_call>{"name": "...", "args": {...}}</tool_call>\nОтвет Сэр: ...
  ```
- Пропорции как в JARVIS: 40% код / 25% reasoning (CoT+think) / 20% tool-call (Linux, AST, FAISS) / 15% персона
- Отфильтровать дедупликацию, повторы токенов, >8192 токенов

### 2.3 Токенизация
- Родной BPE (fst_bpe.json, 32770), не Qwen-токенайзер
- Пре-токенизация в `.npz` uint16-блоки (4096 токенов) → SSD `/content` + Drive-бэкап
- Сессия Colab = чистая машина; Drive — источник истины

---

## 3. Обучение (Training)

### 3.1 Фазы (4 стадии)

1. **S0 Warmup-dense** (~1 epoch, dense fp16+grad-scaler):
   - Все параметры float, без STE — установить базу `W0` и α-распределение
   - T4 (sm_75): fp16, `torch.compile` OFF по умолчанию
2. **S1 STE-binarize** (~2 epochs):
   - `binarize_ratio` анилингуется 0→1: `w_q = (1-β)·w + β·sign(w)·s`
   - Активации A8 QDQ включаются после β≥0.5
3. **S2 Orth-fine** (~0.5 epoch):
   - Фиксируем `W0`, обучаем поля+гиперсеть с `L_orth`
4. **S3 Export**: `sign(W)·s` → 1-bit pack → кастомный GGUF → bitnet.cpp/llama.cpp fork

### 3.2 Loss
- Cross-Entropy (next-token), только ассистент/think сегменты
- `L_orth = ||UᵀU − I||_F` (блочно-диагональная ортогональность полей) + кросс-полевой терм `||U_iᵀU_j||_F`
- Ауксилярный load-balancing НЕ нужен (непрерывная природа α)

### 3.3 Оптимизатор и память
- AdamW, weight_decay 0.01, `lr 2e-4 → 1e-5`, warmup ~100 steps, grad-clip 1.0
- Обучение полей+гиперсети+attn ≈ **2.0B float** (после S0, W0 frozen) → AdamW-состояния ≈ **~8GB** на T4 (16GB — запас), `W0` frozen после S0
- Gradient checkpointing + `grad_accum=8`, batch 8

### 3.4 Скорость генерации (bandwidth-анализ)
- Dense 3B fp16: чтение 6GB/токен → bottleneck шины
- MoF 1-bit: чтение **~400MB** + синтез коррекций в L1/L2 кэше (Compute-bound, а не BW-bound) → **на 20-40% быстрее MoE** (по модели), на 50-80% на 2+ GPU (нет All-to-All, обмен только α-коэффициентами)

---

## 4. Инференс (Runtime)

- **Backend**: bitnet.cpp / fork llama.cpp, ядра W1A8 XOR+POPCNT (SSE/AVX2/GPU CUDA)
- **Вес модели**: ~408MB на MX450
- **KV-кэш**: ~268MB (fp16) / 134MB (fp8), ctx 8192
- **Роутер/гиперсеть**: ~10MB (всегда в кэше)
- **Итого**: ~420-450MB из 2GB MX450

---

## 5. Экосистема JARVIS (Ecosystem)

| Компонент | Железо | Ресурс |
|---|---|---|
| Qwen 27B 1-bit (учитель) | Colab/Kaggle | ~3.5GB VRAM (временный) |
| 3B 1-bit MoF FST-Net | MX450 | ~420MB VRAM |
| KV-кэш (8k) | MX450 | ~268MB |
| Kokoro-82M TTS (ru Ruslan) | CPU ONNX | ~100MB RAM / 5% CPU |
| GGM / STDM (память) | Disk+CPU | 300K+ узлов, FAISS |
| ACSC (валидатор) | CPU | самокритика команд |
| MCP / Unix-сокеты | CPU | pacman, git, процессы |

---

## 6. План реализации (открытые пункты)

1. `config_3b_mof.py` — конфиг ✅
2. `model/core_mof.py` — BitLinear(STE), ContinuousField, GQA+RoPE, гиперсеть, L_orth ✅
3. `engine/trainer_engine/custom_trainer.py` — фазы S0-S2, npz-кэш, Drive-персист, resume ✅
4. `build_jarvis_data.py` — думающие траектории ✅
5. Генерация датасета 150-200K учителем (отдельный скрипт дистилляции — TODO)
6. 1-bit pack + GGUF + bitnet.cpp fork — TODO (после валидации качества на MX450-тесте)

## 7. Критерии готовности (Definition of Done)

- [ ] Модель обучается в Colab (S0→S3) без OOM на T4 16GB
- [ ] Экспорт 1-bit ~410MB, инференс на MX450 < 450MB VRAM
- [ ] Связный русский ответ «Сэр» + корректный `<tool_call>` на 40% тест-выборке
- [ ] GGM/STDM/ACSC/MCP подключены к серверу, end-to-end агентский цикл

# FST-Net JARVIS

Local-first autonomous AI assistant (JARVIS) on FST-Net 800M with MCP, GGM (Graph-Gated Memory), STDM (State-Tree Delta Memory), and ACSC (Async Self-Critique). Small model + external memory = deep behavior without storing the world in weights.

Target: JARVIS Core 3B 1-bit MoF (see `brain/SPEC_3B_MOF.md`) — fits in MX450 2GB VRAM at ~400MB weights.

## Structure

```
fstnet/
├── brain/    # Мозг: модель, конфиги, обучение, оценка, датасеты, tokenizer
├── body/     # Тело Джарвиса: сервер, MCP, run-скрипты
└── memory/   # Память: GGM (граф+FAISS), STDM, ACSC, ggm-инструменты
```

## Architecture

```
OpenCode/Ide → :8000 (JARVIS Server, body/) → :8765 (MCP, body/)
                                                   ├─ STDM (AST delta, O(1) context, memory/)
                                                   ├─ GGM (FAISS + MiniLM, memory/)
                                                   └─ ACSC (self-critique loop, memory/)
```

## Quick Start

### JARVIS Server (OpenAI-compatible)
```bash
cd body
FSTNET_MODEL=../brain/checkpoints/800m/best.pt python3 fstnet_server.py
curl -X POST localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Sir, check the system load."}],"chat_id":"chat1"}'
```

### MCP Server
```bash
cd body
python3 mcp_full.py --port 8765
curl -X POST localhost:8765/mcp \
  -d '{"tool":"ggm_search","chat_id":"chat1","query":"quicksort"}'
```

## Dataset: JARVIS (synthetic, 40/25/20/15)

```bash
cd brain
python3 build_jarvis_data.py --count 60000
```

Mix: 40% coding · 25% reasoning/CoT · 20% tool-calling (`<tool_call>{json}</tool_call>`) · 15% persona (Sir, witty, concise). Each dialog carries the JARVIS system prompt; loss is masked to assistant turns only (multiple `<tool_call>` segments supported).

## Training (Google Colab)

```bash
cd brain
!pip install -q tokenizers tqdm
!python3 build_jarvis_data.py --count 60000
!FSTNET_EPOCHS=5 FSTNET_LR=3e-4 python3 train_colab_800m.py
```

3B 1-bit MoF (Stage 1 + Stage 2) — полный авто-запуск:
```bash
!wget -q https://raw.githubusercontent.com/MrModelOS/fstnet/master/colab_run_full.py
%run colab_run_full.py
```

Optimizations: Drive auto-mount + SSD cache (`.npz`/`.pt`), fp16 on T4 / bf16 on Ampere+, grad-checkpointing, vectorized MoF fields, subsample 100k/эпоху, batch 2×accum 32 (eff 64), seq 512, optional `FSTNET_COMPILE=1`. Checkpoints always duplicated to `MyDrive/fstnet/checkpoints/`.

## Model Config

| Config | Params | d_model | layers | ctx |
|--------|--------|---------|--------|-----|
| `config_800m.py` | 956M | 1536 | 24 | 2048 |
| `config_3b_mof.py` | 3.39B (1-bit ~424MB) | 2048 | 32 | 4096 |

## Local Inference (Arch + Vulkan)

```bash
cd llama.cpp && cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
./build/bin/llama-server -m fstnet-800m-q8.gguf -ngl 99 --ctx-size 2048 -t 4 --memory-f16
```

Quantization: Q8_0 (~547MB) recommended; Q5_K_M acceptable.

## Components

- **MCP Server** (`body/mcp_full.py`) — session isolation, CRUD, context injection
- **GGM** (`memory/ggm.py`) — FAISS vector search, MiniLM embeddings
- **STDM** (`memory/stdm.py`) — AST-based delta memory
- **ACSC** (`memory/acsc.py`) — generate → test → refine
- **JARVIS Server** (`body/fstnet_server.py`) — OpenAI-compatible API + GGM injection
- **GGUF convert** (`brain/convert_to_gguf.py`) — checkpoint → Ollama/llama.cpp

## License

MIT
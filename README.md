# FST-Net 152M

Local-first AI code assistant with MCP (Model Context Protocol), GGM (Graph-Gated Memory), STDM (State-Tree Delta Memory), and ACSC (Async Self-Critique).

## Architecture

```
IDE/OpenCode → :8000 (FST-Net Server) → :8765 (MCP)
                                              ├─ STDM (AST delta O(1))
                                              ├─ GGM (FAISS + MiniLM)
                                              └─ ACSC (self-critique)
```

## Quick Start

### Ollama (easiest)
```bash
ollama run fstnet "def quicksort(arr):"
```

### API Server
```bash
python3 fstnet_server.py
curl -X POST localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"write fibonacci"}],"chat_id":"chat1"}'
```

### MCP Server
```bash
python3 mcp_full.py --port 8765
curl -X POST localhost:8765/mcp \
  -d '{"tool":"ggm_search","chat_id":"chat1","query":"quicksort"}'
```

## Training (Google Colab T4)

```python
!git clone https://github.com/MrModelOS/fstnet.git
%cd fstnet
!pip install -q transformers datasets tokenizers tqdm
!python3 train_colab_152m.py
```

## Model Configs

| Config | Params | d_model | layers | Use case |
|--------|--------|---------|--------|----------|
| `config.py` | 33M | 768 | 1 | Testing |
| `config_100m.py` | 94M | 1024 | 4 | Light |
| `config_150m.py` | 134M | 1024 | 6 | Balanced |
| `config_152m.py` | 151M | 1024 | 7 | **Recommended** |

## Local Inference (Arch/Hyprland + Vulkan)

```bash
# Build llama.cpp with Vulkan
cd llama.cpp && cmake -B build -DGGML_VULKAN=ON && cmake --build build -j

# Run with Q8_0 quant
./build/bin/llama-server -m fstnet-152m-q8.gguf -ngl 99 --ctx-size 2048 -t 4 --memory-f16

# Or with nice (low CPU priority)
nice -n 10 ionice -c 3 ./build/bin/llama-server -m fstnet-152m-q8.gguf -ngl 99
```

## Quantization

| Quant | Size | Quality | Recommended |
|-------|------|---------|-------------|
| Q8_0 | ~160MB | ~FP16 | ✅ Yes |
| Q5_K_M | ~100MB | Good | ⚠️ Acceptable |
| Q4_K | ~80MB | Degraded | ❌ No |

## Components

- **MCP Server** (`mcp_full.py`) — session isolation, CRUD, context injection
- **GGM** — 300K nodes, FAISS vector search, MiniLM embeddings
- **STDM** (`stdm.py`) — AST-based delta memory, O(1) context
- **ACSC** (`acsc.py`) — async self-critique + adversarial testing
- **FST-Net Server** (`fstnet_server.py`) — OpenAI-compatible API

## Datasets

- CodeAlpaca (code generation)
- SlimOrca (reasoning/CoT)
- UltraChat (dialogue)
- GSM8K (math)

## License

MIT

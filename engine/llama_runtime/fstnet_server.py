#!/usr/bin/env python3
"""FST-Net / JARVIS Inference Server with MCP integration.

API:
  POST /v1/chat/completions — OpenAI-compatible endpoint
  POST /v1/generate — raw generation
  GET  /health

Env:
  FSTNET_MODEL=<path.pt>   чекпоинт (по умолч. checkpoints/800m/best.pt)
  FSTNET_PORT=<port>       порт (по умолч. 8000)
"""
import os, json, time, requests, sys
import torch, torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from tokenizers import Tokenizer

import uvicorn

# ── Paths (перенесено в brain/) ────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "brain"))
TOKENIZER_PATH = os.path.join(_HERE, "..", "brain", "tokenizer", "fst_bpe.json")

# ── Config ──────────────────────────────────────────────
MODEL_PATH = os.environ.get("FSTNET_MODEL", "checkpoints/800m/best.pt")
MCP_URL = "http://localhost:8765/mcp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = 256
PORT = int(os.environ.get("FSTNET_PORT", "8000"))

# ── Load model ──────────────────────────────────────────
print(f"Loading model from {MODEL_PATH}...", flush=True)
from config_800m import FSTConfig800M
from model.core import FSTNetCore

ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
config = ckpt.get("config", FSTConfig800M())
model = FSTNetCore(config)
sd = {k: v for k, v in ckpt["model_state"].items() if "causal_mask" not in k}
model.load_state_dict(sd, strict=False)
model = model.to(DEVICE).eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {DEVICE}", flush=True)

tok = Tokenizer.from_file(TOKENIZER_PATH)
tok.post_processor = None
bos_id = tok.token_to_id("<bos>")
im_end_id = tok.token_to_id("<|im_end|>")
eos_id = tok.token_to_id("<eos>")

# ── MCP Client ──────────────────────────────────────────
def mcp_search(chat_id: str, query: str, top_k: int = 2) -> str:
    """Get context from MCP/GGM."""
    try:
        r = requests.post(MCP_URL, json={
            "tool": "ggm_search",
            "chat_id": chat_id,
            "query": query,
            "top_k": top_k
        }, timeout=2)
        data = r.json()
        results = data.get("results", [])
        if not results:
            return ""
        parts = []
        for h in results:
            parts.append(f"### {h.get('concept', '')}\n{h.get('content', '')}")
        return "\n\n".join(parts)
    except Exception:
        return ""

def mcp_insert(chat_id: str, concept: str, content: str):
    """Store fact in MCP/GGM."""
    try:
        requests.post(MCP_URL, json={
            "tool": "ggm_insert",
            "chat_id": chat_id,
            "concept": concept,
            "content": content
        }, timeout=2)
    except Exception:
        pass

# ── Generation ──────────────────────────────────────────
def generate(prompt_ids: List[int], max_new: int = MAX_NEW, temp: float = 0.5, top_k: int = 30) -> List[int]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = input_ids.clone()
    stop_tokens = {t for t in (im_end_id, eos_id) if t is not None}

    for _ in range(max_new):
        if generated.shape[1] > config.max_seq_len:
            context = generated[:, -config.max_seq_len:]
        else:
            context = generated

        with torch.no_grad():
            logits, _ = model(context, target_cycles=6)
            next_logits = logits[:, -1, :] / max(temp, 1e-3)

            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < min_val, float("-inf"))

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() in stop_tokens:
            break

    return generated[0, input_ids.shape[1]:].tolist()

# ── App ─────────────────────────────────────────────────
app = FastAPI(title="FST-Net")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.5
    top_k: int = 30
    chat_id: str = "default"
    use_ggm: bool = True

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.5

@app.get("/health")
def health():
    return {"ok": True, "model": "JARVIS (FST-Net 800M)", "device": DEVICE}

JARVIS_SYSTEM = (
    "You are JARVIS, an ultra-competent, loyal, polite, and witty AI assistant. "
    "You address the user as 'Sir', speak concisely, and assist with coding, system "
    "controls, research, and self-refinement. Use <tool_call>{...}</tool_call> JSON when "
    "an action or environment query is required; a system result will follow, then "
    "answer the user plainly."
)

@app.post("/v1/chat/completions")
def chat_completion(req: ChatRequest):
    t0 = time.time()

    system_parts = [JARVIS_SYSTEM]
    user_query = ""

    for msg in req.messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            user_query = msg.content

    # GGM context injection
    ggm_context = ""
    if req.use_ggm and user_query:
        ggm_context = mcp_search(req.chat_id, user_query, top_k=2)

    # Assemble ChatML prompt
    prompt_text = ""
    if ggm_context:
        prompt_text += f"<|im_start|>system\nMemory (GGM):\n{ggm_context}<|im_end|>\n"
    prompt_text += f"<|im_start|>system\n{system_parts[-1]}<|im_end|>\n"
    prompt_text += f"<|im_start|>user\n{user_query}<|im_end|>\n"
    prompt_text += "<|im_start|>assistant\n"

    ids = [bos_id] + tok.encode(prompt_text).ids
    out_ids = generate(ids, req.max_tokens, req.temperature, req.top_k)
    response = tok.decode(out_ids).strip()

    # Store response in GGM
    if req.use_ggm:
        mcp_insert(req.chat_id, user_query[:50], response[:200])

    return {
        "id": f"jarvis-{int(time.time())}",
        "choices": [{"message": {"role": "assistant", "content": response}}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out_ids)},
        "duration": time.time() - t0
    }

@app.post("/v1/generate")
def generate_endpoint(req: GenerateRequest):
    t0 = time.time()
    prompt = f"<|im_start|>system\n{JARVIS_SYSTEM}<|im_end|>\n<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = [bos_id] + tok.encode(prompt).ids
    out_ids = generate(ids, req.max_tokens, req.temperature)
    response = tok.decode(out_ids).strip()
    return {"response": response, "tokens": len(out_ids), "duration": time.time() - t0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

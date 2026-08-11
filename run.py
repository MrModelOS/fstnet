"""FST-Net interactive + GGM retrieval.

Graph-Gated Memory интексирует локальные проекты и подмешивает релевантный
контекст в каждый запрос, снимая нагрузку по запоминанию фактов с весов модели.

Формат промпта с контекстом GGM:
  <|im_start|>user
  # Context from project knowledge base

  ### file.py — project
  [code snippet]

  ---
  {user question}<|im_end|>
  <|im_start|>assistant
"""
import os
import torch
from tokenizers import Tokenizer
from config import FSTConfig
from model.core import FSTNetCore

try:
    from memory.ggm import GraphGatedMemory, GGMConfig, TfidfSvdEmbedder, MiniLMEmbedder
except ImportError:
    from memory.ggm_stub import GraphGatedMemory
    GGMConfig = None
    TfidfSvdEmbedder = None
    MiniLMEmbedder = None

import numpy as np


def load_tokenizer(path: str = "tokenizer/fst_bpe.json"):
    tok = Tokenizer.from_file(path)
    tok.post_processor = None
    return tok


def load_model(checkpoint_path: str = None):
    config = FSTConfig()
    model = FSTNetCore(config)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_checkpoint_into(ckpt["model_state"])
        print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
    return model, config


def setup_ggm(
    dirs_to_index: list = None,
    use_minilm: bool = False,
    load_path: str = None,
) -> GraphGatedMemory:
    """Создаёт GGM и индексирует проекты, либо загружает с диска."""
    if load_path and os.path.isfile(load_path) and os.path.isfile(load_path + ".meta"):
        try:
            from ggm_tool import load_ggm
            ggm = load_ggm(load_path)
            print(f"GGM loaded from {load_path}", flush=True)
            return ggm
        except Exception as e:
            print(f"GGM load failed ({e}), reindexing...", flush=True)

    if GGMConfig is not None:
        cfg = GGMConfig()
        if use_minilm:
            emb = MiniLMEmbedder()
        else:
            emb = TfidfSvdEmbedder(cfg.d_model)
        ggm = GraphGatedMemory(cfg, embedder=emb)
    else:
        ggm = GraphGatedMemory(d_model=768)

    if dirs_to_index is None:
        dirs_to_index = [os.path.dirname(os.path.abspath(__file__))]

    print(f"GGM: indexing {len(dirs_to_index)} directories...", flush=True)
    n = ggm.index_directory(dirs_to_index, max_chunks_total=300_000)
    st = ggm.stats()
    print(f"GGM ready: {st['nodes']:,} chunks, {st['projects']} projects, "
          f"embedder={st['embedder']}", flush=True)
    return ggm


def interactive_loop(checkpoint_path: str = None, index_dirs: list = None, ggm_path: str = None):
    """Интерактивный цикл: GGM retrieval -> генерация."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model, config = load_model(checkpoint_path)
    model = model.to(device)
    model.eval()

    params, _ = model.count_parameters()
    print(f"Model: {params:,} params (FP32)")

    tokenizer = load_tokenizer(config.tokenizer_path)
    bos = tokenizer.token_to_id("<bos>")
    im_end_id = tokenizer.token_to_id(config.im_end)
    eos_id = tokenizer.token_to_id("<eos>")

    ggm = setup_ggm(index_dirs, load_path=ggm_path)
    use_context = True
    max_context_chars = 280   # ~60 tokens, оставляет место для вопроса и ответа в 256 окне

    print("=" * 55)
    print("FST-Net + GGM Interactive Session")
    print("Commands: /quit, /cycles N, /temp T, /topk K, /tokens N")
    print("          /ggm (stats), /index [dir], /ctx (toggle context)")
    print("=" * 55)

    cycles = 6
    temperature = 0.7
    top_k = 40
    max_new = 120

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input.startswith("/cycles "):
            cycles = int(user_input.split()[1])
            print(f"  cycles = {cycles}")
            continue
        if user_input.startswith("/temp "):
            temperature = float(user_input.split()[1])
            print(f"  temperature = {temperature}")
            continue
        if user_input.startswith("/topk "):
            top_k = int(user_input.split()[1])
            print(f"  top_k = {top_k}")
            continue
        if user_input.startswith("/tokens "):
            max_new = int(user_input.split()[1])
            print(f"  max_new_tokens = {max_new}")
            continue
        if user_input == "/ggm":
            st = ggm.stats()
            for k, v in st.items():
                print(f"  {k}: {v}")
            continue
        if user_input.startswith("/index"):
            dirs = index_dirs
            if len(user_input.split()) > 1:
                dirs = [user_input.split(" ", 1)[1].strip()]
            print(f"  reindexing {dirs}...", flush=True)
            ggm = setup_ggm(dirs)
            continue
        if user_input == "/ctx":
            use_context = not use_context
            print(f"  context injection = {use_context}")
            continue

        # ----- GGM retrieval -----
        context_text = ""
        if use_context and ggm.size > 0:
            hits = ggm.query(user_input, top_k=2, graph_expand=1)
            if hits:
                parts = []
                total_len = 0
                for h in hits:
                    meta = h.get("meta", {})
                    loc = meta.get("file", "") or os.path.basename(meta.get("path", ""))
                    snippet = h["content"].strip()
                    block = f"### {loc}\n{snippet}"
                    if total_len + len(block) > max_context_chars:
                        remain = max_context_chars - total_len
                        if remain > 100:
                            parts.append(block[:remain])
                        break
                    parts.append(block)
                    total_len += len(block)
                if parts:
                    context_text = "# Context from project knowledge base\n\n" + "\n\n".join(parts)

        # ----- build prompt -----
        if context_text:
            user_block = f"{context_text}\n\n---\n{user_input}"
        else:
            user_block = user_input

        prompt = f"{config.im_start}user\n{user_block}{config.im_end}\n{config.im_start}assistant\n"
        ids = [bos] + tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids, used_cycles = model.generate(
                input_ids,
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=top_k,
                target_cycles=cycles,
                eos_ids=(im_end_id, eos_id),
            )

        gen_ids = output_ids[0, input_ids.shape[1]:].tolist()
        generated = tokenizer.decode(gen_ids)

        print(f"\nFST-Net [{used_cycles} cycles, T={temperature}, k={top_k}]:")
        print(generated.strip())

        # добавляем диалог в GGM для будущих обращений
        ggm.add_node(
            ggm.embedder.embed(user_input + "\n" + generated[:500]),
            content=f"Q: {user_input}\nA: {generated[:500]}",
            meta={"type": "conversation"},
        )


if __name__ == "__main__":
    import sys

    idx_dirs = None
    ckpt = None
    ggm_path = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--ckpt" and i + 1 < len(sys.argv):
            ckpt = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--index" and i + 1 < len(sys.argv):
            idx_dirs = [sys.argv[i + 1]]
            i += 2
        elif sys.argv[i] == "--ggm" and i + 1 < len(sys.argv):
            ggm_path = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    interactive_loop(checkpoint_path=ckpt, index_dirs=idx_dirs, ggm_path=ggm_path)

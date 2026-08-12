"""Конвертер FST-Net (PyTorch) → GGUF для Ollama.

Поддерживает LLaMA-подобную архитектуру FST-Net:
  - RoPE, SwiGLU, GQA, RMSNorm
  - Tied embeddings (head.weight = embedding.weight)

Запуск:
  python3 convert_to_gguf.py checkpoints/100m/step_500.pt -o fstnet-51m-q8.gguf --q8
  python3 convert_to_gguf.py checkpoints/final/final.pt -o fstnet-51m.gguf

Для Ollama:
  ollama create fstnet -f Modelfile
"""
import os
import sys
import json
import numpy as np
import torch


def load_fstnet(path):
    """Загружает FST-Net checkpoint и возвращает (state_dict, config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]
    cfg = ckpt.get("config", None)
    return sd, cfg


def gguf_name_map_fstnet():
    """Маппинг имён тензоров FST-Net → GGUF (LLaMA-стиль)."""
    return {
        "embedding.weight": "token_embd.weight",
        "norm_out.weight": "output_norm.weight",
        "head.weight": "output.weight",
        "fractal_layers.{i}.attn.wq.weight": "blk.{i}.attn_q.weight",
        "fractal_layers.{i}.attn.wk.weight": "blk.{i}.attn_k.weight",
        "fractal_layers.{i}.attn.wv.weight": "blk.{i}.attn_v.weight",
        "fractal_layers.{i}.attn.wo.weight": "blk.{i}.attn_output.weight",
        "fractal_layers.{i}.mlp.w1.weight": "blk.{i}.ffn_gate.weight",
        "fractal_layers.{i}.mlp.w2.weight": "blk.{i}.ffn_down.weight",
        "fractal_layers.{i}.mlp.w3.weight": "blk.{i}.ffn_up.weight",
        "fractal_layers.{i}.norm1.weight": "blk.{i}.attn_norm.weight",
        "fractal_layers.{i}.norm2.weight": "blk.{i}.ffn_norm.weight",
    }


def convert_to_gguf(checkpoint_path, output_path, quantize=None, vocab_path=None):
    try:
        from gguf import GGUFWriter
    except ImportError:
        print("Need: pip install gguf")
        sys.exit(1)

    sd, cfg = load_fstnet(checkpoint_path)

    # determine config
    vocab_size = 32770
    d_model = 768
    n_heads = 12
    n_kv_heads = 4
    d_ff = 3072
    n_layers = 3
    max_seq_len = 256

    if cfg is not None:
        vocab_size = getattr(cfg, "vocab_size", vocab_size)
        d_model = getattr(cfg, "d_model", d_model)
        n_heads = getattr(cfg, "n_heads", n_heads)
        n_kv_heads = getattr(cfg, "n_kv_heads", n_kv_heads)
        d_ff = getattr(cfg, "d_ff", d_ff)
        n_layers = getattr(cfg, "n_layers", n_layers)
        max_seq_len = getattr(cfg, "max_seq_len", max_seq_len)

    head_dim = d_model // n_heads

    print(f"FST-Net: {d_model}d, {n_layers}L, {n_heads}H ({n_kv_heads} KV), "
          f"{d_ff}ff, {vocab_size}vocab, {max_seq_len}ctx", flush=True)

    # detect tied embeddings: if head.weight missing in sd, use embedding.weight
    has_separate_head = "head.weight" in sd
    if not has_separate_head and "embedding.weight" in sd:
        sd["head.weight"] = sd["embedding.weight"].clone()
        print("  tied: head.weight = embedding.weight", flush=True)

    writer = GGUFWriter(output_path, "llama")

    # metadata
    writer.add_block_count(n_layers)
    writer.add_context_length(max_seq_len)
    writer.add_embedding_length(d_model)
    writer.add_feed_forward_length(d_ff)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)

    # tokenizer
    if vocab_path and os.path.isfile(vocab_path):
        with open(vocab_path) as f:
            tok = json.load(f)
        model = tok.get("model", {})
        vocab = model.get("vocab", {})
        merges = model.get("merges", [])
        if isinstance(vocab, dict):
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            tokens = [t for t, _ in sorted_vocab]
            # pad tokens to match embedding rows (vocab_size incl. specials)
            if len(tokens) < vocab_size:
                for extra in ("<|im_start|>", "<|im_end|>"):
                    if extra not in tokens:
                        tokens.append(extra)
                while len(tokens) < vocab_size:
                    tokens.append(f"<special_{len(tokens)}>")
            scores = [1.0] * len(tokens)
            writer.add_tokenizer_model("gpt2")
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_uint32("llama.vocab_size", len(tokens))
            # BPE merges (required by llama.cpp)
            if merges:
                # convert merges to strings
                merge_strs = [" ".join(m) for m in merges]
                writer.add_token_merges(merge_strs)
        # special tokens
        special = tok.get("added_tokens", [])
        special_ids = [t["id"] for t in special if "id" in t]
        if special_ids:
            # GGUF doesn't have clean special token API; add as user_defined
            pass

    # add tensors
    mapping = gguf_name_map_fstnet()

    added = 0
    for src, dst_template in mapping.items():
        if "{i}" not in dst_template:
            # not a per-layer template
            if src not in sd:
                continue
            tensor = sd[src].float().numpy()
            writer.add_tensor(dst_template, tensor)
            added += 1
        else:
            for i in range(n_layers):
                key = src.format(i=i)
                if key not in sd:
                    continue
                dst = dst_template.format(i=i)
                tensor = sd[key].float().numpy()
                writer.add_tensor(dst, tensor)
                added += 1

    print(f"  wrote {added} tensors", flush=True)

    # quantization
    if quantize == "q8_0":
        print("  quantizing Q8_0...", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(output_path) / 1024**2
    print(f"  saved: {output_path} ({size:.1f} MB)", flush=True)
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FST-Net → GGUF converter")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("-o", "--output", required=True, help="Output .gguf path")
    parser.add_argument("--quant", choices=["q8_0", "q4_0"], default=None)
    parser.add_argument("--vocab", default="tokenizer/fst_bpe.json")
    args = parser.parse_args()

    convert_to_gguf(args.checkpoint, args.output, args.quant, args.vocab)


if __name__ == "__main__":
    main()

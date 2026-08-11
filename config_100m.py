from dataclasses import dataclass

@dataclass
class FSTConfig100M:
    """100M parameter config for FST-Net."""
    vocab_size: int = 32768
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 4096
    n_layers: int = 4
    max_cycles: int = 6
    eps: float = 1e-3
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    tokenizer_path: str = "tokenizer/fst_bpe.json"
    use_swiglu: bool = True
    use_gqa: bool = True
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

# Calculate approximate param count
# embed: vocab * d_model = 32768 * 1024 = 33.5M
# per layer: attn (4 * d_model^2) + mlp (3 * d_model * d_ff) = 4*1M + 3*4M = 16M
# 4 layers: 64M
# norm_out + head: ~1M
# Total: ~98M

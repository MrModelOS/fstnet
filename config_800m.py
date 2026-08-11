from dataclasses import dataclass

@dataclass
class FSTConfig800M:
    """800M parameter config - autonomous code engineer."""
    vocab_size: int = 32770
    d_model: int = 1536
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 6144
    n_layers: int = 24
    max_cycles: int = 8
    eps: float = 1e-3
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0
    tokenizer_path: str = "tokenizer/fst_bpe.json"
    use_swiglu: bool = True
    use_gqa: bool = True
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

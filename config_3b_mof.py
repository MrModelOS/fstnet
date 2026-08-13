from dataclasses import dataclass


@dataclass
class FSTMoFConfig:
    vocab_size: int = 32770
    tokenizer_path: str = "tokenizer/fst_bpe.json"
    dim: int = 2048
    n_layers: int = 32
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    d_ff: int = 6144
    max_seq_len: int = 4096
    rope_base: float = 10000.0

    n_fields: int = 32
    field_rank: int = 64
    gating_top_k: int = 8
    alpha_temperature: float = 1.0
    orth_scale: float = 0.01
    quant_act: bool = False

    binarize_ratio: float = 1.0

    init_std: float = 0.02
    hidden_alpha: int = 512

    def block_bytes(self):
        attn = (2 * self.dim * self.dim
                + 2 * self.dim * (self.dim // self.n_kv_heads)
                + self.dim * self.dim)
        base = 3 * self.dim * self.d_ff
        fields = (3 * self.n_fields
                  * (self.dim + self.d_ff) * self.field_rank)
        hyper = self.dim * self.hidden_alpha + self.hidden_alpha * self.n_fields
        return attn + base + fields + hyper

    def total_params(self):
        return self.vocab_size * self.dim + self.n_layers * self.block_bytes()

    def bytes_1bit(self):
        return self.total_params() / 8

    def kv_cache_bytes(self, ctx=8192, dtype_bytes=2):
        kv_dim = self.n_kv_heads * self.head_dim
        return self.n_layers * ctx * kv_dim * dtype_bytes

    def active_params(self, gs=None):
        gs = gs or self.gating_top_k
        per = self.dim * self.dim * 2 + self.dim * self.dim // self.n_kv_heads + self.dim * self.dim
        base = 3 * self.dim * self.d_ff
        fields = 3 * gs * (self.dim + self.d_ff) * self.field_rank
        hyper = self.dim * self.hidden_alpha + self.hidden_alpha * self.n_fields
        return self.vocab_size * self.dim + self.n_layers * (per + base + fields + hyper)


if __name__ == "__main__":
    c = FSTMoFConfig()
    tot = c.total_params()
    print(f"total params : {tot/1e9:.3f}B")
    print(f"1-bit storage: {c.bytes_1bit()/1e6:.0f}MB")
    for gs in (4, 8, 16):
        print(f"active @ GS={gs:>2}: {c.active_params(gs)/1e9:.3f}B")
    print(f"KV 8k fp16: {c.kv_cache_bytes()/1e6:.0f}MB")

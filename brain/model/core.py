import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import FSTConfig


def apply_rotary_pos_emb(x, cos, sin):
    """RoPE: (B, T, n_heads, head_dim) -> rotate half-dim pairs."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class RoPEMultiheadAttention(nn.Module):
    """Grouped-Query Attention с RoPE (Llama-style)."""

    def __init__(self, config: FSTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * self.n_heads == config.d_model
        assert config.n_heads % self.n_kv_heads == 0
        self.n_rep = config.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _precompute_freqs(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, self.inv_freq) / math.sqrt(config_rope_scaling(self))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape

        cos, sin = self._precompute_freqs(T, x.device)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE: (B, T, heads, dim) -> (B, heads, T, dim)
        q = apply_rotary_pos_emb(q, cos[None, :, None, :], sin[None, :, None, :]).transpose(1, 2)
        k = apply_rotary_pos_emb(k, cos[None, :, None, :], sin[None, :, None, :]).transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: repeat kv heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Attention (SDPA: flash / math / mem-efficient)
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.dtype).unsqueeze(0).unsqueeze(0)
        is_causal = attn_mask is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask if attn_mask is not None else None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.dropout(self.wo(out))
        return out


def config_rope_scaling(module):
    cfg = getattr(module, "config", None)
    return getattr(cfg, "rope_scaling", 1.0)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DynamicFractalBlock(nn.Module):
    def __init__(self, config: FSTConfig):
        super().__init__()
        self.config = config
        self.attn = RoPEMultiheadAttention(config)
        self.mlp = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.RMSNorm(config.d_model, eps=1e-5)
        self.norm2 = nn.RMSNorm(config.d_model, eps=1e-5)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((config.max_seq_len, config.max_seq_len), float("-inf")),
                diagonal=1,
            ),
        )

    def forward(self, x):
        T = x.shape[1]
        x = x + self.attn(self.norm1(x), attn_mask=self.causal_mask[:T, :T])
        x = x + self.mlp(self.norm2(x))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


class FSTNetCore(nn.Module):
    def __init__(self, config: FSTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.fractal_layers = nn.ModuleList(
            [DynamicFractalBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_out = RMSNorm(config.d_model, eps=1e-5)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, target_cycles=None, return_hidden=False):
        B, T = input_ids.shape
        h = self.drop(self.embedding(input_ids))

        max_c = target_cycles if target_cycles is not None else self.config.max_cycles

        # gradient checkpointing для экономии VRAM
        use_checkpoint = self.training and len(self.fractal_layers) > 1

        # проходим через N слоёв; каждый слой может сделать несколько циклов
        for layer_idx, layer in enumerate(self.fractal_layers):
            cycles_used = 0
            for cycle in range(max_c):
                cycles_used += 1
                h_prev = h

                if use_checkpoint:
                    from torch.utils.checkpoint import checkpoint
                    h = checkpoint(layer, h, use_reentrant=False)
                else:
                    h = layer(h)

                # dynamic early exit (только в eval)
                if not self.training and cycle > 2:
                    with torch.no_grad():
                        cos_sim = F.cosine_similarity(
                            h.flatten(1), h_prev.flatten(1), dim=-1
                        ).mean()
                        if (1.0 - cos_sim.item()) < self.config.eps:
                            break

        h = self.norm_out(h)
        if return_hidden:
            return h, cycles_used
        logits = self.head(h)
        return logits, cycles_used

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        target_cycles: int = 8,
        eos_ids: tuple = None,
    ):
        self.eval()
        generated = input_ids.clone()

        stop_ids = set(eos_ids or ())

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len :]
            else:
                context = generated

            logits, cycles = self.forward(context, target_cycles=target_cycles)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k)
                min_val = top_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < min_val, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in stop_ids:
                break

        return generated, cycles

    def load_checkpoint_into(self, ckpt_state):
        """Загружает веса, расширяя эмбеддинги/head под новый vocab (добавленные токены)."""
        old_vocab = ckpt_state["embedding.weight"].shape[0]
        new_vocab = self.config.vocab_size

        if old_vocab != new_vocab:
            old_emb = ckpt_state["embedding.weight"]
            new_emb = torch.empty(new_vocab, self.config.d_model)
            torch.nn.init.normal_(new_emb, mean=0.0, std=0.02)
            new_emb[:old_vocab] = old_emb
            ckpt_state["embedding.weight"] = new_emb

        ckpt_state["head.weight"] = ckpt_state["embedding.weight"]
        self.load_state_dict(ckpt_state)
        return old_vocab, new_vocab

    def count_parameters(self):
        params = sum(p.numel() for p in self.parameters())
        vram_mb = (params * 2) / (1024**2)
        return params, vram_mb

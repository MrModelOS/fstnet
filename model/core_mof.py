import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_3b_mof import FSTMoFConfig


def ste_sign(x):
    return x + (torch.sign(x) - x).detach()


def quant_act8(x):
    rmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return (x / rmax * 127).round().clamp(-127, 127) / 127 * rmax


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, binarize=1.0, quant_in=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.binarize = binarize
        self.quant_in = quant_in
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -1.0 / in_features ** 0.5, 1.0 / in_features ** 0.5)
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        if self.quant_in:
            x = quant_act8(x)
        w = self.weight
        s = self.weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-12) if self.training else self.scale
        if self.training:
            self.scale.data.copy_(s.data.detach())
        wq = (1 - self.binarize) * w + self.binarize * (ste_sign(w) * s)
        return F.linear(x, wq, self.bias)


class ContinuousField(nn.Module):
    def __init__(self, in_features, out_features, n_fields, field_rank):
        super().__init__()
        self.U = nn.Parameter(torch.empty(n_fields, in_features, field_rank))
        self.V = nn.Parameter(torch.empty(n_fields, field_rank, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)

    def forward(self, x, alpha, indices):
        out = 0.0
        for k, i in enumerate(indices.tolist() if not indices.is_cuda else indices.cpu().tolist()):
            h = x @ self.U[i]
            out = out + alpha[..., k].unsqueeze(-1) * (h @ self.V[i])
        return out

    def orth_loss(self):
        n = self.U.shape[0]
        flat = self.U.reshape(n, -1, self.U.shape[2])
        gram_inner = torch.bmm(flat.transpose(1, 2), flat)
        target = torch.eye(self.U.shape[2], device=self.U.device)
        l_in = (gram_inner - target).square().mean()
        cross = []
        for i in range(n):
            for j in range(i + 1, n):
                cross.append((self.U[i].T @ self.U[j]).square().mean())
        l_cross = torch.stack(cross).mean() if cross else torch.zeros((), device=self.U.device)
        return l_in + l_cross


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(dim, max_seq_len, base=10000.0, device=None, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    t = torch.arange(max_seq_len, dtype=dtype)
    freqs = torch.outer(t, inv)
    return torch.stack((freqs.cos(), freqs.sin()), dim=-1).to(device)


def apply_rope(x, freqs):
    T = x.shape[1]
    f = freqs[:T]
    f = f.reshape(T, 1, -1, 2)
    xh = x.reshape(*x.shape[:-1], -1, 2)
    x_rot = torch.stack((xh[..., 0] * f[..., 0] - xh[..., 1] * f[..., 1],
                         xh[..., 0] * f[..., 1] + xh[..., 1] * f[..., 0]), dim=-1)
    return x_rot.reshape(*x.shape)


class GQAAttention(nn.Module):
    def __init__(self, cfg: FSTMoFConfig):
        super().__init__()
        self.bs = cfg
        self.Wq = BitLinear(cfg.dim, cfg.n_heads * cfg.head_dim)
        self.Wk = BitLinear(cfg.dim, cfg.n_kv_heads * cfg.head_dim)
        self.Wv = BitLinear(cfg.dim, cfg.n_kv_heads * cfg.head_dim)
        self.Wo = BitLinear(cfg.n_heads * cfg.head_dim, cfg.dim)

    def forward(self, x, freqs, mask=None, kv_cache=None):
        b, t, _ = x.shape
        q = self.Wq(x).view(b, t, self.bs.n_heads, self.bs.head_dim)
        k = self.Wk(x).view(b, t, self.bs.n_kv_heads, self.bs.head_dim)
        v = self.Wv(x).view(b, t, self.bs.n_kv_heads, self.bs.head_dim)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0], k), dim=1)
            v = torch.cat((kv_cache[1], v), dim=1)
            return q, k, v
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        rep = self.bs.n_heads // self.bs.n_kv_heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(rep, dim=1)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.bs.head_dim)
        if mask is not None:
            s = s + mask
        a = F.softmax(s, dim=-1)
        o = (a @ v).transpose(1, 2)
        o = o.reshape(b, t, -1)
        return self.Wo(o)


class Hypernet(nn.Module):
    def __init__(self, cfg: FSTMoFConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.hidden_alpha, bias=False)
        self.fc2 = nn.Linear(cfg.hidden_alpha, cfg.n_fields, bias=False)
        self.temp = cfg.alpha_temperature
        self.bs = cfg

    def forward(self, x, kv_cache=None):
        h = F.silu(self.fc1(x))
        logits = self.fc2(h) / self.temp
        probs = F.softmax(logits, dim=-1)
        if self.bs.gating_top_k >= self.bs.n_fields or not self.training:
            return probs, torch.arange(self.bs.n_fields, device=x.device)
        kth_vals, _ = probs.topk(self.bs.gating_top_k, dim=-1)
        keep = (probs >= kth_vals[..., -1:]).to(probs.dtype)
        kept = probs * keep
        kept = kept / kept.sum(-1, keepdim=True).clamp_min(1e-12)
        if kv_cache is not None:
            kv_cache.append(kept)
        return kept, None


class MoFFFN(nn.Module):
    def __init__(self, cfg: FSTMoFConfig):
        super().__init__()
        self.bs = cfg
        self.hyper = Hypernet(cfg)
        self.W0g = BitLinear(cfg.dim, cfg.d_ff)
        self.W0u = BitLinear(cfg.dim, cfg.d_ff)
        self.W0d = BitLinear(cfg.d_ff, cfg.dim)
        self.Fg = ContinuousField(cfg.dim, cfg.d_ff, cfg.n_fields, cfg.field_rank)
        self.Fu = ContinuousField(cfg.dim, cfg.d_ff, cfg.n_fields, cfg.field_rank)
        self.Fd = ContinuousField(cfg.d_ff, cfg.dim, cfg.n_fields, cfg.field_rank)

    def forward(self, x, kv_cache=None):
        a, _ = self.hyper(x, kv_cache)
        idx = torch.arange(self.bs.n_fields, device=x.device)
        g = F.silu(self.W0g(x) + self.Fg(x, a, idx))
        u = self.W0u(x) + self.Fu(x, a, idx)
        h = g * u
        return self.W0d(h) + self.Fd(h, a, idx)

    def orth_loss(self):
        return self.Fg.orth_loss() + self.Fu.orth_loss() + self.Fd.orth_loss()


class TransformerBlock(nn.Module):
    def __init__(self, cfg: FSTMoFConfig, layer_id: int):
        super().__init__()
        self.attn = GQAAttention(cfg)
        self.ffn = MoFFFN(cfg)
        self.n1 = RMSNorm(cfg.dim)
        self.n2 = RMSNorm(cfg.dim)

    def forward(self, x, freqs, mask=None, kv_cache=None):
        x = x + self.attn(self.n1(x), freqs, mask, kv_cache)
        x = x + self.ffn(self.n2(x), kv_cache)
        return x


class FSTMoFModel(nn.Module):
    def __init__(self, cfg: FSTMoFConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.head = BitLinear(cfg.dim, cfg.vocab_size)
        self.register_buffer("freqs", precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base))
        self.apply(self._init_)

    def _init_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, target=None):
        b, t = idx.shape
        x = self.tok_emb(idx)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1) if t > 1 else None
        for blk in self.blocks:
            x = blk(x, self.freqs, mask)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    def orth_loss(self):
        return sum(b.ffn.orth_loss() for b in self.blocks)

    def set_binarize(self, ratio):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.binarize = ratio


if __name__ == "__main__":
    cfg = FSTMoFConfig()
    m = FSTMoFModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    y = torch.randint(0, cfg.vocab_size, (1, 64))
    logits, loss = m(x, y)
    lo = m.orth_loss()
    tot = sum(p.numel() for p in m.parameters())
    print(f"loss={loss.item():.3f} orth={lo.item():.4f} params={tot/1e9:.3f}B logits={logits.shape}")
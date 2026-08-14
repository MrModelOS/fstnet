"""paged_kv.py — страничный (PagedAttention-style) KV-кэш для Native Llama Engine.

Проблема: у FST-Net `FSTMoFModel.generate()` пересчитывает ВЕСЬ контекст на каждый
новый токен (naивный вызов модели без KV). Это O(T^2) на декод-шаг.
Здесь — инкрементальный путь:
  - блоки фиксированного размера (pages): новый токен пишется в свободный слот
    без копирования истории;
  - опциональная INT8-квантизация K/V (8k-контекст: 134MB вместо 268MB в fp16);
  - ModelPagedRunner — инкрементальный generate поверх СУЩЕСТВУЮЩИХ субмодулей
    модели (внимание по страницам, остальные блоки вызываются как есть).

Память (config 3b_mof: n_kv_heads=4, head_dim=128, 32 слоя, ctx=8192):
  fp16 = 268MB, int8 = 134MB.
"""

import math

import torch

from model.core_mof import apply_rope


class PagedKV:
    """Страничный KV-кэш. Длина признака на слой: n_kv_heads * head_dim."""

    def __init__(self, n_layers, n_kv_heads, head_dim, ctx_len=8192,
                 page_size=64, dtype=torch.float16, quantize_int8=False,
                 device=None):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.token_len = n_kv_heads * head_dim
        self.page_size = max(1, page_size)
        self.dtype = dtype
        self.quantize_int8 = quantize_int8
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.core_dtype = torch.int8 if quantize_int8 else dtype
        self._pages = {}     # (layer, block) -> [k_page, v_page]
        self._free = []      # свободные block_id
        self._next_block = 0
        self._pos = 0
        self._max_blocks = math.ceil(ctx_len / self.page_size)

    def _alloc(self):
        if self._free:
            return self._free.pop()
        bid = self._next_block
        self._next_block += 1
        for layer in range(self.n_layers):
            self._pages[(layer, bid)] = [
                torch.zeros((self.page_size, self.token_len),
                            dtype=self.core_dtype, device=self.device),
                torch.zeros((self.page_size, self.token_len),
                            dtype=self.core_dtype, device=self.device),
            ]
        return bid

    def free_block(self, block_id):
        self._free.append(block_id)

    @property
    def n_blocks(self):
        return self._next_block - len(self._free)

    def mem_bytes(self):
        es = 1 if self.quantize_int8 else torch.empty((), dtype=self.dtype).element_size()
        return self.n_blocks * self.n_layers * 2 * self.page_size * self.token_len * es

    def append_layer(self, layer, k, v):
        """Дописывает ОДИН токен K/V для слоя layer. k/v — (n_kv_heads, head_dim)
        или (token_len,). Возвращает позицию либо None при переполнении ctx."""
        if self._pos >= self._max_blocks * self.page_size:
            return None
        block, slot = self._pos // self.page_size, self._pos % self.page_size
        if (layer, block) not in self._pages:
            self._alloc()
        kp, vp = self._pages[(layer, block)]
        kp[slot].copy_(k.reshape(-1).to(self.core_dtype, non_blocking=True))
        vp[slot].copy_(v.reshape(-1).to(self.core_dtype, non_blocking=True))
        return self._pos

    def append(self, k_list, v_list):
        """Дописывает пачку токенов по всем слоям (prefill).
        k_list[layer] — (t, n_kv_heads, head_dim). Возвращает начальную позицию."""
        n = k_list[0].shape[0]
        start = self._pos
        for t in range(n):
            for layer in range(self.n_layers):
                self.append_layer(layer, k_list[layer][t], v_list[layer][t])
            self._pos += 1
        return start

    def get(self, layer):
        """Контекст слоя: (k, v) тензоры (pos, n_kv_heads, head_dim) в исходном dtype."""
        pos = self._pos
        if pos == 0:
            return (torch.zeros((0, self.n_kv_heads, self.head_dim),
                                dtype=self.dtype, device=self.device),
                    torch.zeros((0, self.n_kv_heads, self.head_dim),
                                dtype=self.dtype, device=self.device))
        ks, vs = [], []
        for block in range(self._next_block):
            base = block * self.page_size
            if base >= pos:
                break
            take = min(self.page_size, pos - base)
            kp, vp = self._pages[(layer, block)]
            ks.append(kp[:take].to(self.dtype, non_blocking=True))
            vs.append(vp[:take].to(self.dtype, non_blocking=True))
        flat_k = torch.cat(ks, dim=0)
        flat_v = torch.cat(vs, dim=0)
        return (flat_k.view(-1, self.n_kv_heads, self.head_dim),
                flat_v.view(-1, self.n_kv_heads, self.head_dim))

    def clear(self):
        self._pages.clear()
        self._free.clear()
        self._next_block = 0
        self._pos = 0


class ModelPagedRunner:
    """Инкрементальный generate поверх существующих весов FST-Net.

    prefill() кодирует промпт и заполняет страницы K/V каждого слоя.
    forward_token() декодирует один токен, опираясь только на кэш (без
    пересчёта истории). Остальные слои (Norm / MoFFFN / ContinuousField)
    вызываются как есть — математика не дублируется.
    """

    def __init__(self, model, ctx_len=8192, page_size=64, quantize_int8=False):
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        self.kv = PagedKV(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim,
                          ctx_len=ctx_len, page_size=page_size, dtype=dtype,
                          quantize_int8=quantize_int8, device=device)

    @torch.no_grad()
    def prefill(self, idx):
        """Кодирует idx (b, t), заполняет кэш. Возвращает скрытое состояние."""
        self.kv.clear()
        model = self.model
        b, t = idx.shape
        x = model.tok_emb(idx)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device,
                                     dtype=x.dtype), diagonal=1) if t > 1 else None
        k_list, v_list = [], []
        for blk in model.blocks:
            xn = blk.n1(x)
            k = blk.attn.Wk(xn).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim)
            v = blk.attn.Wv(xn).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim)
            k_rope = apply_rope(k, self.model.freqs)
            k_list.append(k_rope[0])
            v_list.append(v[0])
            q = blk.attn.Wq(xn).view(b, t, self.cfg.n_heads, self.cfg.head_dim)
            q = apply_rope(q, self.model.freqs)
            x = x + blk.attn.Wo(self._attend(q, k_rope, v, mask))
            x = x + blk.ffn(blk.n2(x))
        self.kv.append(k_list, v_list)
        return x

    @torch.no_grad()
    def forward_token(self, idx):
        """idx — (b, 1) следующий токен. Возвращает logits (b, 1, vocab)."""
        model = self.model
        b = idx.shape[0]
        x = model.tok_emb(idx)
        for layer, blk in enumerate(model.blocks):
            xn = blk.n1(x)
            q = blk.attn.Wq(xn).view(b, 1, self.cfg.n_heads, self.cfg.head_dim)
            k = blk.attn.Wk(xn).view(b, 1, self.cfg.n_kv_heads, self.cfg.head_dim)
            v = blk.attn.Wv(xn).view(b, 1, self.cfg.n_kv_heads, self.cfg.head_dim)
            # RoPE для новой позиции
            pos = self.kv._pos
            q = apply_rope(q, self.model.freqs[pos:pos + 1])
            k = apply_rope(k, self.model.freqs[pos:pos + 1])
            # кэш (прошлые токены) + текущий (query-позиция)
            kc, vc = self.kv.get(layer)
            kc = kc.unsqueeze(0); vc = vc.unsqueeze(0)
            k_all = torch.cat((kc, k), dim=1)
            v_all = torch.cat((vc, v), dim=1)
            o = self._attend(q, k_all, v_all)
            x = x + blk.attn.Wo(o)
            x = x + blk.ffn(blk.n2(x))
            # дописываем текущий K/V в кэш ПОСЛЕ внимания (все слои — один слот)
            self.kv.append_layer(layer, k[0, 0], v[0, 0])
        self.kv._pos += 1  # один токен — один шаг по позиции
        x = model.norm(x)
        return model.head(x)

    def _attend(self, q, k, v, mask=None):
        b, t, h, hd = q.shape
        rep = self.cfg.n_heads // self.cfg.n_kv_heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(rep, dim=1)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if mask is not None:
            s = s + mask
        a = torch.softmax(s, dim=-1)
        return (a @ v).transpose(1, 2).reshape(b, t, -1)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=128, temperature=0.8, top_k=50,
                 top_p=0.9, stop_tokens=None):
        stop = set(stop_tokens or [])
        self.prefill(prompt_ids)
        out = list(prompt_ids[0].tolist())
        dev = prompt_ids.device
        cur = prompt_ids[:, -1:]
        for _ in range(max_new):
            logits = self.forward_token(cur)[0, 0]
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = probs.sort(descending=True)
                cum = sp.cumsum(dim=-1) - sp
                keep = cum <= top_p
                sp, si = sp[keep], si[keep]
                sp = sp / sp.sum()
                nxt = si[sp.multinomial(1, replacement=True)]
            else:
                nxt = probs.multinomial(1, replacement=True)
            nxt = nxt.item()
            out.append(nxt)
            cur = torch.tensor([[nxt]], device=dev, dtype=torch.long)
            if nxt in stop:
                break
        return out
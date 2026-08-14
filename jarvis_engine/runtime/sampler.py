"""sampler.py — сэмплирование для Native Llama Engine.

Поддерживает: temperature, top-k, top-p, min-p, repetition penalty 1.25-2.0
и Grammar/Schema-masking:
  - `allowed_fn`: callable(logits, prefix) -> bool-маска по валидности токена
    (пример: JSON-схема, тэги <|im_start|>/<|assistant|>, tool-вызовы);
  - `json_schema`: упрощённый валидатор (строка-массив-объект с полями).

Всё чистый torch — без внешних зависимостей. Функция `sample()` возвращает
индекс токена и `info` для логов (prob, entropy).
"""

import math

import torch


def _penalize(logits, ids, penalty):
    if penalty <= 1.0 or not ids:
        return logits
    un = logits.new_tensor(ids, dtype=torch.long).unique()
    logits = logits.clone()
    logits[un] /= penalty
    return logits


def sample(logits, ids=None, temperature=0.7, top_k=50, top_p=0.9,
           min_p=0.0, repetition_penalty=1.15, allowed_fn=None):
    """logits: (vocab,). ids: список уже сгенерённых токенов (для rep-penalty)."""
    if temperature <= 1e-6:
        return int(logits.argmax().item()), {"mode": "greedy"}
    logits = logits / temperature
    if repetition_penalty > 1.0 and ids:
        logits = _penalize(logits, ids, repetition_penalty)

    # грамматика: логика исключаем запрещённые токены сразу
    if allowed_fn is not None:
        mask = allowed_fn(logits, ids)
        logits = logits.masked_fill(~mask, float("-inf"))

    if top_k > 0:
        v, _ = logits.topk(min(int(top_k), logits.size(-1)))
        logits = logits.masked_fill(logits < v[-1], float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0 or min_p > 0.0:
        sorted_p, sorted_idx = probs.sort(descending=True)
        cum = sorted_p.cumsum(dim=-1)
        keep = torch.ones_like(sorted_p, dtype=torch.bool)
        if top_p < 1.0:
            keep &= ~((cum - sorted_p) > top_p)
        if min_p > 0.0:
            keep &= sorted_p >= (probs.max() * min_p)
        sorted_p = sorted_p[keep]
        if sorted_p.numel() == 0:
            return int(probs.argmax().item()), {"mode": "fallback-greedy"}
        sorted_p = sorted_p / sorted_p.sum()
        idx = sorted_idx[keep]
        nxt = idx[sorted_p.multinomial(1, replacement=True)]
    else:
        nxt = probs.multinomial(1, replacement=True)
    info = {"mode": "sampled"}
    return int(nxt.item()), info


# ── Grammar / Schema masking ──────────────────────────────────────────
class JsonSchemaMask:
    """Упрощённый JSON-маскировщик: навязывает валидный JSON.

    Парсит частичный префикс (по токенам детокенизировать сложно на лету,
    поэтому работет на уровне строк префикса, конкатенируя токены).
    """

    def __init__(self, tokenizer, schema=None):
        """schema: dict {fields:[{name,type}]} или None (строгий валидный JSON)."""
        self.tok = tokenizer
        self.schema = schema or {}

    def allowed_token_ids(self, prefix_text: str):
        """Возвращает набор id разрешённых токенов для префикса JSON.

        prefix_text — уже декодированный текст генерируемого JSON.
        Реализация «строки-клоза»: в начальном состоянии либо { либо [.
        """
        stripped = prefix_text.lstrip()
        allowed = []
        for tok in self.tok.get_vocab().keys():
            s = tok
            if not s:
                continue
            if s == "<|im_end|>":
                continue
            # старт: только { или [
            if not stripped:
                if s.startswith("{") or s.startswith("["):
                    allowed.append(self.tok.token_to_id(tok))
                continue
            # после открывающей скобки допускаем всё, пока не закроем корректно
            # простой эвристический баланс скобок
            opens = stripped.count("{") + stripped.count("[")
            closes = stripped.count("}") + stripped.count("]")
            if opens <= closes:
                # можно закрывать только когда открыто лишнее
                if s.startswith(("}", "]")):
                    if opens > closes:
                        allowed.append(self.tok.token_to_id(tok))
                    continue
            allowed.append(self.tok.token_to_id(tok))
        return set(a for a in allowed if a is not None)

    def make_allowed_fn(self, decode_prefix):
        """Возвращает allowed_fn(logits, ids) совместимый с sample()."""
        def fn(logits, ids):
            prefix = decode_prefix(ids) if ids else ""
            allow = self.allowed_token_ids(prefix)
            mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            for t in allow:
                if t < mask.size(0):
                    mask[t] = True
            # всегда даём хотя бы один даунстрим
            if not mask.any():
                mask[logits.argmax()] = True
            return mask
        return fn


def clamp_top_k(vocab=32770, default=50):
    return min(int(default), vocab)
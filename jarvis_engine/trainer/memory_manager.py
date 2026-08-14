"""memory_manager.py — борьба с OOM при обучении масштабных моделей.

Что делает:
- Gradient Checkpointing: пересчёт активаций на backward вместо их хранения
  (экономия до 60% VRAM ценой ~30% времени).
- Очистка VRAM-кеша между батчами (torch.cuda.empty_cache).
- Прогрев (warmup) без аккумулирования градиентов — для стабильного старта.
- Трассировка пикового потребления VRAM для логов.
"""

import os

import torch
import torch.nn as nn


class MemoryManager:
    def __init__(self, grad_ckpt: bool = False, empty_cache_every: int = 50,
                 device: str = "cuda"):
        self.device = device
        self.empty_cache_every = max(1, empty_cache_every)
        self.enabled = grad_ckpt
        self.peak_mb = 0
        self._steps_since_clean = 0

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Включает gradient checkpointing для всех трансформер-блоков.

        Использует torch.utils.checkpoint.checkpoint для каждого блока.
        Для FSTMoFModel блоки лежат в model.blocks (nn.ModuleList).
        """
        if not self.enabled:
            return model
        for blk in getattr(model, "blocks", []):
            blk.forward = _checkpointed_forward(blk)
        return model

    def before_backward(self):
        """Вызывается перед loss.backward(): принудительная очистка кеша."""
        self._steps_since_clean += 1
        if self._steps_since_clean >= self.empty_cache_every:
            self._steps_since_clean = 0
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def after_step(self):
        """После optim.step(): фиксирует пик VRAM и мягко очищает кеш."""
        if self.device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**2
            self.peak_mb = max(self.peak_mb, alloc)
            if self.peak_mb - alloc > 512:
                torch.cuda.empty_cache()

    def report(self) -> str:
        if self.device != "cuda":
            return ""
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return f"VRAM peak {self.peak_mb:.0f}MB / {total:.0f}MB"


def _checkpointed_forward(block: nn.Module):
    """Обёртка forward блока: активации не хранятся, считаются на backward."""
    import torch.utils.checkpoint as ckpt

    orig = block.__class__.forward

    def forward(self, x, freqs=None, mask=None, kv_cache=None):
        if self.training:
            return ckpt.checkpoint(lambda x_, f_, m_: orig(self, x_, f_, m_), x, freqs, mask)
        return orig(self, x, freqs, mask, kv_cache)

    block.forward = forward.__get__(block, type(block))
    return block.forward


def enable_if_env(device: str = "cuda") -> MemoryManager:
    """Создаёт MemoryManager.

    Gradient checkpointing ВКЛЮЧЁН по умолчанию (для 3.4B на T4 16GB веса+грады
    уже 13.6GB — активации хранить негде). Отключить: FSTNET_GRAD_CKPT=0.
    FSTNET_CKPT_EVERY — как часто чистить кеш (по умолчанию 50 шагов).
    """
    grad_ckpt = os.environ.get("FSTNET_GRAD_CKPT", "1").strip() not in ("0", "false", "no")
    every = int(os.environ.get("FSTNET_CKPT_EVERY", "50"))
    return MemoryManager(grad_ckpt=grad_ckpt, empty_cache_every=every, device=device)
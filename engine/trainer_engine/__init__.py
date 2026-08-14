"""trainer_engine — тренировочная машина FST-Net MoF.

Модули:
  ste_optimizer   — Adafactor (факторизованные состояния, STE-совместим).
  memory_manager  — gradient checkpointing + очистка VRAM-кеша между шагами.
  custom_trainer  — цикл обучения (S0-S2, resume, val, Drive-персист).
"""
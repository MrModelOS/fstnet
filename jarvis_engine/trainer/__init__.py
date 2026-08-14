"""jarvis_engine/trainer — Custom Trainer Engine (Блок 2).

Модули:
  ste_optimizer   — Adafactor (факторизованные состояния, STE-совместим).
  memory_manager  — gradient checkpointing + очистка VRAM-кеша между шагами.
  run_trainer     — многофазный цикл обучения S0→S2 (resume, val, Drive-персист).
"""
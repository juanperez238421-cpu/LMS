# BOT Feedback Training Workflow

## Objetivo
Convertir ejecuciones visuales del bot en dataset util para mejorar politicas de movimiento/click y reutilizar conocimiento en otros juegos.

## Artefactos de entrada

1. Stream JSONL:
- `reports/feedback_training/live/play_runtime_<timestamp>/feedback_stream.jsonl`
- `reports/feedback_training/smoke/smoke_<timestamp>/feedback_stream.jsonl`

2. Evidencia visual:
- `reports/feedback_training/**/screens/*.png`

Cada evento JSONL contiene:
- estado del bot (`bot_state`, `state_reason`)
- accion y resultado (`action`, `action_ok`, `active_keys`)
- feedback observado (`input_probe`, `click_probe`, `cursor_probe`)
- screenshot asociado (`feedback_screenshot`)

## Pipeline recomendado

1. **Ingestion**
- Unificar todos los JSONL en una tabla (`run_id`, `step_id`, `state`, `action`, `ok`, `features`).

2. **Labeling**
- Crear etiquetas binarias:
  - `success_enter_match`
  - `success_click_confirmed`
  - `success_key_feedback`
- Crear etiquetas continuas:
  - `time_to_enter_match_sec`
  - `click_latency_steps`

3. **Feature engineering**
- Features de feedback:
  - deltas `keyDown/keyUp`, `pointerDown/pointerUp`, `click0`, `cursor_moves`
- Features de contexto:
  - `bot_state`, `state_reason`, `last_event_signal`
- Features de accion:
  - tipo de accion (`move`, `click`, `lobby_select`)
  - key presionada

4. **Entrenamiento**
- Baseline inicial: arboles de decision o gradient boosting para predecir `action_ok`.
- Politica secuencial: contextual bandit o RL offline para escoger la siguiente accion segun feedback reciente.

5. **Validacion**
- Holdout por corrida completa (no mezclar pasos de la misma corrida entre train/test).
- KPI minimos:
  - +X% en `success_enter_match`
  - +X% en `success_click_confirmed`
  - -X% en tiempo lobby->in_match

6. **Generalizacion a otros juegos**
- Mantener esquema comun de dataset:
  - `state`, `action`, `feedback`, `outcome`
- Implementar adaptadores por juego:
  - detector de estado de lobby/in_match
  - mapeo de teclas/acciones
  - detector de confirmacion de input

## Ciclo de mejora continua

1. Ejecutar corrida dual con HUD y feedback.
2. Revisar outliers en screenshots + JSONL.
3. Ajustar parametros/heuristicas.
4. Repetir y comparar KPI.

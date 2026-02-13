# README Testing (Paso a paso)

Esta guia define como ejecutar pruebas del bot LMS y como guardar evidencia util para feedback y entrenamiento posterior.

## 1) Preparacion del entorno

1. Abrir PowerShell en la raiz del repo (`d:\Procastrinar\LMS`).
2. Ejecutar bootstrap:

```powershell
.\scripts\bootstrap_repo.ps1
```

3. O activar entorno ya creado:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2) Test 1 (base y rapido)

Objetivo: validar parser/base sin UI real.

```powershell
python -m pytest tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument -q
```

Criterio de exito:
- `1 passed`

## 3) Test 2 (archivo LMS completo con mocks)

Objetivo: validar flujo bot en pruebas automatizadas.

```powershell
python -m pytest tests/test_lms_live_collector_bot.py -q
```

## 4) Test 3 (suite completa)

Objetivo: validar estabilidad global.

```powershell
python -m pytest -q
```

## 5) Prueba visual real (Chrome visible)

### Auto fases (smoke -> play)
```powershell
.\scripts\run_lms_bot_auto.ps1
```

### Dual mode (juego + smoke paralelo)
```powershell
.\scripts\run_lms_bot_dual.ps1
```

Backend de decision opcional (replica RE):
```powershell
python lms_live_collector.py --play-game --bot-decision-backend lms_re --bot-lmsre-mode-name royale_mode --channel chrome
```

Backend de decision AlphaStar (checkpoint local):
```powershell
python lms_live_collector.py --play-game --bot-decision-backend alphastar --bot-alphastar-checkpoint artifacts/alphastar/pi_rl.pt --channel chrome
```

Script rapido para dual mode + AlphaStar:
```powershell
.\scripts\run_lms_bot_dual_alphastar.ps1
```

Modo exploracion AlphaStar (stochastic sampling):
```powershell
python lms_live_collector.py --play-game --bot-decision-backend alphastar --bot-alphastar-checkpoint artifacts/alphastar/pi_rl.pt --bot-alphastar-stochastic --bot-alphastar-temperature 1.15 --channel chrome
```

Protocolo estricto para agentes de menor capacidad (sin editar archivos):
- `docs/workflows/LOW_AGENT_ALPHASTAR_RUN_PROTOCOL.md`
- `docs/workflows/LOW_AGENT_TEST_ONLY_PROTOCOL.md` (solo tests, sin modificar codigo)

Rutina automatizada de tests repetidos para entrenamiento:
```powershell
.\scripts\run_training_test_routines.ps1 -CollectorRuns 20 -MathReplayRuns 10
```

Rutina intensiva (mas corridas, fail-fast):
```powershell
.\scripts\run_training_test_routines_high.ps1 -CollectorRuns 60 -MathReplayRuns 30
```

Run recomendado para esta version (hasta muerte, backend RE de `Game`):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lms_bot_dual.ps1 --bot-decision-backend lms_re --bot-lmsre-mode-name royale_mode --bot-run-until-end --bot-run-stop-on-death-only --bot-run-max-sec 900 --report-every-sec 20
```

Ambos scripts ahora activan:
- HUD visual en tiempo real (`--bot-debug-hud`)
- Captura periodica de screenshots de feedback
- OCR visual periodico sobre screenshots (`--bot-visual-ocr`)
- Guardado JSONL de feedback para entrenamiento
- Render MP4 de la corrida (`timeline.mp4`)

## 6) Donde queda la evidencia

1. Logs de ejecucion:
- `reports/live/*.log`
- `reports/smoke/*`
- `reports/smoke/bot_smoke_test_last.json` (incluye `kills`, `kill_history`, `last_death_cause`, `death_history`)

2. Feedback para entrenamiento:
- `reports/feedback_training/live/play_runtime_<timestamp>/feedback_stream.jsonl`
- `reports/feedback_training/live/play_runtime_<timestamp>/screens/*.png`
- `reports/feedback_training/live/play_runtime_<timestamp>/timeline.mp4`
- `reports/feedback_training/smoke/smoke_<timestamp>/feedback_stream.jsonl`
- `reports/feedback_training/smoke/smoke_<timestamp>/screens/*.png`

## 7) Señales clave que debes revisar

1. Teclas y reaccion:
- En el HUD: teclas `UP/LEFT/DOWN/RIGHT` en amarillo cuando se presionan.
- En logs: `[BOT][MOVE_SMOKE] key=...` y `[BOT] Moviendo en direccion: ...`
- En smoke: `[BOT][SMOKE] ... kills=<n> lastDeath=<causa>`
- En parallel smoke: `[BOT][PARALLEL_SMOKE] ... kills=<n> lastDeath=<causa>`

2. Feedback recibido:
- En el HUD: `Input kd/ku`, `Pointer down/up`, `Click click0`.
- En JSONL: `input_probe`, `click_probe`, `cursor_probe`.
 - En logs/HUD: `motion_score` para validar desplazamiento real y deteccion de atasco.
- En JSONL de smoke: `kills`, `kill_history`, `deaths`, `last_death_cause`, `last_death_source`, `death_history`.

3. Transicion de estado:
- `[BOT][STATE] lobby -> in_match`
- `[BOT][VISION] state_hint=... conf=... names=... damage_hint=...`
- En `feedback_stream.jsonl`: campo `visual_ocr` con estado visual, nombres y numeros detectados.
- En cierre de run hasta muerte: `[BOT][RUN] Fin detectado por muerte del bot. cause=... conf=...`
- Resumen final: `[BOT][RUN] Ejecucion completa finalizada. reason=death_event ... death=<causa>:<conf>`

## 8) Recomendacion para iterar entrenamiento

1. Ejecutar 5-10 corridas dual mode.
2. Agrupar JSONL + screenshots por corrida.
3. Medir tasa de:
- entrada a partida,
- deteccion de click valido,
- confirmacion de keydown/keyup,
- tiempo lobby->in_match.
4. Ajustar parametros (`--bot-ui-poll-ms`, `--bot-action-timeout-ms`, `--bot-smoke-move-hold-ms`) y repetir.
5. Para atascos, ajustar (`--bot-move-motion-threshold`, `--bot-move-stuck-streak`, `--bot-move-escape-steps`).
6. Para OCR visual, ajustar (`--bot-visual-ocr-every-sec`, `--bot-visual-ocr-max-frames`, `--bot-visual-ocr-max-names`).

## 9) Bindings soportados (bot + smoke)

- `W` move up (`KeyW`)
- `A` move left (`KeyA`)
- `S` move down (`KeyS`)
- `D` move right (`KeyD`)
- `Space` attack
- `Shift` sprint
- `R` build wall
- `C` open/close map
- `1` ability 1 (offense)
- `2` ability 2 (mobility)
- `3` ability 3 (defense)
- Right mouse button sprint (`MouseRight`)

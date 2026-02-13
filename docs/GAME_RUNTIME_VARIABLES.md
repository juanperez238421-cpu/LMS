# Game Runtime Variables (JS Probe)

Este documento resume la extraccion de variables relevantes del runtime JS del juego usando Playwright (equivalente a inspeccion tipo F12, sin modificar el juego).

## Objetivo

- Identificar variables y objetos globales utiles para telemetria del bot.
- Recolectar evidencia por fase (`lobby_bootstrap`, `in_match_entry`).
- Mantener snapshots JSON para entrenamiento y depuracion.

## Como se genera

Comando base:

```bash
python lms_live_collector.py \
  --play-game \
  --bot-runtime-probe \
  --bot-runtime-probe-dir reports/runtime_probe \
  --bot-runtime-probe-max-keys 180 \
  --channel chrome \
  --no-persistent
```

Archivos de salida:

- `reports/runtime_probe/lobby_bootstrap_<run_id>_<timestamp>.json`
- `reports/runtime_probe/in_match_entry_<run_id>_<timestamp>.json`

## Campos del snapshot

- `window_key_count`: total de globals observados.
- `runtime_vars[]`: lista priorizada de objetos/variables con hints (`player`, `enemy`, `mana`, `zone`, `safe`, `map`, etc.).
- `local_storage_keys[]`, `session_storage_keys[]`: llaves de storage detectadas.
- `canvas_count`, `iframe_count`: estado de render del runtime.

## Variables objetivo recomendadas para el bot

- Estado de partida: `match`, `lobby`, `round`, `state`.
- Recursos: `mana`, `hp`, `cooldown`.
- Combate: `enemy`, `damage`, `ability`.
- Macro: `zone`, `safe_zone`, `map`, `timer`.

## Uso en la UI Smoke

El monitor paralelo consume bridge en vivo (`window.__botSmokeParallelBridge`) con:

- `mana_current`, `mana_max`
- `zone_countdown_sec`, `safe_zone_x`, `safe_zone_y`, `safe_zone_radius`
- `enemy_detected`, `enemy_conf`, `enemy_dir`, `enemy_x_ratio`, `enemy_y_ratio`
- `map_name`, `bot_state`

Esto permite radar en tiempo real + contador de zona + recomendacion tactica en la interfaz.

## Resultados recientes (2026-02-11)

Batch validado: `reports/live/batch3_20260211_180055/`

Snapshots generados por corrida (2 por match):

- `lobby_bootstrap`
- `in_match_entry`

Hallazgos runtime (repetibles en las 3 corridas):

- `window_key_count`: ~1295
- `canvas_count`: 1
- `iframe_count`: 2
- objetos globales relevantes detectados:
  - `createUnityInstance`
  - `unityFramework`
  - `UnityProgress`
  - `__lmsBotHudState`
  - `__lmsBotCursorProbe`
  - `__lmsClickProbe`
  - `__lmsInputProbe`

Hallazgos de `localStorage` utiles para bot/control:

- `upKey`, `leftKey`, `downKey`, `rightKey`
- `attackKey`, `dashKey`, `specialAttackKey`
- `buildTurretKey`, `buildTrapKey`, `buildSpecialKey`, `wallKey`, `unbuildKey`
- `selectedAbility`, `selectedCharacter`, `selectedSkin`, `selectedPrefabSet`
- `desiredMode`, `photonRegion`, `userName`, `language`, `rmbAction`

Interpretacion:

- El runtime expone bootstrap Unity y preferencias de input/usuario en storage.
- Variables tacticas de combate (hp/mana/enemy/zone) no aparecen como globals estables directos en `window`; para decision en tiempo real sigue siendo mas confiable el pipeline actual: eventos + OCR + vision + bridge.

# LMS Metrics Workflow

## 1) Reprocesar HAR y minar metricas texto

```bash
python lms_events_collector.py \
  --db data/processed/sqlite/lms_events_step1.db \
  --har data/raw/har/lastmagestanding.com.har \
  --endpoint-substring /GameFunction/LogEvents

python lms_text_miner.py \
  --db data/processed/sqlite/lms_events_step1.db \
  --keywords "damage,dmg,kill,kills,elim,frag,craft,item,create" \
  --url-limit 30 \
  --sample-limit 40 \
  --json-path-limit 30 \
  --report-json reports/audit/step1_text_mining.json
```

## 2) Captura en vivo ampliada + minado offline

```bash
python lms_live_collector.py \
  --db data/processed/sqlite/lms_events.db \
  --capture-all-post \
  --capture-all-max-bytes 2000000 \
  --scan-websocket \
  --ws-save-frames \
  --ws-save-binary \
  --ws-print-keyword-hits
```

Luego minar:

```bash
python lms_text_miner.py \
  --db data/processed/sqlite/lms_events.db \
  --keywords "damage,dmg,kill,kills,elim,frag,craft,item,create"
```

## 3) OCR sobre screenshots (manual o integrado)

OCR manual desde imagen:

```bash
# Dependencias OCR:
#   pip install opencv-python numpy pytesseract
#   (opcional) pip install easyocr
#   instalar binario tesseract en el sistema

python lms_ocr_extractor.py \
  --input data/processed/ocr/match_end_sample.png \
  --config config/lms_ocr_rois.example.json \
  --json-only
```

OCR capturando con Playwright:

```bash
python lms_ocr_extractor.py \
  --playwright-url https://lastmagestanding.com/ \
  --playwright-selector canvas \
  --playwright-output data/processed/ocr/playwright_capture.png \
  --config config/lms_ocr_rois.example.json
```

OCR integrado al final de partida:

```bash
python lms_live_collector.py \
  --db data/processed/sqlite/lms_events.db \
  --ocr-on-match-end \
  --ocr-config config/lms_ocr_rois.example.json \
  --ocr-output-dir data/processed/ocr
```

## 4) Evidencia WS (sin reverse engineering)

```bash
python lms_ws_analyzer.py \
  --db data/processed/sqlite/lms_events.db \
  --limit 50 \
  --export-samples data/raw/ws/ws_samples \
  --export-count 20
```

Si `keyword_hit` sigue en `None` y `decoded_text` no trae JSON util, tratar WS como binario no explotable sin reverse engineering y priorizar OCR.

## 5) Bot web en 2 fases automaticas (smoke -> play)

Prueba deterministica de cursor/click primero y luego juego automatico:

```bash
python lms_live_collector.py \
  --bot-auto-phases \
  --channel chrome \
  --no-persistent
```

Script PowerShell equivalente:

```powershell
.\scripts\run_lms_bot_auto.ps1
```

Modo dual para inspeccionar juego y monitor de cursor/click al mismo tiempo:

```bash
python lms_live_collector.py \
  --play-game \
  --bot-parallel-smoke \
  --channel chrome \
  --no-persistent
```

Script PowerShell dual:

```powershell
.\scripts\run_lms_bot_dual.ps1
```

Login Google previo a jugar (automatico en script dual):

```powershell
$env:LMS_GOOGLE_EMAIL = "tu_correo_google"
$env:LMS_GOOGLE_PASSWORD = "tu_password_google"
.\scripts\run_lms_bot_dual.ps1
```

Notas:
- El script activa `--bot-google-login` automaticamente si detecta ambas variables.
- Si Google pide challenge/2FA, el bot lo reporta y continua sin confirmar login.

En modo dual, al entrar en `in_match` el bot aplica ciclo de movimiento smoke en juego real:
`ArrowUp -> ArrowDown -> ArrowLeft -> ArrowRight` (repite).

Puedes definir patron personalizado, por ejemplo:
`ArrowUp,ArrowUp,ArrowLeft,ArrowRight,ArrowDown`.

## 6) Runtime Probe (entorno JS tipo F12)

Captura variables globales relevantes detectadas en el runtime del juego (lobby + entrada a partida):

```bash
python lms_live_collector.py \
  --play-game \
  --bot-runtime-probe \
  --bot-runtime-probe-dir reports/runtime_probe \
  --bot-runtime-probe-max-keys 180 \
  --channel chrome \
  --no-persistent
```

Salida:
- JSON por snapshot en `reports/runtime_probe/`
- fases esperadas: `lobby_bootstrap`, `in_match_entry`
- campos clave: `runtime_vars[]`, `local_storage_keys`, `session_storage_keys`, `window_key_count`

## 7) Prueba de 3 corridas consecutivas (chrome + smoke)

```powershell
.\scripts\run_lms_bot_batch.ps1 -Count 3 -RunMaxSec 240
```

Salida:
- carpeta `reports/live/batch3_YYYYMMDD_HHMMSS`
- logs por partida: `match_1.log`, `match_2.log`, `match_3.log`
- manifiesto: `manifest.jsonl`

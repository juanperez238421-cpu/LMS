# LMS Web Telemetry GUI (v2)

Web dashboard for live telemetry from JSONL.

## Design constraints
- No extra bot load: only JSONL tail + WebSocket broadcast.
- No OCR/capture work in bot tick.
- Backend throttles WS with `LMS_TELEMETRY_WS_HZ`.
- Bounded queue with drop-oldest under pressure.

## Features
- Tabs: `Vitals`, `Combat`, `Zone`, `Entities`, `Raw`
- Event filters: `all`, `actions only`, `damage only`
- Freeze on death: freeze UI when HP reaches 0 or death signal appears
- Log rotation support in tailer (reopen on recreate/rotate/truncate)
- `/config` endpoint with runtime viewer config

## Smoke check
```powershell
python -c "import fastapi, uvicorn, orjson"
```

## Run only web GUI
```powershell
.\tools\gui_web\run_web_gui.ps1 -TelemetryPath "reports\runtime\telemetry_live.jsonl" -Port 8008 -WsHz 10 -History 1200
```

Open:
`http://127.0.0.1:8008`

## Run live bot + telemetry + web GUI
```powershell
.\tools\run_live_with_telemetry.ps1 -WebGui -WebGuiPort 8008 -WebGuiWsHz 10
```

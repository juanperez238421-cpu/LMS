# Repository Structure

This repository has two major tracks:

1. `src/botgame/`: server-authoritative simulation and ML training code.
2. Root LMS scripts (`lms_*.py`): web telemetry capture, OCR, websocket analysis, and browser bot experiments.

## Top-Level Layout

```text
.
|- config/                       # Runtime/config templates
|- data/
|  |- raw/
|  |  |- har/                    # HAR exports
|  |  `- ws/                     # Exported websocket frame samples
|  `- processed/
|     |- ocr/                    # OCR screenshots and extracted assets
|     `- sqlite/                 # SQLite databases produced by LMS tools
|- docs/
|  |- architecture/
|  `- workflows/
|- reports/
|  |- audit/                     # Mining/audit reports
|  |- live/                      # Live play/bot logs
|  |- smoke/                     # Smoke/dual monitor outputs
|  `- ops/                       # Operational diagnostics
|- scripts/                      # Operational scripts (setup, run, train, organize)
|- src/botgame/                  # Simulation, bots, training, server
|- tests/                        # Unit tests
|- lms_events_collector.py       # HAR/events ingestion tool
|- lms_live_collector.py         # Live browser collector + bot automation
|- lms_ocr_extractor.py          # OCR extraction utility
|- lms_text_miner.py             # Offline text mining over captured data
`- lms_ws_analyzer.py            # WebSocket frame analyzer
```

## Storage Conventions

- Database defaults should target `data/processed/sqlite/`.
- Screenshots/OCR outputs should target `data/processed/ocr/`.
- HAR and raw websocket exports should target `data/raw/`.
- Runtime logs should target `reports/live/` and smoke outputs `reports/smoke/`.

## Operational Scripts

- `scripts/bootstrap_repo.ps1`: bootstrap dependencies and test environment.
- `scripts/organize_repo.ps1`: normalize/re-segment generated artifacts.
- `scripts/run_lms_bot_auto.ps1`: smoke -> play sequence.
- `scripts/run_lms_bot_dual.ps1`: game + smoke monitor in parallel.

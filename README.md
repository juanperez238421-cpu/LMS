# LMS Bot Platform

This repository contains two complementary systems:

1. `src/botgame`: a server-authoritative bot simulation and ML training stack.
2. Root LMS tooling (`lms_*.py`): live telemetry capture, OCR extraction, websocket analysis, and browser bot automation for `https://lastmagestanding.com/`.

## Repository Layout

```text
.
|- config/                         # Config templates (OCR ROIs, etc.)
|- data/
|  |- raw/har/                     # HAR captures
|  |- raw/ws/                      # Exported websocket payload samples
|  |- processed/sqlite/            # SQLite databases
|  `- processed/ocr/               # OCR screenshots/artifacts
|- docs/
|  |- architecture/                # Design and structure docs
|  `- workflows/                   # Workflow references
|- reports/
|  |- audit/                       # Mining and audit reports
|  |- live/                        # Live run logs
|  |- smoke/                       # Smoke-test logs/json/png
|  `- ops/                         # Operational diagnostics
|- scripts/                        # Setup, organization, run helpers
|- src/botgame/                    # Core simulation + training code
|- tests/                          # Unit tests
|- lms_events_collector.py
|- lms_live_collector.py
|- lms_ocr_extractor.py
|- lms_text_miner.py
`- lms_ws_analyzer.py
```

## Quick Start (Windows PowerShell)

```powershell
.\scripts\bootstrap_repo.ps1
```

This bootstrap script:
- creates/uses `.venv`,
- installs project and LMS dependencies,
- installs Playwright Chromium,
- runs tests.

## Core Workflows

- Metrics and capture workflow: `LMS_METRICS_WORKFLOW.md`
- Repository structure details: `docs/architecture/REPOSITORY_STRUCTURE.md`
- Feedback-to-training workflow: `docs/workflows/BOT_FEEDBACK_TRAINING.md`
- AlphaStar mass-training workflow: `docs/workflows/ALPHASTAR_MASS_TRAINING.md`

## Bot Run Modes

1. Auto phases (`smoke -> play`):
```powershell
.\scripts\run_lms_bot_auto.ps1
```

2. Dual mode (`play-game + parallel smoke monitor`):
```powershell
.\scripts\run_lms_bot_dual.ps1
```

You can override the in-match movement pattern:
```powershell
.\scripts\run_lms_bot_dual.ps1 --bot-smoke-move-pattern "ArrowUp,ArrowUp,ArrowLeft,ArrowRight,ArrowDown"
```

## Data and Reports Conventions

- Databases: `data/processed/sqlite/*.db`
- HAR captures: `data/raw/har/*.har`
- WS sample exports: `data/raw/ws/*`
- OCR files: `data/processed/ocr/*`
- Live logs: `reports/live/*.log`
- Smoke outputs: `reports/smoke/*`

## Keep It Organized

To re-segment generated artifacts at any time:

```powershell
.\scripts\organize_repo.ps1
```

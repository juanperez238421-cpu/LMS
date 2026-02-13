# Context Files for AI Analysis

This directory contains copies of key files from the LMS project that are highly relevant for analysis by an Artificial Intelligence (AI) model, especially in the context of bot behavior, game interaction, and training.

## Included Files:

*   **lms_live_collector.py**: Core logic for the live bot's operation.
*   **lms_text_miner.py**: Handles text extraction within the game.
*   **lms_ocr_extractor.py**: Manages Optical Character Recognition (OCR) for reading on-screen information.
*   **lms_ws_analyzer.py**: Analyzes WebSocket communication for game events.
*   **lms_ocr_rois.example.json**: Example configuration for Regions of Interest (ROIs) for OCR.
*   **config_lms_ocr_rois.example.json**: Another example configuration for OCR ROIs, from the `config` directory.
*   **run_lms_bot_auto.ps1**: PowerShell script for running the bot in auto mode.
*   **run_lms_bot_dual.ps1**: PowerShell script for running the bot in dual (game + smoke) mode.
*   **train_imitation.ps1**: PowerShell script related to imitation learning training.
*   **train_rl.ps1**: PowerShell script related to reinforcement learning training.
*   **pyproject.toml**: Project dependencies and metadata.
*   **requirements-lms.txt**: Python package requirements.

## Dynamically Generated Report Files:

These files are crucial for AI training and analysis but are not copied here because they are generated dynamically during bot runs and can be very large (images, videos). You can find them in the `reports/feedback_training/live/` directory, typically organized by timestamped subfolders:

*   **`reports/feedback_training/live/play_runtime_<timestamp>/feedback_stream.jsonl`**: Contains structured feedback data from bot runs, essential for training.
*   **`reports/feedback_training/live/play_runtime_<timestamp>/screens/*.png`**: Screenshots captured during gameplay, serving as visual input for AI.
*   **`reports/feedback_training/live/play_runtime_<timestamp>/timeline.mp4`**: Video recordings of bot runs, providing a complete visual and temporal context.
*   **`reports/runtime_probe/`**: Contains runtime snapshots for deeper analysis.

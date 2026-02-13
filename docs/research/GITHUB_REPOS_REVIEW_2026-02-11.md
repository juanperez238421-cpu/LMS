# GitHub Repos Review for LMS Bot (2026-02-11)

## Goal
Increase realtime responsiveness, improve UI text/state detection robustness, and set a practical path for learning from match feedback.

## Reviewed repositories

1. RapidFuzz
   - Repo: https://github.com/rapidfuzz/RapidFuzz
   - Why: Fast fuzzy text matching for noisy OCR output.
   - Status: Adopted now (`rapidfuzz` added to requirements and integrated for death phrase matching).

2. orjson
   - Repo: https://github.com/ijl/orjson
   - Why: Faster JSON decode for high-frequency event payload parsing.
   - Status: Adopted now (`orjson` added to requirements and integrated with safe fallback to stdlib json).

3. BetterCam
   - Repo: https://github.com/RootKit-Org/BetterCam
   - Why: Very high FPS screen capture on Windows (useful if we move heavy vision outside Playwright DOM/canvas flow).
   - Status: Not yet integrated. Candidate for Phase 2 realtime vision pipeline.

4. python-mss
   - Repo: https://github.com/BoboTiG/python-mss
   - Why: Cross-platform fast screenshot capture with simple API.
   - Status: Not yet integrated. Candidate fallback backend if BetterCam is not available.

5. Ultralytics YOLO
   - Repo: https://github.com/ultralytics/ultralytics
   - Why: Real-time object detection for enemies/UI markers.
   - Status: Not yet integrated. Candidate for enemy detection upgrade.

6. Stable-Baselines3
   - Repo: https://github.com/DLR-RM/stable-baselines3
   - Why: Reliable RL training baselines for policy learning from feedback data.
   - Status: Not yet integrated. Candidate for offline training loop from saved runs.

7. imitation
   - Repo: https://github.com/HumanCompatibleAI/imitation
   - Why: Imitation learning / reward learning to bootstrap policies from recorded data.
   - Status: Not yet integrated. Candidate for behavior cloning stage.

## Immediate implementation completed in this repo

- `rapidfuzz` integrated to improve OCR phrase detection resilience.
- `orjson` integrated to reduce JSON parsing overhead in hot paths.
- Knowledge DB flow kept active for incremental policy updates from runtime feedback.

## Recommended next implementation phases

1. Phase 2 realtime vision:
   - Add optional BetterCam/MSS capture backend behind a CLI flag.
   - Feed sampled frames to a detector pipeline with strict rate control.

2. Phase 3 detection model:
   - Add YOLO-based enemy and UI element detection.
   - Store detections in feedback JSONL and SQLite for policy context.

3. Phase 4 learning loop:
   - Build offline trainer (SB3 + imitation) consuming `reports/feedback_training/live/*`.
   - Export a lightweight policy artifact used at runtime for move/action ranking.


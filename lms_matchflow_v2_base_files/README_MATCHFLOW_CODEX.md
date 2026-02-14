# MatchFlow v2 + Map Identification (5 maps)

## Goal
Identify **phase** (lobby/countdown/in-match/dead/end) and identify **map name before match starts**,
without wasting frames on expensive perception outside IN_MATCH.

## Map identification strategy
1) Preferred (fastest & most robust): OCR a small ROI that shows the **map name**.
   - `map_catalog.match_map_name_from_ocr(text)` maps it to one of the 5 known maps.
2) Fallback: minimap fingerprint (dHash on 3 stable regions) vs Big/Small assets.
   - `MapFingerprintDB([Path("assets/Mapas/Big"), Path("assets/Mapas/Small")]).match_minimap(minimap_crop)`

## Assets
Put your map PNGs under:
- `assets/Mapas/Big`
- `assets/Mapas/Small`

Expected names:
- CASTILLOS DESÉRTICOS.png
- LA FINCA.png
- ASENTAMIENTO DESÉRTICO.png
- FORTALEZA DE LAVA.png
- JARDINES CONGELADOS.png

## ROIs
Use `lms_screen_rois.matchflow.example.json` (or `lms_screen_rois.matchflow.json`) as template and calibrate:
- `map_name_label` (OCR)
- `minimap` (image crop)

## Runtime wiring
Use `runtime_flow.MatchFlowRuntime`:
- `detect_map(...)` uses OCR-first then fingerprint fallback
- `update_phase(...)` outputs `game_phase`, `phase_confidence`, `phase_reason`
- `should_run_full_pipeline()` gates expensive perception
- `maybe_extract_end_metrics(...)` emits end metrics when `END_OF_COMBAT`

## Tests
pytest -q

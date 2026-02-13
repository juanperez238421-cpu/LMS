# LMS Telemetry Patch (drop-in)

This folder contains a ready-to-copy baseline for:
- a **non-blocking** JSONL telemetry writer (publisher)
- a low-overhead **secondary UI** (Rich TUI subscriber)
- a simulator to test the UI without the game

## Files
- `src/botgame/telemetry/schema.py`
- `src/botgame/telemetry/jsonl_writer.py`
- `tools/telemetry_tui.py`
- `tools/telemetry_sim.py`

## Minimal wiring into `lms_live_collector.py`

### 1) Add CLI flags in `build_parser()` (near other bot flags)

```py
parser.add_argument("--telemetry-jsonl", default="", help="Write runtime telemetry frames to JSONL.")
parser.add_argument("--telemetry-rate-hz", type=float, default=10.0, help="Max telemetry publish rate.")
```

### 2) Import + init once (after args parsed)

```py
from botgame.telemetry.schema import TelemetryFrame
from botgame.telemetry.jsonl_writer import JsonlTelemetryWriter, TelemetryWriterConfig

telemetry = None
if getattr(args, "telemetry_jsonl", ""):
    telemetry = JsonlTelemetryWriter(
        TelemetryWriterConfig(
            path=args.telemetry_jsonl,
            rate_hz=float(args.telemetry_rate_hz),
            flush_every_s=0.5,
            queue_max=256,
            enabled=True,
        )
    ).start()
```

### 3) Emit inside the main in-match tick (exact location)

In your current file, the damage deltas are computed around:

```py
current_damage_done_total = float(bot_event_signals.get("damage_done_total", 0.0) or 0.0)
current_damage_taken_total = float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0)
damage_done_delta = current_damage_done_total - float(knowledge_prev_damage_done)
damage_taken_delta = current_damage_taken_total - float(knowledge_prev_damage_taken)
```

Right after that block is the best place to emit:

```py
if telemetry:
    # HP proxy: this codebase currently uses "100 - damage_taken_total" as a fallback.
    hp_pct = max(0.0, min(100.0, 100.0 - current_damage_taken_total))
    hp_conf = 0.35
    hp_src = "fallback"

    # Stamina/mana proxy comes from ability_state when available (already computed in codebase).
    mana_now = float(ability_state.get("mana", 0.0) or 0.0)
    max_mana = max(1.0, float(ability_state.get("max_mana", 100.0) or 100.0))
    stamina_pct = max(0.0, min(100.0, (mana_now / max_mana) * 100.0))
    stamina_conf = 0.6
    stamina_src = "heur"

    frame = TelemetryFrame(
        ts_ms=int(time.time() * 1000),
        run_id=str(knowledge_run_id or ""),
        match_id=str(match_id or ""),
        hp_pct=hp_pct, hp_conf=hp_conf, hp_src=hp_src,
        stamina_pct=stamina_pct, stamina_conf=stamina_conf, stamina_src=stamina_src,
        dmg_in_total=current_damage_taken_total,
        dmg_out_total=current_damage_done_total,
        dmg_in_tick=max(0.0, damage_taken_delta),
        dmg_out_tick=max(0.0, damage_done_delta),
        decision=str(bot_state or ""),
        action_last=str(last_action_label or ""),
        action_ok=bool(last_action_ok) if last_action_ok is not None else None,
        motion_score=float(motion_eval),
        zone_outside=bool(bot_event_signals.get("zone_outside_safe", False)),
        zone_toxic=bool(bot_event_signals.get("zone_toxic_detected", False)),
        zone_countdown_s=float(bot_event_signals.get("zone_countdown", 0.0) or 0.0) or None,
    )
    telemetry.maybe_emit(frame)
```

### 4) Shutdown
In the `finally`/shutdown path:
```py
if telemetry:
    telemetry.close()
```

## Running

### Sim test
```bash
python tools/telemetry_sim.py --out reports/runtime/telemetry_sim.jsonl --rate 10 --secs 120
python tools/telemetry_tui.py --path reports/runtime/telemetry_sim.jsonl --fps 10
```

### Live
Run the collector/bot with:
```bash
python lms_live_collector.py ... --telemetry-jsonl reports/runtime/telemetry_live.jsonl --telemetry-rate-hz 10
```

In a second terminal:
```bash
python tools/telemetry_tui.py --path reports/runtime/telemetry_live.jsonl --fps 10
```

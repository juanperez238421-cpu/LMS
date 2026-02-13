from __future__ import annotations

import argparse
import time
import math
import random

from botgame.telemetry.schema import TelemetryFrame, EntityTrack
from botgame.telemetry.jsonl_writer import JsonlTelemetryWriter, TelemetryWriterConfig


def now_ms() -> int:
    return int(time.time() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--secs", type=float, default=60.0)
    args = ap.parse_args()

    w = JsonlTelemetryWriter(TelemetryWriterConfig(path=args.out, rate_hz=args.rate, enabled=True)).start()

    t0 = time.perf_counter()
    hp = 100.0
    dmg_in_total = 0.0
    dmg_out_total = 0.0

    try:
        while (time.perf_counter() - t0) < args.secs:
            t = time.perf_counter() - t0
            dmg_in = 0.0
            dmg_out = 0.0
            if random.random() < 0.15:
                dmg_in = random.uniform(2, 12)
                dmg_in_total += dmg_in
                hp = max(0.0, hp - dmg_in * 0.8)
            if random.random() < 0.20:
                dmg_out = random.uniform(3, 15)
                dmg_out_total += dmg_out

            enemy_vis = random.random() < 0.5
            enemy_conf = random.random() if enemy_vis else 0.0
            enemy_xy = (
                (
                    max(0.0, min(1.0, 0.5 + (0.35 * math.cos(t * 0.9)))),
                    max(0.0, min(1.0, 0.5 + (0.35 * math.sin(t * 0.9)))),
                )
                if enemy_vis
                else None
            )
            enemy_dist_norm = (
                min(1.0, math.hypot(enemy_xy[0] - 0.5, enemy_xy[1] - 0.5) / 0.70710678118)
                if enemy_xy
                else None
            )

            frame = TelemetryFrame(
                ts_ms=now_ms(),
                run_id="SIM",
                match_id="SIM",
                hp_pct=hp,
                hp_conf=0.9,
                hp_src="sim",
                stamina_pct=50 + 50 * math.sin(t * 0.7),
                stamina_conf=0.7,
                stamina_src="sim",
                dmg_in_total=dmg_in_total,
                dmg_out_total=dmg_out_total,
                dmg_in_tick=dmg_in,
                dmg_out_tick=dmg_out,
                enemy_visible=enemy_vis,
                enemy_conf=enemy_conf,
                enemy_dir_deg=(math.sin(t) * 180) if enemy_vis else None,
                enemy_xy=enemy_xy,
                enemy_dist_norm=enemy_dist_norm,
                enemy_age_ms=(0.0 if enemy_vis else random.uniform(200, 1200)),
                ally_count=1,
                enemy_count=(1 if enemy_vis else 0),
                decision="ENGAGE" if enemy_vis else "SCOUT",
                action_last=random.choice(["", "Digit1", "Digit2", "Digit3", "Shift", "MouseLeft"]),
                action_ok=True,
                motion_score=abs(math.sin(t * 0.3)),
                stuck=False,
                collision_proxy=False,
                entities=(
                    [
                        EntityTrack(
                            name="Self",
                            kind="bot",
                            team="ally",
                            hp_pct=hp,
                            conf=0.95,
                            distance_norm=0.0,
                            anchor_xy=(0.5, 0.5),
                        ),
                        EntityTrack(
                            name="EnemyA",
                            kind="player",
                            team="enemy",
                            hp_pct=random.uniform(10, 100),
                            conf=enemy_conf,
                            distance_norm=enemy_dist_norm,
                            anchor_xy=enemy_xy,
                        ),
                    ]
                    if enemy_vis
                    else [
                        EntityTrack(
                            name="Self",
                            kind="bot",
                            team="ally",
                            hp_pct=hp,
                            conf=0.95,
                            distance_norm=0.0,
                            anchor_xy=(0.5, 0.5),
                        )
                    ]
                ),
            )
            w.maybe_emit(frame)
            time.sleep(1.0 / args.rate)
    finally:
        w.close()


if __name__ == "__main__":
    main()

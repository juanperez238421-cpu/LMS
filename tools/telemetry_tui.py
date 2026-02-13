from __future__ import annotations

import argparse
import time
from typing import Optional, Dict, Any

import orjson
from rich.live import Live
from rich.table import Table
from rich.panel import Panel


def tail_jsonl(path: str, sleep_s: float = 0.05):
    with open(path, "rb") as f:
        f.seek(0, 2)  # end
        while True:
            line = f.readline()
            if not line:
                time.sleep(sleep_s)
                continue
            yield line


def fmt_pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:5.1f}%"


def build_table(last: Dict[str, Any]) -> Table:
    t = Table(title="LMS Telemetry (TUI)", show_lines=False)
    t.add_column("Group", style="bold")
    t.add_column("Value")

    t.add_row("HP", f"{fmt_pct(last.get('hp_pct'))}  conf={last.get('hp_conf',0):.2f} src={last.get('hp_src','?')}")
    t.add_row("Stamina", f"{fmt_pct(last.get('stamina_pct'))}  conf={last.get('stamina_conf',0):.2f} src={last.get('stamina_src','?')}")

    t.add_row("Damage In", f"tick={last.get('dmg_in_tick',0):.1f}  total={last.get('dmg_in_total',0):.1f}")
    t.add_row("Damage Out", f"tick={last.get('dmg_out_tick',0):.1f}  total={last.get('dmg_out_total',0):.1f}")

    t.add_row("Enemy", f"visible={last.get('enemy_visible',False)} conf={last.get('enemy_conf',0):.2f} dir={last.get('enemy_dir_deg','-')}")
    t.add_row("Decision", f"{last.get('decision','')}")
    t.add_row("Action", f"{last.get('action_last','')} ok={last.get('action_ok','-')}")

    t.add_row("Zone", f"outside={last.get('zone_outside',False)} toxic={last.get('zone_toxic',False)} cd={last.get('zone_countdown_s','-')}")
    t.add_row("Motion", f"score={last.get('motion_score','-')} stuck={last.get('stuck','-')} collision={last.get('collision_proxy','-')}")

    ents = last.get("entities", [])
    t.add_row("Entities", f"{len(ents)} tracked")
    if ents:
        for e in ents[:3]:
            t.add_row("  -", f"{e.get('name','')} ({e.get('kind','?')}) hp={e.get('hp_pct','-')} conf={e.get('conf',0):.2f}")

    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="telemetry_*.jsonl")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    last: Dict[str, Any] = {}
    frame_period = 1.0 / max(args.fps, 1e-6)
    last_render = 0.0

    with Live(refresh_per_second=args.fps) as live:
        for raw in tail_jsonl(args.path):
            try:
                obj = orjson.loads(raw)
                last = obj
            except Exception:
                continue

            now = time.perf_counter()
            if (now - last_render) >= frame_period:
                live.update(Panel(build_table(last), border_style="cyan"))
                last_render = now


if __name__ == "__main__":
    main()


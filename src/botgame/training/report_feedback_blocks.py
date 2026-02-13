from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class FeedbackRunMetrics:
    run_id: str
    feedback_path: str
    duration_sec: float
    ticks: int
    screenshots: int
    stop_reason: str
    death_cause: str
    damage_done: float
    damage_taken: float
    enemy_detected_ticks: int
    zone_detected_ticks: int
    attack_click_actions: int
    move_only_actions: int
    knowledge_reward_sum: float
    knowledge_reward_avg: float
    telemetry_hp_present_ticks: int
    telemetry_complete_ticks: int
    telemetry_damage_done_nonzero_ticks: int
    telemetry_damage_taken_nonzero_ticks: int
    telemetry_damage_done_regressions: int
    telemetry_damage_taken_regressions: int
    telemetry_hp_present_rate: float
    telemetry_complete_rate: float
    telemetry_damage_done_nonzero_rate: float
    telemetry_damage_taken_nonzero_rate: float
    telemetry_acceptance_pass: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "feedback_path": self.feedback_path,
            "duration_sec": self.duration_sec,
            "ticks": self.ticks,
            "screenshots": self.screenshots,
            "stop_reason": self.stop_reason,
            "death_cause": self.death_cause,
            "damage_done": self.damage_done,
            "damage_taken": self.damage_taken,
            "enemy_detected_ticks": self.enemy_detected_ticks,
            "zone_detected_ticks": self.zone_detected_ticks,
            "attack_click_actions": self.attack_click_actions,
            "move_only_actions": self.move_only_actions,
            "knowledge_reward_sum": self.knowledge_reward_sum,
            "knowledge_reward_avg": self.knowledge_reward_avg,
            "telemetry_hp_present_ticks": self.telemetry_hp_present_ticks,
            "telemetry_complete_ticks": self.telemetry_complete_ticks,
            "telemetry_damage_done_nonzero_ticks": self.telemetry_damage_done_nonzero_ticks,
            "telemetry_damage_taken_nonzero_ticks": self.telemetry_damage_taken_nonzero_ticks,
            "telemetry_damage_done_regressions": self.telemetry_damage_done_regressions,
            "telemetry_damage_taken_regressions": self.telemetry_damage_taken_regressions,
            "telemetry_hp_present_rate": self.telemetry_hp_present_rate,
            "telemetry_complete_rate": self.telemetry_complete_rate,
            "telemetry_damage_done_nonzero_rate": self.telemetry_damage_done_nonzero_rate,
            "telemetry_damage_taken_nonzero_rate": self.telemetry_damage_taken_nonzero_rate,
            "telemetry_acceptance_pass": self.telemetry_acceptance_pass,
        }


def collect_feedback_run_metrics(run_dir: Path) -> FeedbackRunMetrics | None:
    feedback_path = run_dir / "feedback_stream.jsonl"
    if not feedback_path.exists():
        return None
    events = list(_read_jsonl(feedback_path))
    if not events:
        return None

    ticks: List[Dict[str, Any]] = []
    session_start = None
    session_end = None
    for event in events:
        if event.get("event") == "session_start":
            session_start = event
            continue
        if event.get("event") == "session_end":
            session_end = event
            continue
        if event.get("event"):
            continue
        ticks.append(event)

    start_ts = _to_float((session_start or {}).get("ts"), 0.0)
    end_ts = _to_float((session_end or {}).get("ts"), 0.0)
    if start_ts > 0.0 and end_ts > 0.0 and end_ts >= start_ts:
        duration_sec = end_ts - start_ts
    elif ticks:
        first_tick_ts = _to_float(ticks[0].get("ts"), 0.0)
        last_tick_ts = _to_float(ticks[-1].get("ts"), first_tick_ts)
        duration_sec = max(0.0, last_tick_ts - first_tick_ts)
    else:
        duration_sec = 0.0

    screenshots = len(list((run_dir / "screens").glob("*.png")))
    tick_count = len(ticks)

    attack_click_actions = 0
    move_only_actions = 0
    enemy_detected_ticks = 0
    zone_detected_ticks = 0
    knowledge_reward_values: List[float] = []
    damage_done_tick_last = 0.0
    damage_taken_tick_last = 0.0
    telemetry_hp_present_ticks = 0
    telemetry_complete_ticks = 0
    telemetry_damage_done_nonzero_ticks = 0
    telemetry_damage_taken_nonzero_ticks = 0
    telemetry_damage_done_regressions = 0
    telemetry_damage_taken_regressions = 0

    for tick in ticks:
        action_name = str(tick.get("action", "") or "")
        if "attack_click" in action_name:
            attack_click_actions += 1
        if "move_only" in action_name:
            move_only_actions += 1
        enemy_signal = tick.get("enemy_signal", {}) or {}
        if bool(enemy_signal.get("detected", False)):
            enemy_detected_ticks += 1
        if bool(tick.get("zone_toxic_detected", False)):
            zone_detected_ticks += 1
        knowledge_reward_values.append(_to_float(tick.get("knowledge_reward"), 0.0))
        damage_done_tick_last = _to_float(tick.get("damage_done_total"), damage_done_tick_last)
        damage_taken_tick_last = _to_float(tick.get("damage_taken_total"), damage_taken_tick_last)

        telemetry_tick = tick.get("telemetry_quality", {}) or {}
        if isinstance(telemetry_tick, dict):
            hp_present = bool(
                telemetry_tick.get(
                    "hp_present",
                    tick.get("health_current") is not None,
                )
            )
            complete = bool(
                telemetry_tick.get(
                    "complete",
                    hp_present and ("damage_done_total" in tick) and ("damage_taken_total" in tick),
                )
            )
            damage_done_nonzero = bool(
                telemetry_tick.get(
                    "damage_done_nonzero",
                    _to_float(tick.get("damage_done_total"), 0.0) > 0.0,
                )
            )
            damage_taken_nonzero = bool(
                telemetry_tick.get(
                    "damage_taken_nonzero",
                    _to_float(tick.get("damage_taken_total"), 0.0) > 0.0,
                )
            )
            if hp_present:
                telemetry_hp_present_ticks += 1
            if complete:
                telemetry_complete_ticks += 1
            if damage_done_nonzero:
                telemetry_damage_done_nonzero_ticks += 1
            if damage_taken_nonzero:
                telemetry_damage_taken_nonzero_ticks += 1
            if bool(telemetry_tick.get("damage_done_regressed", False)):
                telemetry_damage_done_regressions += 1
            if bool(telemetry_tick.get("damage_taken_regressed", False)):
                telemetry_damage_taken_regressions += 1

    damage_done = _to_float((session_end or {}).get("damage_done_total"), damage_done_tick_last)
    damage_taken = _to_float((session_end or {}).get("damage_taken_total"), damage_taken_tick_last)
    stop_reason = str((session_end or {}).get("run_stop_reason", "") or "")
    death_info = (session_end or {}).get("death", {}) or {}
    death_cause = str(death_info.get("cause", "") or "")
    knowledge_reward_sum = float(sum(knowledge_reward_values))
    knowledge_reward_avg = _mean(knowledge_reward_values)
    telemetry_hp_present_rate = float(telemetry_hp_present_ticks) / float(max(1, tick_count))
    telemetry_complete_rate = float(telemetry_complete_ticks) / float(max(1, tick_count))
    telemetry_damage_done_nonzero_rate = float(telemetry_damage_done_nonzero_ticks) / float(max(1, tick_count))
    telemetry_damage_taken_nonzero_rate = float(telemetry_damage_taken_nonzero_ticks) / float(max(1, tick_count))

    telemetry_session = (session_end or {}).get("telemetry_quality", {}) or {}
    if isinstance(telemetry_session, dict):
        telemetry_hp_present_rate = _to_float(
            telemetry_session.get("hp_present_rate"),
            telemetry_hp_present_rate,
        )
        telemetry_complete_rate = _to_float(
            telemetry_session.get("complete_rate"),
            telemetry_complete_rate,
        )
        telemetry_damage_done_nonzero_rate = _to_float(
            telemetry_session.get("damage_done_nonzero_rate"),
            telemetry_damage_done_nonzero_rate,
        )
        telemetry_damage_taken_nonzero_rate = _to_float(
            telemetry_session.get("damage_taken_nonzero_rate"),
            telemetry_damage_taken_nonzero_rate,
        )
        telemetry_damage_done_regressions = int(
            telemetry_session.get("damage_done_regressions", telemetry_damage_done_regressions) or 0
        )
        telemetry_damage_taken_regressions = int(
            telemetry_session.get("damage_taken_regressions", telemetry_damage_taken_regressions) or 0
        )
    telemetry_acceptance_pass = bool(
        telemetry_hp_present_rate >= 0.98
        and telemetry_complete_rate >= 0.98
        and telemetry_damage_done_regressions == 0
        and telemetry_damage_taken_regressions == 0
    )
    if isinstance(telemetry_session, dict):
        acceptance_obj = telemetry_session.get("acceptance", {}) or {}
        if isinstance(acceptance_obj, dict) and ("pass" in acceptance_obj):
            telemetry_acceptance_pass = bool(acceptance_obj.get("pass"))

    return FeedbackRunMetrics(
        run_id=run_dir.name,
        feedback_path=str(feedback_path.as_posix()),
        duration_sec=float(duration_sec),
        ticks=tick_count,
        screenshots=screenshots,
        stop_reason=stop_reason,
        death_cause=death_cause,
        damage_done=float(damage_done),
        damage_taken=float(damage_taken),
        enemy_detected_ticks=enemy_detected_ticks,
        zone_detected_ticks=zone_detected_ticks,
        attack_click_actions=attack_click_actions,
        move_only_actions=move_only_actions,
        knowledge_reward_sum=knowledge_reward_sum,
        knowledge_reward_avg=knowledge_reward_avg,
        telemetry_hp_present_ticks=int(telemetry_hp_present_ticks),
        telemetry_complete_ticks=int(telemetry_complete_ticks),
        telemetry_damage_done_nonzero_ticks=int(telemetry_damage_done_nonzero_ticks),
        telemetry_damage_taken_nonzero_ticks=int(telemetry_damage_taken_nonzero_ticks),
        telemetry_damage_done_regressions=int(telemetry_damage_done_regressions),
        telemetry_damage_taken_regressions=int(telemetry_damage_taken_regressions),
        telemetry_hp_present_rate=float(telemetry_hp_present_rate),
        telemetry_complete_rate=float(telemetry_complete_rate),
        telemetry_damage_done_nonzero_rate=float(telemetry_damage_done_nonzero_rate),
        telemetry_damage_taken_nonzero_rate=float(telemetry_damage_taken_nonzero_rate),
        telemetry_acceptance_pass=bool(telemetry_acceptance_pass),
    )


def summarize_feedback_blocks(runs: List[FeedbackRunMetrics], block_size: int) -> List[Dict[str, Any]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not runs:
        return []

    blocks: List[Dict[str, Any]] = []
    for start in range(0, len(runs), block_size):
        block = runs[start : start + block_size]
        ticks_total = sum(r.ticks for r in block)
        attack_total = sum(r.attack_click_actions for r in block)
        move_total = sum(r.move_only_actions for r in block)
        death_event_count = sum(1 for r in block if r.stop_reason == "death_event")
        damage_done_positive = sum(1 for r in block if r.damage_done > 0.0)
        telemetry_pass_count = sum(1 for r in block if bool(r.telemetry_acceptance_pass))

        blocks.append(
            {
                "block_index": (start // block_size) + 1,
                "run_start": block[0].run_id,
                "run_end": block[-1].run_id,
                "runs": len(block),
                "avg_duration_sec": _mean([r.duration_sec for r in block]),
                "avg_ticks": _mean([float(r.ticks) for r in block]),
                "avg_screenshots": _mean([float(r.screenshots) for r in block]),
                "avg_damage_done": _mean([r.damage_done for r in block]),
                "avg_damage_taken": _mean([r.damage_taken for r in block]),
                "damage_done_positive_rate": float(damage_done_positive / max(1, len(block))),
                "death_event_rate": float(death_event_count / max(1, len(block))),
                "avg_enemy_detected_ticks": _mean([float(r.enemy_detected_ticks) for r in block]),
                "avg_zone_detected_ticks": _mean([float(r.zone_detected_ticks) for r in block]),
                "attack_click_action_ratio": float(attack_total / max(1, ticks_total)),
                "move_only_action_ratio": float(move_total / max(1, ticks_total)),
                "avg_knowledge_reward_sum": _mean([r.knowledge_reward_sum for r in block]),
                "avg_knowledge_reward_per_tick": _mean([r.knowledge_reward_avg for r in block]),
                "avg_telemetry_hp_present_rate": _mean([r.telemetry_hp_present_rate for r in block]),
                "avg_telemetry_complete_rate": _mean([r.telemetry_complete_rate for r in block]),
                "avg_telemetry_damage_done_nonzero_rate": _mean([r.telemetry_damage_done_nonzero_rate for r in block]),
                "avg_telemetry_damage_taken_nonzero_rate": _mean([r.telemetry_damage_taken_nonzero_rate for r in block]),
                "telemetry_damage_done_regressions": int(sum(r.telemetry_damage_done_regressions for r in block)),
                "telemetry_damage_taken_regressions": int(sum(r.telemetry_damage_taken_regressions for r in block)),
                "telemetry_acceptance_pass_rate": float(telemetry_pass_count / max(1, len(block))),
            }
        )
    return blocks


def build_feedback_block_report(
    feedback_root: Path,
    block_size: int,
    last_n_runs: int = 0,
) -> Dict[str, Any]:
    run_dirs = sorted([p for p in feedback_root.glob("play_runtime_*") if p.is_dir()], key=lambda p: p.name)
    if last_n_runs > 0:
        run_dirs = run_dirs[-last_n_runs:]
    run_metrics: List[FeedbackRunMetrics] = []
    for run_dir in run_dirs:
        metrics = collect_feedback_run_metrics(run_dir)
        if metrics is not None:
            run_metrics.append(metrics)

    blocks = summarize_feedback_blocks(run_metrics, block_size=block_size)
    summary: Dict[str, Any] = {
        "feedback_root": str(feedback_root.as_posix()),
        "total_runs": len(run_metrics),
        "block_size": int(block_size),
        "blocks": blocks,
        "runs": [r.to_dict() for r in run_metrics],
    }
    if len(blocks) >= 2:
        first = blocks[0]
        last = blocks[-1]
        summary["first_vs_last"] = {
            "delta_avg_duration_sec": float(last["avg_duration_sec"] - first["avg_duration_sec"]),
            "delta_avg_ticks": float(last["avg_ticks"] - first["avg_ticks"]),
            "delta_avg_damage_done": float(last["avg_damage_done"] - first["avg_damage_done"]),
            "delta_avg_damage_taken": float(last["avg_damage_taken"] - first["avg_damage_taken"]),
            "delta_damage_done_positive_rate": float(last["damage_done_positive_rate"] - first["damage_done_positive_rate"]),
            "delta_death_event_rate": float(last["death_event_rate"] - first["death_event_rate"]),
            "delta_attack_click_action_ratio": float(last["attack_click_action_ratio"] - first["attack_click_action_ratio"]),
            "delta_move_only_action_ratio": float(last["move_only_action_ratio"] - first["move_only_action_ratio"]),
            "delta_avg_knowledge_reward_per_tick": float(
                last["avg_knowledge_reward_per_tick"] - first["avg_knowledge_reward_per_tick"]
            ),
            "delta_avg_telemetry_hp_present_rate": float(
                last["avg_telemetry_hp_present_rate"] - first["avg_telemetry_hp_present_rate"]
            ),
            "delta_avg_telemetry_complete_rate": float(
                last["avg_telemetry_complete_rate"] - first["avg_telemetry_complete_rate"]
            ),
            "delta_avg_telemetry_damage_done_nonzero_rate": float(
                last["avg_telemetry_damage_done_nonzero_rate"] - first["avg_telemetry_damage_done_nonzero_rate"]
            ),
            "delta_avg_telemetry_damage_taken_nonzero_rate": float(
                last["avg_telemetry_damage_taken_nonzero_rate"] - first["avg_telemetry_damage_taken_nonzero_rate"]
            ),
            "delta_telemetry_acceptance_pass_rate": float(
                last["telemetry_acceptance_pass_rate"] - first["telemetry_acceptance_pass_rate"]
            ),
        }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize live feedback metrics in blocks of runs.")
    parser.add_argument("--feedback-root", type=Path, default=Path("reports/feedback_training/live"))
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--last-n-runs", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = build_feedback_block_report(
        feedback_root=args.feedback_root,
        block_size=int(args.block_size),
        last_n_runs=int(args.last_n_runs),
    )
    output = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

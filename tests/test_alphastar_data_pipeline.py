import json
from pathlib import Path

import pytest

from botgame.training.alphastar.action_codec import FactorizedActionCodec
from botgame.training.alphastar.data import load_live_feedback_sequences
from botgame.training.report_feedback_blocks import build_feedback_block_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _decode_step(codec: FactorizedActionCodec, seq, idx: int):
    encoded = {name: int(seq.actions[name][idx]) for name in codec.head_sizes}
    return codec.decode_action(encoded)


def test_live_feedback_parses_keys_ability_and_session_end_outcome(tmp_path: Path):
    feedback_file = tmp_path / "play_runtime_20260213_000001" / "feedback_stream.jsonl"
    _write_jsonl(
        feedback_file,
        [
            {"event": "session_start", "ts": 1.0},
            {
                "ts": 2.0,
                "bot_state": "in_match",
                "action": "in_match:move_only:KeyW+KeyA",
                "action_ok": True,
                "active_keys": ["KeyW", "KeyA", "Digit2"],
                "ability_last": "Digit2",
            },
            {
                "ts": 3.0,
                "bot_state": "in_match",
                "action": "in_match:attack_click:mouse",
                "action_ok": True,
                "active_keys": ["MouseRight", "KeyR"],
                "ability_last": "KeyR",
            },
            {"event": "session_end", "ts": 4.0, "run_stop_reason": "death_event"},
        ],
    )

    codec = FactorizedActionCodec(ability_size=5)
    sequences = load_live_feedback_sequences(
        feedback_root=tmp_path,
        codec=codec,
        manifest_outcomes={},
    )
    assert len(sequences) == 1
    seq = sequences[0]
    assert seq.length == 2
    assert float(seq.dones[-1]) == 1.0
    assert float(seq.extras["terminal_outcome"]) == -1.0

    first = _decode_step(codec, seq, 0)
    second = _decode_step(codec, seq, 1)

    assert first.move_x < -0.5
    assert first.move_y < -0.5
    assert first.ability_id == 2
    assert first.fire is False

    assert second.fire is True
    assert second.ability_id == 4
    assert float(seq.rewards[-1]) == pytest.approx(-0.95, abs=1e-6)


def test_live_feedback_manifest_zero_falls_back_to_session_end(tmp_path: Path):
    feedback_file = tmp_path / "play_runtime_20260213_000002" / "feedback_stream.jsonl"
    _write_jsonl(
        feedback_file,
        [
            {"event": "session_start", "ts": 10.0},
            {"ts": 11.0, "bot_state": "in_match", "action": "in_match:attack_click:mouse", "action_ok": True},
            {"ts": 12.0, "bot_state": "in_match", "action": "in_match:attack_click:mouse", "action_ok": True},
            {"event": "session_end", "ts": 13.0, "run_stop_reason": "death_event"},
        ],
    )
    codec = FactorizedActionCodec(ability_size=5)
    sequences = load_live_feedback_sequences(
        feedback_root=tmp_path,
        codec=codec,
        manifest_outcomes={str(feedback_file.as_posix()): 0.0},
    )
    assert len(sequences) == 1
    assert float(sequences[0].extras["terminal_outcome"]) == -1.0


def test_live_feedback_manifest_nonzero_overrides_session_end(tmp_path: Path):
    feedback_file = tmp_path / "play_runtime_20260213_000003" / "feedback_stream.jsonl"
    _write_jsonl(
        feedback_file,
        [
            {"event": "session_start", "ts": 20.0},
            {"ts": 21.0, "bot_state": "in_match", "action": "in_match:attack_click:mouse", "action_ok": True},
            {"ts": 22.0, "bot_state": "in_match", "action": "in_match:attack_click:mouse", "action_ok": True},
            {"event": "session_end", "ts": 23.0, "run_stop_reason": "death_event"},
        ],
    )
    codec = FactorizedActionCodec(ability_size=5)
    sequences = load_live_feedback_sequences(
        feedback_root=tmp_path,
        codec=codec,
        manifest_outcomes={str(feedback_file.as_posix()): 1.0},
    )
    assert len(sequences) == 1
    assert float(sequences[0].extras["terminal_outcome"]) == 1.0


def test_feedback_block_report_has_first_vs_last_delta(tmp_path: Path):
    root = tmp_path / "live"
    run1 = root / "play_runtime_20260213_010000"
    run2 = root / "play_runtime_20260213_020000"
    _write_jsonl(
        run1 / "feedback_stream.jsonl",
        [
            {"event": "session_start", "ts": 100.0},
            {"ts": 101.0, "action": "in_match:attack_click:mouse", "action_ok": True, "knowledge_reward": 0.1},
            {"ts": 102.0, "action": "in_match:move_only:KeyW", "action_ok": True, "knowledge_reward": 0.2},
            {"event": "session_end", "ts": 110.0, "run_stop_reason": "death_event", "damage_done_total": 0.0, "damage_taken_total": 1.0, "death": {"cause": "enemy_unknown"}},
        ],
    )
    _write_jsonl(
        run2 / "feedback_stream.jsonl",
        [
            {"event": "session_start", "ts": 200.0},
            {"ts": 201.0, "action": "in_match:attack_click:mouse", "action_ok": True, "knowledge_reward": 0.6},
            {"ts": 202.0, "action": "in_match:attack_click:mouse", "action_ok": True, "knowledge_reward": 0.7},
            {"event": "session_end", "ts": 215.0, "run_stop_reason": "win", "damage_done_total": 8.0, "damage_taken_total": 0.0, "death": {"cause": ""}},
        ],
    )
    (run1 / "screens").mkdir(parents=True, exist_ok=True)
    (run2 / "screens").mkdir(parents=True, exist_ok=True)
    (run1 / "screens" / "0001.png").write_bytes(b"\x89PNG")
    (run2 / "screens" / "0001.png").write_bytes(b"\x89PNG")
    (run2 / "screens" / "0002.png").write_bytes(b"\x89PNG")

    report = build_feedback_block_report(feedback_root=root, block_size=1)
    assert report["total_runs"] == 2
    assert len(report["blocks"]) == 2
    assert "first_vs_last" in report
    assert report["first_vs_last"]["delta_avg_damage_done"] > 0.0
    assert report["first_vs_last"]["delta_death_event_rate"] < 0.0
    assert "avg_telemetry_hp_present_rate" in report["blocks"][0]
    assert "delta_avg_telemetry_complete_rate" in report["first_vs_last"]


def test_feedback_block_report_telemetry_acceptance_rates(tmp_path: Path):
    root = tmp_path / "live"
    run1 = root / "play_runtime_20260213_030000"
    run2 = root / "play_runtime_20260213_040000"
    _write_jsonl(
        run1 / "feedback_stream.jsonl",
        [
            {"event": "session_start", "ts": 300.0},
            {
                "ts": 301.0,
                "action": "in_match:attack_click:mouse",
                "action_ok": True,
                "damage_done_total": 0.0,
                "damage_taken_total": 0.0,
                "health_current": None,
                "telemetry_quality": {"hp_present": False, "complete": False},
            },
            {
                "event": "session_end",
                "ts": 320.0,
                "run_stop_reason": "death_event",
                "damage_done_total": 0.0,
                "damage_taken_total": 100.0,
                "telemetry_quality": {
                    "hp_present_rate": 0.0,
                    "complete_rate": 0.0,
                    "damage_done_regressions": 0,
                    "damage_taken_regressions": 0,
                    "acceptance": {"pass": False},
                },
            },
        ],
    )
    _write_jsonl(
        run2 / "feedback_stream.jsonl",
        [
            {"event": "session_start", "ts": 400.0},
            {
                "ts": 401.0,
                "action": "in_match:attack_click:mouse",
                "action_ok": True,
                "damage_done_total": 3.0,
                "damage_taken_total": 7.0,
                "health_current": 93.0,
                "telemetry_quality": {
                    "hp_present": True,
                    "complete": True,
                    "damage_done_nonzero": True,
                    "damage_taken_nonzero": True,
                },
            },
            {
                "event": "session_end",
                "ts": 420.0,
                "run_stop_reason": "win",
                "damage_done_total": 6.0,
                "damage_taken_total": 8.0,
                "telemetry_quality": {
                    "hp_present_rate": 1.0,
                    "complete_rate": 1.0,
                    "damage_done_regressions": 0,
                    "damage_taken_regressions": 0,
                    "acceptance": {"pass": True},
                },
            },
        ],
    )

    report = build_feedback_block_report(feedback_root=root, block_size=1)
    assert report["total_runs"] == 2
    assert report["runs"][0]["telemetry_acceptance_pass"] is False
    assert report["runs"][1]["telemetry_acceptance_pass"] is True
    assert report["blocks"][0]["telemetry_acceptance_pass_rate"] == 0.0
    assert report["blocks"][1]["telemetry_acceptance_pass_rate"] == 1.0
    assert report["first_vs_last"]["delta_telemetry_acceptance_pass_rate"] == pytest.approx(1.0, abs=1e-6)

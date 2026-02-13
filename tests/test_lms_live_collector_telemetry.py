from lms_live_collector import (
    build_runtime_telemetry_snapshot,
    create_runtime_telemetry_monitor,
    summarize_runtime_telemetry_monitor,
    update_runtime_telemetry_monitor,
)


def test_runtime_telemetry_derives_health_from_damage_taken():
    snapshot = build_runtime_telemetry_snapshot(
        signals={
            "damage_done_total": 12.0,
            "damage_taken_total": 27.0,
            "health_max": 100.0,
            "damage_done_source": "event",
            "damage_taken_source": "event",
        },
        now_mono=100.0,
        previous=None,
    )
    assert snapshot["health_current"] == 73.0
    assert snapshot["health_source"] == "derived_from_damage_taken"
    assert snapshot["complete"] is True
    assert snapshot["hp_present"] is True


def test_runtime_telemetry_clamps_damage_regressions_to_previous():
    previous = {
        "damage_done_total": 15.0,
        "damage_taken_total": 22.0,
        "health_current": 78.0,
        "health_max": 100.0,
    }
    snapshot = build_runtime_telemetry_snapshot(
        signals={
            "damage_done_total": 2.0,
            "damage_taken_total": 4.0,
            "health_max": 100.0,
        },
        now_mono=120.0,
        previous=previous,
    )
    assert snapshot["damage_done_total"] == 15.0
    assert snapshot["damage_taken_total"] == 22.0
    assert snapshot["damage_done_regressed"] is True
    assert snapshot["damage_taken_regressed"] is True


def test_runtime_telemetry_summary_fails_on_regressions():
    monitor = create_runtime_telemetry_monitor()
    snap_ok = build_runtime_telemetry_snapshot(
        signals={"damage_done_total": 0.0, "damage_taken_total": 0.0, "health_max": 100.0},
        now_mono=10.0,
        previous=None,
    )
    update_runtime_telemetry_monitor(monitor, snap_ok)

    snap_regressed = build_runtime_telemetry_snapshot(
        signals={"damage_done_total": 0.0, "damage_taken_total": 0.0, "health_max": 100.0},
        now_mono=11.0,
        previous={"damage_done_total": 5.0, "damage_taken_total": 7.0, "health_current": 93.0, "health_max": 100.0},
    )
    update_runtime_telemetry_monitor(monitor, snap_regressed)
    summary = summarize_runtime_telemetry_monitor(monitor, latest_snapshot=snap_regressed)
    assert summary["ticks_total"] == 2
    assert summary["damage_done_regressions"] == 1
    assert summary["damage_taken_regressions"] == 1
    assert summary["acceptance"]["pass"] is False

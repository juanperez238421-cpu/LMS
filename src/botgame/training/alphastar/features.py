from __future__ import annotations

from typing import Any

import numpy as np

from botgame.common.types import Observation

OBS_DIM = 19


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def observation_to_features(observation: Observation | dict[str, Any]) -> np.ndarray:
    """Converts world observations to a fixed-size vector."""
    if isinstance(observation, Observation):
        obs_dict: dict[str, Any] = {
            "self_state": observation.self_state.__dict__,
            "zone_state": observation.zone_state.__dict__,
            "visible_entities": [entity.__dict__ for entity in observation.visible_entities],
            "visible_items": [item.__dict__ for item in observation.visible_items],
            "tick_id": observation.tick_id,
        }
    else:
        obs_dict = observation

    self_state = obs_dict.get("self_state", {}) or {}
    zone_state = obs_dict.get("zone_state", {}) or {}
    visible_entities = list(obs_dict.get("visible_entities", []) or [])
    visible_items = list(obs_dict.get("visible_items", []) or [])

    px, py = (self_state.get("position", [0.0, 0.0]) + [0.0, 0.0])[:2]
    vx, vy = (self_state.get("velocity", [0.0, 0.0]) + [0.0, 0.0])[:2]
    hp = _safe_float(self_state.get("hp", 0.0))
    ammo = _safe_float(self_state.get("ammo", 0.0))
    zx, zy = (zone_state.get("position", [0.0, 0.0]) + [0.0, 0.0])[:2]
    zr = _safe_float(zone_state.get("radius", 0.0))
    is_safe = 1.0 if bool(zone_state.get("is_safe", False)) else 0.0

    enemy_x = 0.0
    enemy_y = 0.0
    enemy_hp = 0.0
    enemy_dist = 0.0
    if visible_entities:
        first_enemy = next((e for e in visible_entities if e.get("type") == "enemy"), visible_entities[0])
        ex, ey = (first_enemy.get("position", [0.0, 0.0]) + [0.0, 0.0])[:2]
        enemy_x = _safe_float(ex)
        enemy_y = _safe_float(ey)
        enemy_hp = _safe_float(first_enemy.get("hp", 0.0))
        enemy_dist = float(np.hypot(enemy_x - _safe_float(px), enemy_y - _safe_float(py)))

    item_x = 0.0
    item_y = 0.0
    item_dist = 0.0
    if visible_items:
        first_item = visible_items[0]
        ix, iy = (first_item.get("position", [0.0, 0.0]) + [0.0, 0.0])[:2]
        item_x = _safe_float(ix)
        item_y = _safe_float(iy)
        item_dist = float(np.hypot(item_x - _safe_float(px), item_y - _safe_float(py)))

    return np.array(
        [
            _safe_float(px),
            _safe_float(py),
            _safe_float(vx),
            _safe_float(vy),
            hp,
            ammo,
            _safe_float(zx),
            _safe_float(zy),
            zr,
            is_safe,
            enemy_x,
            enemy_y,
            enemy_hp,
            item_x,
            item_y,
            float(len(visible_entities)),
            float(len(visible_items)),
            enemy_dist,
            item_dist,
        ],
        dtype=np.float32,
    )


def live_event_to_features(event: dict[str, Any]) -> np.ndarray:
    """Maps live collector feedback events into the same fixed vector space."""
    ui_state = event.get("ui_state", {}) or {}
    input_probe = event.get("input_probe", {}) or {}
    click_probe = event.get("click_probe", {}) or {}
    cursor_probe = event.get("cursor_probe", {}) or {}
    bot_state = str(event.get("bot_state", "unknown") or "unknown")
    in_match = 1.0 if bot_state == "in_match" else 0.0
    lobby = 1.0 if bot_state == "lobby" else 0.0

    return np.array(
        [
            float(event.get("mono", 0.0) % 1000.0) / 1000.0,
            in_match,
            lobby,
            1.0 if bool(ui_state.get("canvas_visible", False)) else 0.0,
            float(event.get("action_ok", False)),
            float(len(event.get("active_keys", []) or [])),
            0.0,
            0.0,
            0.0,
            0.0,
            float(input_probe.get("keyDown", 0) or 0),
            float(input_probe.get("pointerMove", 0) or 0),
            float(click_probe.get("click0", 0) or 0),
            float(cursor_probe.get("moves", 0) or 0),
            float(cursor_probe.get("visible", False)),
            float(input_probe.get("focusEvents", 0) or 0),
            float(input_probe.get("blurEvents", 0) or 0),
            float(ui_state.get("play_visible", False)),
            float(ui_state.get("character_visible", False)),
        ],
        dtype=np.float32,
    )


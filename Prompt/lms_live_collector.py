#!/usr/bin/env python3
import argparse
import base64
from datetime import datetime
import gzip
import json
import math
import os
import re
import random
import subprocess
import sqlite3
import sys
import time
import unicodedata
import zlib
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs

from lms_events_collector import (
    ensure_schema,
    ingest_payload_records,
    insert_raw_event,
    insert_raw_request,
    iter_json_objects_from_text,
    print_match_report,
    print_round_activity_report,
)

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
    from playwright.sync_api import Frame, Page, Locator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Falta playwright. Instala con:\n"
        "  pip install playwright\n"
        "  python -m playwright install chromium"
    ) from exc

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None

try:
    from botgame.bots.lms_reverse_engineered import LMSReverseEngineeredBot
    from botgame.common.types import (
        Entity as BGEntity,
        Item as BGItem,
        Observation as BGObservation,
        SelfState as BGSelfState,
        ZoneState as BGZoneState,
    )
except Exception:  # pragma: no cover
    LMSReverseEngineeredBot = None
    BGEntity = None
    BGItem = None
    BGObservation = None
    BGSelfState = None
    BGZoneState = None


BOT_MARKERS = ("bot", "npc", "ai", "cpu")
PLAYER_KEY_HINTS = ("player", "user", "username", "nickname", "display")
CHARACTER_KEY_HINTS = ("character", "hero", "champion", "class")
BOT_KEY_HINTS = ("is_bot", "bot", "npc", "is_ai", "ai")
DEFAULT_WS_KEYWORDS = (
    "damage,kill,kills,elimination,craft,item,loot,player,enemy,weapon,armor"
)
DEFAULT_SMOKE_MOVE_PATTERN = "KeyW,KeyS,KeyA,KeyD"
DEFAULT_CAPTURE_ALL_MAX_BYTES = 2_000_000
DEFAULT_BOT_FEEDBACK_DIR = "reports/feedback_training/live"
DEFAULT_BOT_SMOKE_FEEDBACK_DIR = "reports/feedback_training/smoke"
DEFAULT_BOT_KNOWLEDGE_DB = "data/processed/sqlite/bot_knowledge.db"
DEFAULT_BOT_RUNTIME_PROBE_DIR = "reports/runtime_probe"
DEFAULT_KNOWLEDGE_MIN_SAMPLES = 3
DEFAULT_KNOWLEDGE_EXPLORATION = 0.16
MATCH_EVENT_HINTS = (
    "games_played/round_spawn",
    "games_played/round_start",
    "games_played/round_after_spawn",
    "frame_rate/playing",
    "active_engagement",
    "player_match_result/pmr_received",
)
LOBBY_EVENT_HINTS = (
    "games_played/state_lobby",
    "loading/lobby_loaded",
    "loading/lobby_interactable",
    "games_played/play_button_hit",
)
LOADING_EVENT_HINTS = (
    "connection/connecting",
    "loading/config_loaded",
    "loading/unity_loaded",
    "authentication/authentication_start",
)
MATCH_END_EVENT_HINTS = (
    "games_played/match_end",
    "player_match_result/pmr_received",
    "player_match_result/match_end",
)
DEATH_EVENT_HINTS = (
    "player_death",
    "player_dead",
    "death",
    "died",
    "dead",
    "eliminated",
    "defeat",
    "defeat_screen",
    "defeated",
    "round_lost",
    "match_lost",
    "you_lost",
    "you_lose",
    "you lose",
    "game_over",
    "knocked",
    "knockout",
    "you_died",
)
MOVE_KEY_ALIASES = {
    "up": "ArrowUp",
    "arrowup": "ArrowUp",
    "down": "ArrowDown",
    "arrowdown": "ArrowDown",
    "left": "ArrowLeft",
    "arrowleft": "ArrowLeft",
    "right": "ArrowRight",
    "arrowright": "ArrowRight",
    "w": "ArrowUp",
    "s": "ArrowDown",
    "a": "ArrowLeft",
    "d": "ArrowRight",
    "keyw": "KeyW",
    "keys": "KeyS",
    "keya": "KeyA",
    "keyd": "KeyD",
}
MOVE_VECTORS = {
    "KeyW": (0, -1),
    "KeyS": (0, 1),
    "KeyA": (-1, 0),
    "KeyD": (1, 0),
}
ORTHOGONAL_STRAFE = {
    "KeyW": "KeyD",
    "KeyD": "KeyS",
    "KeyS": "KeyA",
    "KeyA": "KeyW",
}
OPPOSITE_KEY = {
    "KeyW": "KeyS",
    "KeyS": "KeyW",
    "KeyA": "KeyD",
    "KeyD": "KeyA",
}
ABILITY_CLASSIFICATION = {
    "Digit1": "offense",
    "Digit2": "mobility",
    "Digit3": "defense",
    "Shift": "dash",
    "MouseRight": "dash_mouse",
}


def ensure_live_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_players (
            name TEXT PRIMARY KEY,
            sightings INTEGER NOT NULL DEFAULT 1,
            first_seen INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            last_seen INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_characters (
            name TEXT PRIMARY KEY,
            sightings INTEGER NOT NULL DEFAULT 1,
            first_seen INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            last_seen INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            source TEXT
        )
        """
    )
    conn.commit()


def ensure_ws_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ws_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at_ms INTEGER NOT NULL,
            direction TEXT NOT NULL,
            ws_request_id TEXT,
            ws_url TEXT,
            opcode INTEGER,
            payload_len INTEGER,
            decoded_kind TEXT,
            keyword_hit TEXT,
            text_preview TEXT,
            payload_b64 TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ws_frames_keyword ON ws_frames(keyword_hit)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ws_frames_reqid ON ws_frames(ws_request_id)"
    )
    conn.commit()


def configure_sqlite_for_realtime(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA temp_store=MEMORY")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA mmap_size=268435456")
    except Exception:
        pass


def ensure_bot_knowledge_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_match_runs (
            run_id TEXT PRIMARY KEY,
            started_at_ms INTEGER NOT NULL,
            ended_at_ms INTEGER,
            mode TEXT,
            stop_reason TEXT,
            map_name TEXT,
            damage_done REAL,
            damage_taken REAL,
            steps INTEGER,
            avg_motion REAL,
            enemy_seen_steps INTEGER,
            zone_observed_steps INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_step_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            bot_state TEXT,
            context_key TEXT,
            action_kind TEXT,
            action_key TEXT,
            action_label TEXT,
            motion_score REAL,
            enemy_seen INTEGER,
            enemy_conf REAL,
            enemy_dir TEXT,
            zone_countdown_sec REAL,
            safe_zone_x REAL,
            safe_zone_y REAL,
            safe_zone_radius REAL,
            ability_key TEXT,
            ability_used INTEGER,
            ability_ready_snapshot TEXT,
            damage_done_total REAL,
            damage_taken_total REAL,
            reward REAL
        )
        """
    )
    # Backward-compatible schema upgrades for existing local DBs.
    for alter_sql in (
        "ALTER TABLE bot_step_feedback ADD COLUMN death_cause TEXT",
        "ALTER TABLE bot_step_feedback ADD COLUMN death_cause_conf REAL",
        "ALTER TABLE bot_step_feedback ADD COLUMN death_attacker_name TEXT",
        "ALTER TABLE bot_step_feedback ADD COLUMN death_attacker_is_bot INTEGER",
        "ALTER TABLE bot_step_feedback ADD COLUMN own_guardian TEXT",
        "ALTER TABLE bot_step_feedback ADD COLUMN enemy_guardian TEXT",
        "ALTER TABLE bot_step_feedback ADD COLUMN dash_cooldown_sec REAL",
        "ALTER TABLE bot_step_feedback ADD COLUMN loot_context TEXT",
        "ALTER TABLE bot_match_runs ADD COLUMN death_cause TEXT",
        "ALTER TABLE bot_match_runs ADD COLUMN death_cause_conf REAL",
        "ALTER TABLE bot_match_runs ADD COLUMN own_guardian TEXT",
        "ALTER TABLE bot_match_runs ADD COLUMN death_attacker_name TEXT",
    ):
        try:
            conn.execute(alter_sql)
        except Exception:
            pass
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bot_step_feedback_runid_ts
        ON bot_step_feedback(run_id, ts_ms)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_policy_stats (
            context_key TEXT NOT NULL,
            action_key TEXT NOT NULL,
            samples INTEGER NOT NULL,
            reward_sum REAL NOT NULL,
            reward_avg REAL NOT NULL,
            success_rate REAL NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (context_key, action_key)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bot_policy_stats_updated
        ON bot_policy_stats(updated_at_ms)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_death_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            cause TEXT,
            cause_conf REAL,
            attacker_name TEXT,
            attacker_is_bot INTEGER,
            zone_toxic INTEGER,
            zone_outside INTEGER,
            enemy_recent INTEGER,
            map_name TEXT,
            event_name TEXT,
            visual_state TEXT,
            visual_conf REAL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bot_death_events_runid_ts
        ON bot_death_events(run_id, ts_ms)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_guardian_ability_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            owner_type TEXT,
            guardian_name TEXT,
            ability_key TEXT,
            ability_class TEXT,
            cooldown_sec REAL,
            ready INTEGER,
            source TEXT,
            event_path TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bot_guardian_obs_runid_ts
        ON bot_guardian_ability_observations(run_id, ts_ms)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_loot_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            loot_type TEXT,
            loot_name TEXT,
            source TEXT,
            context_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_bot_loot_obs_runid_ts
        ON bot_loot_observations(run_id, ts_ms)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_guardian_catalog (
            guardian_name TEXT PRIMARY KEY,
            sightings INTEGER NOT NULL DEFAULT 1,
            first_seen_ms INTEGER NOT NULL,
            last_seen_ms INTEGER NOT NULL,
            source TEXT
        )
        """
    )
    conn.commit()


def load_policy_cache(conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict[str, float]]:
    cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    try:
        rows = conn.execute(
            """
            SELECT context_key, action_key, samples, reward_sum, reward_avg, success_rate
            FROM bot_policy_stats
            """
        ).fetchall()
    except Exception:
        return cache
    for row in rows:
        try:
            context_key = str(row[0] or "")
            action_key = str(row[1] or "")
            cache[(context_key, action_key)] = {
                "samples": float(row[2] or 0.0),
                "reward_sum": float(row[3] or 0.0),
                "reward_avg": float(row[4] or 0.0),
                "success_rate": float(row[5] or 0.0),
            }
        except Exception:
            continue
    return cache


def pick_action_from_policy(
    candidates: List[str],
    context_key: str,
    policy_cache: Dict[Tuple[str, str], Dict[str, float]],
    min_samples: int,
    exploration: float,
    context_action_penalties: Optional[Dict[Tuple[str, str], float]] = None,
    action_penalties: Optional[Dict[str, float]] = None,
    map_action_penalties: Optional[Dict[Tuple[str, str], float]] = None,
    map_name: str = "",
    penalty_weight: float = 1.0,
) -> Tuple[str, float, int]:
    if not candidates:
        return "", 0.0, 0
    uniq_candidates: List[str] = []
    for action in candidates:
        action_s = str(action or "").strip()
        if action_s and action_s not in uniq_candidates:
            uniq_candidates.append(action_s)
    if not uniq_candidates:
        return "", 0.0, 0

    explore = random.random() < max(0.0, min(1.0, float(exploration)))
    if explore:
        picked = random.choice(uniq_candidates)
        return picked, -0.01, 0

    best_action = uniq_candidates[0]
    best_score = -1e9
    best_samples = 0
    min_s = max(1, int(min_samples))
    for action in uniq_candidates:
        stats = policy_cache.get((context_key, action))
        if not stats:
            score = -0.015
            samples = 0
        else:
            samples = int(stats.get("samples", 0.0) or 0.0)
            reward_avg = float(stats.get("reward_avg", 0.0) or 0.0)
            success_rate = float(stats.get("success_rate", 0.0) or 0.0)
            if samples < min_s:
                score = reward_avg - 0.01
            else:
                score = reward_avg + ((success_rate - 0.5) * 0.18) + min(0.10, math.log1p(samples) / 36.0)
        if penalty_weight > 0.0:
            penalty_total = 0.0
            if context_action_penalties:
                penalty_total += float(context_action_penalties.get((context_key, action), 0.0) or 0.0)
            if action_penalties:
                penalty_total += float(action_penalties.get(action, 0.0) or 0.0)
            if map_action_penalties:
                map_key = str(map_name or "").strip().lower()
                if map_key:
                    penalty_total += float(map_action_penalties.get((map_key, action), 0.0) or 0.0)
            if penalty_total > 0.0:
                score -= penalty_total * float(penalty_weight)
        if score > best_score:
            best_score = score
            best_action = action
            best_samples = samples
    return best_action, best_score, best_samples


def update_policy_cache_and_store(
    conn: sqlite3.Connection,
    policy_cache: Dict[Tuple[str, str], Dict[str, float]],
    context_key: str,
    action_key: str,
    reward: float,
    success: bool,
    now_ms: int,
) -> None:
    key = (str(context_key or ""), str(action_key or ""))
    old = policy_cache.get(key, {})
    samples = int(old.get("samples", 0.0) or 0.0) + 1
    reward_sum = float(old.get("reward_sum", 0.0) or 0.0) + float(reward)
    reward_avg = reward_sum / float(max(1, samples))
    old_success_rate = float(old.get("success_rate", 0.0) or 0.0)
    success_num = 1.0 if success else 0.0
    success_rate = ((old_success_rate * float(samples - 1)) + success_num) / float(max(1, samples))
    record = {
        "samples": float(samples),
        "reward_sum": float(reward_sum),
        "reward_avg": float(reward_avg),
        "success_rate": float(success_rate),
    }
    policy_cache[key] = record
    conn.execute(
        """
        INSERT INTO bot_policy_stats(
            context_key, action_key, samples, reward_sum, reward_avg, success_rate, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(context_key, action_key) DO UPDATE SET
            samples = excluded.samples,
            reward_sum = excluded.reward_sum,
            reward_avg = excluded.reward_avg,
            success_rate = excluded.success_rate,
            updated_at_ms = excluded.updated_at_ms
        """,
        (
            key[0],
            key[1],
            samples,
            reward_sum,
            reward_avg,
            success_rate,
            int(now_ms),
        ),
    )


def build_policy_context_key(
    bot_state: str,
    enemy_detected: bool,
    enemy_near: bool,
    enemy_dir: str,
    escape_mode: bool,
    zone_countdown_sec: float,
    visual_state_hint: str,
    zone_escape_mode: bool = False,
    zone_outside_safe: bool = False,
    zone_signal_source: str = "none",
) -> str:
    if zone_countdown_sec < 0:
        zone_band = "none"
    elif zone_countdown_sec <= 8.0:
        zone_band = "critical"
    elif zone_countdown_sec <= 18.0:
        zone_band = "high"
    elif zone_countdown_sec <= 34.0:
        zone_band = "mid"
    else:
        zone_band = "low"
    return (
        f"st={str(bot_state or 'unknown')}"
        f"|enemy={1 if enemy_detected else 0}"
        f"|near={1 if enemy_near else 0}"
        f"|dir={str(enemy_dir or 'CENTER')}"
        f"|escape={1 if escape_mode else 0}"
        f"|zone={zone_band}"
        f"|zesc={1 if zone_escape_mode else 0}"
        f"|zout={1 if zone_outside_safe else 0}"
        f"|zsrc={str(zone_signal_source or 'none')}"
        f"|vis={str(visual_state_hint or 'unknown')}"
    )


def compute_stuck_penalty_score(
    samples: int,
    stuck_samples: int,
    avg_motion: float,
    avg_reward: float,
    motion_threshold: float,
) -> float:
    n = max(0, int(samples))
    if n <= 0:
        return 0.0
    stuck_ratio = float(stuck_samples) / float(n)
    threshold = max(0.2, float(motion_threshold))
    low_motion_ratio = max(0.0, (threshold - float(avg_motion)) / threshold)
    neg_reward = max(0.0, -float(avg_reward))
    confidence = min(1.0, math.log1p(float(n)) / 2.40)
    raw_penalty = (
        (stuck_ratio * 0.44)
        + (low_motion_ratio * 0.24)
        + (min(1.0, neg_reward / 2.8) * 0.18)
    ) * confidence
    return max(0.0, min(0.65, raw_penalty))


def load_historical_move_penalties(
    conn: sqlite3.Connection,
    motion_threshold: float,
    max_rows: int,
    min_samples: int,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], Dict[Tuple[str, str], float]]:
    context_action_penalties: Dict[Tuple[str, str], float] = {}
    action_penalties: Dict[str, float] = {}
    map_action_penalties: Dict[Tuple[str, str], float] = {}
    limit_rows = max(200, int(max_rows))
    min_n = max(1, int(min_samples))
    threshold = max(0.2, float(motion_threshold))
    try:
        rows = conn.execute(
            """
            WITH recent AS (
                SELECT context_key, action_key, motion_score, reward
                FROM bot_step_feedback
                WHERE action_kind='move'
                  AND action_key IS NOT NULL
                  AND action_key <> ''
                ORDER BY id DESC
                LIMIT ?
            )
            SELECT
                context_key,
                action_key,
                COUNT(*) AS n,
                SUM(CASE WHEN motion_score IS NULL OR motion_score < ? THEN 1 ELSE 0 END) AS stuck_n,
                AVG(CASE WHEN motion_score IS NULL THEN 0.0 ELSE motion_score END) AS avg_motion,
                AVG(COALESCE(reward, 0.0)) AS avg_reward
            FROM recent
            GROUP BY context_key, action_key
            """,
            (limit_rows, threshold),
        ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        try:
            context_key = str(row[0] or "")
            action_key = str(row[1] or "")
            n = int(row[2] or 0)
            if n < min_n or not context_key or not action_key:
                continue
            penalty = compute_stuck_penalty_score(
                samples=n,
                stuck_samples=int(row[3] or 0),
                avg_motion=float(row[4] or 0.0),
                avg_reward=float(row[5] or 0.0),
                motion_threshold=threshold,
            )
            if penalty >= 0.01:
                context_action_penalties[(context_key, action_key)] = penalty
        except Exception:
            continue

    try:
        rows = conn.execute(
            """
            WITH recent AS (
                SELECT action_key, motion_score, reward
                FROM bot_step_feedback
                WHERE action_kind='move'
                  AND action_key IS NOT NULL
                  AND action_key <> ''
                ORDER BY id DESC
                LIMIT ?
            )
            SELECT
                action_key,
                COUNT(*) AS n,
                SUM(CASE WHEN motion_score IS NULL OR motion_score < ? THEN 1 ELSE 0 END) AS stuck_n,
                AVG(CASE WHEN motion_score IS NULL THEN 0.0 ELSE motion_score END) AS avg_motion,
                AVG(COALESCE(reward, 0.0)) AS avg_reward
            FROM recent
            GROUP BY action_key
            """,
            (limit_rows, threshold),
        ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        try:
            action_key = str(row[0] or "")
            n = int(row[1] or 0)
            if n < min_n or not action_key:
                continue
            penalty = compute_stuck_penalty_score(
                samples=n,
                stuck_samples=int(row[2] or 0),
                avg_motion=float(row[3] or 0.0),
                avg_reward=float(row[4] or 0.0),
                motion_threshold=threshold,
            )
            if penalty >= 0.01:
                action_penalties[action_key] = penalty
        except Exception:
            continue

    try:
        rows = conn.execute(
            """
            WITH recent AS (
                SELECT
                    COALESCE(r.map_name, '') AS map_name,
                    s.action_key,
                    s.motion_score,
                    s.reward
                FROM bot_step_feedback s
                LEFT JOIN bot_match_runs r
                  ON r.run_id = s.run_id
                WHERE s.action_kind='move'
                  AND s.action_key IS NOT NULL
                  AND s.action_key <> ''
                ORDER BY s.id DESC
                LIMIT ?
            )
            SELECT
                map_name,
                action_key,
                COUNT(*) AS n,
                SUM(CASE WHEN motion_score IS NULL OR motion_score < ? THEN 1 ELSE 0 END) AS stuck_n,
                AVG(CASE WHEN motion_score IS NULL THEN 0.0 ELSE motion_score END) AS avg_motion,
                AVG(COALESCE(reward, 0.0)) AS avg_reward
            FROM recent
            WHERE map_name <> ''
            GROUP BY map_name, action_key
            """,
            (limit_rows, threshold),
        ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        try:
            map_name = str(row[0] or "").strip().lower()
            action_key = str(row[1] or "")
            n = int(row[2] or 0)
            if n < min_n or not map_name or not action_key:
                continue
            penalty = compute_stuck_penalty_score(
                samples=n,
                stuck_samples=int(row[3] or 0),
                avg_motion=float(row[4] or 0.0),
                avg_reward=float(row[5] or 0.0),
                motion_threshold=threshold,
            )
            if penalty >= 0.01:
                map_action_penalties[(map_name, action_key)] = penalty
        except Exception:
            continue

    return context_action_penalties, action_penalties, map_action_penalties


def update_runtime_move_penalties(
    stats_dict: Dict[str, Dict[str, float]],
    penalty_context_map: Dict[Tuple[str, str], float],
    penalty_action_map: Dict[str, float],
    penalty_map_action_map: Dict[Tuple[str, str], float],
    context_key: str,
    action_key: str,
    map_name: str,
    is_stuck: bool,
    motion_score: Optional[float],
    motion_threshold: float,
) -> None:
    ctx = str(context_key or "")
    act = str(action_key or "")
    if not ctx or not act:
        return
    map_key = str(map_name or "").strip().lower()
    metric_key_ctx = f"ctx::{ctx}::{act}"
    metric_key_act = f"act::{act}"
    metric_key_map = f"map::{map_key}::{act}" if map_key else ""
    update_targets = [metric_key_ctx, metric_key_act]
    if metric_key_map:
        update_targets.append(metric_key_map)
    for metric_key in update_targets:
        rec = stats_dict.get(metric_key)
        if rec is None:
            rec = {"samples": 0.0, "stuck": 0.0, "motion_sum": 0.0}
        rec["samples"] = float(rec.get("samples", 0.0) or 0.0) + 1.0
        if is_stuck:
            rec["stuck"] = float(rec.get("stuck", 0.0) or 0.0) + 1.0
        if motion_score is not None:
            rec["motion_sum"] = float(rec.get("motion_sum", 0.0) or 0.0) + float(motion_score)
        stats_dict[metric_key] = rec

    def penalty_for(metric_key: str) -> float:
        rec = stats_dict.get(metric_key)
        if not rec:
            return 0.0
        samples = int(rec.get("samples", 0.0) or 0.0)
        if samples <= 0:
            return 0.0
        stuck_n = int(rec.get("stuck", 0.0) or 0.0)
        motion_avg = float(rec.get("motion_sum", 0.0) or 0.0) / float(max(1, samples))
        return compute_stuck_penalty_score(
            samples=samples,
            stuck_samples=stuck_n,
            avg_motion=motion_avg,
            avg_reward=0.0,
            motion_threshold=motion_threshold,
        )

    penalty_context_map[(ctx, act)] = penalty_for(metric_key_ctx)
    penalty_action_map[act] = penalty_for(metric_key_act)
    if map_key:
        penalty_map_action_map[(map_key, act)] = penalty_for(metric_key_map)

def upsert_names(conn: sqlite3.Connection, table: str, names: Set[str], source: str) -> None:
    if not names:
        return
    clean_names = sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})
    if not clean_names:
        return
    for name in clean_names:
        conn.execute(
            f"""
            INSERT INTO {table} (name, sightings, source)
            VALUES (?, 1, ?)
            ON CONFLICT(name) DO UPDATE SET
                sightings = {table}.sightings + 1,
                last_seen = strftime('%s','now'),
                source = excluded.source
            """,
            (name, source),
        )
    conn.commit()


def looks_like_bot(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in BOT_MARKERS)


def is_probably_player_key(key: str) -> bool:
    k = key.lower()
    if k in ("map_name", "event_name", "character_name"):
        return False
    if "name" in k and any(h in k for h in PLAYER_KEY_HINTS):
        return True
    return k in ("player", "username", "display_name", "nickname")


def is_probably_character_key(key: str) -> bool:
    k = key.lower()
    if "name" in k and "character" in k:
        return True
    return any(h in k for h in CHARACTER_KEY_HINTS)


def extract_entities(node: Any, players: Set[str], characters: Set[str], parent_key: str = "") -> None:
    if isinstance(node, dict):
        # Heuristic for objects like {"player_name":"X", "is_bot":true}
        node_keys = {str(k).lower(): k for k in node.keys()}
        bot_flag = False
        for bot_key in BOT_KEY_HINTS:
            if bot_key in node_keys:
                value = node[node_keys[bot_key]]
                if isinstance(value, bool) and value:
                    bot_flag = True
                    break
                if isinstance(value, (int, float)) and value == 1:
                    bot_flag = True
                    break
                if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes"):
                    bot_flag = True
                    break

        for key, value in node.items():
            key_s = str(key)
            key_l = key_s.lower()

            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    if is_probably_character_key(key_l) and not looks_like_bot(candidate):
                        characters.add(candidate)
                    if is_probably_player_key(key_l) and not bot_flag and not looks_like_bot(candidate):
                        players.add(candidate)

            extract_entities(value, players, characters, parent_key=key_l)
        return

    if isinstance(node, list):
        parent = parent_key.lower()
        for item in node:
            if isinstance(item, str):
                candidate = item.strip()
                if not candidate or looks_like_bot(candidate):
                    continue
                if "player" in parent or "user" in parent:
                    players.add(candidate)
                if "character" in parent or "hero" in parent or "champion" in parent:
                    characters.add(candidate)
            else:
                extract_entities(item, players, characters, parent_key=parent)
        return


def print_top_events(conn: sqlite3.Connection, limit: int = 10) -> None:
    rows = conn.execute(
        """
        SELECT event_name, COUNT(*) AS n
        FROM events
        GROUP BY event_name
        ORDER BY n DESC, event_name
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("Sin eventos en base por ahora.")
        return
    print("Top event_name:")
    for event_name, count in rows:
        print(f"- {event_name}: {count}")


def print_detected_entities(title: str, names: Set[str], limit: int = 20) -> None:
    if not names:
        print(f"{title}: sin datos")
        return
    sorted_names = sorted(names)
    if len(sorted_names) > limit:
        preview = ", ".join(sorted_names[:limit])
        print(f"{title}: {preview} ... (+{len(sorted_names) - limit} mas)")
    else:
        print(f"{title}: {', '.join(sorted_names)}")


def fast_json_loads(raw_text: str) -> Any:
    if orjson is not None:
        try:
            return orjson.loads(raw_text)
        except Exception:
            pass
    return json.loads(raw_text)


def normalize_text_for_match(text: str) -> str:
    text_raw = str(text or "")
    normalized = unicodedata.normalize("NFKD", text_raw)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii", errors="ignore")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    return ascii_text


def fuzzy_contains_phrase(text_norm: str, phrase_norm: str, threshold: int = 90) -> bool:
    text_s = str(text_norm or "").strip()
    phrase_s = str(phrase_norm or "").strip()
    if not text_s or not phrase_s:
        return False
    if phrase_s in text_s:
        return True
    if fuzz is None:
        return False
    try:
        score = int(fuzz.partial_ratio(phrase_s, text_s))
    except Exception:
        return False
    return score >= int(max(1, min(100, threshold)))


def try_parse_payload_text(raw_text: str) -> Optional[Dict[str, Any]]:
    raw_text = raw_text.strip()
    if not raw_text:
        return None

    try:
        parsed = fast_json_loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"events": parsed}
    except Exception:
        pass

    # Fallback for form payloads like data=<json>.
    form = parse_qs(raw_text, keep_blank_values=True)
    for key in ("data", "payload", "events"):
        values = form.get(key)
        if not values:
            continue
        for candidate in values:
            try:
                parsed = fast_json_loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"events": parsed}
            except Exception:
                continue

    return None


def extract_request_payload(request) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        payload_obj = request.post_data_json
        if isinstance(payload_obj, dict):
            return payload_obj, "post_data_json"
        if isinstance(payload_obj, list):
            return {"events": payload_obj}, "post_data_json_list"
    except Exception:
        pass

    try:
        post_data = request.post_data
        if isinstance(post_data, str):
            parsed = try_parse_payload_text(post_data)
            if parsed is not None:
                return parsed, "post_data"
    except Exception:
        pass

    try:
        post_buffer = request.post_data_buffer
        if isinstance(post_buffer, (bytes, bytearray)) and post_buffer:
            decoded = post_buffer.decode("utf-8", errors="ignore")
            parsed = try_parse_payload_text(decoded)
            if parsed is not None:
                return parsed, "post_data_buffer"
    except Exception:
        pass

    return None, "no_payload"


def parse_keyword_csv(raw: str) -> Set[str]:
    parts = [p.strip().lower() for p in raw.split(",")]
    return {p for p in parts if p}

def parse_move_pattern_csv(raw: str) -> List[str]:
    parts = [p.strip() for p in str(raw or "").split(",")]
    parsed: List[str] = []
    for part in parts:
        if not part:
            continue
        key = MOVE_KEY_ALIASES.get(part.lower(), part)
        if key in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"):
            key = {
                "ArrowUp": "KeyW",
                "ArrowDown": "KeyS",
                "ArrowLeft": "KeyA",
                "ArrowRight": "KeyD",
            }[key]
        if key in ("KeyW", "KeyA", "KeyS", "KeyD"):
            parsed.append(key)
    if parsed:
        return parsed
    default_parts = [p.strip() for p in DEFAULT_SMOKE_MOVE_PATTERN.split(",")]
    fallback: List[str] = []
    for item in default_parts:
        if not item:
            continue
        mapped = MOVE_KEY_ALIASES.get(item.lower(), item)
        if mapped in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"):
            mapped = {
                "ArrowUp": "KeyW",
                "ArrowDown": "KeyS",
                "ArrowLeft": "KeyA",
                "ArrowRight": "KeyD",
            }[mapped]
        if mapped in ("KeyW", "KeyA", "KeyS", "KeyD"):
            fallback.append(mapped)
    return fallback or ["KeyW", "KeyS", "KeyA", "KeyD"]


def decode_ws_payload(
    opcode: int, payload_data: str
) -> Tuple[Optional[str], str, Optional[str], int]:
    if payload_data is None:
        return None, "empty", None, 0

    if opcode == 1:
        text = payload_data if isinstance(payload_data, str) else str(payload_data)
        return text, "text", None, len(text)

    if not isinstance(payload_data, str):
        return None, "non_string_binary", None, 0

    try:
        raw = base64.b64decode(payload_data)
    except Exception:
        return None, "binary_base64_decode_failed", None, len(payload_data)

    payload_b64 = payload_data
    payload_len = len(raw)

    candidates = [("binary_utf8", raw)]
    try:
        candidates.append(("binary_gzip_utf8", gzip.decompress(raw)))
    except Exception:
        pass
    for wbits, kind in (
        (zlib.MAX_WBITS, "binary_zlib_utf8"),
        (-zlib.MAX_WBITS, "binary_deflate_utf8"),
    ):
        try:
            candidates.append((kind, zlib.decompress(raw, wbits=wbits)))
        except Exception:
            pass

    for kind, blob in candidates:
        try:
            text = blob.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if text and text.strip():
            return text, kind, payload_b64, payload_len

    return None, "binary_undecoded", payload_b64, payload_len


def content_type_from_request(request) -> str:
    try:
        headers = request.headers
        if isinstance(headers, dict):
            for key, value in headers.items():
                if str(key).lower() == "content-type":
                    return str(value or "").strip().lower()
    except Exception:
        pass
    return ""


def should_capture_post_body(content_type: str) -> bool:
    if not content_type:
        return True
    text_like = (
        "application/json",
        "text/",
        "application/x-www-form-urlencoded",
        "application/graphql",
    )
    return any(marker in content_type for marker in text_like)


def read_request_body_text(request) -> Optional[str]:
    try:
        post_buffer = request.post_data_buffer
        if isinstance(post_buffer, (bytes, bytearray)):
            return bytes(post_buffer).decode("utf-8", errors="ignore")
    except Exception:
        pass

    try:
        post_data = request.post_data
        if isinstance(post_data, str):
            return post_data
    except Exception:
        pass

    return None


def persist_match_stats(
    conn: sqlite3.Connection,
    event_id: int,
    screenshot_path: Optional[str],
    ocr_result: Dict[str, Any],
    source: str,
) -> None:
    metrics = ocr_result.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    def metric_value(name: str) -> Optional[float]:
        metric = metrics.get(name)
        if isinstance(metric, dict):
            candidate = metric.get("value")
        else:
            candidate = metric
        if isinstance(candidate, (int, float)):
            return float(candidate)
        return None

    damage_done = metric_value("damage_done")
    damage_taken = metric_value("damage_taken")
    kills = metric_value("kills")
    crafted = metric_value("crafted")
    items_crafted = metric_value("items_crafted")

    confidence_candidates = []
    for metric in metrics.values():
        if isinstance(metric, dict):
            conf = metric.get("confidence")
            if isinstance(conf, (int, float)):
                confidence_candidates.append(float(conf))
    ocr_confidence = (
        sum(confidence_candidates) / len(confidence_candidates)
        if confidence_candidates
        else None
    )

    conn.execute(
        """
        INSERT INTO match_stats (
            event_id,
            damage_done,
            damage_taken,
            kills,
            crafted,
            items_crafted,
            ocr_confidence,
            screenshot_path,
            source,
            raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            damage_done=excluded.damage_done,
            damage_taken=excluded.damage_taken,
            kills=excluded.kills,
            crafted=excluded.crafted,
            items_crafted=excluded.items_crafted,
            ocr_confidence=excluded.ocr_confidence,
            screenshot_path=excluded.screenshot_path,
            source=excluded.source,
            raw_json=excluded.raw_json,
            captured_at=strftime('%s','now')
        """,
        (
            event_id,
            damage_done,
            damage_taken,
            kills,
            crafted,
            items_crafted,
            ocr_confidence,
            screenshot_path,
            source,
            json.dumps(ocr_result, ensure_ascii=True),
        ),
    )
    conn.commit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Captura en vivo de LogEvents -> SQLite (sin exportar HAR)."
    )
    parser.add_argument("--db", default="data/processed/sqlite/lms_events.db", help="Ruta SQLite.")
    parser.add_argument(
        "--game-url",
        default="https://lastmagestanding.com/",
        help="URL del juego.",
    )
    parser.add_argument(
        "--endpoint-substring",
        default="/GameFunction/LogEvents",
        help="Texto para identificar requests de eventos.",
    )
    parser.add_argument(
        "--capture-all-post",
        action="store_true",
        help="Captura body de todos los POST (JSON/text) y los guarda en raw_requests.",
    )
    parser.add_argument(
        "--capture-all-max-bytes",
        type=int,
        default=DEFAULT_CAPTURE_ALL_MAX_BYTES,
        help="Limite de bytes para guardar body_text cuando --capture-all-post esta activo.",
    )
    parser.add_argument(
        "--profile-dir",
        default="playwright_profile",
        help=(
            "Perfil persistente del navegador para mantener login/cookies. "
            "Usa ruta vacia para contexto temporal."
        ),
    )
    parser.add_argument(
        "--channel",
        default="chromium",
        help="Canal del navegador: chromium, chrome, msedge.",
    )
    parser.add_argument(
        "--stealth-login",
        action="store_true",
        help=(
            "Quita flags tipicas de automatizacion para mejorar login OAuth "
            "(util cuando Google bloquea acceso)."
        ),
    )
    parser.add_argument(
        "--bot-google-login",
        action="store_true",
        help=(
            "Antes de jugar, intenta autenticar cuenta Google en el navegador "
            "para que el juego use la sesion logueada."
        ),
    )
    parser.add_argument(
        "--bot-google-email",
        default="",
        help=(
            "Email de cuenta Google para login previo (si vacio, usa env LMS_GOOGLE_EMAIL)."
        ),
    )
    parser.add_argument(
        "--bot-google-password",
        default="",
        help=(
            "Password de cuenta Google para login previo (si vacio, usa env LMS_GOOGLE_PASSWORD)."
        ),
    )
    parser.add_argument(
        "--bot-google-login-timeout-sec",
        type=float,
        default=110.0,
        help="Timeout total del intento de login Google previo a jugar.",
    )
    parser.add_argument(
        "--no-chromium-sandbox",
        action="store_true",
        help="Desactiva sandbox de Chromium. No recomendado.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecutar navegador sin UI (no recomendado si juegas manual).",
    )
    parser.add_argument(
        "--report-every-sec",
        type=int,
        default=60,
        help="Intervalo de reporte automatico.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=10,
        help="Limite de filas para reporte de partidas.",
    )
    parser.add_argument(
        "--print-each-request",
        action="store_true",
        help="Imprime cada request LogEvents detectado.",
    )
    parser.add_argument(
        "--print-event-names",
        action="store_true",
        help="Imprime eventName detectados en tiempo real.",
    )
    parser.add_argument(
        "--print-entity-updates",
        action="store_true",
        help="Imprime nuevos jugadores/personajes detectados en vivo.",
    )
    parser.add_argument(
        "--no-persistent",
        action="store_true",
        help="No usar perfil persistente; inicia contexto temporal.",
    )
    parser.add_argument(
        "--scan-websocket",
        action="store_true",
        help="Escanea mensajes websocket para detectar jugadores/personajes.",
    )
    parser.add_argument(
        "--ws-player-heuristics",
        action="store_true",
        help=(
            "Activa deteccion heuristica de nombres desde websocket. "
            "Es experimental y puede dar falsos positivos."
        ),
    )
    parser.add_argument(
        "--cdp-url",
        default="",
        help=(
            "URL CDP para conectarse a un Chrome ya abierto, ejemplo "
            "http://127.0.0.1:9222. Evita login en navegador automatizado."
        ),
    )
    parser.add_argument(
        "--cdp-new-page",
        action="store_true",
        help="Al usar --cdp-url, abre una nueva pestaÃƒÂ±a para --game-url.",
    )
    parser.add_argument(
        "--ws-keywords",
        default=DEFAULT_WS_KEYWORDS,
        help="Palabras clave para detectar metricas en frames websocket.",
    )
    parser.add_argument(
        "--ws-save-frames",
        action="store_true",
        help="Guardar frames websocket en tabla ws_frames.",
    )
    parser.add_argument(
        "--ws-max-saved",
        type=int,
        default=2000,
        help="Maximo de frames websocket a guardar en SQLite.",
    )
    parser.add_argument(
        "--ws-save-binary",
        action="store_true",
        help="Guardar payload base64 de frames binarios.",
    )
    parser.add_argument(
        "--ws-print-keyword-hits",
        action="store_true",
        help="Imprime frames websocket cuando detecta keywords.",
    )
    parser.add_argument(
        "--ocr-on-match-end",
        action="store_true",
        help="Al detectar games_played/match_end, captura screenshot del canvas y ejecuta OCR.",
    )
    parser.add_argument(
        "--ocr-config",
        default="",
        help="Ruta al JSON de ROIs para lms_ocr_extractor.py.",
    )
    parser.add_argument(
        "--ocr-output-dir",
        default="data/processed/ocr",
        help="Directorio para screenshots capturadas al final de partida.",
    )
    parser.add_argument(
        "--ocr-canvas-selector",
        default="canvas",
        help="Selector CSS del canvas del juego para screenshot.",
    )
    parser.add_argument(
        "--ocr-script",
        default="lms_ocr_extractor.py",
        help="Ruta al script OCR.",
    )
    parser.add_argument(
        "--ocr-timeout-sec",
        type=int,
        default=30,
        help="Timeout del proceso OCR por captura.",
    )
    parser.add_argument(
        "--play-game",
        action="store_true",
        help="Activar el bot para jugar una partida automaticamente."
    )
    parser.add_argument(
        "--bot-iframe-selector",
        default="#game_iframe",
        help="Selector CSS del iframe del juego para el modo --play-game.",
    )
    parser.add_argument(
        "--bot-frame-timeout-ms",
        type=int,
        default=30000,
        help="Timeout para resolver el Frame del juego en modo bot.",
    )
    parser.add_argument(
        "--bot-action-timeout-ms",
        type=int,
        default=12000,
        help="Timeout para localizar/clicar controles UI del juego en modo bot.",
    )
    parser.add_argument(
        "--bot-click-mode",
        choices=("mouse", "dom", "hybrid"),
        default="hybrid",
        help="Modo de click para ataque: mouse, dom o hybrid.",
    )
    parser.add_argument(
        "--bot-log-ui-state",
        action="store_true",
        help="Imprime estado UI detectado dentro del iframe (debug bot).",
    )
    parser.add_argument(
        "--bot-ui-poll-ms",
        type=int,
        default=750,
        help="Intervalo de polling UI del bot en ms.",
    )
    parser.add_argument(
        "--bot-visual-cursor",
        dest="bot_visual_cursor",
        action="store_true",
        default=True,
        help="Muestra un cursor visual del bot dentro del frame del juego.",
    )
    parser.add_argument(
        "--no-bot-visual-cursor",
        dest="bot_visual_cursor",
        action="store_false",
        help="Desactiva el cursor visual del bot.",
    )
    parser.add_argument(
        "--bot-mouse-move-steps",
        type=int,
        default=10,
        help="Cantidad de pasos para mover mouse antes del click del bot.",
    )
    parser.add_argument(
        "--bot-cursor-transition-ms",
        type=int,
        default=45,
        help="Duracion de transicion del cursor visual (ms). Menor = respuesta mas rapida.",
    )
    parser.add_argument(
        "--bot-cursor-log-interval-sec",
        type=float,
        default=2.0,
        help="Intervalo para log de confirmacion de movimiento del cursor visual.",
    )
    parser.add_argument(
        "--bot-cursor-idle-amplitude",
        type=float,
        default=10.0,
        help="Amplitud (px) para micro-movimiento del cursor visual en espera.",
    )
    parser.add_argument(
        "--bot-smoke-test",
        action="store_true",
        help="Ejecuta prueba deterministica de cursor/click sobre una pagina controlada.",
    )
    parser.add_argument(
        "--bot-smoke-steps",
        type=int,
        default=16,
        help="Cantidad de pasos de movimiento en smoke test.",
    )
    parser.add_argument(
        "--bot-smoke-step-ms",
        type=int,
        default=220,
        help="Pausa por paso en smoke test (ms).",
    )
    parser.add_argument(
        "--bot-smoke-output",
        default="reports/smoke/bot_smoke_test_last.json",
        help="Ruta JSON de salida para resultados del smoke test.",
    )
    parser.add_argument(
        "--bot-smoke-screenshot",
        default="reports/smoke/bot_smoke_test_last.png",
        help="Ruta screenshot final del smoke test.",
    )
    parser.add_argument(
        "--bot-auto-phases",
        action="store_true",
        help="Fase 1 smoke test; si pasa, Fase 2 play-game automaticamente.",
    )
    parser.add_argument(
        "--bot-auto-stop-on-smoke-fail",
        action="store_true",
        default=True,
        help="En modo auto, detiene flujo si smoke test falla.",
    )
    parser.add_argument(
        "--no-bot-auto-stop-on-smoke-fail",
        dest="bot_auto_stop_on_smoke_fail",
        action="store_false",
        help="En modo auto, continua a play-game aunque smoke falle.",
    )
    parser.add_argument(
        "--bot-parallel-smoke",
        action="store_true",
        help="Ejecuta smoke monitor en paralelo mientras corre play-game.",
    )
    parser.add_argument(
        "--bot-smoke-move-hold-ms",
        type=int,
        default=180,
        help="Duracion de pulsacion por direccion en in_match cuando --bot-parallel-smoke esta activo.",
    )
    parser.add_argument(
        "--bot-smoke-move-pattern",
        default=DEFAULT_SMOKE_MOVE_PATTERN,
        help=(
            "Patron de direcciones para in_match en modo smoke paralelo, separado por coma. "
            "Ejemplo: KeyW,KeyW,KeyA,KeyD,KeyS (tambien acepta Arrow/WASD)"
        ),
    )
    parser.add_argument(
        "--bot-move-base-hold-ms",
        type=int,
        default=260,
        help="Duracion base de movimiento en in_match (ms).",
    )
    parser.add_argument(
        "--bot-move-click-every-steps",
        type=int,
        default=2,
        help="Frecuencia de click de ataque en in_match (cada N pasos; 1=siempre).",
    )
    parser.add_argument(
        "--bot-move-motion-sample-every",
        type=int,
        default=2,
        help="Evalua movimiento visual cada N pasos para reducir latencia.",
    )
    parser.add_argument(
        "--bot-move-motion-threshold",
        type=float,
        default=1.9,
        help="Umbral minimo de cambio visual (0-255) para considerar que hubo desplazamiento real.",
    )
    parser.add_argument(
        "--bot-move-stuck-streak",
        type=int,
        default=2,
        help="Cantidad de ciclos con bajo movimiento para activar maniobra de escape.",
    )
    parser.add_argument(
        "--bot-move-escape-steps",
        type=int,
        default=3,
        help="Pasos de movimiento de escape al detectar atasco.",
    )
    parser.add_argument(
        "--bot-opening-move-sec",
        type=float,
        default=11.0,
        help="Segundos de patron de apertura agresivo al entrar a partida.",
    )
    parser.add_argument(
        "--bot-opening-hold-multiplier",
        type=float,
        default=1.9,
        help="Multiplicador de hold durante fase de apertura sin enemigo.",
    )
    parser.add_argument(
        "--bot-collision-streak-threshold",
        type=int,
        default=2,
        help="Streak de baja movilidad para marcar colision probable.",
    )
    parser.add_argument(
        "--bot-collision-escape-extra-steps",
        type=int,
        default=1,
        help="Pasos extra de escape cuando hay colision probable.",
    )
    parser.add_argument(
        "--bot-stuck-repeat-action-streak",
        type=int,
        default=3,
        help="Cantidad de repeticiones de la misma accion de movimiento para activar senal de atasco.",
    )
    parser.add_argument(
        "--bot-stuck-confirm-streak",
        type=int,
        default=2,
        help="Ticks consecutivos con senales de atasco para confirmar bloqueo.",
    )
    parser.add_argument(
        "--bot-stuck-recovery-steps",
        type=int,
        default=7,
        help="Cantidad de pasos de plan de recuperacion cuando se confirma atasco.",
    )
    parser.add_argument(
        "--bot-stuck-motion-factor",
        type=float,
        default=0.92,
        help="Factor del umbral de motion para considerar bajo movimiento en deteccion multi-senal.",
    )
    parser.add_argument(
        "--bot-move-diagonal",
        dest="bot_move_diagonal",
        action="store_true",
        default=True,
        help="Habilita movimiento diagonal combinado para reducir colisiones.",
    )
    parser.add_argument(
        "--no-bot-move-diagonal",
        dest="bot_move_diagonal",
        action="store_false",
        help="Desactiva movimiento diagonal combinado.",
    )
    parser.add_argument(
        "--bot-enemy-vision",
        dest="bot_enemy_vision",
        action="store_true",
        default=True,
        help="Activa deteccion visual heuristica de enemigos desde canvas.",
    )
    parser.add_argument(
        "--no-bot-enemy-vision",
        dest="bot_enemy_vision",
        action="store_false",
        help="Desactiva deteccion visual heuristica de enemigos.",
    )
    parser.add_argument(
        "--bot-enemy-vision-interval-ms",
        type=int,
        default=420,
        help="Intervalo de escaneo visual de enemigo (ms).",
    )
    parser.add_argument(
        "--bot-enemy-red-ratio-threshold",
        type=float,
        default=0.008,
        help="Umbral minimo de pixeles rojos para marcar deteccion enemiga.",
    )
    parser.add_argument(
        "--bot-enemy-min-area",
        type=float,
        default=55.0,
        help="Area minima de contorno rojo para validar enemigo.",
    )
    parser.add_argument(
        "--bot-run-until-end",
        action="store_true",
        help="Mantiene el bot activo hasta detectar match_end o muerte del bot.",
    )
    parser.add_argument(
        "--bot-run-stop-on-death-only",
        action="store_true",
        help="Con --bot-run-until-end, ignora match_end y solo detiene al detectar muerte del bot.",
    )
    parser.add_argument(
        "--bot-run-max-sec",
        type=int,
        default=1200,
        help="Limite maximo (segundos) para --bot-run-until-end antes de cortar por seguridad.",
    )
    parser.add_argument(
        "--bot-open-map-every-sec",
        type=float,
        default=14.0,
        help="Intervalo para abrir/cerrar mapa con tecla C durante in_match (0 desactiva).",
    )
    parser.add_argument(
        "--bot-ability-every-sec",
        type=float,
        default=3.8,
        help="Intervalo base para intento de habilidades 1/2/3.",
    )
    parser.add_argument(
        "--bot-decision-backend",
        choices=("legacy", "lms_re"),
        default="legacy",
        help=(
            "Backend de decision para in_match. "
            "'legacy' usa heuristicas actuales; 'lms_re' usa LMSReverseEngineeredBot."
        ),
    )
    parser.add_argument(
        "--bot-lmsre-mode-name",
        default="royale_mode",
        help="Modo usado por LMSReverseEngineeredBot para cargar reglas (ej: royale_mode, red_vs_blue_mode).",
    )
    parser.add_argument(
        "--bot-debug-hud",
        dest="bot_debug_hud",
        action="store_true",
        default=True,
        help="Muestra HUD visual de feedback/reaccion en tiempo real dentro del frame.",
    )
    parser.add_argument(
        "--no-bot-debug-hud",
        dest="bot_debug_hud",
        action="store_false",
        help="Desactiva HUD visual de feedback/reaccion.",
    )
    parser.add_argument(
        "--bot-feedback-dir",
        default=DEFAULT_BOT_FEEDBACK_DIR,
        help="Directorio base para guardar stream de feedback y screenshots de entrenamiento.",
    )
    parser.add_argument(
        "--bot-feedback-jsonl",
        default="",
        help="Ruta JSONL explicita para guardar feedback runtime (si vacio, auto en --bot-feedback-dir).",
    )
    parser.add_argument(
        "--bot-feedback-screenshot-every-sec",
        type=float,
        default=0.0,
        help="Intervalo de screenshots del juego con HUD para feedback (0 desactiva).",
    )
    parser.add_argument(
        "--bot-feedback-max-screenshots",
        type=int,
        default=120,
        help="Maximo de screenshots de feedback runtime por corrida.",
    )
    parser.add_argument(
        "--bot-visual-ocr",
        dest="bot_visual_ocr",
        action="store_true",
        default=True,
        help="Extrae texto de screenshots temporales (nombres/daÃ±o/estado) para feedback en tiempo real.",
    )
    parser.add_argument(
        "--no-bot-visual-ocr",
        dest="bot_visual_ocr",
        action="store_false",
        help="Desactiva OCR visual en screenshots temporales.",
    )
    parser.add_argument(
        "--bot-visual-ocr-every-sec",
        type=float,
        default=2.0,
        help="Intervalo minimo para procesar OCR visual sobre screenshots runtime.",
    )
    parser.add_argument(
        "--bot-visual-ocr-max-frames",
        type=int,
        default=180,
        help="Maximo de frames procesados por OCR visual en una corrida.",
    )
    parser.add_argument(
        "--bot-visual-ocr-max-names",
        type=int,
        default=8,
        help="Maximo de nombres OCR guardados por frame.",
    )
    parser.add_argument(
        "--bot-feedback-render-video",
        action="store_true",
        help="Renderiza un video MP4 de la secuencia de screenshots runtime al finalizar.",
    )
    parser.add_argument(
        "--bot-feedback-video-fps",
        type=float,
        default=6.0,
        help="FPS del video MP4 de feedback runtime.",
    )
    parser.add_argument(
        "--bot-realtime-mode",
        dest="bot_realtime_mode",
        action="store_true",
        default=True,
        help=(
            "Activa optimizaciones de baja latencia para reaccion en tiempo real "
            "(websocket CDP + menor polling + ruta de red optimizada)."
        ),
    )
    parser.add_argument(
        "--no-bot-realtime-mode",
        dest="bot_realtime_mode",
        action="store_false",
        help="Desactiva optimizaciones de baja latencia del bot.",
    )
    parser.add_argument(
        "--bot-knowledge-db",
        default=DEFAULT_BOT_KNOWLEDGE_DB,
        help="Ruta SQLite para memoria/aprendizaje incremental del bot.",
    )
    parser.add_argument(
        "--bot-knowledge-enabled",
        dest="bot_knowledge_enabled",
        action="store_true",
        default=True,
        help="Guarda y reutiliza conocimiento de movimiento/habilidades por contexto.",
    )
    parser.add_argument(
        "--no-bot-knowledge-enabled",
        dest="bot_knowledge_enabled",
        action="store_false",
        help="Desactiva almacenamiento y uso de conocimiento historico.",
    )
    parser.add_argument(
        "--bot-knowledge-min-samples",
        type=int,
        default=DEFAULT_KNOWLEDGE_MIN_SAMPLES,
        help="Minimo de muestras para confiar en una accion historica.",
    )
    parser.add_argument(
        "--bot-knowledge-exploration",
        type=float,
        default=DEFAULT_KNOWLEDGE_EXPLORATION,
        help="Probabilidad de explorar acciones nuevas (0-1).",
    )
    parser.add_argument(
        "--bot-knowledge-flush-every",
        type=int,
        default=12,
        help="Cantidad de steps antes de flush/commit de memoria incremental.",
    )
    parser.add_argument(
        "--bot-history-stuck-max-rows",
        type=int,
        default=12000,
        help="Cantidad de steps historicos para estimar penalizaciones anti-atasco por accion.",
    )
    parser.add_argument(
        "--bot-history-stuck-min-samples",
        type=int,
        default=4,
        help="Minimo de muestras historicas para aplicar penalizacion anti-atasco.",
    )
    parser.add_argument(
        "--bot-history-stuck-weight",
        type=float,
        default=1.0,
        help="Peso aplicado a penalizaciones anti-atasco historicas y runtime.",
    )
    parser.add_argument(
        "--bot-smoke-feedback-dir",
        default=DEFAULT_BOT_SMOKE_FEEDBACK_DIR,
        help="Directorio base para evidencia visual del smoke test (steps + telemetry).",
    )
    parser.add_argument(
        "--bot-runtime-probe",
        dest="bot_runtime_probe",
        action="store_true",
        default=True,
        help="Captura snapshot de variables globales relevantes del runtime JS del juego.",
    )
    parser.add_argument(
        "--no-bot-runtime-probe",
        dest="bot_runtime_probe",
        action="store_false",
        help="Desactiva snapshot de runtime JS del juego.",
    )
    parser.add_argument(
        "--bot-runtime-probe-dir",
        default=DEFAULT_BOT_RUNTIME_PROBE_DIR,
        help="Directorio para guardar snapshots JSON del runtime JS.",
    )
    parser.add_argument(
        "--bot-runtime-probe-max-keys",
        type=int,
        default=140,
        help="Maximo de globals candidatos que se guardan por snapshot runtime.",
    )
    return parser


def _build_launch_kwargs(args: argparse.Namespace, channel: str) -> Dict[str, object]:
    launch_kwargs: Dict[str, object] = {
        "headless": args.headless,
    }

    if channel in ("chromium", ""):
        launch_kwargs["chromium_sandbox"] = not args.no_chromium_sandbox
    else:
        launch_kwargs["chromium_sandbox"] = False

    if channel and channel != "chromium":
        launch_kwargs["channel"] = channel

    if args.stealth_login:
        launch_kwargs["ignore_default_args"] = ["--enable-automation"]
        launch_kwargs["args"] = ["--disable-blink-features=AutomationControlled"]

    return launch_kwargs


def launch_context_with_fallback(
    p,
    args: argparse.Namespace,
    profile_dir: str,
) -> Tuple[object, object, str]:
    requested_channel = (args.channel or "chromium").strip().lower()
    channel_order = [requested_channel]
    if requested_channel != "chromium":
        channel_order.append("chromium")

    last_error: Optional[Exception] = None
    for channel in channel_order:
        launch_kwargs = _build_launch_kwargs(args, channel)

        if profile_dir and not args.no_persistent:
            try:
                context = p.chromium.launch_persistent_context(
                    user_data_dir=profile_dir,
                    **launch_kwargs,
                )
                page = context.new_page()
                return context, page, f"channel={channel}, persistent=on"
            except Exception as exc:
                last_error = exc
                print(f"[WARN] Fallo contexto persistente con {channel}: {exc}")

        try:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context()
            page = context.new_page()
            return context, page, f"channel={channel}, persistent=off"
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Fallo browser temporal con {channel}: {exc}")

    if isinstance(last_error, Exception):
        raise last_error
    raise RuntimeError("No se pudo iniciar navegador con ninguna configuracion.")


def attach_over_cdp(
    p,
    cdp_url: str,
    game_url: str,
    open_new_page: bool,
) -> Tuple[object, object, object, str]:
    browser = p.chromium.connect_over_cdp(cdp_url)
    contexts = browser.contexts
    context = contexts[0] if contexts else browser.new_context()

    pages = context.pages
    if open_new_page or not pages:
        page = context.new_page()
        if game_url:
            page.goto(game_url, wait_until="domcontentloaded")
    else:
        page = pages[0]

    return browser, context, page, f"cdp={cdp_url}, attached=yes"


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.play_game) and bool(args.bot_realtime_mode):
        args.scan_websocket = True
        args.ws_player_heuristics = True
        args.bot_ui_poll_ms = int(min(int(args.bot_ui_poll_ms), 140))
        args.bot_enemy_vision_interval_ms = int(min(int(args.bot_enemy_vision_interval_ms), 180))
    conn = sqlite3.connect(Path(args.db))
    configure_sqlite_for_realtime(conn)
    ensure_schema(conn)
    ensure_live_schema(conn)
    ensure_ws_schema(conn)
    knowledge_conn: Optional[sqlite3.Connection] = None
    knowledge_policy_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    historical_move_context_penalties: Dict[Tuple[str, str], float] = {}
    historical_move_action_penalties: Dict[str, float] = {}
    historical_move_map_action_penalties: Dict[Tuple[str, str], float] = {}
    knowledge_pending_writes = 0
    if bool(args.bot_knowledge_enabled):
        try:
            knowledge_path = Path(args.bot_knowledge_db)
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            knowledge_conn = sqlite3.connect(knowledge_path)
            configure_sqlite_for_realtime(knowledge_conn)
            ensure_bot_knowledge_schema(knowledge_conn)
            knowledge_policy_cache = load_policy_cache(knowledge_conn)
            (
                historical_move_context_penalties,
                historical_move_action_penalties,
                historical_move_map_action_penalties,
            ) = load_historical_move_penalties(
                conn=knowledge_conn,
                motion_threshold=max(0.2, float(args.bot_move_motion_threshold)),
                max_rows=max(200, int(args.bot_history_stuck_max_rows)),
                min_samples=max(1, int(args.bot_history_stuck_min_samples)),
            )
            print(
                "[BOT][KNOWLEDGE] DB listo en "
                f"{knowledge_path} stats={len(knowledge_policy_cache)}"
            )
            print(
                "[BOT][KNOWLEDGE] penalizaciones anti-atasco "
                f"context={len(historical_move_context_penalties)} "
                f"action={len(historical_move_action_penalties)} "
                f"map_action={len(historical_move_map_action_penalties)}"
            )
        except Exception as exc:
            knowledge_conn = None
            knowledge_policy_cache = {}
            historical_move_context_penalties = {}
            historical_move_action_penalties = {}
            historical_move_map_action_penalties = {}
            print(f"[BOT][KNOWLEDGE][WARN] No se pudo iniciar DB de conocimiento: {exc}")
    capture_all_max_bytes = max(1024, int(args.capture_all_max_bytes))
    ocr_timeout_sec = max(5, int(args.ocr_timeout_sec))
    ocr_script_path = Path(args.ocr_script)

    stats: Dict[str, int] = {
        "requests_seen": 0,
        "events_saved": 0,
        "payload_errors": 0,
        "no_payload": 0,
        "no_events": 0,
        "raw_posts_saved": 0,
        "raw_events_saved": 0,
        "raw_posts_skipped": 0,
        "ws_frames": 0,
        "ws_saved": 0,
        "ws_keyword_hits": 0,
        "ocr_queued": 0,
        "ocr_runs": 0,
        "ocr_saved": 0,
        "ocr_errors": 0,
    }
    ws_keywords = parse_keyword_csv(args.ws_keywords)
    ws_url_by_request_id: Dict[str, str] = {}
    ws_pending_writes = 0
    seen_players: Set[str] = set()
    seen_characters: Set[str] = set()
    pending_match_end_events: List[int] = []
    last_report = time.monotonic()
    bot_event_signals: Dict[str, Any] = {
        "last_event_name": "",
        "last_event_ts": 0.0,
        "lobby_ts": 0.0,
        "loading_ts": 0.0,
        "match_ts": 0.0,
        "match_end_ts": 0.0,
        "death_ts": 0.0,
        "death_cause": "unknown",
        "death_cause_confidence": 0.0,
        "death_cause_source": "none",
        "death_attacker_name": "",
        "death_attacker_is_bot": None,
        "damage_done_latest": 0.0,
        "damage_taken_latest": 0.0,
        "damage_done_total": 0.0,
        "damage_taken_total": 0.0,
        "map_name": "",
        "own_guardian": "",
        "enemy_guardian": "",
        "zone_countdown_sec": -1.0,
        "safe_zone_x": None,
        "safe_zone_y": None,
        "safe_zone_radius": None,
        "player_pos_x": None,
        "player_pos_y": None,
        "zone_outside_safe": False,
        "zone_toxic_detected": False,
        "zone_toxic_confidence": 0.0,
        "zone_signal_source": "none",
        "zone_signal_ts": 0.0,
        "last_rank": None,
        "mana_current": None,
        "mana_max": None,
        "loot_last_type": "",
        "loot_last_name": "",
        "loot_ts": 0.0,
        "loot_count": 0,
        "ability_ready": {
            "Digit1": None,
            "Digit2": None,
            "Digit3": None,
            "KeyR": None,
            "Shift": None,
        },
        "ability_cooldown_sec": {
            "Digit1": None,
            "Digit2": None,
            "Digit3": None,
            "KeyR": None,
            "Shift": None,
        },
        "ability_signal_ts": 0.0,
        "visual_state_hint": "unknown",
        "visual_state_confidence": 0.0,
        "visual_names": [],
        "visual_damage_hint": 0.0,
        "visual_ocr_ts": 0.0,
    }

    def iter_event_key_values(node: Any, prefix: str = "") -> List[Tuple[str, Any]]:
        pairs: List[Tuple[str, Any]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                key_s = str(key)
                next_prefix = f"{prefix}.{key_s}" if prefix else key_s
                pairs.append((next_prefix, value))
                pairs.extend(iter_event_key_values(value, next_prefix))
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                next_prefix = f"{prefix}[{idx}]"
                pairs.extend(iter_event_key_values(item, next_prefix))
        return pairs

    def accumulate_metric_with_reset(metric_key: str, latest_value: float) -> None:
        latest_field = f"{metric_key}_latest"
        total_field = f"{metric_key}_total"
        previous = float(bot_event_signals.get(latest_field, 0.0) or 0.0)
        if latest_value >= previous:
            delta = latest_value - previous
        else:
            # likely reset between rounds, keep positive progression
            delta = max(0.0, latest_value)
        bot_event_signals[latest_field] = latest_value
        bot_event_signals[total_field] = float(bot_event_signals.get(total_field, 0.0) or 0.0) + max(0.0, delta)

    def update_bot_metrics_from_event(event_obj: Dict[str, Any]) -> None:
        def ability_key_from_path(path_l: str) -> str:
            text = str(path_l or "")
            if any(h in text for h in ("ability1", "ability_1", "digit1", "slot1", "skill1", "spell1")):
                return "Digit1"
            if any(h in text for h in ("ability2", "ability_2", "digit2", "slot2", "skill2", "spell2")):
                return "Digit2"
            if any(h in text for h in ("ability3", "ability_3", "digit3", "slot3", "skill3", "spell3")):
                return "Digit3"
            if any(h in text for h in ("build_wall", "wall_ability", "keyr", "abilityr", "ability_r", "wall")):
                return "KeyR"
            if any(
                h in text
                for h in (
                    "dash",
                    "sprint",
                    "shift",
                    "roll",
                    "blink",
                    "mobility",
                    "dodge",
                    "teleport",
                )
            ):
                return "Shift"
            return ""

        guardian_tokens = (
            "character_selected",
            "character_name",
            "character",
            "hero",
            "champion",
            "guardian",
            "class_name",
        )
        attacker_tokens = (
            "killer",
            "killed_by",
            "attacker",
            "eliminated_by",
            "last_damage_source",
            "damage_source",
            "source_player",
            "source_character",
            "instigator",
        )
        loot_tokens = (
            "loot",
            "cofre",
            "chest",
            "drop",
            "pickup",
            "reward",
            "item",
            "scroll",
            "rune",
            "orb",
        )

        def classify_loot_type(path_l: str, value_l: str) -> str:
            joined = f"{path_l} {value_l}"
            if any(h in joined for h in ("chest", "cofre", "crate")):
                return "chest"
            if any(h in joined for h in ("scroll", "rune", "orb", "ability", "spell")):
                return "ability_drop"
            if any(h in joined for h in ("weapon", "wand", "staff", "gear", "armor")):
                return "gear"
            return "generic_loot"

        pairs = iter_event_key_values(event_obj)
        for path, value in pairs:
            path_l = str(path or "").lower()
            compact_path = re.sub(r"[^a-z0-9]+", "", path_l)
            if isinstance(value, str):
                value_s = str(value).strip()
                if not value_s:
                    continue
                value_l = value_s.lower()
                if ("map_name" in path_l or path_l.endswith(".map")) and len(value_s) > 0:
                    bot_event_signals["map_name"] = value_s
                if any(tok in path_l for tok in guardian_tokens):
                    if any(h in path_l for h in ("enemy", "opponent", "killer", "attacker", "target")):
                        bot_event_signals["enemy_guardian"] = value_s
                    else:
                        bot_event_signals["own_guardian"] = value_s
                if any(tok in path_l for tok in attacker_tokens) and len(value_s) >= 2:
                    bot_event_signals["death_attacker_name"] = value_s
                    bot_event_signals["death_cause_source"] = "event_attacker"
                    if looks_like_bot(value_s):
                        bot_event_signals["death_attacker_is_bot"] = True
                if any(tok in path_l for tok in loot_tokens) or any(tok in value_l for tok in loot_tokens):
                    bot_event_signals["loot_last_name"] = value_s[:120]
                    bot_event_signals["loot_last_type"] = classify_loot_type(path_l, value_l)
                    bot_event_signals["loot_ts"] = time.monotonic()
                    bot_event_signals["loot_count"] = int(bot_event_signals.get("loot_count", 0) or 0) + 1
                continue
            if not isinstance(value, (int, float)):
                continue

            num = float(value)
            if math.isnan(num) or math.isinf(num):
                continue

            if any(h in path_l for h in ("damage_done", "damage_dealt", "dmg_done", "damageout")):
                if num >= 0:
                    accumulate_metric_with_reset("damage_done", num)
            if any(h in path_l for h in ("damage_taken", "damage_received", "dmg_taken", "damagein")):
                if num >= 0:
                    accumulate_metric_with_reset("damage_taken", num)

            if any(h in path_l for h in (
                "zone_shrink",
                "shrink_countdown",
                "seconds_to_shrink",
                "shrink_timer",
                "time_to_shrink",
            )) or any(h in compact_path for h in ("stormcountdown", "gascountdown", "toxiccountdown", "safezonecountdown")):
                if 0.0 <= num <= 9999.0:
                    bot_event_signals["zone_countdown_sec"] = num
                    bot_event_signals["zone_signal_source"] = "event"
                    bot_event_signals["zone_signal_ts"] = time.monotonic()
            if any(h in path_l for h in ("safe_zone_x", "zone_center_x", "safezonex")) or any(
                h in compact_path for h in ("safezonecenterx", "zonecenterx", "safecirclex")
            ):
                bot_event_signals["safe_zone_x"] = num
                bot_event_signals["zone_signal_source"] = "event"
                bot_event_signals["zone_signal_ts"] = time.monotonic()
            if any(h in path_l for h in ("safe_zone_y", "zone_center_y", "safezoney")) or any(
                h in compact_path for h in ("safezonecentery", "zonecentery", "safecircley")
            ):
                bot_event_signals["safe_zone_y"] = num
                bot_event_signals["zone_signal_source"] = "event"
                bot_event_signals["zone_signal_ts"] = time.monotonic()
            if any(h in path_l for h in ("safe_zone_radius", "zone_radius", "safezoneradius")) or any(
                h in compact_path for h in ("safecircleradius", "safezoner", "zonecircleradius")
            ):
                if num >= 0:
                    bot_event_signals["safe_zone_radius"] = num
                    bot_event_signals["zone_signal_source"] = "event"
                    bot_event_signals["zone_signal_ts"] = time.monotonic()
            if (
                ("enemy" not in compact_path and "target" not in compact_path)
                and (
                    any(
                        h in compact_path
                        for h in (
                            "playerposx",
                            "playerpositionx",
                            "selfposx",
                            "myposx",
                            "localplayerx",
                            "playerx",
                            "selfx",
                        )
                    )
                    or (
                        ("player" in path_l or "self" in path_l or "local" in path_l)
                        and path_l.endswith(".x")
                    )
                )
            ):
                bot_event_signals["player_pos_x"] = num
            if (
                ("enemy" not in compact_path and "target" not in compact_path)
                and (
                    any(
                        h in compact_path
                        for h in (
                            "playerposy",
                            "playerpositiony",
                            "selfposy",
                            "myposy",
                            "localplayery",
                            "playery",
                            "selfy",
                        )
                    )
                    or (
                        ("player" in path_l or "self" in path_l or "local" in path_l)
                        and path_l.endswith(".y")
                    )
                )
            ):
                bot_event_signals["player_pos_y"] = num
            if any(h in compact_path for h in ("outsidezone", "outsideofsafezone", "outofsafezone", "instorm", "ingas", "intoxiccloud")):
                bot_event_signals["zone_outside_safe"] = bool(num >= 0.5)
                bot_event_signals["zone_toxic_detected"] = bool(num >= 0.5)
                bot_event_signals["zone_toxic_confidence"] = 0.8 if num >= 0.5 else 0.0
                bot_event_signals["zone_signal_source"] = "event"
                bot_event_signals["zone_signal_ts"] = time.monotonic()
            if any(h in compact_path for h in ("killerisbot", "attackerisbot", "killedbybot", "sourceisbot", "isbotkiller")):
                bot_event_signals["death_attacker_is_bot"] = bool(num >= 0.5)
                bot_event_signals["death_cause_source"] = "event_attacker"
            if any(h in compact_path for h in ("insafezone", "withinsafezone")):
                bot_event_signals["zone_outside_safe"] = bool(num < 0.5)
                bot_event_signals["zone_signal_source"] = "event"
                bot_event_signals["zone_signal_ts"] = time.monotonic()
            if any(h in path_l for h in ("leaderboard_rank", "rank")) and 0 <= num <= 500:
                bot_event_signals["last_rank"] = num
            if any(h in path_l for h in ("mana", "mp", "energy")):
                if any(h in path_l for h in ("max_mana", "mana_max", "maxmana", "max_mp", "max_energy")):
                    if num > 0:
                        bot_event_signals["mana_max"] = num
                elif any(h in path_l for h in ("mana", "mp", "energy")):
                    if num >= 0:
                        bot_event_signals["mana_current"] = num

            ability_key = ability_key_from_path(path_l)
            if ability_key:
                ready_hints = ("ready", "available", "can_use", "usable", "off_cooldown")
                cooldown_hints = ("cooldown", "cd", "remaining", "time_left", "time_to_ready")
                if any(h in path_l for h in ready_hints):
                    bot_event_signals["ability_ready"][ability_key] = bool(num >= 0.5)
                    bot_event_signals["ability_signal_ts"] = time.monotonic()
                if any(h in path_l for h in cooldown_hints):
                    if 0.0 <= num <= 120.0:
                        bot_event_signals["ability_cooldown_sec"][ability_key] = num
                        if num <= 0.18:
                            bot_event_signals["ability_ready"][ability_key] = True
                        elif num >= 0.35:
                            bot_event_signals["ability_ready"][ability_key] = False
                        bot_event_signals["ability_signal_ts"] = time.monotonic()
            if any(h in compact_path for h in ("chestopened", "lootpickup", "itempickup", "dropcollected", "rewardclaim")):
                if num >= 0.5:
                    bot_event_signals["loot_last_name"] = path_l
                    bot_event_signals["loot_last_type"] = classify_loot_type(path_l, path_l)
                    bot_event_signals["loot_ts"] = time.monotonic()
                    bot_event_signals["loot_count"] = int(bot_event_signals.get("loot_count", 0) or 0) + 1

    def note_bot_event_signal(event_name: str) -> None:
        event_name_s = str(event_name or "").strip()
        if not event_name_s:
            return
        now_mono = time.monotonic()
        event_name_lower = event_name_s.lower()
        bot_event_signals["last_event_name"] = event_name_s
        bot_event_signals["last_event_ts"] = now_mono

        if any(hint in event_name_lower for hint in MATCH_EVENT_HINTS):
            bot_event_signals["match_ts"] = now_mono
        if any(hint in event_name_lower for hint in MATCH_END_EVENT_HINTS):
            bot_event_signals["match_end_ts"] = now_mono
        if any(hint in event_name_lower for hint in LOBBY_EVENT_HINTS):
            bot_event_signals["lobby_ts"] = now_mono
        if any(hint in event_name_lower for hint in LOADING_EVENT_HINTS) or event_name_lower.startswith("loading/"):
            bot_event_signals["loading_ts"] = now_mono
        if any(hint in event_name_lower for hint in DEATH_EVENT_HINTS):
            bot_event_signals["death_ts"] = now_mono
            if any(h in event_name_lower for h in ("storm", "gas", "toxic", "zone", "poison")):
                bot_event_signals["death_cause"] = "toxic_zone"
                bot_event_signals["death_cause_confidence"] = max(
                    float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                    0.82,
                )
                bot_event_signals["death_cause_source"] = "event_name"
            elif any(h in event_name_lower for h in ("player", "killed", "eliminated", "attacker", "enemy")):
                bot_event_signals["death_cause"] = "enemy_unknown"
                bot_event_signals["death_cause_confidence"] = max(
                    float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                    0.58,
                )
                bot_event_signals["death_cause_source"] = "event_name"

    def infer_death_from_event(event_obj: Dict[str, Any], event_name: str) -> bool:
        event_name_l = str(event_name or "").lower()
        if any(h in event_name_l for h in DEATH_EVENT_HINTS):
            return True
        if "match_end" not in event_name_l:
            return False
        rank_val = bot_event_signals.get("last_rank")
        try:
            rank_num = float(rank_val) if rank_val is not None else None
        except Exception:
            rank_num = None
        if rank_num is not None and rank_num > 1.0:
            return True
        # Fallback: match_end in battle royale usually implies elimination.
        return True

    def infer_death_cause(
        enemy_recent: bool,
        zone_outside_safe: bool,
        zone_toxic_detected: bool,
        visual_feedback: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, str]:
        attacker_name = str(bot_event_signals.get("death_attacker_name", "") or "").strip()
        attacker_is_bot = bot_event_signals.get("death_attacker_is_bot", None)
        source = "inferred"

        toxic_hits = 0
        toxic_ratio = 0.0
        if isinstance(visual_feedback, dict):
            toxic_hits = int(visual_feedback.get("toxic_zone_hits", 0) or 0)
            toxic_ratio = float(visual_feedback.get("toxic_color_ratio", 0.0) or 0.0)
        toxic_visual = toxic_hits > 0 or toxic_ratio >= 0.028
        toxic_now = bool(zone_outside_safe or zone_toxic_detected or toxic_visual)

        if toxic_now and enemy_recent:
            return "mixed_zone_enemy", 0.80, "mixed"
        if toxic_now:
            conf = 0.88 if zone_outside_safe else (0.78 if zone_toxic_detected else 0.66)
            return "toxic_zone", conf, "zone"
        if enemy_recent or attacker_name:
            if attacker_is_bot is True:
                return "enemy_bot", 0.86, "attacker_flag"
            if attacker_is_bot is False:
                return "enemy_player", 0.84, "attacker_flag"
            if attacker_name:
                if looks_like_bot(attacker_name):
                    return "enemy_bot", 0.78, "attacker_name"
                return "enemy_player", 0.74, "attacker_name"
            return "enemy_unknown", 0.62, "enemy_recent"
        return "unknown", 0.22, source

    def refresh_death_cause_context(
        now_mono: float,
        enemy_recent: bool,
        zone_outside_safe: bool,
        zone_toxic_detected: bool,
        visual_feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        death_ts = float(bot_event_signals.get("death_ts", 0.0) or 0.0)
        if death_ts <= 0.0:
            return
        if (now_mono - death_ts) > 20.0:
            return
        cause, conf, source = infer_death_cause(
            enemy_recent=enemy_recent,
            zone_outside_safe=zone_outside_safe,
            zone_toxic_detected=zone_toxic_detected,
            visual_feedback=visual_feedback,
        )
        prev_conf = float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0)
        if conf >= prev_conf or str(bot_event_signals.get("death_cause", "unknown") or "unknown") == "unknown":
            bot_event_signals["death_cause"] = str(cause)
            bot_event_signals["death_cause_confidence"] = float(conf)
            bot_event_signals["death_cause_source"] = str(source)

    def process_detected_entities(players: Set[str], characters: Set[str], source: str) -> None:
        new_players = players - seen_players
        new_characters = characters - seen_characters
        seen_players.update(players)
        seen_characters.update(characters)

        if players:
            upsert_names(conn, "live_players", players, source)
        if characters:
            upsert_names(conn, "live_characters", characters, source)

        if args.print_entity_updates:
            if new_players:
                print(f"[PLAYERS] Nuevos: {', '.join(sorted(new_players))}")
            if new_characters:
                print(f"[CHARS] Nuevos: {', '.join(sorted(new_characters))}")

    def process_payload(payload: Dict[str, Any], source: str) -> None:
        try:
            inserted = ingest_payload_records(conn, payload, source=source)
            stats["events_saved"] += len(inserted)
        except ValueError:
            stats["no_events"] += 1
            return
        except Exception:
            stats["payload_errors"] += 1
            return

        events = [event for _event_id, event in inserted if isinstance(event, dict)]

        if args.print_event_names:
            names = sorted({str(event.get("eventName", "unknown")) for event in events})
            if names:
                print(f"[EV] {', '.join(names)}")

        players: Set[str] = set()
        characters: Set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_name_value = str(event.get("eventName", ""))
            note_bot_event_signal(event_name_value)
            update_bot_metrics_from_event(event)
            if infer_death_from_event(event, event_name_value):
                bot_event_signals["death_ts"] = time.monotonic()
            extract_entities(event, players, characters)

        process_detected_entities(players, characters, source)

        if args.ocr_on_match_end:
            for event_id, event in inserted:
                if str(event.get("eventName", "")) == "games_played/match_end":
                    pending_match_end_events.append(event_id)
                    stats["ocr_queued"] += 1
                    print(f"[OCR] queued event_id={event_id}")

    def iter_event_objects_from_payload(payload_obj: Any) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        stack: List[Any] = [payload_obj]
        visited = 0
        while stack and visited < 400:
            visited += 1
            current = stack.pop()
            if isinstance(current, dict):
                if isinstance(current.get("eventName"), str):
                    found.append(current)
                for key in ("events", "event", "payload", "data", "records", "items", "logs"):
                    val = current.get(key)
                    if isinstance(val, list):
                        stack.extend(val[:80])
                    elif isinstance(val, dict):
                        stack.append(val)
            elif isinstance(current, list):
                stack.extend(current[:80])
        return found

    def process_live_payload_signals(payload_obj: Any, source: str) -> int:
        events = iter_event_objects_from_payload(payload_obj)
        if not events:
            return 0
        players: Set[str] = set()
        characters: Set[str] = set()
        for event in events:
            event_name_value = str(event.get("eventName", "") or "")
            if not event_name_value:
                continue
            note_bot_event_signal(event_name_value)
            update_bot_metrics_from_event(event)
            if infer_death_from_event(event, event_name_value):
                bot_event_signals["death_ts"] = time.monotonic()
            extract_entities(event, players, characters)
        process_detected_entities(players, characters, source=source)
        return len(events)

    def pick_active_page(page_obj):
        if page_obj is not None:
            try:
                if not page_obj.is_closed():
                    return page_obj
            except Exception:
                pass
        try:
            if context is None:
                return None
            for p in context.pages:
                try:
                    if not p.is_closed():
                        return p
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def capture_all_post(request) -> None:
        if not args.capture_all_post:
            return

        content_type = content_type_from_request(request)
        if not should_capture_post_body(content_type):
            stats["raw_posts_skipped"] += 1
            return

        body_text = read_request_body_text(request)
        if body_text is None:
            stats["raw_posts_skipped"] += 1
            return

        raw_source = "live_capture:all_post"
        raw_request_id = insert_raw_request(
            conn=conn,
            url=str(request.url),
            method="POST",
            content_type=content_type or None,
            body_text=body_text,
            source=raw_source,
            max_body_bytes=capture_all_max_bytes,
        )
        stats["raw_posts_saved"] += 1

        for obj in iter_json_objects_from_text(body_text):
            insert_raw_event(
                conn=conn,
                raw_request_id=raw_request_id,
                url=str(request.url),
                obj=obj,
                source=raw_source,
            )
            stats["raw_events_saved"] += 1
        conn.commit()

    def route_handler(route, request) -> None:
        try:
            if request.method.upper() != "POST":
                return

            capture_all_post(request)

            if args.endpoint_substring not in request.url:
                return

            stats["requests_seen"] += 1
            if args.print_each_request:
                print(f"[REQ] {request.method} {request.url}")

            payload, origin = extract_request_payload(request)
            if payload is None:
                stats["no_payload"] += 1
            else:
                process_payload(payload, source=f"live_capture:{origin}")
        except Exception:
            stats["payload_errors"] += 1
        finally:
            try:
                route.continue_()
            except Exception:
                pass

    def run_ocr_for_event(page_obj, event_id: int) -> None:
        stats["ocr_runs"] += 1
        screenshot_path: Optional[Path] = None
        try:
            output_dir = Path(args.ocr_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = output_dir / f"match_end_{event_id}_{int(time.time() * 1000)}.png"

            active_page = pick_active_page(page_obj)
            if active_page is None:
                raise RuntimeError("No hay pagina activa para screenshot OCR.")

            clip = None
            try:
                locator = active_page.locator(args.ocr_canvas_selector).first
                locator.wait_for(state="visible", timeout=3000)
                box = locator.bounding_box()
                if box and box.get("width", 0) > 1 and box.get("height", 0) > 1:
                    clip = {
                        "x": box["x"],
                        "y": box["y"],
                        "width": box["width"],
                        "height": box["height"],
                    }
            except Exception:
                clip = None

            if clip:
                active_page.screenshot(path=str(screenshot_path), clip=clip)
            else:
                active_page.screenshot(path=str(screenshot_path), full_page=True)

            cmd = [sys.executable, str(ocr_script_path), "--input", str(screenshot_path), "--json-only"]
            if args.ocr_config:
                cmd.extend(["--config", args.ocr_config])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=ocr_timeout_sec,
                check=False,
            )
            if result.returncode != 0:
                stats["ocr_errors"] += 1
                print(
                    f"[WARN] OCR fallo para event_id={event_id}: "
                    f"code={result.returncode} stderr={result.stderr[:220]}"
                )
                return

            stdout = result.stdout.strip()
            if not stdout:
                stats["ocr_errors"] += 1
                print(f"[WARN] OCR sin salida para event_id={event_id}")
                return

            try:
                ocr_result = fast_json_loads(stdout)
            except Exception:
                ocr_result = fast_json_loads(stdout.splitlines()[-1])

            if not isinstance(ocr_result, dict):
                stats["ocr_errors"] += 1
                print(f"[WARN] OCR retorno no-objeto para event_id={event_id}")
                return

            persist_match_stats(
                conn=conn,
                event_id=event_id,
                screenshot_path=str(screenshot_path),
                ocr_result=ocr_result,
                source="ocr_match_end_live",
            )
            stats["ocr_saved"] += 1
            print(f"[OCR] event_id={event_id} screenshot={screenshot_path.name}")
        except Exception as exc:
            stats["ocr_errors"] += 1
            print(f"[WARN] OCR exception event_id={event_id}: {exc}")

    def drain_ocr_queue(page_obj, force: bool = False) -> None:
        if not pending_match_end_events:
            return
        processed = 0
        while pending_match_end_events:
            event_id = pending_match_end_events.pop(0)
            run_ocr_for_event(page_obj, event_id)
            processed += 1
            if not force and processed >= 1:
                break

    def commit_ws_if_needed(force: bool = False) -> None:
        nonlocal ws_pending_writes
        if ws_pending_writes >= 50 or (force and ws_pending_writes > 0):
            conn.commit()
            ws_pending_writes = 0

    def capture_ws_frame(direction: str, request_id: str, response: Dict[str, Any]) -> None:
        nonlocal ws_pending_writes
        stats["ws_frames"] += 1

        try:
            opcode = int(response.get("opcode", -1))
        except Exception:
            opcode = -1
        payload_data = response.get("payloadData", "")
        decoded_text, decoded_kind, payload_b64, payload_len = decode_ws_payload(opcode, payload_data)

        keyword_hit = None
        if decoded_text:
            lowered = decoded_text.lower()
            for kw in ws_keywords:
                if kw in lowered:
                    keyword_hit = kw
                    stats["ws_keyword_hits"] += 1
                    break

        if keyword_hit and args.ws_print_keyword_hits:
            snippet = decoded_text.replace("\n", " ").replace("\r", " ")[:220]
            ws_url = ws_url_by_request_id.get(request_id, "")
            print(
                f"[WS:{direction}] hit={keyword_hit} kind={decoded_kind} "
                f"opcode={opcode} url={ws_url} snippet={snippet}"
            )

        if decoded_text and (args.ws_player_heuristics or args.play_game):
            for obj in iter_json_objects_from_text(decoded_text):
                if not isinstance(obj, dict):
                    continue
                if args.ws_player_heuristics:
                    try:
                        players: Set[str] = set()
                        characters: Set[str] = set()
                        extract_entities(obj, players, characters)
                        process_detected_entities(players, characters, source="websocket_cdp")
                    except Exception:
                        pass
                if args.play_game and args.bot_realtime_mode:
                    try:
                        process_live_payload_signals(obj, source="websocket_events")
                    except Exception:
                        pass

        should_save = args.ws_save_frames or bool(keyword_hit)
        if not should_save:
            return
        if stats["ws_saved"] >= args.ws_max_saved:
            return

        text_preview = decoded_text[:5000] if decoded_text else None
        payload_to_store = payload_b64 if args.ws_save_binary else None

        conn.execute(
            """
            INSERT INTO ws_frames (
                captured_at_ms,
                direction,
                ws_request_id,
                ws_url,
                opcode,
                payload_len,
                decoded_kind,
                keyword_hit,
                text_preview,
                payload_b64
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time() * 1000),
                direction,
                request_id,
                ws_url_by_request_id.get(request_id),
                opcode,
                payload_len,
                decoded_kind,
                keyword_hit,
                text_preview,
                payload_to_store,
            ),
        )
        stats["ws_saved"] += 1
        ws_pending_writes += 1
        commit_ws_if_needed(force=False)

    def generate_report_if_needed(current_page_obj) -> None:
        nonlocal last_report
        if args.report_every_sec <= 0:
            return
        if time.monotonic() - last_report < args.report_every_sec:
            return

        _generate_report_output(conn, stats, seen_characters, seen_players, args)
        last_report = time.monotonic()
        active_page = pick_active_page(current_page_obj)
        if active_page is not None:
            drain_ocr_queue(page_obj=active_page, force=False)

    def _generate_report_output(conn: sqlite3.Connection, stats: Dict[str, int], seen_characters: Set[str], seen_players: Set[str], args: argparse.Namespace) -> None:
        total_in_db = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        print("\n== Resumen en vivo ==")
        print(
            f"requests_seen={stats['requests_seen']} "
            f"events_saved={stats['events_saved']} "
            f"no_payload={stats['no_payload']} "
            f"no_events={stats['no_events']} "
            f"payload_errors={stats['payload_errors']} "
            f"raw_posts_saved={stats['raw_posts_saved']} "
            f"raw_events_saved={stats['raw_events_saved']} "
            f"raw_posts_skipped={stats['raw_posts_skipped']} "
            f"ws_frames={stats['ws_frames']} "
            f"ws_keyword_hits={stats['ws_keyword_hits']} "
            f"ws_saved={stats['ws_saved']} "
            f"ocr_queued={stats['ocr_queued']} "
            f"ocr_runs={stats['ocr_runs']} "
            f"ocr_saved={stats['ocr_saved']} "
            f"ocr_errors={stats['ocr_errors']} "
            f"db_events={total_in_db}"
        )
        print_top_events(conn, limit=10)
        print_detected_entities("Personajes detectados", seen_characters)
        print_detected_entities("Jugadores detectados (sin bots)", seen_players)
        print("\nUltimas partidas (match_end):")
        has_match = print_match_report(conn, limit=args.report_limit)
        if not has_match:
            print("\nActividad de ronda (fallback):")
            print_round_activity_report(conn, limit=args.report_limit)
        print("\n====================\n")
        commit_ws_if_needed(force=False)

    def resolve_game_frame(page_obj: Page, iframe_selector: str, timeout_ms: int) -> Optional[Frame]:
        if page_obj is None:
            raise RuntimeError("No hay pagina activa para resolver iframe de juego.")

        timeout_ms = max(1000, int(timeout_ms))
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        iframe_locator = page_obj.locator(iframe_selector).first
        last_error: Optional[Exception] = None

        try:
            iframe_locator.wait_for(state="attached", timeout=min(timeout_ms, 4000))
        except Exception as exc:
            last_error = exc

        while time.monotonic() < deadline:
            try:
                handle = iframe_locator.element_handle(timeout=1200)
                if handle is not None:
                    frame_obj = handle.content_frame()
                    if frame_obj is not None:
                        try:
                            frame_obj.wait_for_load_state(
                                "domcontentloaded",
                                timeout=min(timeout_ms, 3000),
                            )
                        except Exception:
                            pass
                        return frame_obj
            except Exception as exc:
                last_error = exc
            time.sleep(0.2)

        # Fallback: some builds render Unity directly on the main frame (no iframe).
        try:
            canvas_locator = page_obj.locator("canvas#unity-canvas, canvas").first
            canvas_locator.wait_for(state="attached", timeout=1200)
            print("[BOT][INFO] No iframe encontrado; usando main_frame con canvas Unity.")
            return page_obj.main_frame
        except Exception:
            pass

        if last_error is not None:
            print(
                f"[BOT][ERROR] No se pudo resolver Frame para iframe "
                f"'{iframe_selector}': {last_error}"
            )
        else:
            print(f"[BOT][ERROR] No se pudo resolver Frame para iframe '{iframe_selector}'.")
        return None

    def frame_point_to_page_target(
        page_obj: Page,
        iframe_selector: str,
        frame_x: float,
        frame_y: float,
        source: str,
    ) -> Dict[str, float]:
        iframe_box = None
        try:
            iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
        except Exception:
            iframe_box = None

        if iframe_box:
            page_x = float(iframe_box["x"]) + float(frame_x)
            page_y = float(iframe_box["y"]) + float(frame_y)
        else:
            page_x = float(frame_x)
            page_y = float(frame_y)

        return {
            "frame_x": float(frame_x),
            "frame_y": float(frame_y),
            "page_x": float(page_x),
            "page_y": float(page_y),
            "source": source,
        }

    def build_target_from_locator(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
        locator: Locator,
        source: str,
    ) -> Optional[Dict[str, float]]:
        try:
            center = locator.evaluate(
                """
                (el) => {
                  if (!el) return null;
                  const r = el.getBoundingClientRect();
                  if (!r || r.width <= 1 || r.height <= 1) return null;
                  return {x: r.left + (r.width / 2), y: r.top + (r.height / 2)};
                }
                """
            )
            if not isinstance(center, dict):
                return None
            return frame_point_to_page_target(
                page_obj=page_obj,
                iframe_selector=iframe_selector,
                frame_x=float(center.get("x", 0.0)),
                frame_y=float(center.get("y", 0.0)),
                source=source,
            )
        except Exception:
            return None

    def click_first_visible_in_frame(
        frame_obj: Frame,
        selectors: List[str],
        label: str,
        timeout_ms: int,
        page_obj: Optional[Page] = None,
        iframe_selector: str = "",
        click_mode: str = "hybrid",
        mouse_move_steps: int = 8,
        visual_cursor: bool = False,
    ) -> bool:
        last_error: Optional[Exception] = None
        timeout_ms = max(300, int(timeout_ms))
        per_selector_timeout = max(75, min(timeout_ms, 500))

        for selector in selectors:
            try:
                locator = frame_obj.locator(selector).first
                try:
                    is_visible = bool(locator.is_visible(timeout=per_selector_timeout))
                except Exception:
                    is_visible = False
                if not is_visible:
                    continue
                if page_obj is not None:
                    target = build_target_from_locator(
                        page_obj=page_obj,
                        frame_obj=frame_obj,
                        iframe_selector=iframe_selector,
                        locator=locator,
                        source=f"ui:{label}:{selector}",
                    )
                    if target is not None:
                        ok_click, used_mode = perform_attack_click(
                            page_obj=page_obj,
                            frame_obj=frame_obj,
                            target=target,
                            click_mode=click_mode,
                            mouse_move_steps=mouse_move_steps,
                            visual_cursor=visual_cursor,
                            allow_unverified_mouse=True,
                        )
                        if ok_click:
                            print(f"[BOT] Click ok ({label}) via {used_mode}: {selector}")
                            return True
                locator.click(timeout=min(timeout_ms, 500), force=True)
                print(f"[BOT] Click ok ({label}): {selector}")
                return True
            except Exception as exc:
                last_error = exc
                continue

        if page_obj is not None:
            for selector in selectors:
                try:
                    page_locator = page_obj.locator(selector).first
                    try:
                        is_visible = bool(page_locator.is_visible(timeout=per_selector_timeout))
                    except Exception:
                        is_visible = False
                    if not is_visible:
                        continue
                    page_locator.click(timeout=min(timeout_ms, 500), force=True)
                    print(f"[BOT] Click ok ({label}) via page: {selector}")
                    return True
                except Exception as exc:
                    last_error = exc
                    continue

        print(f"[BOT][WARN] No se pudo clicar ({label}). Last error: {last_error}")
        return False

    def focus_game_frame(page_obj: Page, frame_obj: Frame, iframe_selector: str) -> None:
        try:
            frame_obj.focus("body", timeout=1200)
            return
        except Exception:
            pass
        try:
            page_obj.locator(iframe_selector).first.click(timeout=1200)
        except Exception:
            pass

    def collect_bot_ui_state(frame_obj: Frame) -> Dict[str, Any]:
        script = """
        () => {
          const isVisible = (sel) => {
            const el = document.querySelector(sel);
            if (!el) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden' &&
                   rect.width > 2 && rect.height > 2;
          };
          return {
            character_visible: isVisible('div.character-select-view div.character-container') ||
                               isVisible('div.character-container'),
            select_visible: isVisible('div.character-select-view div.select-button') ||
                            isVisible(\"button[data-testid='select-button']\") ||
                            isVisible('button'),
            play_visible: isVisible('div.game-view div.play-button') ||
                          isVisible('div.play-button') ||
                          isVisible(\"button[data-testid='play-button']\"),
            canvas_visible: isVisible('canvas'),
            body_class: document.body ? document.body.className : '',
          };
        }
        """
        try:
            result = frame_obj.evaluate(script)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {
            "character_visible": False,
            "select_visible": False,
            "play_visible": False,
            "canvas_visible": False,
            "body_class": "",
        }

    def resolve_google_credentials(args: argparse.Namespace) -> Tuple[str, str]:
        email = str(args.bot_google_email or "").strip()
        password = str(args.bot_google_password or "")
        if not email:
            email = str(os.environ.get("LMS_GOOGLE_EMAIL", "") or "").strip()
        if not password:
            password = str(os.environ.get("LMS_GOOGLE_PASSWORD", "") or "")
        return email, password

    def mask_email(email: str) -> str:
        value = str(email or "").strip()
        if "@" not in value:
            if len(value) <= 4:
                return "*" * len(value)
            return value[:2] + ("*" * max(1, len(value) - 4)) + value[-2:]
        user, domain = value.split("@", 1)
        if len(user) <= 2:
            user_masked = user[:1] + "*"
        else:
            user_masked = user[:2] + ("*" * max(1, len(user) - 2))
        return f"{user_masked}@{domain}"

    def _google_is_authenticated(page_obj: Page) -> bool:
        try:
            page_obj.goto("https://myaccount.google.com/", wait_until="domcontentloaded", timeout=22000)
            url_l = str(page_obj.url or "").lower()
            if "accounts.google.com" in url_l and ("signin" in url_l or "challenge" in url_l):
                return False
            has_email_input = False
            try:
                has_email_input = page_obj.locator("input[type='email'], input[name='identifier']").first.is_visible(timeout=900)
            except Exception:
                has_email_input = False
            return not has_email_input
        except Exception:
            return False

    def _click_first_visible(locators: List[Locator], timeout_ms: int = 1400) -> bool:
        for loc in locators:
            try:
                if loc.is_visible(timeout=timeout_ms):
                    loc.click(timeout=timeout_ms, force=True)
                    return True
            except Exception:
                continue
        return False

    def ensure_google_login_pre_game(
        page_obj: Page,
        email: str,
        password: str,
        timeout_sec: float,
    ) -> bool:
        email_s = str(email or "").strip()
        email_safe = mask_email(email_s)
        password_s = str(password or "")
        if not email_s or not password_s:
            print(
                "[BOT][GOOGLE_LOGIN][WARN] Credenciales no definidas. "
                "Usa --bot-google-email/--bot-google-password o env LMS_GOOGLE_EMAIL/LMS_GOOGLE_PASSWORD."
            )
            return False
        if _google_is_authenticated(page_obj):
            print(f"[BOT][GOOGLE_LOGIN] Sesion Google ya activa ({email_safe}).")
            return True

        deadline = time.monotonic() + max(20.0, float(timeout_sec))
        try:
            page_obj.goto(
                "https://accounts.google.com/ServiceLogin?hl=en&service=mail&continue=https%3A%2F%2Fwww.google.com%2F",
                wait_until="domcontentloaded",
                timeout=28000,
            )
        except Exception as exc:
            print(f"[BOT][GOOGLE_LOGIN][WARN] No se pudo abrir login Google: {exc}")
            return False

        while time.monotonic() < deadline:
            url_l = str(page_obj.url or "").lower()
            if "challenge" in url_l:
                print("[BOT][GOOGLE_LOGIN][WARN] Google requiere challenge/2FA. Completa manualmente y reintenta.")
                return False

            # If account chooser is shown, prefer target account tile.
            try:
                account_tile = page_obj.locator(
                    f"[data-identifier='{email_s}'], div[data-email='{email_s}'], li[data-identifier='{email_s}']"
                ).first
                if account_tile.is_visible(timeout=600):
                    account_tile.click(timeout=1200, force=True)
                    time.sleep(0.6)
                    continue
            except Exception:
                pass

            email_input = page_obj.locator("input[type='email'], input[name='identifier']").first
            try:
                if email_input.is_visible(timeout=600):
                    try:
                        email_input.fill(email_s, timeout=1400)
                    except Exception:
                        email_input.click(timeout=1200)
                        email_input.press("Control+A")
                        email_input.type(email_s, delay=18)
                    _click_first_visible(
                        [
                            page_obj.locator("#identifierNext button").first,
                            page_obj.locator("button:has-text('Siguiente')").first,
                            page_obj.locator("button:has-text('Next')").first,
                        ],
                        timeout_ms=1300,
                    )
                    try:
                        email_input.press("Enter")
                    except Exception:
                        pass
                    time.sleep(0.8)
                    continue
            except Exception:
                pass

            pwd_input = page_obj.locator("input[type='password'], input[name='Passwd']").first
            try:
                if pwd_input.is_visible(timeout=800):
                    try:
                        pwd_input.fill(password_s, timeout=1600)
                    except Exception:
                        pwd_input.click(timeout=1200)
                        pwd_input.press("Control+A")
                        pwd_input.type(password_s, delay=20)
                    _click_first_visible(
                        [
                            page_obj.locator("#passwordNext button").first,
                            page_obj.locator("button:has-text('Siguiente')").first,
                            page_obj.locator("button:has-text('Next')").first,
                        ],
                        timeout_ms=1300,
                    )
                    try:
                        pwd_input.press("Enter")
                    except Exception:
                        pass
                    time.sleep(1.0)
                    if _google_is_authenticated(page_obj):
                        print(f"[BOT][GOOGLE_LOGIN] Login OK ({email_safe}).")
                        return True
                    continue
            except Exception:
                pass

            if _google_is_authenticated(page_obj):
                print(f"[BOT][GOOGLE_LOGIN] Login OK ({email_safe}).")
                return True

            time.sleep(0.7)

        ok = _google_is_authenticated(page_obj)
        if ok:
            print(f"[BOT][GOOGLE_LOGIN] Login OK ({email_safe}).")
            return True
        print("[BOT][GOOGLE_LOGIN][WARN] Timeout de login Google.")
        return False

    def collect_runtime_variable_probe(frame_obj: Frame, max_keys: int) -> Dict[str, Any]:
        script = """
        (payload) => {
          const maxKeys = Math.max(20, Math.min(800, Number(payload.maxKeys || 140)));
          const hints = Array.isArray(payload.hints) ? payload.hints.map((x) => String(x || '').toLowerCase()) : [];
          const skipPrefixes = ['webkit', 'on', 'moz', 'ms'];
          const safeType = (v) => {
            if (v === null) return 'null';
            if (Array.isArray(v)) return 'array';
            const t = typeof v;
            if (t !== 'object') return t;
            try {
              if (v instanceof Element) return 'dom_element';
              if (v instanceof Window) return 'window';
              if (v instanceof Document) return 'document';
            } catch (_) {}
            return 'object';
          };
          const preview = (v) => {
            const t = safeType(v);
            if (t === 'string') return String(v).slice(0, 120);
            if (t === 'number' || t === 'boolean') return String(v);
            if (t === 'array') return `array(len=${Number(v.length || 0)})`;
            if (t === 'function') return 'function';
            if (t === 'dom_element') {
              try {
                return `<${String(v.tagName || 'element').toLowerCase()} id='${String(v.id || '')}' class='${String(v.className || '')}'>`;
              } catch (_) {
                return 'dom_element';
              }
            }
            if (t === 'object') {
              try {
                const keys = Object.keys(v).slice(0, 6);
                return `{${keys.join(',')}}`;
              } catch (_) {
                return 'object';
              }
            }
            return t;
          };
          const scoreKey = (k) => {
            const lower = String(k || '').toLowerCase();
            if (!lower) return 0;
            let score = 0;
            for (const hint of hints) {
              if (lower.includes(hint)) score += 1;
            }
            if (lower.includes('__')) score += 1;
            if (lower.includes('unity') || lower.includes('webpack')) score += 2;
            return score;
          };
          const scanNested = (v) => {
            try {
              if (!v || typeof v !== 'object') return [];
              const keys = Object.keys(v);
              const out = [];
              for (const nk of keys) {
                const lower = String(nk || '').toLowerCase();
                let matched = false;
                for (const hint of hints) {
                  if (lower.includes(hint)) {
                    matched = true;
                    break;
                  }
                }
                if (matched) out.push(String(nk));
                if (out.length >= 16) break;
              }
              return out;
            } catch (_) {
              return [];
            }
          };
          const keys = Object.getOwnPropertyNames(window || {});
          const candidates = [];
          for (const k of keys) {
            const key = String(k || '');
            const lower = key.toLowerCase();
            if (!lower) continue;
            let skip = false;
            for (const pref of skipPrefixes) {
              if (lower.startsWith(pref)) {
                skip = true;
                break;
              }
            }
            if (skip) continue;
            const score = scoreKey(key);
            if (score <= 0) continue;
            let val = null;
            try {
              val = window[key];
            } catch (_) {
              val = null;
            }
            const valType = safeType(val);
            const nested = scanNested(val);
            candidates.push({
              key,
              score,
              type: valType,
              preview: preview(val),
              nested_keys: nested,
              nested_count: Number(nested.length || 0),
            });
          }
          candidates.sort((a, b) => {
            if (b.score !== a.score) return b.score - a.score;
            if (b.nested_count !== a.nested_count) return b.nested_count - a.nested_count;
            return String(a.key).localeCompare(String(b.key));
          });
          const runtimeVars = candidates.slice(0, maxKeys);
          const telemetry = {
            ts_ms: Date.now(),
            href: String(location.href || ''),
            title: String(document.title || ''),
            window_key_count: Number(keys.length || 0),
            canvas_count: Number(document.querySelectorAll('canvas').length || 0),
            iframe_count: Number(document.querySelectorAll('iframe').length || 0),
            local_storage_keys: (() => {
              try {
                return Object.keys(localStorage || {}).slice(0, 60);
              } catch (_) {
                return [];
              }
            })(),
            session_storage_keys: (() => {
              try {
                return Object.keys(sessionStorage || {}).slice(0, 60);
              } catch (_) {
                return [];
              }
            })(),
            ua: String(navigator.userAgent || ''),
            runtime_vars: runtimeVars,
          };
          return telemetry;
        }
        """
        hints = [
            "player",
            "enemy",
            "mana",
            "health",
            "hp",
            "zone",
            "safe",
            "map",
            "match",
            "lobby",
            "damage",
            "ability",
            "cooldown",
            "state",
            "round",
            "timer",
            "name",
            "unity",
            "webpack",
        ]
        try:
            result = frame_obj.evaluate(
                script,
                {
                    "maxKeys": max(20, min(800, int(max_keys))),
                    "hints": hints,
                },
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {
            "ts_ms": int(time.time() * 1000),
            "href": "",
            "title": "",
            "window_key_count": 0,
            "canvas_count": 0,
            "iframe_count": 0,
            "local_storage_keys": [],
            "session_storage_keys": [],
            "ua": "",
            "runtime_vars": [],
        }

    def write_runtime_probe_snapshot(
        base_dir: str,
        phase: str,
        run_id: str,
        snapshot: Dict[str, Any],
    ) -> Optional[Path]:
        try:
            out_dir = Path(base_dir or DEFAULT_BOT_RUNTIME_PROBE_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_phase = re.sub(r"[^a-z0-9_\\-]+", "_", str(phase or "runtime").strip().lower())
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{safe_phase}_{run_id}_{ts}.json"
            payload = dict(snapshot or {})
            payload["phase"] = safe_phase
            payload["run_id"] = str(run_id or "")
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=True, indent=2)
            return out_path
        except Exception as exc:
            print(f"[BOT][RUNTIME_PROBE][WARN] No se pudo guardar snapshot: {exc}")
            return None

    def decide_bot_action(ui_state: Dict[str, Any]) -> str:
        if ui_state.get("character_visible"):
            return "character_select"
        if ui_state.get("select_visible"):
            return "confirm_select"
        if ui_state.get("play_visible"):
            return "press_play"
        if ui_state.get("canvas_visible"):
            return "canvas_unknown"
        return "idle"

    def is_recent_signal(ts_value: float, within_sec: float, now_mono: float) -> bool:
        return float(ts_value) > 0.0 and (now_mono - float(ts_value)) <= float(within_sec)

    def detect_bot_game_state(
        ui_state: Dict[str, Any],
        play_target: Optional[Dict[str, float]],
        now_mono: float,
        play_transition_started_at: Optional[float],
        visual_state_hint: str = "unknown",
        visual_state_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        body_class = str(ui_state.get("body_class", "") or "").lower()
        lobby_dom = bool(
            ui_state.get("character_visible")
            or ui_state.get("select_visible")
            or ui_state.get("play_visible")
        )
        canvas_visible = bool(ui_state.get("canvas_visible"))
        play_source = str(play_target.get("source", "") if isinstance(play_target, dict) else "")
        play_conf = float(play_target.get("confidence", 0.0) or 0.0) if isinstance(play_target, dict) else 0.0
        vision_play = bool(play_source == "vision_yellow_button")
        dom_play = bool(play_source == "dom_text_play")
        visual_hint = str(visual_state_hint or "unknown").strip().lower()
        visual_conf = max(0.0, min(1.0, float(visual_state_confidence or 0.0)))
        transition_elapsed = (
            (now_mono - play_transition_started_at)
            if play_transition_started_at is not None
            else None
        )
        has_recent_match = is_recent_signal(bot_event_signals.get("match_ts", 0.0), 75.0, now_mono)
        has_recent_lobby = is_recent_signal(bot_event_signals.get("lobby_ts", 0.0), 45.0, now_mono)
        has_recent_loading = is_recent_signal(bot_event_signals.get("loading_ts", 0.0), 30.0, now_mono)
        match_ts = float(bot_event_signals.get("match_ts", 0.0) or 0.0)
        lobby_ts = float(bot_event_signals.get("lobby_ts", 0.0) or 0.0)
        lobby_signal_newer = lobby_ts > 0.0 and (lobby_ts >= (match_ts - 0.8))
        last_event_lower = str(bot_event_signals.get("last_event_name", "") or "").lower()
        event_lobbyish = any(h in last_event_lower for h in LOBBY_EVENT_HINTS) or last_event_lower.startswith("loading/lobby")
        event_loadingish = any(h in last_event_lower for h in LOADING_EVENT_HINTS) or last_event_lower.startswith("loading/")
        event_matchish = (
            any(h in last_event_lower for h in MATCH_EVENT_HINTS)
            or "round_spawn" in last_event_lower
            or "round_start" in last_event_lower
            or "round_after_spawn" in last_event_lower
            or "frame_rate/playing" in last_event_lower
            or "active_engagement" in last_event_lower
        )

        if "playing" in body_class or "in_game" in body_class or "ingame" in body_class:
            return {"state": "in_match", "reason": "body_class_playing"}

        # Lobby evidence should win against stale in-match events, but
        # avoid false lobby flips when vision button detection is noisy.
        if lobby_dom:
            if transition_elapsed is not None and transition_elapsed <= 2.5:
                return {"state": "loading", "reason": "lobby_dom_during_transition"}
            return {"state": "lobby", "reason": "dom_lobby"}
        if dom_play and (lobby_signal_newer or (not has_recent_match)):
            return {"state": "lobby", "reason": "dom_text_play"}
        if vision_play and play_conf >= 0.72 and (not has_recent_match):
            if transition_elapsed is not None and transition_elapsed <= 2.5:
                return {"state": "loading", "reason": "vision_play_during_transition"}
            return {"state": "lobby", "reason": f"vision_play:{play_conf:.2f}"}

        if has_recent_match and canvas_visible and event_matchish:
            if transition_elapsed is not None and transition_elapsed <= 2.0:
                return {"state": "loading", "reason": "match_event_during_transition"}
            return {"state": "in_match", "reason": f"event_match:{bot_event_signals.get('last_event_name', '')}"}

        if has_recent_lobby and (
            lobby_signal_newer
            or lobby_dom
            or dom_play
            or (vision_play and play_conf >= 0.68)
        ):
            if transition_elapsed is not None and transition_elapsed <= 3.0:
                return {"state": "loading", "reason": "recent_lobby_during_transition"}
            if event_matchish and canvas_visible:
                return {"state": "in_match", "reason": "lobby_signal_overridden_by_match_event"}
            if event_loadingish and canvas_visible and (not lobby_dom) and (not dom_play):
                return {"state": "loading", "reason": "recent_lobby_loading_event"}
            return {
                "state": "lobby",
                "reason": f"event:{bot_event_signals.get('last_event_name', '')}",
            }

        if has_recent_match and canvas_visible:
            strong_match_signal = (not has_recent_lobby) or (match_ts > (lobby_ts + 1.6))
            if visual_hint == "in_match" and visual_conf >= 0.60:
                strong_match_signal = True
            if event_lobbyish and (not dom_play) and (not lobby_dom):
                strong_match_signal = False
            if not strong_match_signal:
                return {"state": "loading", "reason": "match_signal_weak"}
            return {
                "state": "in_match",
                "reason": f"event:{bot_event_signals.get('last_event_name', '')}",
            }

        if has_recent_lobby and (lobby_signal_newer or (not has_recent_match)):
            return {
                "state": "lobby",
                "reason": f"event:{bot_event_signals.get('last_event_name', '')}",
            }

        if play_transition_started_at is not None:
            elapsed = transition_elapsed if transition_elapsed is not None else (now_mono - play_transition_started_at)
            if elapsed <= 10.0:
                return {"state": "loading", "reason": "post_play_transition"}
            if elapsed <= 65.0 and canvas_visible:
                return {"state": "in_match", "reason": "post_play_settle"}

        if visual_hint == "death" and visual_conf >= 0.76:
            return {"state": "loading", "reason": f"visual:{visual_hint}:{visual_conf:.2f}"}
        if visual_hint == "in_match" and visual_conf >= 0.72 and canvas_visible:
            return {
                "state": "in_match",
                "reason": f"visual:{visual_hint}:{visual_conf:.2f}",
            }
        if visual_hint == "lobby" and visual_conf >= 0.72 and (not has_recent_match):
            return {"state": "lobby", "reason": f"visual:{visual_hint}:{visual_conf:.2f}"}

        if has_recent_loading:
            return {
                "state": "loading",
                "reason": f"event:{bot_event_signals.get('last_event_name', '')}",
            }

        if canvas_visible:
            if visual_hint == "in_match" and visual_conf >= 0.60:
                return {"state": "in_match", "reason": f"canvas+visual:{visual_conf:.2f}"}
            return {"state": "loading", "reason": "canvas_visible_no_play"}
        return {"state": "loading", "reason": "fallback"}

    def get_canvas_target(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
        x_ratio: float,
        y_ratio: float,
    ) -> Optional[Dict[str, float]]:
        try:
            canvas_rect = frame_obj.evaluate(
                """
                () => {
                  const c = document.querySelector('canvas');
                  if (!c) return null;
                  const r = c.getBoundingClientRect();
                  return {x: r.left, y: r.top, width: r.width, height: r.height};
                }
                """
            )
            if not isinstance(canvas_rect, dict):
                return None
            if canvas_rect.get("width", 0) <= 1 or canvas_rect.get("height", 0) <= 1:
                return None

            frame_x = float(canvas_rect["x"]) + (float(canvas_rect["width"]) * x_ratio)
            frame_y = float(canvas_rect["y"]) + (float(canvas_rect["height"]) * y_ratio)

            iframe_box = None
            try:
                iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
            except Exception:
                iframe_box = None
            if iframe_box:
                page_x = float(iframe_box["x"]) + frame_x
                page_y = float(iframe_box["y"]) + frame_y
            else:
                page_x = frame_x
                page_y = frame_y

            return {
                "frame_x": frame_x,
                "frame_y": frame_y,
                "page_x": page_x,
                "page_y": page_y,
                "source": "canvas_ratio",
            }
        except Exception:
            return None

    def detect_play_button_target(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
    ) -> Optional[Dict[str, float]]:
        # Geometry fallback first: JUGAR is usually centered near bottom.
        fallback_target = get_canvas_target(
            page_obj,
            frame_obj,
            iframe_selector,
            x_ratio=0.50,
            y_ratio=0.92,
        )
        if isinstance(fallback_target, dict):
            fallback_target["confidence"] = 0.15

        # Prefer direct DOM text target when available.
        try:
            dom_rect = frame_obj.evaluate(
                """
                () => {
                  const visibleRect = (el) => {
                    if (!el) return null;
                    const style = window.getComputedStyle(el);
                    if (!style) return null;
                    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || '1') < 0.05) return null;
                    const r = el.getBoundingClientRect();
                    if (!r || r.width < 24 || r.height < 16) return null;
                    return r;
                  };
                  const terms = /(play|jugar|start)/i;
                  const candidates = [];
                  const nodes = document.querySelectorAll('button, [role="button"], a, div, span');
                  for (const el of nodes) {
                    const text = String((el.innerText || el.textContent || '')).trim();
                    if (!text || !terms.test(text)) continue;
                    const r = visibleRect(el);
                    if (!r) continue;
                    const centerX = r.left + (r.width / 2);
                    const centerY = r.top + (r.height / 2);
                    const centerBias = 1.0 - Math.min(1.0, Math.abs((centerX / Math.max(1, window.innerWidth)) - 0.5) * 2.0);
                    const bottomBias = Math.max(0.0, Math.min(1.0, (centerY / Math.max(1, window.innerHeight))));
                    const sizeScore = Math.min(1.0, (r.width * r.height) / (240 * 68));
                    const score = (0.45 * centerBias) + (0.35 * bottomBias) + (0.20 * sizeScore);
                    candidates.push({
                      x: centerX,
                      y: centerY,
                      width: r.width,
                      height: r.height,
                      text: text.slice(0, 32),
                      score,
                    });
                  }
                  if (!candidates.length) return null;
                  candidates.sort((a, b) => b.score - a.score);
                  const best = candidates[0];
                  if (best.score < 0.45) return null;
                  return best;
                }
                """
            )
            if isinstance(dom_rect, dict):
                frame_x = float(dom_rect.get("x", 0.0))
                frame_y = float(dom_rect.get("y", 0.0))
                if frame_x > 1.0 and frame_y > 1.0:
                    iframe_box = None
                    try:
                        iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
                    except Exception:
                        iframe_box = None
                    if iframe_box:
                        page_x = float(iframe_box["x"]) + frame_x
                        page_y = float(iframe_box["y"]) + frame_y
                    else:
                        page_x = frame_x
                        page_y = frame_y
                    return {
                        "frame_x": frame_x,
                        "frame_y": frame_y,
                        "page_x": page_x,
                        "page_y": page_y,
                        "source": "dom_text_play",
                        "confidence": max(0.0, min(1.0, float(dom_rect.get("score", 0.8)))),
                    }
        except Exception:
            pass

        if cv2 is None or np is None:
            return fallback_target

        try:
            canvas_rect = frame_obj.evaluate(
                """
                () => {
                  const c = document.querySelector('canvas');
                  if (!c) return null;
                  const r = c.getBoundingClientRect();
                  return {x: r.left, y: r.top, width: r.width, height: r.height};
                }
                """
            )
            if not isinstance(canvas_rect, dict):
                return fallback_target
            if canvas_rect.get("width", 0) <= 20 or canvas_rect.get("height", 0) <= 20:
                return fallback_target

            iframe_box = None
            try:
                iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
            except Exception:
                iframe_box = None
            if iframe_box:
                clip_x = float(iframe_box["x"]) + float(canvas_rect["x"])
                clip_y = float(iframe_box["y"]) + float(canvas_rect["y"])
            else:
                clip_x = float(canvas_rect["x"])
                clip_y = float(canvas_rect["y"])
            clip_w = float(canvas_rect["width"])
            clip_h = float(canvas_rect["height"])

            raw_png = page_obj.screenshot(
                clip={"x": clip_x, "y": clip_y, "width": clip_w, "height": clip_h},
                type="png",
            )
            image = cv2.imdecode(np.frombuffer(raw_png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return fallback_target

            h, w = image.shape[:2]
            y0 = int(h * 0.78)
            x0 = int(w * 0.25)
            x1 = int(w * 0.75)
            roi = image[y0:h, x0:x1]
            if roi.size == 0:
                return fallback_target

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (15, 70, 120), (45, 255, 255))
            yellow_ratio = float(mask.mean() / 255.0)
            if yellow_ratio < 0.05:
                return fallback_target

            frame_x: float
            frame_y: float
            roi_h, roi_w = mask.shape[:2]
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            best_box = None
            best_area = 0.0
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area <= 0:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                aspect = float(bw) / float(max(1, bh))
                if (
                    bw >= int(roi_w * 0.20)
                    and bh >= int(roi_h * 0.08)
                    and aspect >= 2.0
                    and area > best_area
                ):
                    best_area = area
                    best_box = (bx, by, bw, bh)

            if best_box is not None:
                bx, by, bw, bh = best_box
                frame_x = float(x0 + bx + (bw / 2.0))
                frame_y = float(y0 + by + (bh / 2.0))
            else:
                moments = cv2.moments(mask)
                if moments.get("m00", 0) <= 0:
                    return fallback_target
                cx_roi = float(moments["m10"] / moments["m00"])
                cy_roi = float(moments["m01"] / moments["m00"])
                frame_x = float(x0) + cx_roi
                frame_y = float(y0) + cy_roi

            if iframe_box:
                page_x = float(iframe_box["x"]) + frame_x
                page_y = float(iframe_box["y"]) + frame_y
            else:
                page_x = frame_x
                page_y = frame_y

            conf = min(
                0.98,
                max(
                    0.20,
                    (0.55 * min(1.0, yellow_ratio / 0.12))
                    + (0.45 * min(1.0, best_area / max(1.0, float(roi_w * roi_h) * 0.25))),
                ),
            )

            return {
                "frame_x": frame_x,
                "frame_y": frame_y,
                "page_x": page_x,
                "page_y": page_y,
                "source": "vision_yellow_button",
                "confidence": float(conf),
            }
        except Exception:
            return fallback_target

    def ensure_bot_cursor_overlay(frame_obj: Frame, transition_ms: int = 45) -> bool:
        script = """
        ({transitionMs}) => {
          if (!document || !document.body) return false;
          const transition = Math.max(0, Number(transitionMs || 45));
          if (!window.__lmsBotCursorStyle) {
            const style = document.createElement('style');
            style.id = '__lmsBotCursorStyle';
            style.textContent = `
              #__lmsBotCursor {
                position: fixed;
                width: 18px;
                height: 18px;
                border: 2px solid #00f5ff;
                border-radius: 50%;
                box-shadow: 0 0 0 2px rgba(0, 245, 255, 0.2), 0 0 12px rgba(0, 245, 255, 0.8);
                background: rgba(0, 245, 255, 0.12);
                transform: translate(-50%, -50%);
                transition: left ${transition}ms linear, top ${transition}ms linear;
                pointer-events: none;
                z-index: 2147483647;
                opacity: 0.95;
                will-change: left, top;
              }
              #__lmsBotCursor::after {
                content: 'BOT';
                position: absolute;
                top: -18px;
                left: 50%;
                transform: translateX(-50%);
                color: #00f5ff;
                font-size: 10px;
                font-family: sans-serif;
                font-weight: 700;
                text-shadow: 0 0 6px rgba(0, 245, 255, 0.95);
                white-space: nowrap;
              }
            `;
            (document.head || document.body).appendChild(style);
            window.__lmsBotCursorStyle = true;
          }
          let cursor = document.getElementById('__lmsBotCursor');
          const cx = Math.floor(window.innerWidth * 0.5);
          const cy = Math.floor(window.innerHeight * 0.5);
          if (!cursor) {
            cursor = document.createElement('div');
            cursor.id = '__lmsBotCursor';
            cursor.style.left = `${cx}px`;
            cursor.style.top = `${cy}px`;
            document.body.appendChild(cursor);
          }
          if (!window.__lmsBotCursorProbe) {
            window.__lmsBotCursorProbe = {
              moves: 0,
              lastX: cx,
              lastY: cy,
              lastSource: 'init',
              visible: true,
              updatedAt: Date.now()
            };
          } else {
            window.__lmsBotCursorProbe.visible = true;
            window.__lmsBotCursorProbe.updatedAt = Date.now();
          }
          return true;
        }
        """
        try:
            return bool(frame_obj.evaluate(script, {"transitionMs": max(0, int(transition_ms))}))
        except Exception:
            return False

    def move_bot_cursor_overlay(
        frame_obj: Frame,
        frame_x: float,
        frame_y: float,
        source: str = "unknown",
    ) -> bool:
        script = """
        ({x, y, source}) => {
          const cursor = document.getElementById('__lmsBotCursor');
          if (!cursor) return {ok: false};
          const clampedX = Math.max(1, Math.min(window.innerWidth - 1, Number(x)));
          const clampedY = Math.max(1, Math.min(window.innerHeight - 1, Number(y)));
          cursor.style.left = `${clampedX}px`;
          cursor.style.top = `${clampedY}px`;

          if (!window.__lmsBotCursorProbe) {
            window.__lmsBotCursorProbe = {
              moves: 0,
              lastX: clampedX,
              lastY: clampedY,
              lastSource: String(source || 'unknown'),
              visible: true,
              updatedAt: Date.now()
            };
          } else {
            const p = window.__lmsBotCursorProbe;
            const prevX = Number(p.lastX ?? clampedX);
            const prevY = Number(p.lastY ?? clampedY);
            const dx = Math.abs(prevX - clampedX);
            const dy = Math.abs(prevY - clampedY);
            if ((dx + dy) >= 0.5) {
              p.moves = Number(p.moves || 0) + 1;
            }
            p.lastX = clampedX;
            p.lastY = clampedY;
            p.lastSource = String(source || 'unknown');
            p.visible = true;
            p.updatedAt = Date.now();
          }
          return {ok: true};
        }
        """
        try:
            result = frame_obj.evaluate(
                script,
                {"x": frame_x, "y": frame_y, "source": str(source or "unknown")},
            )
            if isinstance(result, dict):
                return bool(result.get("ok"))
            return bool(result)
        except Exception:
            return False

    def read_bot_cursor_probe(frame_obj: Frame) -> Dict[str, Any]:
        try:
            result = frame_obj.evaluate(
                """
                () => window.__lmsBotCursorProbe || {
                  moves: 0,
                  lastX: 0,
                  lastY: 0,
                  lastSource: '',
                  visible: false,
                  updatedAt: 0
                }
                """
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {
            "moves": 0,
            "lastX": 0.0,
            "lastY": 0.0,
            "lastSource": "",
            "visible": False,
            "updatedAt": 0,
        }

    def ensure_bot_debug_hud(frame_obj: Frame) -> bool:
        script = """
        () => {
          if (!document || !document.body) return false;
          if (!window.__lmsBotHudStyle) {
            const style = document.createElement('style');
            style.id = '__lmsBotHudStyle';
            style.textContent = `
              #__lmsBotHud {
                position: fixed;
                top: 10px;
                right: 10px;
                width: 330px;
                max-width: min(38vw, 330px);
                border-radius: 12px;
                padding: 10px 10px 8px 10px;
                background: rgba(2, 6, 23, 0.82);
                border: 1px solid rgba(56, 189, 248, 0.65);
                color: #e2e8f0;
                font-family: Consolas, Menlo, Monaco, monospace;
                font-size: 11px;
                line-height: 1.28;
                z-index: 2147483646;
                backdrop-filter: blur(3px);
                pointer-events: none;
              }
              #__lmsBotHud .h-title {
                font-weight: 700;
                letter-spacing: 0.3px;
                color: #7dd3fc;
                margin-bottom: 6px;
              }
              #__lmsBotHud .h-row { margin: 2px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
              #__lmsBotHud .h-label { color: #93c5fd; }
              #__lmsBotHud .h-value { color: #f8fafc; }
              #__lmsBotHud .h-keys {
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 4px;
                margin-top: 6px;
                margin-bottom: 6px;
              }
              #__lmsBotHud .h-key {
                border: 1px solid rgba(148, 163, 184, 0.45);
                border-radius: 6px;
                text-align: center;
                padding: 2px 0;
                color: #cbd5e1;
                background: rgba(30, 41, 59, 0.62);
                font-weight: 600;
              }
              #__lmsBotHud .h-key.active {
                border-color: rgba(250, 204, 21, 0.95);
                color: #111827;
                background: linear-gradient(180deg, #fde047, #f59e0b);
                box-shadow: 0 0 10px rgba(250, 204, 21, 0.45);
              }
              #__lmsBotHud .h-feed {
                margin-top: 6px;
                border-top: 1px dashed rgba(148, 163, 184, 0.35);
                padding-top: 5px;
                max-height: 86px;
                overflow: hidden;
              }
              #__lmsBotHud .h-feed-line {
                color: #cbd5e1;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
              }
            `;
            (document.head || document.body).appendChild(style);
            window.__lmsBotHudStyle = true;
          }
          let hud = document.getElementById('__lmsBotHud');
          if (!hud) {
            hud = document.createElement('div');
            hud.id = '__lmsBotHud';
            hud.innerHTML = `
              <div class="h-title">BOT FEEDBACK HUD</div>
              <div class="h-row"><span class="h-label">State:</span> <span id="__lmsHudState" class="h-value">init</span></div>
              <div class="h-row"><span class="h-label">Reason:</span> <span id="__lmsHudReason" class="h-value">-</span></div>
              <div class="h-row"><span class="h-label">Action:</span> <span id="__lmsHudAction" class="h-value">idle</span></div>
              <div class="h-row"><span class="h-label">Input:</span> <span id="__lmsHudInput" class="h-value">kd=0 ku=0</span></div>
              <div class="h-row"><span class="h-label">Pointer:</span> <span id="__lmsHudPointer" class="h-value">down=0 up=0</span></div>
              <div class="h-row"><span class="h-label">Cursor:</span> <span id="__lmsHudCursor" class="h-value">moves=0 src=-</span></div>
              <div class="h-row"><span class="h-label">Click:</span> <span id="__lmsHudClick" class="h-value">click0=0 target=-</span></div>
              <div class="h-row"><span class="h-label">Enemy:</span> <span id="__lmsHudEnemy" class="h-value">seen=0 conf=0.00 dir=-</span></div>
              <div class="h-row"><span class="h-label">Combat:</span> <span id="__lmsHudDamage" class="h-value">out=0 in=0 mana=100</span></div>
              <div class="h-row"><span class="h-label">Zone:</span> <span id="__lmsHudZone" class="h-value">counter=- safe=(-,-,-)</span></div>
              <div class="h-row"><span class="h-label">Death:</span> <span id="__lmsHudDeath" class="h-value">cause=unknown conf=0.00 src=-</span></div>
              <div class="h-row"><span class="h-label">Abilities:</span> <span id="__lmsHudAbility" class="h-value">last=- class=-</span></div>
              <div class="h-keys">
                <div id="__lmsHudKey_KeyW" class="h-key">W</div>
                <div id="__lmsHudKey_KeyA" class="h-key">A</div>
                <div id="__lmsHudKey_KeyS" class="h-key">S</div>
                <div id="__lmsHudKey_KeyD" class="h-key">D</div>
                <div id="__lmsHudKey_Shift" class="h-key">SHIFT</div>
                <div id="__lmsHudKey_Space" class="h-key">SPACE</div>
                <div id="__lmsHudKey_KeyR" class="h-key">R</div>
                <div id="__lmsHudKey_KeyC" class="h-key">C</div>
                <div id="__lmsHudKey_Digit1" class="h-key">1</div>
                <div id="__lmsHudKey_Digit2" class="h-key">2</div>
                <div id="__lmsHudKey_Digit3" class="h-key">3</div>
                <div id="__lmsHudKey_MouseRight" class="h-key">RMB</div>
              </div>
              <div id="__lmsHudFeed" class="h-feed"></div>
            `;
            document.body.appendChild(hud);
          }
          if (!window.__lmsBotHudState) {
            window.__lmsBotHudState = {feed: []};
          }
          return true;
        }
        """
        try:
            return bool(frame_obj.evaluate(script))
        except Exception:
            return False

    def update_bot_debug_hud(frame_obj: Frame, payload: Dict[str, Any]) -> None:
        script = """
        (data) => {
          if (!document || !document.body) return;
          const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = String(value ?? '');
          };
          const safeNum = (v) => Number(v || 0);
          const safeStr = (v) => String(v ?? '');
          const active = Array.isArray(data.active_keys) ? data.active_keys.map((x) => String(x)) : [];
          const keyIds = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Shift', 'Space', 'KeyR', 'KeyC', 'Digit1', 'Digit2', 'Digit3', 'MouseRight'];
          for (const key of keyIds) {
            const el = document.getElementById(`__lmsHudKey_${key}`);
            if (!el) continue;
            if (active.includes(key)) el.classList.add('active');
            else el.classList.remove('active');
          }

          setText('__lmsHudState', safeStr(data.state || 'unknown'));
          setText('__lmsHudReason', safeStr(data.reason || '-'));
          setText('__lmsHudAction', safeStr(data.action || 'idle'));
          setText(
            '__lmsHudInput',
            `kd=${safeNum(data.key_down)} ku=${safeNum(data.key_up)} last=${safeStr(data.last_key || '-')}`
          );
          setText(
            '__lmsHudPointer',
            `down=${safeNum(data.pointer_down)} up=${safeNum(data.pointer_up)} move=${safeNum(data.pointer_move)}`
          );
          setText(
            '__lmsHudCursor',
            `moves=${safeNum(data.cursor_moves)} src=${safeStr(data.cursor_source || '-')}`
          );
          setText(
            '__lmsHudClick',
            `click0=${safeNum(data.click0)} target=${safeStr(data.click_target || '-')}`
          );
          setText(
            '__lmsHudEnemy',
            `seen=${safeNum(data.enemy_seen)} conf=${Number(data.enemy_conf || 0).toFixed(2)} dir=${safeStr(data.enemy_dir || '-')} vis=${safeStr(data.visual_state || 'unknown')}:${Number(data.visual_conf || 0).toFixed(2)}`
          );
          setText(
            '__lmsHudDamage',
            `out=${Number(data.damage_done || 0).toFixed(1)} in=${Number(data.damage_taken || 0).toFixed(1)} mana=${Number(data.mana || 0).toFixed(1)}`
          );
          setText(
            '__lmsHudZone',
            `counter=${safeStr(data.zone_counter || '-')} safe=${safeStr(data.safe_zone || '(-,-,-)')}`
          );
          setText(
            '__lmsHudDeath',
            `cause=${safeStr(data.death_cause || 'unknown')} conf=${Number(data.death_conf || 0).toFixed(2)} src=${safeStr(data.death_source || '-')}`
          );
          setText(
            '__lmsHudAbility',
            `last=${safeStr(data.ability_last || '-')} class=${safeStr(data.ability_class || '-')} dash_cd=${Number(data.dash_cd || 0).toFixed(2)}`
          );

          const feedEl = document.getElementById('__lmsHudFeed');
          if (feedEl) {
            if (!window.__lmsBotHudState) window.__lmsBotHudState = {feed: []};
            const feed = window.__lmsBotHudState.feed || [];
            const line = safeStr(data.feed_line || '').trim();
            if (line) {
              feed.push(line);
              while (feed.length > 6) feed.shift();
              window.__lmsBotHudState.feed = feed;
            }
            feedEl.innerHTML = feed
              .map((lineText) => `<div class="h-feed-line">${lineText}</div>`)
              .join('');
          }
        }
        """
        try:
            frame_obj.evaluate(script, payload)
        except Exception:
            pass

    def install_input_feedback_probe(frame_obj: Frame) -> None:
        script = """
        () => {
          if (window.__lmsInputProbe) return;
          window.__lmsInputProbe = {
            keyDown: 0,
            keyUp: 0,
            lastKeyDown: '',
            lastKeyUp: '',
            pointerDown: 0,
            pointerUp: 0,
            pointerMove: 0,
            focusEvents: 0,
            blurEvents: 0,
            lastEventTs: Date.now()
          };
          const p = window.__lmsInputProbe;
          window.addEventListener('keydown', (e) => {
            p.keyDown += 1;
            p.lastKeyDown = String(e.key || e.code || '');
            p.lastEventTs = Date.now();
          }, true);
          window.addEventListener('keyup', (e) => {
            p.keyUp += 1;
            p.lastKeyUp = String(e.key || e.code || '');
            p.lastEventTs = Date.now();
          }, true);
          window.addEventListener('pointerdown', () => { p.pointerDown += 1; p.lastEventTs = Date.now(); }, true);
          window.addEventListener('pointerup', () => { p.pointerUp += 1; p.lastEventTs = Date.now(); }, true);
          window.addEventListener('pointermove', () => { p.pointerMove += 1; p.lastEventTs = Date.now(); }, true);
          window.addEventListener('focus', () => { p.focusEvents += 1; p.lastEventTs = Date.now(); }, true);
          window.addEventListener('blur', () => { p.blurEvents += 1; p.lastEventTs = Date.now(); }, true);
        }
        """
        try:
            frame_obj.evaluate(script)
        except Exception:
            pass

    def read_input_feedback_probe(frame_obj: Frame) -> Dict[str, Any]:
        try:
            result = frame_obj.evaluate(
                """
                () => window.__lmsInputProbe || {
                  keyDown: 0,
                  keyUp: 0,
                  lastKeyDown: '',
                  lastKeyUp: '',
                  pointerDown: 0,
                  pointerUp: 0,
                  pointerMove: 0,
                  focusEvents: 0,
                  blurEvents: 0,
                  lastEventTs: 0
                }
                """
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {
            "keyDown": 0,
            "keyUp": 0,
            "lastKeyDown": "",
            "lastKeyUp": "",
            "pointerDown": 0,
            "pointerUp": 0,
            "pointerMove": 0,
            "focusEvents": 0,
            "blurEvents": 0,
            "lastEventTs": 0,
        }

    def create_feedback_session(base_dir: str, mode: str, explicit_jsonl: str = "") -> Optional[Dict[str, Any]]:
        base = str(base_dir or "").strip()
        explicit = str(explicit_jsonl or "").strip()
        if not base and not explicit:
            return None
        try:
            if explicit:
                jsonl_path = Path(explicit)
                run_dir = jsonl_path.parent
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = Path(base) / f"{mode}_{ts}"
                jsonl_path = run_dir / "feedback_stream.jsonl"
            run_dir.mkdir(parents=True, exist_ok=True)
            screen_dir = run_dir / "screens"
            screen_dir.mkdir(parents=True, exist_ok=True)
            return {
                "run_dir": run_dir,
                "jsonl_path": jsonl_path,
                "screen_dir": screen_dir,
                "last_shot_at": 0.0,
                "shot_count": 0,
            }
        except Exception as exc:
            print(f"[BOT][FEEDBACK][WARN] No se pudo crear sesion de feedback: {exc}")
            return None

    def append_feedback_event(
        session: Optional[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> None:
        if not session:
            return
        try:
            jsonl_path = Path(session["jsonl_path"])
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:
            print(f"[BOT][FEEDBACK][WARN] No se pudo guardar evento feedback: {exc}")

    def maybe_capture_feedback_screenshot(
        session: Optional[Dict[str, Any]],
        page_obj: Page,
        now_mono: float,
        every_sec: float,
        max_screenshots: int,
        label: str,
    ) -> Optional[str]:
        if not session:
            return None
        interval = float(every_sec or 0.0)
        if interval <= 0.0:
            return None
        max_shots = max(1, int(max_screenshots))
        current = int(session.get("shot_count", 0))
        if current >= max_shots:
            return None
        last_at = float(session.get("last_shot_at", 0.0))
        if last_at > 0.0 and (now_mono - last_at) < interval:
            return None
        safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(label or "tick"))
        shot_index = current + 1
        filename = f"{shot_index:04d}_{safe_label}.png"
        shot_path = Path(session["screen_dir"]) / filename
        try:
            page_obj.screenshot(path=str(shot_path), full_page=True)
            session["shot_count"] = shot_index
            session["last_shot_at"] = now_mono
            return str(shot_path)
        except Exception as exc:
            print(f"[BOT][FEEDBACK][WARN] screenshot feedback fallo: {exc}")
            return None

    def render_feedback_video(session: Optional[Dict[str, Any]], fps: float) -> str:
        if not session or cv2 is None or np is None:
            return ""
        try:
            screen_dir = Path(session["screen_dir"])
            run_dir = Path(session["run_dir"])
            frames = sorted(screen_dir.glob("*.png"))
            if len(frames) < 2:
                return ""
            first = cv2.imread(str(frames[0]))
            if first is None or first.size == 0:
                return ""
            height, width = first.shape[:2]
            out_path = run_dir / "timeline.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, float(fps)),
                (int(width), int(height)),
            )
            if not writer.isOpened():
                return ""
            try:
                for frame_path in frames:
                    img = cv2.imread(str(frame_path))
                    if img is None or img.size == 0:
                        continue
                    if img.shape[1] != width or img.shape[0] != height:
                        img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
                    writer.write(img)
            finally:
                writer.release()
            return str(out_path)
        except Exception as exc:
            print(f"[BOT][FEEDBACK][WARN] No se pudo renderizar video feedback: {exc}")
            return ""

    def extract_visual_feedback_from_screenshot(
        image_path: str,
        max_names: int,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "ok": False,
            "engine": "none",
            "state_hint": "unknown",
            "state_confidence": 0.0,
            "lobby_hits": 0,
            "in_match_hits": 0,
            "safe_zone_hits": 0,
            "toxic_zone_hits": 0,
            "toxic_color_ratio": 0.0,
            "toxic_top_ratio": 0.0,
            "toxic_bottom_ratio": 0.0,
            "toxic_left_ratio": 0.0,
            "toxic_right_ratio": 0.0,
            "toxic_escape_keys": [],
            "names": [],
            "damage_numbers": [],
            "raw_excerpt": "",
        }
        if not image_path:
            return result
        if cv2 is None or np is None or pytesseract is None:
            return result
        try:
            try:
                current_tesseract_cmd = str(getattr(pytesseract.pytesseract, "tesseract_cmd", "") or "").strip()
                if (not current_tesseract_cmd) or current_tesseract_cmd.lower() == "tesseract":
                    for candidate in (
                        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
                    ):
                        if Path(candidate).exists():
                            pytesseract.pytesseract.tesseract_cmd = candidate
                            break
            except Exception:
                pass
            img = cv2.imread(str(image_path))
            if img is None or img.size == 0:
                return result
            h, w = img.shape[:2]
            if h <= 2 or w <= 2:
                return result

            rois = [
                (0, 0, w, max(12, int(h * 0.24))),  # top HUD / names
                (0, int(h * 0.24), w, max(12, int(h * 0.24))),  # mid feed
                (0, int(h * 0.75), w, max(12, int(h * 0.22))),  # bottom bars
            ]
            texts: List[str] = []
            for x, y, rw, rh in rois:
                x2 = min(w, max(1, x + rw))
                y2 = min(h, max(1, y + rh))
                crop = img[y:y2, x:x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                bin_img = cv2.adaptiveThreshold(
                    blur,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    5,
                )
                txt = pytesseract.image_to_string(
                    bin_img,
                    config="--oem 3 --psm 6",
                )
                if txt and txt.strip():
                    texts.append(txt.strip())

            merged = "\n".join(texts).strip()
            if not merged:
                return result

            lower = merged.lower()
            lower_norm = normalize_text_for_match(merged)
            lobby_tokens = (
                "play",
                "jugar",
                "lobby",
                "salon",
                "select",
                "character",
                "ready",
                "queue",
                "matchmaking",
            )
            match_tokens = (
                "damage",
                "dmg",
                "hp",
                "mana",
                "kill",
                "kills",
                "zone",
                "safe",
                "alive",
                "map",
            )
            safe_zone_tokens = (
                "safe zone",
                "zona segura",
                "inside zone",
                "dentro de la zona",
                "stay in zone",
            )
            toxic_zone_tokens = (
                "toxic",
                "toxic cloud",
                "nube toxica",
                "nube tóxica",
                "poison",
                "gas",
                "storm",
                "outside zone",
                "fuera de zona",
                "fuera de la zona",
                "veneno",
            )
            death_tokens = (
                "you died",
                "has muerto",
                "eliminated",
                "eliminado",
                "defeated",
                "defeat",
                "defeat screen",
                "killed",
                "muerto",
                "derrotado",
                "derrota",
                "has sido eliminado",
                "you lose",
                "you lost",
                "game over",
                "estas fuera",
                "spectating",
            )
            lobby_hits = sum(lower_norm.count(token) for token in lobby_tokens)
            in_match_hits = sum(lower_norm.count(token) for token in match_tokens)
            safe_zone_hits = sum(
                lower_norm.count(normalize_text_for_match(token))
                for token in safe_zone_tokens
            )
            toxic_zone_hits = sum(
                lower_norm.count(normalize_text_for_match(token))
                for token in toxic_zone_tokens
            )
            death_hits = sum(lower_norm.count(token) for token in death_tokens)
            if fuzzy_contains_phrase(lower_norm, "estas fuera", threshold=89):
                death_hits += 2

            toxic_color_ratio = 0.0
            toxic_top_ratio = 0.0
            toxic_bottom_ratio = 0.0
            toxic_left_ratio = 0.0
            toxic_right_ratio = 0.0
            toxic_escape_keys: List[str] = []
            try:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h_ch = hsv[:, :, 0]
                s_ch = hsv[:, :, 1]
                v_ch = hsv[:, :, 2]
                toxic_mask = (
                    (((h_ch >= 120) & (h_ch <= 170)) | ((h_ch >= 8) & (h_ch <= 18)))
                    & (s_ch >= 55)
                    & (v_ch >= 35)
                )
                border_w = max(6, int(w * 0.18))
                border_h = max(6, int(h * 0.18))
                border_mask = np.zeros((h, w), dtype=np.uint8)
                border_mask[:border_h, :] = 1
                border_mask[-border_h:, :] = 1
                border_mask[:, :border_w] = 1
                border_mask[:, -border_w:] = 1
                border_pixels = int(np.count_nonzero(border_mask))
                if border_pixels > 0:
                    toxic_border = int(np.count_nonzero(toxic_mask & (border_mask > 0)))
                    toxic_color_ratio = float(toxic_border) / float(border_pixels)
                top_pixels = max(1, int(border_h * w))
                bottom_pixels = max(1, int(border_h * w))
                left_pixels = max(1, int(border_w * h))
                right_pixels = max(1, int(border_w * h))
                toxic_top_ratio = float(np.count_nonzero(toxic_mask[:border_h, :])) / float(top_pixels)
                toxic_bottom_ratio = float(np.count_nonzero(toxic_mask[-border_h:, :])) / float(bottom_pixels)
                toxic_left_ratio = float(np.count_nonzero(toxic_mask[:, :border_w])) / float(left_pixels)
                toxic_right_ratio = float(np.count_nonzero(toxic_mask[:, -border_w:])) / float(right_pixels)

                x_bias = toxic_left_ratio - toxic_right_ratio
                y_bias = toxic_top_ratio - toxic_bottom_ratio
                side_eps = 0.005
                x_key = ""
                y_key = ""
                if abs(x_bias) >= side_eps:
                    x_key = "KeyD" if x_bias > 0 else "KeyA"
                if abs(y_bias) >= side_eps:
                    y_key = "KeyS" if y_bias > 0 else "KeyW"
                if y_key:
                    toxic_escape_keys.append(y_key)
                if x_key:
                    toxic_escape_keys.append(x_key)
            except Exception:
                toxic_color_ratio = 0.0
                toxic_top_ratio = 0.0
                toxic_bottom_ratio = 0.0
                toxic_left_ratio = 0.0
                toxic_right_ratio = 0.0
                toxic_escape_keys = []
            if toxic_color_ratio >= 0.028:
                toxic_zone_hits += 1
            state_hint = "unknown"
            state_conf = 0.0
            if death_hits >= 1:
                state_hint = "death"
                state_conf = min(0.97, 0.62 + (0.11 * float(death_hits)))
            elif lobby_hits >= 2 and lobby_hits > (in_match_hits + 1):
                state_hint = "lobby"
                state_conf = min(0.95, 0.45 + (0.08 * float(lobby_hits)))
            elif in_match_hits >= 2 and in_match_hits >= (lobby_hits + 1):
                state_hint = "in_match"
                state_conf = min(0.95, 0.45 + (0.08 * float(in_match_hits)))

            damage_values: List[float] = []
            for match in re.finditer(
                r"(?:damage|dmg|da[oÃ±]o)[^0-9]{0,10}([0-9]{1,6}(?:[.,][0-9]{1,2})?)",
                lower,
            ):
                value_raw = str(match.group(1) or "").replace(",", ".")
                try:
                    val = float(value_raw)
                except Exception:
                    continue
                if 0 <= val <= 200000:
                    damage_values.append(val)

            # Fallback: keep some standalone large numbers as possible counters
            if not damage_values:
                for raw_num in re.findall(r"\b[0-9]{2,6}\b", merged):
                    try:
                        nval = float(raw_num)
                    except Exception:
                        continue
                    if 20 <= nval <= 200000:
                        damage_values.append(nval)
                        if len(damage_values) >= 4:
                            break

            stop_words = {
                "Play",
                "Jugar",
                "Select",
                "Loading",
                "Ready",
                "Lobby",
                "BOT",
                "FEEDBACK",
                "HUD",
                "State",
                "Reason",
                "Action",
                "Damage",
                "Mana",
                "Health",
                "Zone",
                "Safe",
                "Map",
                "Match",
            }
            name_candidates: List[str] = []
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,15}\b", merged):
                if token in stop_words:
                    continue
                if token.isupper() and len(token) <= 3:
                    continue
                if token.lower() in ("www", "http", "https"):
                    continue
                if token not in name_candidates:
                    name_candidates.append(token)
                if len(name_candidates) >= max(1, int(max_names)):
                    break

            result.update(
                {
                    "ok": True,
                    "engine": "pytesseract",
                    "state_hint": state_hint,
                    "state_confidence": float(state_conf),
                    "lobby_hits": int(lobby_hits),
                    "in_match_hits": int(in_match_hits),
                    "safe_zone_hits": int(safe_zone_hits),
                    "toxic_zone_hits": int(toxic_zone_hits),
                    "toxic_color_ratio": float(toxic_color_ratio),
                    "toxic_top_ratio": float(toxic_top_ratio),
                    "toxic_bottom_ratio": float(toxic_bottom_ratio),
                    "toxic_left_ratio": float(toxic_left_ratio),
                    "toxic_right_ratio": float(toxic_right_ratio),
                    "toxic_escape_keys": list(toxic_escape_keys[:2]),
                    "death_hits": int(death_hits),
                    "names": name_candidates,
                    "damage_numbers": [float(v) for v in damage_values[:8]],
                    "raw_excerpt": merged[:500],
                }
            )
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result

    def install_click_probe(frame_obj: Frame) -> None:
        script = """
        () => {
          if (window.__lmsClickProbe) return;
          window.__lmsClickProbe = {down0:0, up0:0, click0:0, lastTarget:''};
          const onDown = (e) => {
            if (e.button === 0) {
              window.__lmsClickProbe.down0 += 1;
              window.__lmsClickProbe.lastTarget = (e.target && e.target.tagName) || '';
            }
          };
          const onUp = (e) => {
            if (e.button === 0) {
              window.__lmsClickProbe.up0 += 1;
              window.__lmsClickProbe.lastTarget = (e.target && e.target.tagName) || '';
            }
          };
          const onClick = (e) => {
            if (e.button === 0) {
              window.__lmsClickProbe.click0 += 1;
              window.__lmsClickProbe.lastTarget = (e.target && e.target.tagName) || '';
            }
          };
          window.addEventListener('mousedown', onDown, true);
          window.addEventListener('mouseup', onUp, true);
          window.addEventListener('click', onClick, true);
        }
        """
        try:
            frame_obj.evaluate(script)
        except Exception:
            pass

    def read_click_probe(frame_obj: Frame) -> Dict[str, Any]:
        try:
            result = frame_obj.evaluate(
                "() => window.__lmsClickProbe || {down0:0, up0:0, click0:0, lastTarget:''}"
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}

    def get_attack_target(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
    ) -> Optional[Dict[str, float]]:
        try:
            canvas_rect = frame_obj.evaluate(
                """
                () => {
                  const c = document.querySelector('canvas');
                  if (!c) return null;
                  const r = c.getBoundingClientRect();
                  return {x: r.left, y: r.top, width: r.width, height: r.height};
                }
                """
            )
            if isinstance(canvas_rect, dict) and canvas_rect.get("width", 0) > 1 and canvas_rect.get("height", 0) > 1:
                frame_x = float(canvas_rect["x"]) + (float(canvas_rect["width"]) / 2.0)
                frame_y = float(canvas_rect["y"]) + (float(canvas_rect["height"]) / 2.0)
                iframe_box = None
                try:
                    iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
                except Exception:
                    iframe_box = None
                if iframe_box:
                    page_x = float(iframe_box["x"]) + frame_x
                    page_y = float(iframe_box["y"]) + frame_y
                else:
                    # canvas rect from main_frame is already in page viewport coordinates
                    page_x = frame_x
                    page_y = frame_y
                return {
                    "frame_x": frame_x,
                    "frame_y": frame_y,
                    "page_x": page_x,
                    "page_y": page_y,
                }
        except Exception:
            pass

        try:
            iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
            if iframe_box and iframe_box.get("width", 0) > 1 and iframe_box.get("height", 0) > 1:
                return {
                    "frame_x": float(iframe_box["width"]) / 2.0,
                    "frame_y": float(iframe_box["height"]) / 2.0,
                    "page_x": float(iframe_box["x"]) + (float(iframe_box["width"]) / 2.0),
                    "page_y": float(iframe_box["y"]) + (float(iframe_box["height"]) / 2.0),
                }
        except Exception:
            pass

        viewport_size = page_obj.viewport_size
        if viewport_size:
            return {
                "frame_x": float(viewport_size["width"]) / 2.0,
                "frame_y": float(viewport_size["height"]) / 2.0,
                "page_x": float(viewport_size["width"]) / 2.0,
                "page_y": float(viewport_size["height"]) / 2.0,
            }
        return None

    def build_cursor_patrol_target(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
        now_mono: float,
    ) -> Optional[Dict[str, float]]:
        x_ratio = 0.50 + (0.10 * math.sin(now_mono * 1.7))
        y_ratio = 0.82 + (0.06 * math.cos(now_mono * 1.3))
        x_ratio = max(0.20, min(0.80, x_ratio))
        y_ratio = max(0.65, min(0.94, y_ratio))
        target = get_canvas_target(
            page_obj=page_obj,
            frame_obj=frame_obj,
            iframe_selector=iframe_selector,
            x_ratio=x_ratio,
            y_ratio=y_ratio,
        )
        if target is None:
            return None
        target["source"] = "cursor_patrol"
        return target

    def add_idle_wiggle_target(
        target: Optional[Dict[str, float]],
        now_mono: float,
        amplitude_px: float,
    ) -> Optional[Dict[str, float]]:
        if target is None:
            return None
        amplitude = max(1.0, float(amplitude_px))
        dx = amplitude * math.sin(now_mono * 3.0)
        dy = (amplitude * 0.60) * math.cos(now_mono * 2.4)

        adjusted: Dict[str, float] = dict(target)
        adjusted["frame_x"] = float(target["frame_x"]) + dx
        adjusted["frame_y"] = float(target["frame_y"]) + dy
        adjusted["page_x"] = float(target["page_x"]) + dx
        adjusted["page_y"] = float(target["page_y"]) + dy
        source = str(target.get("source", "unknown"))
        adjusted["source"] = f"{source}:wiggle"
        return adjusted

    def normalize_move_combo(keys: List[str]) -> List[str]:
        uniq: List[str] = []
        for key in keys:
            if key in MOVE_VECTORS and key not in uniq:
                uniq.append(key)
        if not uniq:
            return ["KeyW"]

        # avoid opposite keys in same combo (cancel each other)
        if len(uniq) >= 2:
            filtered: List[str] = []
            for key in uniq:
                opp = OPPOSITE_KEY.get(key)
                if opp and opp in filtered:
                    continue
                filtered.append(key)
            uniq = filtered if filtered else [uniq[0]]
        return uniq[:2]

    def build_zone_recovery_move_candidates(
        player_x: float,
        player_y: float,
        safe_x: float,
        safe_y: float,
    ) -> List[List[str]]:
        dx = float(safe_x) - float(player_x)
        dy = float(safe_y) - float(player_y)
        x_key = ""
        y_key = ""
        eps = 0.015
        if abs(dx) >= eps:
            x_key = "KeyD" if dx > 0 else "KeyA"
        if abs(dy) >= eps:
            # Assume Y+ moves down (KeyS). Keep inverse option as fallback.
            y_key = "KeyS" if dy > 0 else "KeyW"
        candidates: List[List[str]] = []
        if y_key and x_key:
            candidates.append(normalize_move_combo([y_key, x_key]))
            candidates.append(normalize_move_combo([OPPOSITE_KEY.get(y_key, y_key), x_key]))
        if y_key:
            candidates.append(normalize_move_combo([y_key]))
        if x_key:
            candidates.append(normalize_move_combo([x_key]))
        if y_key:
            candidates.append(normalize_move_combo([OPPOSITE_KEY.get(y_key, y_key)]))
        uniq: List[List[str]] = []
        seen: Set[str] = set()
        for combo in candidates:
            key = "+".join(combo)
            if key and key not in seen:
                seen.add(key)
                uniq.append(combo)
        return uniq or [normalize_move_combo(["KeyW", "KeyD"])]

    def build_visual_toxic_recovery_move_candidates(
        toxic_top_ratio: float,
        toxic_bottom_ratio: float,
        toxic_left_ratio: float,
        toxic_right_ratio: float,
    ) -> List[List[str]]:
        top_ratio = max(0.0, float(toxic_top_ratio))
        bottom_ratio = max(0.0, float(toxic_bottom_ratio))
        left_ratio = max(0.0, float(toxic_left_ratio))
        right_ratio = max(0.0, float(toxic_right_ratio))
        x_bias = left_ratio - right_ratio
        y_bias = top_ratio - bottom_ratio
        side_eps = 0.005
        x_key = ""
        y_key = ""
        if abs(x_bias) >= side_eps:
            x_key = "KeyD" if x_bias > 0 else "KeyA"
        if abs(y_bias) >= side_eps:
            y_key = "KeyS" if y_bias > 0 else "KeyW"

        candidates: List[List[str]] = []
        if y_key and x_key:
            candidates.append(normalize_move_combo([y_key, x_key]))
        if y_key:
            candidates.append(normalize_move_combo([y_key]))
        if x_key:
            candidates.append(normalize_move_combo([x_key]))

        if not candidates:
            side_values = {
                "top": top_ratio,
                "bottom": bottom_ratio,
                "left": left_ratio,
                "right": right_ratio,
            }
            dominant_side = max(side_values, key=side_values.get)
            if side_values[dominant_side] >= 0.02:
                if dominant_side == "top":
                    candidates.append(normalize_move_combo(["KeyS"]))
                elif dominant_side == "bottom":
                    candidates.append(normalize_move_combo(["KeyW"]))
                elif dominant_side == "left":
                    candidates.append(normalize_move_combo(["KeyD"]))
                else:
                    candidates.append(normalize_move_combo(["KeyA"]))

        uniq: List[List[str]] = []
        seen: Set[str] = set()
        for combo in candidates:
            key = "+".join(combo)
            if key and key not in seen:
                seen.add(key)
                uniq.append(combo)
        return uniq

    def choose_in_match_move_combo(
        primary_key: str,
        step_index: int,
        use_diagonal: bool,
        escape_mode: bool,
    ) -> List[str]:
        primary = primary_key if primary_key in MOVE_VECTORS else "KeyW"
        if escape_mode:
            # reverse + strafe to disengage from collisions
            reverse_key = OPPOSITE_KEY.get(primary, "KeyS")
            strafe_key = ORTHOGONAL_STRAFE.get(reverse_key, "KeyA")
            return normalize_move_combo([reverse_key, strafe_key])
        if use_diagonal and (step_index % 2 == 0):
            strafe_key = ORTHOGONAL_STRAFE.get(primary, "KeyD")
            return normalize_move_combo([primary, strafe_key])
        return normalize_move_combo([primary])

    def lms_re_action_to_move_combo(action_obj: Any) -> List[str]:
        keys: List[str] = []
        try:
            move_x = float(getattr(action_obj, "move_x", 0.0) or 0.0)
            move_y = float(getattr(action_obj, "move_y", 0.0) or 0.0)
        except Exception:
            return normalize_move_combo(["KeyW"])
        eps = 0.16
        if move_y <= -eps:
            keys.append("KeyW")
        elif move_y >= eps:
            keys.append("KeyS")
        if move_x <= -eps:
            keys.append("KeyA")
        elif move_x >= eps:
            keys.append("KeyD")
        return normalize_move_combo(keys or ["KeyW"])

    def lms_re_action_to_ability_key(action_obj: Any) -> str:
        try:
            ability_id = getattr(action_obj, "ability_id", None)
            if ability_id is None:
                return ""
            ability_num = int(ability_id)
        except Exception:
            return ""
        if ability_num == 1:
            return "Digit1"
        if ability_num == 2:
            return "Digit2"
        if ability_num == 3:
            return "Digit3"
        return ""

    def build_lms_re_observation_from_live_signals(
        tick_id: int,
        bot_event_signals_ref: Dict[str, Any],
        enemy_signal_ref: Dict[str, Any],
        enemy_recent: bool,
        ability_state_ref: Dict[str, Any],
        now_mono: float,
    ) -> Optional[Any]:
        if (
            BGObservation is None
            or BGSelfState is None
            or BGZoneState is None
            or BGEntity is None
            or BGItem is None
        ):
            return None
        try:
            px_raw = bot_event_signals_ref.get("player_pos_x")
            py_raw = bot_event_signals_ref.get("player_pos_y")
            sx_raw = bot_event_signals_ref.get("safe_zone_x")
            sy_raw = bot_event_signals_ref.get("safe_zone_y")
            sr_raw = bot_event_signals_ref.get("safe_zone_radius")

            if px_raw is None:
                px_raw = sx_raw if sx_raw is not None else 0.0
            if py_raw is None:
                py_raw = sy_raw if sy_raw is not None else 0.0
            if sx_raw is None:
                sx_raw = px_raw
            if sy_raw is None:
                sy_raw = py_raw

            px = float(px_raw)
            py = float(py_raw)
            sx = float(sx_raw)
            sy = float(sy_raw)

            zone_radius = 8.0
            if sr_raw is not None:
                try:
                    zone_radius = max(0.4, float(sr_raw))
                except Exception:
                    zone_radius = 8.0

            cooldown_ref = bot_event_signals_ref.get("ability_cooldown_sec", {})
            cooldowns_norm: Dict[str, float] = {"fire": 0.0}
            if isinstance(cooldown_ref, dict):
                for raw_key, raw_val in cooldown_ref.items():
                    try:
                        cd_val = max(0.0, float(raw_val))
                    except Exception:
                        continue
                    key_norm = str(raw_key or "").strip().lower()
                    if key_norm in ("digit1", "ability1", "slot1", "skill1"):
                        cooldowns_norm["digit1"] = cd_val
                    elif key_norm in ("digit2", "ability2", "slot2", "skill2", "shift", "dash"):
                        cooldowns_norm["digit2"] = cd_val
                    elif key_norm in ("digit3", "ability3", "slot3", "skill3"):
                        cooldowns_norm["digit3"] = cd_val

            dash_remaining = max(0.0, float(ability_state_ref.get("next_sprint_at", 0.0) or 0.0) - now_mono)
            cooldowns_norm["digit2"] = max(float(cooldowns_norm.get("digit2", 0.0) or 0.0), dash_remaining)

            mana_now = max(0.0, float(ability_state_ref.get("mana", 0.0) or 0.0))
            max_mana_now = max(1.0, float(ability_state_ref.get("max_mana", 100.0) or 100.0))
            ammo_proxy = int(max(0.0, min(15.0, (mana_now / max_mana_now) * 12.0)))
            if enemy_recent:
                ammo_proxy = max(ammo_proxy, 3)

            self_state = BGSelfState(
                position=[px, py],
                velocity=[0.0, 0.0],
                hp=max(1.0, 100.0 - float(bot_event_signals_ref.get("damage_taken_total", 0.0) or 0.0)),
                cooldowns=cooldowns_norm,
                ammo=ammo_proxy,
            )

            zone_state = BGZoneState(
                position=[sx, sy],
                radius=zone_radius,
                is_safe=not bool(bot_event_signals_ref.get("zone_outside_safe", False)),
            )

            visible_entities: List[Any] = []
            if enemy_recent:
                dir_name = str(enemy_signal_ref.get("dir", "CENTER") or "CENTER").upper()
                x_ratio = float(enemy_signal_ref.get("x_ratio", 0.50) or 0.50)
                y_ratio = float(enemy_signal_ref.get("y_ratio", 0.50) or 0.50)
                rel_x = (x_ratio - 0.50) * 12.0
                rel_y = (y_ratio - 0.50) * 12.0
                if abs(rel_x) < 0.3 and abs(rel_y) < 0.3:
                    if dir_name == "RIGHT":
                        rel_x = 6.0
                    elif dir_name == "LEFT":
                        rel_x = -6.0
                    elif dir_name == "DOWN":
                        rel_y = 6.0
                    elif dir_name == "UP":
                        rel_y = -6.0
                    else:
                        rel_x = 4.0
                        rel_y = -1.5
                visible_entities.append(
                    BGEntity(
                        id="enemy_live",
                        position=[px + rel_x, py + rel_y],
                        type="enemy",
                        hp=100.0,
                    )
                )

            visible_items: List[Any] = []
            loot_name = str(bot_event_signals_ref.get("loot_last_name", "") or "").strip()
            loot_type = str(bot_event_signals_ref.get("loot_last_type", "") or "").strip()
            loot_ts = float(bot_event_signals_ref.get("loot_ts", 0.0) or 0.0)
            if loot_type and loot_ts > 0.0 and (now_mono - loot_ts) <= 8.0:
                visible_items.append(
                    BGItem(
                        id=(loot_name[:32] if loot_name else "loot_live"),
                        position=[px + 1.8, py + 0.6],
                        type=loot_type,
                    )
                )

            return BGObservation(
                self_state=self_state,
                zone_state=zone_state,
                tick_id=int(tick_id),
                visible_entities=visible_entities,
                visible_items=visible_items,
            )
        except Exception:
            return None

    def build_stuck_recovery_plan(previous_combo: List[str], attempt_index: int) -> List[List[str]]:
        combo = normalize_move_combo(previous_combo)
        primary = combo[0] if combo else "KeyW"
        reverse_key = OPPOSITE_KEY.get(primary, "KeyS")
        strafe_key = ORTHOGONAL_STRAFE.get(primary, "KeyD")
        reverse_strafe = ORTHOGONAL_STRAFE.get(reverse_key, "KeyA")
        phase = int(attempt_index) % 4
        if phase == 0:
            return [
                normalize_move_combo([strafe_key]),
                normalize_move_combo([reverse_key, strafe_key]),
                normalize_move_combo([reverse_key]),
                normalize_move_combo([reverse_key, reverse_strafe]),
            ]
        if phase == 1:
            return [
                normalize_move_combo([reverse_key]),
                normalize_move_combo([reverse_key, reverse_strafe]),
                normalize_move_combo([reverse_strafe]),
                normalize_move_combo([primary, reverse_strafe]),
            ]
        if phase == 2:
            return [
                normalize_move_combo([reverse_strafe]),
                normalize_move_combo([primary, reverse_strafe]),
                normalize_move_combo([strafe_key]),
                normalize_move_combo([primary, strafe_key]),
            ]
        return [
            normalize_move_combo([strafe_key]),
            normalize_move_combo([primary, strafe_key]),
            normalize_move_combo([reverse_key, strafe_key]),
            normalize_move_combo([primary]),
        ]

    def capture_canvas_motion_frame(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
    ) -> Optional[Any]:
        if cv2 is None or np is None:
            return None
        try:
            canvas_rect = frame_obj.evaluate(
                """
                () => {
                  const c = document.querySelector('canvas');
                  if (!c) return null;
                  const r = c.getBoundingClientRect();
                  if (!r || r.width < 10 || r.height < 10) return null;
                  return {x: r.left, y: r.top, width: r.width, height: r.height};
                }
                """
            )
            if not isinstance(canvas_rect, dict):
                return None
            if float(canvas_rect.get("width", 0.0)) < 10.0 or float(canvas_rect.get("height", 0.0)) < 10.0:
                return None

            iframe_box = None
            try:
                iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
            except Exception:
                iframe_box = None

            if iframe_box:
                clip_x = float(iframe_box["x"]) + float(canvas_rect["x"])
                clip_y = float(iframe_box["y"]) + float(canvas_rect["y"])
            else:
                clip_x = float(canvas_rect["x"])
                clip_y = float(canvas_rect["y"])
            clip_w = float(canvas_rect["width"])
            clip_h = float(canvas_rect["height"])
            if clip_w <= 2.0 or clip_h <= 2.0:
                return None

            raw_png = page_obj.screenshot(
                clip={"x": clip_x, "y": clip_y, "width": clip_w, "height": clip_h},
                type="png",
            )
            image = cv2.imdecode(np.frombuffer(raw_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
            h, w = image.shape[:2]
            if h < 10 or w < 10:
                return None
            # avoid HUD/edges and focus on play area
            y0 = int(h * 0.18)
            y1 = int(h * 0.82)
            x0 = int(w * 0.12)
            x1 = int(w * 0.88)
            crop = image[y0:y1, x0:x1]
            if crop.size <= 0:
                crop = image
            resized = cv2.resize(crop, (96, 54), interpolation=cv2.INTER_AREA)
            return resized
        except Exception:
            return None

    def compute_motion_score(prev_frame: Optional[Any], next_frame: Optional[Any]) -> Optional[float]:
        if cv2 is None or np is None:
            return None
        if prev_frame is None or next_frame is None:
            return None
        try:
            if prev_frame.shape != next_frame.shape:
                return None
            diff = cv2.absdiff(prev_frame, next_frame)
            score = float(np.mean(diff))
            if math.isnan(score) or math.isinf(score):
                return None
            return score
        except Exception:
            return None

    def detect_enemy_signal_from_canvas(
        page_obj: Page,
        frame_obj: Frame,
        iframe_selector: str,
        red_ratio_threshold: float,
        min_contour_area: float,
    ) -> Dict[str, Any]:
        if cv2 is None or np is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "near": False,
                "x_ratio": 0.5,
                "y_ratio": 0.5,
                "direction_key": "KeyW",
                "dir": "CENTER",
                "red_ratio": 0.0,
                "area_ratio": 0.0,
            }
        try:
            canvas_rect = frame_obj.evaluate(
                """
                () => {
                  const c = document.querySelector('canvas');
                  if (!c) return null;
                  const r = c.getBoundingClientRect();
                  if (!r || r.width < 20 || r.height < 20) return null;
                  return {x: r.left, y: r.top, width: r.width, height: r.height};
                }
                """
            )
            if not isinstance(canvas_rect, dict):
                raise ValueError("no_canvas")

            iframe_box = None
            try:
                iframe_box = page_obj.locator(iframe_selector).first.bounding_box()
            except Exception:
                iframe_box = None

            if iframe_box:
                clip_x = float(iframe_box["x"]) + float(canvas_rect["x"])
                clip_y = float(iframe_box["y"]) + float(canvas_rect["y"])
            else:
                clip_x = float(canvas_rect["x"])
                clip_y = float(canvas_rect["y"])
            clip_w = float(canvas_rect["width"])
            clip_h = float(canvas_rect["height"])
            raw_png = page_obj.screenshot(
                clip={"x": clip_x, "y": clip_y, "width": clip_w, "height": clip_h},
                type="png",
            )
            image = cv2.imdecode(np.frombuffer(raw_png, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("decode_fail")
            h, w = image.shape[:2]
            if h < 20 or w < 20:
                raise ValueError("small_canvas")

            rx0 = int(w * 0.10)
            rx1 = int(w * 0.90)
            ry0 = int(h * 0.15)
            ry1 = int(h * 0.88)
            roi = image[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                raise ValueError("roi_empty")

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, (0, 95, 75), (12, 255, 255))
            mask2 = cv2.inRange(hsv, (160, 95, 75), (179, 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            red_ratio = float(mask.mean() / 255.0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            best_area = 0.0
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < max(8.0, float(min_contour_area)):
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                aspect = float(bw) / float(max(1, bh))
                if aspect < 0.25 or aspect > 4.0:
                    continue
                if area > best_area:
                    best_area = area
                    best = (bx, by, bw, bh)

            if best is None or red_ratio < max(0.001, float(red_ratio_threshold)):
                return {
                    "detected": False,
                    "confidence": min(0.4, red_ratio * 25.0),
                    "near": False,
                    "x_ratio": 0.5,
                    "y_ratio": 0.5,
                    "direction_key": "KeyW",
                    "dir": "CENTER",
                    "red_ratio": red_ratio,
                    "area_ratio": 0.0,
                }

            bx, by, bw, bh = best
            cx = float(rx0 + bx + (bw / 2.0))
            cy = float(ry0 + by + (bh / 2.0))
            x_ratio = max(0.0, min(1.0, cx / float(w)))
            y_ratio = max(0.0, min(1.0, cy / float(h)))
            dx = x_ratio - 0.5
            dy = y_ratio - 0.5
            area_ratio = best_area / float(max(1.0, roi.shape[0] * roi.shape[1]))

            if abs(dx) >= abs(dy):
                direction_key = "KeyD" if dx >= 0 else "KeyA"
            else:
                direction_key = "KeyS" if dy >= 0 else "KeyW"

            dir_h = "RIGHT" if dx > 0.08 else ("LEFT" if dx < -0.08 else "CENTER")
            dir_v = "DOWN" if dy > 0.08 else ("UP" if dy < -0.08 else "CENTER")
            direction_label = (
                f"{dir_v}_{dir_h}"
                if dir_h != "CENTER" and dir_v != "CENTER"
                else (dir_h if dir_h != "CENTER" else dir_v)
            )
            if direction_label == "CENTER":
                direction_label = "CENTER"

            near = bool(area_ratio >= 0.014 or (abs(dx) < 0.18 and abs(dy) < 0.18))
            confidence = min(1.0, (red_ratio * 35.0) + (area_ratio * 220.0))

            return {
                "detected": True,
                "confidence": confidence,
                "near": near,
                "x_ratio": x_ratio,
                "y_ratio": y_ratio,
                "direction_key": direction_key,
                "dir": direction_label,
                "red_ratio": red_ratio,
                "area_ratio": area_ratio,
            }
        except Exception:
            return {
                "detected": False,
                "confidence": 0.0,
                "near": False,
                "x_ratio": 0.5,
                "y_ratio": 0.5,
                "direction_key": "KeyW",
                "dir": "CENTER",
                "red_ratio": 0.0,
                "area_ratio": 0.0,
            }

    def dispatch_dom_left_click(frame_obj: Frame, frame_x: float, frame_y: float) -> bool:
        script = """
        ({x, y}) => {
          const target = document.elementFromPoint(x, y) || document.querySelector('canvas') || document.body;
          if (!target) return false;
          const mk = (type, buttons) => new MouseEvent(type, {
            bubbles: true,
            cancelable: true,
            button: 0,
            buttons: buttons,
            clientX: x,
            clientY: y,
            view: window
          });
          target.dispatchEvent(mk('mousemove', 0));
          target.dispatchEvent(mk('mousedown', 1));
          target.dispatchEvent(mk('mouseup', 0));
          target.dispatchEvent(mk('click', 0));
          return true;
        }
        """
        try:
            result = frame_obj.evaluate(script, {"x": frame_x, "y": frame_y})
            return bool(result)
        except Exception:
            return False

    def perform_attack_click(
        page_obj: Page,
        frame_obj: Frame,
        target: Dict[str, float],
        click_mode: str,
        mouse_move_steps: int,
        visual_cursor: bool,
        allow_unverified_mouse: bool = False,
    ) -> Tuple[bool, str]:
        install_click_probe(frame_obj)
        before = read_click_probe(frame_obj)

        def probe_changed(prev: Dict[str, Any]) -> bool:
            now = read_click_probe(frame_obj)
            return int(now.get("click0", 0)) > int(prev.get("click0", 0))

        if visual_cursor:
            ensure_bot_cursor_overlay(frame_obj)
            move_bot_cursor_overlay(
                frame_obj,
                frame_x=float(target["frame_x"]),
                frame_y=float(target["frame_y"]),
                source=str(target.get("source", "attack_click")),
            )

        try:
            page_obj.mouse.move(
                float(target["page_x"]),
                float(target["page_y"]),
                steps=max(1, int(mouse_move_steps)),
            )
        except Exception:
            pass

        if click_mode in ("mouse", "hybrid"):
            try:
                page_obj.mouse.click(target["page_x"], target["page_y"], button="left")
                if probe_changed(before):
                    return True, "mouse"
                if allow_unverified_mouse:
                    return True, "mouse_unverified"
            except Exception:
                pass
            if click_mode == "mouse":
                return False, "mouse"

        ok_dom = dispatch_dom_left_click(
            frame_obj,
            frame_x=target["frame_x"],
            frame_y=target["frame_y"],
        )
        if ok_dom and probe_changed(before):
            return True, "dom"
        return False, "dom"

    def run_bot_smoke_test(page_obj: Page) -> bool:
        smoke_html = """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>Bot Smoke Test</title>
          <style>
            html, body { margin: 0; height: 100%; overflow: hidden; font-family: Arial, sans-serif; }
            body {
              background: radial-gradient(circle at 20% 20%, #0f172a, #0b1020 55%, #030712);
              color: #e2e8f0;
            }
            #panel {
              position: fixed;
              inset: 20px auto auto 20px;
              width: 380px;
              background: rgba(15, 23, 42, 0.82);
              border: 1px solid rgba(148, 163, 184, 0.35);
              border-radius: 12px;
              padding: 14px;
              backdrop-filter: blur(4px);
            }
            #play {
              margin-top: 10px;
              width: 100%;
              border: 0;
              border-radius: 10px;
              padding: 12px;
              font-size: 16px;
              font-weight: 700;
              cursor: pointer;
              color: #0b1020;
              background: linear-gradient(180deg, #facc15, #f59e0b);
            }
            #hpWrap {
              margin-top: 10px;
              border-radius: 8px;
              border: 1px solid rgba(239, 68, 68, 0.45);
              background: rgba(15, 23, 42, 0.6);
              overflow: hidden;
            }
            #hpBar {
              height: 12px;
              width: 100%;
              background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
              transform-origin: left center;
            }
            #hpText {
              margin-top: 6px;
              font-size: 12px;
              color: #fca5a5;
            }
            #manaWrap {
              margin-top: 8px;
              border-radius: 8px;
              border: 1px solid rgba(59, 130, 246, 0.45);
              background: rgba(15, 23, 42, 0.6);
              overflow: hidden;
            }
            #manaBar {
              height: 10px;
              width: 100%;
              background: linear-gradient(90deg, #60a5fa, #3b82f6, #1d4ed8);
              transform-origin: left center;
            }
            #manaText {
              margin-top: 5px;
              font-size: 12px;
              color: #93c5fd;
            }
            #zoneText {
              margin-top: 6px;
              font-size: 12px;
              color: #c4b5fd;
            }
            #mapPanel {
              margin-top: 8px;
              width: 220px;
              height: 220px;
              border: 1px solid rgba(129, 140, 248, 0.45);
              border-radius: 10px;
              background: rgba(2, 6, 23, 0.85);
              display: none;
            }
            #mapPanel.open {
              display: block;
            }
            #mapCanvas {
              width: 220px;
              height: 220px;
              border-radius: 10px;
              display: block;
            }
            #controlsText {
              margin-top: 8px;
              font-size: 11px;
              color: #cbd5e1;
              line-height: 1.25;
            }
            #target {
              position: fixed;
              right: 30px;
              bottom: 30px;
              width: 240px;
              height: 130px;
              border: 2px dashed rgba(45, 212, 191, 0.8);
              border-radius: 14px;
              display: grid;
              place-items: center;
              font-weight: 700;
            }
            #enemy {
              position: fixed;
              width: 18px;
              height: 18px;
              border-radius: 50%;
              border: 2px solid rgba(254, 202, 202, 0.95);
              background: rgba(239, 68, 68, 0.95);
              box-shadow: 0 0 12px rgba(239, 68, 68, 0.8);
              transform: translate(-50%, -50%);
              pointer-events: none;
              z-index: 9998;
            }
          </style>
        </head>
        <body>
          <div id="panel">
            <div>Smoke test: cursor + click</div>
            <button id="play">JUGAR</button>
            <div id="status">ready</div>
            <div id="hpWrap"><div id="hpBar"></div></div>
            <div id="hpText">hp=100 in=0.0 out=0.0</div>
            <div id="manaWrap"><div id="manaBar"></div></div>
            <div id="manaText">mana=100 abilities=0/0/0 wall=0 sprint=0</div>
            <div id="zoneText">map=SmokeArena counter=12.0 safe=(0.50,0.50,r=0.42)</div>
            <div id="mapPanel"><canvas id="mapCanvas" width="220" height="220"></canvas></div>
            <div id="controlsText">
              W/A/S/D move | Space attack | Shift or RMB sprint | R wall | C map | 1/2/3 abilities
            </div>
          </div>
          <div id="target">DROP ZONE</div>
          <div id="enemy"></div>
          <script>
            window.__botSmoke = {
              buttonClicks: 0,
              pointerDown: 0,
              pointerUp: 0,
              pointerMoves: 0,
              lastX: 0,
              lastY: 0,
              hp: 100,
              maxHp: 100,
              mana: 100,
              maxMana: 100,
              damageReceived: 0,
              damageDealt: 0,
              deaths: 0,
              kills: 0,
              killHistory: [],
              lastKillAt: 0,
              lastKillBy: '',
              lastDeathCause: 'none',
              lastDeathSource: 'none',
              lastDeathAt: 0,
              deathHistory: [],
              enemyVisible: 0,
              enemyThreat: 0,
              enemyDir: 'CENTER',
              zoneCountdownSec: 12.0,
              zoneRadius: 0.42,
              safeZoneX: 0.50,
              safeZoneY: 0.50,
              mapName: 'SmokeArena',
              zoneToxic: 0,
              zoneOutside: 0,
              zoneSignalSource: 'none',
              mapOpen: 0,
              ability1Uses: 0,
              ability2Uses: 0,
              ability3Uses: 0,
              buildWallUses: 0,
              sprintUses: 0,
              lastAbility: '',
              lastAbilityClass: '',
              simState: 'lobby',
              abilityCatalog: {
                Digit1: 'offense',
                Digit2: 'mobility',
                Digit3: 'defense'
              }
            };
            const metrics = window.__botSmoke;
            const play = document.getElementById('play');
            const status = document.getElementById('status');
            const hpBar = document.getElementById('hpBar');
            const hpText = document.getElementById('hpText');
            const manaBar = document.getElementById('manaBar');
            const manaText = document.getElementById('manaText');
            const zoneText = document.getElementById('zoneText');
            const enemyEl = document.getElementById('enemy');
            const mapPanel = document.getElementById('mapPanel');
            const mapCanvas = document.getElementById('mapCanvas');
            const mapCtx = mapCanvas ? mapCanvas.getContext('2d') : null;
            const keysDown = Object.create(null);
            let mapToggleLock = false;
            let wallUntil = 0;
            let abilityLockUntil = 0;
            let attackLockUntil = 0;
            let sprintMouseUntil = 0;
            let enemyHp = 140;
            let playerX = 0.50;
            let playerY = 0.50;
            let enemyX = 0.66;
            let enemyY = 0.60;
            play.addEventListener('click', () => {
              metrics.buttonClicks += 1;
              metrics.simState = 'in_match';
              status.textContent = 'jugarClicks=' + metrics.buttonClicks;
            });
            window.addEventListener('pointerdown', (ev) => {
              metrics.pointerDown += 1;
              if (ev && Number(ev.button) === 2) {
                sprintMouseUntil = Date.now() + 350;
                metrics.sprintUses += 1;
              }
            }, true);
            window.addEventListener('pointerup', () => { metrics.pointerUp += 1; }, true);
            window.addEventListener('pointermove', (ev) => {
              metrics.pointerMoves += 1;
              metrics.lastX = Number(ev.clientX || 0);
              metrics.lastY = Number(ev.clientY || 0);
            }, true);
            window.addEventListener('keydown', (ev) => {
              const code = String(ev.code || '');
              if (!code) return;
              keysDown[code] = true;
              if (code === 'KeyC' && !mapToggleLock) {
                mapToggleLock = true;
                metrics.mapOpen = metrics.mapOpen ? 0 : 1;
              }
            }, true);
            window.addEventListener('keyup', (ev) => {
              const code = String(ev.code || '');
              if (!code) return;
              delete keysDown[code];
              if (code === 'KeyC') mapToggleLock = false;
            }, true);
            let t = 0;
            const dt = 0.17;
            const useMana = (amount) => {
              const cost = Math.max(0, Number(amount || 0));
              if (externalManaMode) {
                return Number(metrics.mana || 0) >= cost;
              }
              if (metrics.mana >= cost) {
                metrics.mana = Math.max(0, Number(metrics.mana || 0) - cost);
                return true;
              }
              return false;
            };
            const drawMap = () => {
              if (!mapCtx) return;
              const cw = mapCanvas.width;
              const ch = mapCanvas.height;
              mapCtx.clearRect(0, 0, cw, ch);
              mapCtx.fillStyle = '#ffffff';
              mapCtx.fillRect(0, 0, cw, ch);
              mapCtx.strokeStyle = 'rgba(17,17,17,0.25)';
              for (let i = 0; i <= 4; i += 1) {
                const x = (cw / 4) * i;
                const y = (ch / 4) * i;
                mapCtx.beginPath(); mapCtx.moveTo(x, 0); mapCtx.lineTo(x, ch); mapCtx.stroke();
                mapCtx.beginPath(); mapCtx.moveTo(0, y); mapCtx.lineTo(cw, y); mapCtx.stroke();
              }
              const zx = metrics.safeZoneX * cw;
              const zy = metrics.safeZoneY * ch;
              const zr = metrics.zoneRadius * (Math.min(cw, ch) * 0.5);
              mapCtx.beginPath();
              mapCtx.fillStyle = 'rgba(17,17,17,0.06)';
              mapCtx.strokeStyle = '#111111';
              mapCtx.lineWidth = 2;
              mapCtx.arc(zx, zy, zr, 0, Math.PI * 2);
              mapCtx.fill();
              mapCtx.stroke();

              mapCtx.beginPath();
              mapCtx.fillStyle = '#111111';
              mapCtx.arc(enemyX * cw, enemyY * ch, 6, 0, Math.PI * 2);
              mapCtx.fill();
              mapCtx.beginPath();
              mapCtx.fillStyle = '#666666';
              mapCtx.arc(playerX * cw, playerY * ch, 5, 0, Math.PI * 2);
              mapCtx.fill();
            };
            const drawRadar = () => {
              if (!radarCtx || !radarCanvas) return;
              const cw = radarCanvas.width;
              const ch = radarCanvas.height;
              const cx = cw * 0.5;
              const cy = ch * 0.5;
              const radius = Math.min(cw, ch) * 0.47;
              radarCtx.clearRect(0, 0, cw, ch);
              radarCtx.fillStyle = '#ffffff';
              radarCtx.fillRect(0, 0, cw, ch);

              radarCtx.strokeStyle = '#111111';
              radarCtx.lineWidth = 1;
              radarCtx.beginPath();
              radarCtx.arc(cx, cy, radius, 0, Math.PI * 2);
              radarCtx.stroke();
              radarCtx.beginPath();
              radarCtx.arc(cx, cy, radius * 0.68, 0, Math.PI * 2);
              radarCtx.stroke();
              radarCtx.beginPath();
              radarCtx.arc(cx, cy, radius * 0.36, 0, Math.PI * 2);
              radarCtx.stroke();
              radarCtx.beginPath();
              radarCtx.moveTo(cx - radius, cy);
              radarCtx.lineTo(cx + radius, cy);
              radarCtx.stroke();
              radarCtx.beginPath();
              radarCtx.moveTo(cx, cy - radius);
              radarCtx.lineTo(cx, cy + radius);
              radarCtx.stroke();

              radarSweep += 0.1;
              const sweepX = cx + Math.cos(radarSweep) * radius;
              const sweepY = cy + Math.sin(radarSweep) * radius;
              radarCtx.strokeStyle = 'rgba(17,17,17,0.35)';
              radarCtx.beginPath();
              radarCtx.moveTo(cx, cy);
              radarCtx.lineTo(sweepX, sweepY);
              radarCtx.stroke();

              const relEnemyX = clamp((enemyX - playerX) * 2.0, -1, 1);
              const relEnemyY = clamp((enemyY - playerY) * 2.0, -1, 1);
              const enemyDotX = cx + (relEnemyX * radius * 0.88);
              const enemyDotY = cy + (relEnemyY * radius * 0.88);
              const enemyAlpha = clamp((Number(metrics.enemyThreat || 0.0) * 0.85) + 0.15, 0.15, 1.0);
              if (Number(metrics.enemyVisible || 0) > 0) {
                radarCtx.fillStyle = `rgba(17,17,17,${enemyAlpha.toFixed(3)})`;
                radarCtx.beginPath();
                radarCtx.arc(enemyDotX, enemyDotY, 4.6, 0, Math.PI * 2);
                radarCtx.fill();
              }

              const safeRelX = clamp((Number(metrics.safeZoneX || 0.5) - playerX) * 2.0, -1, 1);
              const safeRelY = clamp((Number(metrics.safeZoneY || 0.5) - playerY) * 2.0, -1, 1);
              const safeDotX = cx + (safeRelX * radius * 0.82);
              const safeDotY = cy + (safeRelY * radius * 0.82);
              radarCtx.strokeStyle = '#666666';
              radarCtx.beginPath();
              radarCtx.arc(safeDotX, safeDotY, 4.2, 0, Math.PI * 2);
              radarCtx.stroke();

              radarCtx.fillStyle = '#111111';
              radarCtx.beginPath();
              radarCtx.arc(cx, cy, 4.8, 0, Math.PI * 2);
              radarCtx.fill();

              if (radarLegend) {
                const safeDirX = safeRelX >= 0.18 ? 'RIGHT' : (safeRelX <= -0.18 ? 'LEFT' : '');
                const safeDirY = safeRelY >= 0.18 ? 'DOWN' : (safeRelY <= -0.18 ? 'UP' : '');
                const safeDir = [safeDirY, safeDirX].filter(Boolean).join('_') || 'CENTER';
                radarLegend.textContent =
                  'radar enemy=' + Number(metrics.enemyThreat || 0).toFixed(2) +
                  ' dir=' + String(metrics.enemyDir || 'CENTER') +
                  ' safe_dir=' + safeDir;
              }
            };
            setInterval(() => {
              t += 0.14;
              const bridge = window.__botSmokeParallelBridge || null;
              const bridgeFresh = !!bridge && (Date.now() - Number(bridge.ts || 0)) <= 3200;
              if (bridgeFresh && Number.isFinite(Number(bridge.mana_current))) {
                const bridgeMaxRaw = Number(bridge.mana_max || 30);
                const bridgeMax = Number.isFinite(bridgeMaxRaw) && bridgeMaxRaw > 0 ? bridgeMaxRaw : 30;
                metrics.maxMana = clamp(bridgeMax, 1, 30);
                metrics.mana = clamp(Number(bridge.mana_current || 0), 0, metrics.maxMana);
                externalManaMode = true;
              } else {
                metrics.maxMana = 30;
                externalManaMode = false;
              }
              externalZoneMode = false;
              externalEnemyMode = false;
              if (bridgeFresh) {
                const zc = Number(bridge.zone_countdown_sec);
                if (Number.isFinite(zc) && zc >= 0) {
                  metrics.zoneCountdownSec = zc;
                  externalZoneMode = true;
                }
                const szx = Number(bridge.safe_zone_x);
                const szy = Number(bridge.safe_zone_y);
                const szr = Number(bridge.safe_zone_radius);
                if (Number.isFinite(szx) && Number.isFinite(szy)) {
                  metrics.safeZoneX = clamp(szx, 0.0, 1.0);
                  metrics.safeZoneY = clamp(szy, 0.0, 1.0);
                }
                if (Number.isFinite(szr) && szr > 0) {
                  metrics.zoneRadius = clamp(szr, 0.08, 0.95);
                }
                const mapName = String(bridge.map_name || '').trim();
                if (mapName) metrics.mapName = mapName.slice(0, 80);
                const enemyConf = Number(bridge.enemy_conf);
                const enemySeen = Number(bridge.enemy_detected || 0);
                if (Number.isFinite(enemyConf)) {
                  metrics.enemyThreat = clamp(enemyConf, 0.0, 1.0);
                  metrics.enemyVisible = enemySeen > 0 || enemyConf >= 0.08 ? 1 : 0;
                  externalEnemyMode = true;
                } else if (Number.isFinite(enemySeen)) {
                  metrics.enemyVisible = enemySeen > 0 ? 1 : 0;
                  externalEnemyMode = true;
                }
                const enemyDir = String(bridge.enemy_dir || '').trim().toUpperCase();
                if (enemyDir) metrics.enemyDir = enemyDir;
                const exr = Number(bridge.enemy_x_ratio);
                const eyr = Number(bridge.enemy_y_ratio);
                if (Number.isFinite(exr) && Number.isFinite(eyr)) {
                  enemyX = clamp(exr, 0.0, 1.0);
                  enemyY = clamp(eyr, 0.0, 1.0);
                  metrics.enemyXRatio = enemyX;
                  metrics.enemyYRatio = enemyY;
                  externalEnemyMode = true;
                }
                const bridgeState = String(bridge.bot_state || '').trim().toLowerCase();
                if (bridgeState === 'in_match' || bridgeState === 'lobby' || bridgeState === 'loading' || bridgeState === 'death') {
                  metrics.simState = bridgeState;
                }
              } else {
                metrics.zoneSignalSource = 'sim';
                metrics.zoneToxic = 0;
              }
              metrics.zoneSource = externalZoneMode ? 'live' : 'sim';
              metrics.enemySource = externalEnemyMode ? 'live' : 'sim';
              const w = window.innerWidth || 1280;
              const h = window.innerHeight || 720;
              if (!externalEnemyMode) {
                const exSim = (w * 0.60) + (Math.cos(t * 1.7) * w * 0.18);
                const eySim = (h * 0.60) + (Math.sin(t * 1.2) * h * 0.16);
                enemyX = Math.max(0.05, Math.min(0.95, exSim / w));
                enemyY = Math.max(0.05, Math.min(0.95, eySim / h));
              }
              metrics.enemyXRatio = enemyX;
              metrics.enemyYRatio = enemyY;
              const ex = enemyX * w;
              const ey = enemyY * h;
              enemyEl.style.left = ex + 'px';
              enemyEl.style.top = ey + 'px';

              let bx = w * 0.5;
              let by = h * 0.5;
              const cursor = document.getElementById('__lmsBotCursor');
              if (cursor) {
                const x = parseFloat(cursor.style.left || '');
                const y = parseFloat(cursor.style.top || '');
                if (!Number.isNaN(x)) bx = x;
                if (!Number.isNaN(y)) by = y;
              }
              playerX = Math.max(0.02, Math.min(0.98, bx / w));
              playerY = Math.max(0.02, Math.min(0.98, by / h));

              let moveX = 0;
              let moveY = 0;
              if (keysDown['KeyW']) moveY -= 1;
              if (keysDown['KeyS']) moveY += 1;
              if (keysDown['KeyA']) moveX -= 1;
              if (keysDown['KeyD']) moveX += 1;
              const sprintActive = !!keysDown['ShiftLeft'] || !!keysDown['ShiftRight'] || (Date.now() < sprintMouseUntil);
              if (sprintActive) {
                metrics.sprintUses += 0.12;
                if (!externalManaMode) {
                  metrics.mana = Math.max(0, Number(metrics.mana || 0) - (dt * 4.2));
                }
              }
              const speed = sprintActive ? 0.022 : 0.013;
              if (moveX !== 0 || moveY !== 0) {
                const mag = Math.sqrt((moveX * moveX) + (moveY * moveY)) || 1;
                playerX = Math.max(0.02, Math.min(0.98, playerX + ((moveX / mag) * speed)));
                playerY = Math.max(0.02, Math.min(0.98, playerY + ((moveY / mag) * speed)));
              }

              const nowTs = Date.now();
              if ((keysDown['Digit1'] || keysDown['Numpad1']) && nowTs >= abilityLockUntil && useMana(22)) {
                metrics.ability1Uses += 1;
                metrics.lastAbility = 'Digit1';
                metrics.lastAbilityClass = 'offense';
                abilityLockUntil = nowTs + 900;
                enemyHp = Math.max(0, enemyHp - 18);
                metrics.damageDealt += 18;
              }
              if ((keysDown['Digit2'] || keysDown['Numpad2']) && nowTs >= abilityLockUntil && useMana(18)) {
                metrics.ability2Uses += 1;
                metrics.lastAbility = 'Digit2';
                metrics.lastAbilityClass = 'mobility';
                abilityLockUntil = nowTs + 900;
                sprintMouseUntil = nowTs + 900;
              }
              if ((keysDown['Digit3'] || keysDown['Numpad3']) && nowTs >= abilityLockUntil && useMana(26)) {
                metrics.ability3Uses += 1;
                metrics.lastAbility = 'Digit3';
                metrics.lastAbilityClass = 'defense';
                abilityLockUntil = nowTs + 900;
                wallUntil = nowTs + 1300;
              }
              if (keysDown['KeyR'] && nowTs >= abilityLockUntil && useMana(14)) {
                metrics.buildWallUses += 1;
                metrics.lastAbility = 'KeyR';
                metrics.lastAbilityClass = 'utility_wall';
                abilityLockUntil = nowTs + 1000;
                wallUntil = nowTs + 1600;
              }
              const attackPressed = !!keysDown['Space'];
              if (attackPressed && nowTs >= attackLockUntil) {
                attackLockUntil = nowTs + 280;
                const adx = enemyX - playerX;
                const ady = enemyY - playerY;
                const ad = Math.sqrt((adx * adx) + (ady * ady));
                if (ad <= 0.18) {
                  const dmg = 9 + (Math.random() * 6);
                  enemyHp = Math.max(0, enemyHp - dmg);
                  metrics.damageDealt += dmg;
                }
              }
              if (enemyHp <= 0) {
                metrics.kills = Number(metrics.kills || 0) + 1;
                metrics.lastKillAt = Date.now();
                metrics.lastKillBy = String(
                  metrics.lastAbilityClass ||
                  (attackPressed ? 'basic_attack' : 'unknown')
                );
                const killHistory = Array.isArray(metrics.killHistory) ? metrics.killHistory : [];
                killHistory.push({
                  ts: Number(metrics.lastKillAt || Date.now()),
                  by: String(metrics.lastKillBy || 'unknown'),
                  damage_out_total: Number(metrics.damageDealt || 0)
                });
                if (killHistory.length > 40) killHistory.splice(0, killHistory.length - 40);
                metrics.killHistory = killHistory;
                enemyHp = 140;
              }

              if (!externalZoneMode) {
                metrics.zoneCountdownSec = Number(metrics.zoneCountdownSec || 0) - dt;
                if (metrics.zoneCountdownSec <= 0) {
                  metrics.zoneCountdownSec = 12.0;
                  metrics.zoneRadius = Math.max(0.12, Number(metrics.zoneRadius || 0.42) * 0.90);
                  metrics.safeZoneX = Math.max(0.20, Math.min(0.80, 0.50 + (Math.sin(t * 0.7) * 0.20)));
                  metrics.safeZoneY = Math.max(0.20, Math.min(0.80, 0.50 + (Math.cos(t * 0.9) * 0.20)));
                }
              }

              const dx = ex - bx;
              const dy = ey - by;
              const dist = Math.sqrt((dx * dx) + (dy * dy));
              const threat = Math.max(0, 1 - (dist / 220));
              const moving = (moveX !== 0 || moveY !== 0);
              const enemyDamageTick = threat * (moving ? 1.0 : 2.6);
              const pzx = playerX - Number(metrics.safeZoneX || 0.5);
              const pzy = playerY - Number(metrics.safeZoneY || 0.5);
              const pzd = Math.sqrt((pzx * pzx) + (pzy * pzy));
              const outsideSafeNow = pzd > Number(metrics.zoneRadius || 0.42);
              metrics.zoneOutside = externalZoneMode
                ? (Number(metrics.zoneOutside || 0) > 0 ? 1 : 0)
                : (outsideSafeNow ? 1 : 0);
              const zoneDamageTick = pzd > Number(metrics.zoneRadius || 0.42)
                ? (0.9 + ((pzd - Number(metrics.zoneRadius || 0.42)) * 6.5))
                : 0.0;
              if (!externalZoneMode) {
                metrics.zoneToxic = zoneDamageTick > 0.45 ? 1 : 0;
              }
              const wallReduction = (Date.now() <= wallUntil) ? 0.55 : 1.0;
              const enemyContribution = enemyDamageTick * wallReduction;
              const zoneContribution = zoneDamageTick * wallReduction;
              const damageTick = (enemyDamageTick + zoneDamageTick) * wallReduction;
              if (damageTick > 0.05) {
                metrics.hp = Math.max(0, Number(metrics.hp || 0) - damageTick);
                metrics.damageReceived = Number(metrics.damageReceived || 0) + damageTick;
                if (!externalEnemyMode) {
                  metrics.enemyVisible = 1;
                  metrics.enemyThreat = threat;
                }
              } else {
                if (!externalEnemyMode) {
                  metrics.enemyVisible = 0;
                  metrics.enemyThreat = Math.max(0, threat * 0.5);
                }
              }
              if (metrics.hp <= 0) {
                let deathCause = 'unknown';
                if (zoneContribution > 0.05 && enemyContribution > 0.05) {
                  deathCause = 'mixed_zone_enemy';
                } else if (zoneContribution > 0.05) {
                  deathCause = Number(metrics.zoneToxic || 0) > 0 ? 'toxic_zone' : 'zone_outside_safe';
                } else if (enemyContribution > 0.05) {
                  deathCause = 'enemy';
                }
                metrics.lastDeathCause = deathCause;
                metrics.lastDeathSource = deathCause === 'unknown' ? 'sim_unknown' : 'sim_damage';
                metrics.lastDeathAt = Date.now();
                const deathHistory = Array.isArray(metrics.deathHistory) ? metrics.deathHistory : [];
                deathHistory.push({
                  ts: Number(metrics.lastDeathAt || Date.now()),
                  cause: String(metrics.lastDeathCause || 'unknown'),
                  source: String(metrics.lastDeathSource || 'sim_damage'),
                  zone: Number(zoneContribution || 0),
                  enemy: Number(enemyContribution || 0),
                  toxic: Number(metrics.zoneToxic || 0),
                  outside: Number(metrics.zoneOutside || 0)
                });
                if (deathHistory.length > 30) deathHistory.splice(0, deathHistory.length - 30);
                metrics.deathHistory = deathHistory;
                metrics.deaths = Number(metrics.deaths || 0) + 1;
                metrics.simState = 'death_respawn';
                metrics.hp = metrics.maxHp;
              } else if (metrics.buttonClicks > 0) {
                metrics.simState = 'in_match';
              }
              const absDx = Math.abs(dx);
              const absDy = Math.abs(dy);
              if (!externalEnemyMode) {
                if (absDx >= absDy) metrics.enemyDir = dx >= 0 ? 'RIGHT' : 'LEFT';
                else metrics.enemyDir = dy >= 0 ? 'DOWN' : 'UP';
              }

              const hpRatio = Math.max(0, Math.min(1, Number(metrics.hp || 0) / Number(metrics.maxHp || 100)));
              hpBar.style.transform = `scaleX(${hpRatio})`;
              const manaRatio = Math.max(0, Math.min(1, Number(metrics.mana || 0) / Number(metrics.maxMana || 30)));
              manaBar.style.transform = `scaleX(${manaRatio})`;
              if (!externalManaMode) {
                metrics.mana = Math.min(metrics.maxMana, Number(metrics.mana || 0) + (dt * 2.8));
              }
              hpText.textContent =
                'hp=' + Number(metrics.hp || 0).toFixed(1) +
                ' in=' + Number(metrics.damageReceived || 0).toFixed(1) +
                ' out=' + Number(metrics.damageDealt || 0).toFixed(1) +
                ' kills=' + Number(metrics.kills || 0).toFixed(0) +
                ' threat=' + Number(metrics.enemyThreat || 0).toFixed(2) +
                ' dir=' + String(metrics.enemyDir || 'CENTER');
              manaText.textContent =
                'mana=' + Number(metrics.mana || 0).toFixed(1) +
                '/' + Number(metrics.maxMana || 30).toFixed(0) +
                ' (' + Number(manaRatio * 100).toFixed(0) + '%)' +
                ' src=' + (externalManaMode ? 'live' : 'sim') +
                ' abilities=' + Number(metrics.ability1Uses || 0).toFixed(0) + '/' +
                Number(metrics.ability2Uses || 0).toFixed(0) + '/' +
                Number(metrics.ability3Uses || 0).toFixed(0) +
                ' wall=' + Number(metrics.buildWallUses || 0).toFixed(0) +
                ' sprint=' + Number(metrics.sprintUses || 0).toFixed(0);
              zoneText.textContent =
                'map=' + String(metrics.mapName || 'SmokeArena') +
                ' counter=' + Number(metrics.zoneCountdownSec || 0).toFixed(1) +
                ' safe=(' + Number(metrics.safeZoneX || 0).toFixed(2) + ',' +
                Number(metrics.safeZoneY || 0).toFixed(2) + ',r=' +
                Number(metrics.zoneRadius || 0).toFixed(2) + ')';
              if (metrics.mapOpen) mapPanel.classList.add('open');
              else mapPanel.classList.remove('open');
              drawMap();
              status.textContent =
                'state=' + String(metrics.simState || 'lobby') +
                ' jugar=' + metrics.buttonClicks +
                ' hp=' + Number(metrics.hp || 0).toFixed(1) +
                ' kills=' + Number(metrics.kills || 0).toFixed(0) +
                ' death=' + String(metrics.lastDeathCause || 'none') +
                ' mana=' + Number(metrics.mana || 0).toFixed(1) + '/' + Number(metrics.maxMana || 30).toFixed(0) +
                ' counter=' + Number(metrics.zoneCountdownSec || 0).toFixed(1);
            }, 170);
          </script>
        </body>
        </html>
        """
        page_obj.set_content(smoke_html, wait_until="domcontentloaded")
        frame_obj = page_obj.main_frame
        install_click_probe(frame_obj)
        install_input_feedback_probe(frame_obj)
        if args.bot_debug_hud:
            ensure_bot_debug_hud(frame_obj)

        smoke_feedback_session = create_feedback_session(
            base_dir=args.bot_smoke_feedback_dir,
            mode="smoke",
        )
        if smoke_feedback_session:
            print(
                "[BOT][SMOKE] Feedback run dir: "
                f"{smoke_feedback_session.get('run_dir')}"
            )
            append_feedback_event(
                smoke_feedback_session,
                {
                    "event": "session_start",
                    "ts": time.time(),
                    "mode": "smoke",
                    "steps": int(args.bot_smoke_steps),
                    "step_ms": int(args.bot_smoke_step_ms),
                    "move_pattern": parse_move_pattern_csv(args.bot_smoke_move_pattern),
                },
            )

        cursor_ok = ensure_bot_cursor_overlay(
            frame_obj,
            transition_ms=args.bot_cursor_transition_ms,
        )
        if not cursor_ok:
            print("[BOT][SMOKE][WARN] No se pudo crear overlay de cursor.")
        else:
            print("[BOT][SMOKE] Overlay de cursor listo.")

        steps = max(4, int(args.bot_smoke_steps))
        step_ms = max(50, int(args.bot_smoke_step_ms))
        smoke_move_keys = parse_move_pattern_csv(args.bot_smoke_move_pattern)
        key_cursor = 0
        ratios = [
            (0.22, 0.26),
            (0.72, 0.26),
            (0.72, 0.74),
            (0.22, 0.74),
            (0.50, 0.52),
            (0.30, 0.50),
            (0.68, 0.50),
            (0.50, 0.30),
        ]
        play_button_target: Optional[Dict[str, float]] = None
        try:
            play_button_target = build_target_from_locator(
                page_obj=page_obj,
                frame_obj=frame_obj,
                iframe_selector=args.bot_iframe_selector,
                locator=frame_obj.locator("#play").first,
                source="smoke_button",
            )
        except Exception:
            play_button_target = None

        for idx in range(steps):
            x_ratio, y_ratio = ratios[idx % len(ratios)]
            key_to_press = smoke_move_keys[key_cursor % len(smoke_move_keys)]
            key_cursor += 1
            step_active_keys: List[str] = [key_to_press]
            target = get_canvas_target(
                page_obj=page_obj,
                frame_obj=frame_obj,
                iframe_selector=args.bot_iframe_selector,
                x_ratio=x_ratio,
                y_ratio=y_ratio,
            )
            if target is None:
                viewport = page_obj.viewport_size or {"width": 1280, "height": 720}
                target = {
                    "frame_x": float(viewport["width"]) * x_ratio,
                    "frame_y": float(viewport["height"]) * y_ratio,
                    "page_x": float(viewport["width"]) * x_ratio,
                    "page_y": float(viewport["height"]) * y_ratio,
                    "source": "smoke_viewport_fallback",
                }

            move_bot_cursor_overlay(
                frame_obj,
                frame_x=float(target["frame_x"]),
                frame_y=float(target["frame_y"]),
                source=f"smoke_move_{idx}",
            )
            try:
                try:
                    frame_obj.focus("body", timeout=800)
                except Exception:
                    pass
                if (idx % 3) == 0:
                    page_obj.keyboard.down("Shift")
                    step_active_keys.append("Shift")
                    try:
                        page_obj.mouse.down(button="right")
                        page_obj.mouse.up(button="right")
                    except Exception:
                        pass
                    step_active_keys.append("MouseRight")
                page_obj.keyboard.down(key_to_press)
                time.sleep(max(0.02, min(0.12, step_ms / 3000.0)))
                page_obj.keyboard.up(key_to_press)
                if "Shift" in step_active_keys:
                    page_obj.keyboard.up("Shift")

                # attack binding (Space)
                page_obj.keyboard.down("Space")
                time.sleep(0.02)
                page_obj.keyboard.up("Space")
                step_active_keys.append("Space")

                # ability bindings (1/2/3), wall (R), map (C)
                if (idx % 4) == 0:
                    ability_key = ["Digit1", "Digit2", "Digit3"][(idx // 4) % 3]
                    page_obj.keyboard.down(ability_key)
                    page_obj.keyboard.up(ability_key)
                    step_active_keys.append(ability_key)
                if (idx % 5) == 0:
                    page_obj.keyboard.down("KeyR")
                    page_obj.keyboard.up("KeyR")
                    step_active_keys.append("KeyR")
                if (idx % 6) == 0:
                    page_obj.keyboard.down("KeyC")
                    page_obj.keyboard.up("KeyC")
                    step_active_keys.append("KeyC")

                page_obj.mouse.move(
                    float(target["page_x"]),
                    float(target["page_y"]),
                    steps=max(1, int(args.bot_mouse_move_steps)),
                )
            except Exception as exc:
                print(f"[BOT][SMOKE][WARN] mouse.move fallo en step={idx}: {exc}")

            do_click = (idx % 2) == 0
            if do_click:
                click_target = target
                if play_button_target is not None and (idx % 4) == 0:
                    click_target = play_button_target
                    move_bot_cursor_overlay(
                        frame_obj,
                        frame_x=float(click_target["frame_x"]),
                        frame_y=float(click_target["frame_y"]),
                        source=f"smoke_button_focus_{idx}",
                    )
                    try:
                        page_obj.mouse.move(
                            float(click_target["page_x"]),
                            float(click_target["page_y"]),
                            steps=max(1, int(args.bot_mouse_move_steps)),
                        )
                    except Exception:
                        pass
                try:
                    page_obj.mouse.click(
                        float(click_target["page_x"]),
                        float(click_target["page_y"]),
                        button="left",
                    )
                except Exception as exc:
                    print(f"[BOT][SMOKE][WARN] mouse.click fallo en step={idx}: {exc}")
                dispatch_dom_left_click(
                    frame_obj,
                    frame_x=float(click_target["frame_x"]),
                    frame_y=float(click_target["frame_y"]),
                )

            cursor_probe = read_bot_cursor_probe(frame_obj)
            click_probe = read_click_probe(frame_obj)
            input_probe = read_input_feedback_probe(frame_obj)
            smoke_metrics = frame_obj.evaluate("() => window.__botSmoke || {}")
            if not isinstance(smoke_metrics, dict):
                smoke_metrics = {}
            if args.bot_debug_hud:
                update_bot_debug_hud(
                    frame_obj,
                    {
                        "state": f"smoke:{str(smoke_metrics.get('simState', 'lobby') or 'lobby')}",
                        "reason": "step_loop",
                        "action": f"step={idx + 1} key={key_to_press}",
                        "key_down": int(input_probe.get("keyDown", 0)),
                        "key_up": int(input_probe.get("keyUp", 0)),
                        "last_key": str(input_probe.get("lastKeyDown", "") or "-"),
                        "pointer_down": int(input_probe.get("pointerDown", 0)),
                        "pointer_up": int(input_probe.get("pointerUp", 0)),
                        "pointer_move": int(input_probe.get("pointerMove", 0)),
                        "cursor_moves": int(cursor_probe.get("moves", 0)),
                        "cursor_source": str(cursor_probe.get("lastSource", "") or "-"),
                        "click0": int(click_probe.get("click0", 0)),
                        "click_target": str(click_probe.get("lastTarget", "") or "-"),
                        "enemy_seen": int(smoke_metrics.get("enemyVisible", 0)),
                        "enemy_conf": float(smoke_metrics.get("enemyThreat", 0.0)),
                        "enemy_dir": str(smoke_metrics.get("enemyDir", "") or "-"),
                        "damage_done": float(smoke_metrics.get("damageDealt", 0.0)),
                        "damage_taken": float(smoke_metrics.get("damageReceived", 0.0)),
                        "kills": int(smoke_metrics.get("kills", 0)),
                        "last_death": str(smoke_metrics.get("lastDeathCause", "none") or "none"),
                        "mana": float(smoke_metrics.get("mana", 0.0)),
                        "zone_counter": f"{float(smoke_metrics.get('zoneCountdownSec', 0.0)):.1f}s",
                        "safe_zone": (
                            f"({float(smoke_metrics.get('safeZoneX', 0.0)):.2f},"
                            f"{float(smoke_metrics.get('safeZoneY', 0.0)):.2f},"
                            f"r={float(smoke_metrics.get('zoneRadius', 0.0)):.2f})"
                        ),
                        "ability_last": str(smoke_metrics.get("lastAbility", "") or "-"),
                        "ability_class": str(smoke_metrics.get("lastAbilityClass", "") or "-"),
                        "active_keys": step_active_keys,
                        "feed_line": (
                            f"s{idx + 1} st={str(smoke_metrics.get('simState', 'lobby') or 'lobby')} mv={key_to_press} "
                            f"in={float(smoke_metrics.get('damageReceived', 0.0)):.1f} "
                            f"out={float(smoke_metrics.get('damageDealt', 0.0)):.1f} "
                            f"mana={float(smoke_metrics.get('mana', 0.0)):.1f}"
                        ),
                    },
                )
            print(
                "[BOT][SMOKE] "
                f"step={idx + 1}/{steps} "
                f"moves={int(cursor_probe.get('moves', 0))} "
                f"click0={int(click_probe.get('click0', 0))} "
                f"buttonClicks={int(smoke_metrics.get('buttonClicks', 0))} "
                f"pointerDown={int(smoke_metrics.get('pointerDown', 0))} "
                f"pointerUp={int(smoke_metrics.get('pointerUp', 0))} "
                f"state={str(smoke_metrics.get('simState', 'lobby') or 'lobby')} "
                f"damageOut={float(smoke_metrics.get('damageDealt', 0.0)):.1f} "
                f"damage={float(smoke_metrics.get('damageReceived', 0.0)):.1f} "
                f"kills={int(smoke_metrics.get('kills', 0))} "
                f"lastDeath={str(smoke_metrics.get('lastDeathCause', 'none') or 'none')} "
                f"hp={float(smoke_metrics.get('hp', 0.0)):.1f} "
                f"mana={float(smoke_metrics.get('mana', 0.0)):.1f} "
                f"counter={float(smoke_metrics.get('zoneCountdownSec', 0.0)):.1f}"
            )
            if smoke_feedback_session is not None:
                append_feedback_event(
                    smoke_feedback_session,
                    {
                        "ts": time.time(),
                        "mode": "smoke",
                        "step": idx + 1,
                        "steps_total": steps,
                        "key": key_to_press,
                        "sim_state": str(smoke_metrics.get("simState", "lobby") or "lobby"),
                        "cursor_probe": cursor_probe,
                        "click_probe": click_probe,
                        "input_probe": input_probe,
                        "smoke_metrics": smoke_metrics,
                        "damage_dealt": float(smoke_metrics.get("damageDealt", 0.0)),
                        "damage_received": float(smoke_metrics.get("damageReceived", 0.0)),
                        "kills": int(smoke_metrics.get("kills", 0)),
                        "kill_history": list(smoke_metrics.get("killHistory", []) or []),
                        "deaths": int(smoke_metrics.get("deaths", 0)),
                        "last_death_cause": str(smoke_metrics.get("lastDeathCause", "none") or "none"),
                        "last_death_source": str(smoke_metrics.get("lastDeathSource", "none") or "none"),
                        "death_history": list(smoke_metrics.get("deathHistory", []) or []),
                        "hp": float(smoke_metrics.get("hp", 0.0)),
                        "mana": float(smoke_metrics.get("mana", 0.0)),
                        "zone_counter_sec": float(smoke_metrics.get("zoneCountdownSec", 0.0)),
                        "safe_zone": {
                            "x": float(smoke_metrics.get("safeZoneX", 0.0)),
                            "y": float(smoke_metrics.get("safeZoneY", 0.0)),
                            "radius": float(smoke_metrics.get("zoneRadius", 0.0)),
                        },
                        "map_name": str(smoke_metrics.get("mapName", "")),
                        "ability_last": str(smoke_metrics.get("lastAbility", "")),
                        "ability_class": str(smoke_metrics.get("lastAbilityClass", "")),
                        "ability_usage": {
                            "ability1": int(smoke_metrics.get("ability1Uses", 0)),
                            "ability2": int(smoke_metrics.get("ability2Uses", 0)),
                            "ability3": int(smoke_metrics.get("ability3Uses", 0)),
                            "build_wall": int(smoke_metrics.get("buildWallUses", 0)),
                            "sprint": float(smoke_metrics.get("sprintUses", 0.0)),
                        },
                    },
                )
                try:
                    step_name = f"step_{idx + 1:03d}_{key_to_press}.png"
                    step_path = Path(smoke_feedback_session["screen_dir"]) / step_name
                    page_obj.screenshot(path=str(step_path), full_page=True)
                except Exception as exc:
                    print(f"[BOT][SMOKE][WARN] screenshot step fallo: {exc}")
            time.sleep(step_ms / 1000.0)

        cursor_probe = read_bot_cursor_probe(frame_obj)
        click_probe = read_click_probe(frame_obj)
        input_probe = read_input_feedback_probe(frame_obj)
        smoke_metrics = frame_obj.evaluate("() => window.__botSmoke || {}")
        if not isinstance(smoke_metrics, dict):
            smoke_metrics = {}

        results = {
            "cursor_probe": cursor_probe,
            "click_probe": click_probe,
            "smoke_metrics": smoke_metrics,
            "sim_state": str(smoke_metrics.get("simState", "lobby") or "lobby"),
            "input_probe": input_probe,
            "steps": steps,
            "step_ms": step_ms,
            "kills": int(smoke_metrics.get("kills", 0)),
            "kill_history": list(smoke_metrics.get("killHistory", []) or []),
            "deaths": int(smoke_metrics.get("deaths", 0)),
            "last_death_cause": str(smoke_metrics.get("lastDeathCause", "none") or "none"),
            "last_death_source": str(smoke_metrics.get("lastDeathSource", "none") or "none"),
            "death_history": list(smoke_metrics.get("deathHistory", []) or []),
            "timestamp": int(time.time()),
            "feedback_run_dir": str(smoke_feedback_session.get("run_dir")) if smoke_feedback_session else "",
            "feedback_jsonl": str(smoke_feedback_session.get("jsonl_path")) if smoke_feedback_session else "",
            "feedback_screens_dir": str(smoke_feedback_session.get("screen_dir")) if smoke_feedback_session else "",
        }
        output_path = Path(args.bot_smoke_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")

        screenshot_path = Path(args.bot_smoke_screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            page_obj.screenshot(path=str(screenshot_path), full_page=True)
        except Exception as exc:
            print(f"[BOT][SMOKE][WARN] screenshot fallo: {exc}")

        cursor_moves = int(cursor_probe.get("moves", 0))
        click_count = int(click_probe.get("click0", 0))
        button_clicks = int(smoke_metrics.get("buttonClicks", 0))
        pointer_down = int(smoke_metrics.get("pointerDown", 0))
        ok = cursor_moves >= max(3, steps // 3) and (click_count > 0 or pointer_down > 0) and button_clicks > 0
        status = "PASS" if ok else "WARN"
        print(
            "[BOT][SMOKE] "
            f"{status} cursor_moves={cursor_moves} click0={click_count} "
            f"buttonClicks={button_clicks} pointerDown={pointer_down} "
            f"state={str(smoke_metrics.get('simState', 'lobby') or 'lobby')} "
            f"kills={int(smoke_metrics.get('kills', 0))} "
            f"lastDeath={str(smoke_metrics.get('lastDeathCause', 'none') or 'none')} "
            f"json={output_path} screenshot={screenshot_path}"
        )
        return ok

    def setup_parallel_smoke_monitor(context_obj) -> Optional[Dict[str, Any]]:
        smoke_html = """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>Parallel Smoke Monitor</title>
          <style>
            html, body { margin: 0; height: 100%; overflow: hidden; font-family: 'Segoe UI', Tahoma, sans-serif; }
            body {
              background: #ffffff;
              color: #111111;
            }
            #panel {
              position: fixed;
              inset: 20px auto auto 20px;
              width: 380px;
              background: #ffffff;
              border: 1px solid #111111;
              border-radius: 8px;
              padding: 14px;
              z-index: 9999;
              box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            }
            #title {
              font-size: 14px;
              font-weight: 700;
              margin-bottom: 8px;
            }
            #play {
              margin-top: 10px;
              width: 100%;
              border: 1px solid #111111;
              border-radius: 6px;
              padding: 12px;
              font-size: 14px;
              font-weight: 700;
              cursor: pointer;
              color: #111111;
              background: #ffffff;
            }
            #status {
              margin-top: 8px;
              font-size: 12px;
            }
            #hpWrap {
              margin-top: 10px;
              border-radius: 6px;
              border: 1px solid #111111;
              background: #f5f5f5;
              overflow: hidden;
            }
            #hpBar {
              height: 12px;
              width: 100%;
              background: #333333;
              transform-origin: left center;
            }
            #hpText {
              margin-top: 6px;
              font-size: 12px;
              color: #111111;
            }
            #manaWrap {
              margin-top: 8px;
              border-radius: 6px;
              border: 1px solid #111111;
              background: #f5f5f5;
              overflow: hidden;
            }
            #manaBar {
              height: 10px;
              width: 100%;
              background: #111111;
              transform-origin: left center;
            }
            #manaText {
              margin-top: 5px;
              font-size: 12px;
              color: #111111;
            }
            #zoneText {
              margin-top: 6px;
              font-size: 12px;
              color: #111111;
            }
            #zoneClock {
              margin-top: 6px;
              font-size: 13px;
              font-weight: 700;
              letter-spacing: 0.2px;
              color: #111111;
            }
            #decisionText {
              margin-top: 6px;
              font-size: 12px;
              border: 1px solid #111111;
              border-radius: 6px;
              padding: 6px 8px;
              background: #fafafa;
            }
            #radarPanel {
              margin-top: 8px;
              width: 220px;
              border: 1px solid #111111;
              border-radius: 6px;
              background: #ffffff;
              padding: 6px;
            }
            #radarCanvas {
              width: 208px;
              height: 208px;
              display: block;
              border: 1px solid #111111;
              border-radius: 50%;
              background: #ffffff;
            }
            #radarLegend {
              margin-top: 6px;
              font-size: 11px;
              color: #111111;
              line-height: 1.25;
            }
            #mapPanel {
              margin-top: 8px;
              width: 220px;
              height: 220px;
              border: 1px solid #111111;
              border-radius: 6px;
              background: #ffffff;
              display: none;
            }
            #mapPanel.open {
              display: block;
            }
            #mapCanvas {
              width: 220px;
              height: 220px;
              border-radius: 10px;
              display: block;
            }
            #controlsText {
              margin-top: 8px;
              font-size: 11px;
              color: #111111;
              line-height: 1.25;
            }
            #target {
              position: fixed;
              right: 30px;
              bottom: 30px;
              width: 240px;
              height: 130px;
              border: 2px dashed #111111;
              border-radius: 8px;
              display: grid;
              place-items: center;
              font-weight: 700;
              background: #ffffff;
              color: #111111;
            }
            #enemy {
              position: fixed;
              width: 18px;
              height: 18px;
              border-radius: 50%;
              border: 2px solid #111111;
              background: #111111;
              box-shadow: 0 0 0 2px #ffffff;
              transform: translate(-50%, -50%);
              pointer-events: none;
              z-index: 9998;
            }
          </style>
        </head>
        <body>
          <div id="panel">
            <div id="title">Parallel Smoke Monitor</div>
            <button id="play">JUGAR</button>
            <div id="status">ready</div>
            <div id="hpWrap"><div id="hpBar"></div></div>
            <div id="hpText">hp=100 in=0.0 out=0.0</div>
            <div id="manaWrap"><div id="manaBar"></div></div>
            <div id="manaText">mana=30.0/30 (100%) src=sim abilities=0/0/0 wall=0 sprint=0</div>
            <div id="zoneText">map=SmokeArena counter=12.0 safe=(0.50,0.50,r=0.42)</div>
            <div id="zoneClock">ZONE SHRINK IN: 12.0s</div>
            <div id="decisionText">decision=SCOUT confidence=0.00 zone_src=sim enemy_src=sim</div>
            <div id="radarPanel">
              <canvas id="radarCanvas" width="208" height="208"></canvas>
              <div id="radarLegend">radar enemy=0.00 dir=CENTER safe_dir=CENTER</div>
            </div>
            <div id="mapPanel"><canvas id="mapCanvas" width="220" height="220"></canvas></div>
            <div id="controlsText">
              W/A/S/D move | Space attack | Shift or RMB sprint | R wall | C map | 1/2/3 abilities
            </div>
          </div>
          <div id="target">DROP ZONE</div>
          <div id="enemy"></div>
          <script>
            window.__botSmokeParallel = {
              buttonClicks: 0,
              pointerDown: 0,
              pointerUp: 0,
              pointerMoves: 0,
              lastX: 0,
              lastY: 0,
              hp: 100,
              maxHp: 100,
              mana: 30,
              maxMana: 30,
              damageReceived: 0,
              damageDealt: 0,
              deaths: 0,
              kills: 0,
              killHistory: [],
              lastKillAt: 0,
              lastKillBy: '',
              lastDeathCause: 'none',
              lastDeathSource: 'none',
              lastDeathAt: 0,
              deathHistory: [],
              enemyVisible: 0,
              enemyThreat: 0,
              enemyDir: 'CENTER',
              enemyXRatio: 0.66,
              enemyYRatio: 0.60,
              zoneCountdownSec: 12.0,
              zoneRadius: 0.42,
              safeZoneX: 0.50,
              safeZoneY: 0.50,
              mapName: 'SmokeArena',
              zoneToxic: 0,
              zoneOutside: 0,
              zoneSignalSource: 'none',
              zoneSource: 'sim',
              enemySource: 'sim',
              decisionHint: 'SCOUT',
              mapOpen: 0,
              ability1Uses: 0,
              ability2Uses: 0,
              ability3Uses: 0,
              buildWallUses: 0,
              sprintUses: 0,
              lastAbility: '',
              lastAbilityClass: '',
              simState: 'lobby'
            };
            const metrics = window.__botSmokeParallel;
            const play = document.getElementById('play');
            const status = document.getElementById('status');
            const hpBar = document.getElementById('hpBar');
            const hpText = document.getElementById('hpText');
            const manaBar = document.getElementById('manaBar');
            const manaText = document.getElementById('manaText');
            const zoneText = document.getElementById('zoneText');
            const zoneClock = document.getElementById('zoneClock');
            const decisionText = document.getElementById('decisionText');
            const enemyEl = document.getElementById('enemy');
            const radarCanvas = document.getElementById('radarCanvas');
            const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;
            const radarLegend = document.getElementById('radarLegend');
            const mapPanel = document.getElementById('mapPanel');
            const mapCanvas = document.getElementById('mapCanvas');
            const mapCtx = mapCanvas ? mapCanvas.getContext('2d') : null;
            const keysDown = Object.create(null);
            let mapToggleLock = false;
            let wallUntil = 0;
            let abilityLockUntil = 0;
            let attackLockUntil = 0;
            let sprintMouseUntil = 0;
            let enemyHp = 140;
            let playerX = 0.50;
            let playerY = 0.50;
            let enemyX = 0.66;
            let enemyY = 0.60;
            let externalManaMode = false;
            let externalZoneMode = false;
            let externalEnemyMode = false;
            let radarSweep = 0;
            const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, Number(value || 0)));
            play.addEventListener('click', () => {
              metrics.buttonClicks += 1;
              metrics.simState = 'in_match';
              status.textContent = 'jugarClicks=' + metrics.buttonClicks;
            });
            window.addEventListener('pointerdown', (ev) => {
              metrics.pointerDown += 1;
              if (ev && Number(ev.button) === 2) {
                sprintMouseUntil = Date.now() + 350;
                metrics.sprintUses += 1;
              }
            }, true);
            window.addEventListener('pointerup', () => { metrics.pointerUp += 1; }, true);
            window.addEventListener('pointermove', (ev) => {
              metrics.pointerMoves += 1;
              metrics.lastX = Number(ev.clientX || 0);
              metrics.lastY = Number(ev.clientY || 0);
            }, true);
            window.addEventListener('keydown', (ev) => {
              const code = String(ev.code || '');
              if (!code) return;
              keysDown[code] = true;
              if (code === 'KeyC' && !mapToggleLock) {
                mapToggleLock = true;
                metrics.mapOpen = metrics.mapOpen ? 0 : 1;
              }
            }, true);
            window.addEventListener('keyup', (ev) => {
              const code = String(ev.code || '');
              if (!code) return;
              delete keysDown[code];
              if (code === 'KeyC') mapToggleLock = false;
            }, true);
            let t = 0;
            const dt = 0.17;
            const useMana = (amount) => {
              const cost = Math.max(0, Number(amount || 0));
              if (externalManaMode) {
                return Number(metrics.mana || 0) >= cost;
              }
              if (metrics.mana >= cost) {
                metrics.mana = Math.max(0, Number(metrics.mana || 0) - cost);
                return true;
              }
              return false;
            };
            const drawMap = () => {
              if (!mapCtx) return;
              const cw = mapCanvas.width;
              const ch = mapCanvas.height;
              mapCtx.clearRect(0, 0, cw, ch);
              mapCtx.fillStyle = '#ffffff';
              mapCtx.fillRect(0, 0, cw, ch);
              mapCtx.strokeStyle = 'rgba(17,17,17,0.25)';
              for (let i = 0; i <= 4; i += 1) {
                const x = (cw / 4) * i;
                const y = (ch / 4) * i;
                mapCtx.beginPath(); mapCtx.moveTo(x, 0); mapCtx.lineTo(x, ch); mapCtx.stroke();
                mapCtx.beginPath(); mapCtx.moveTo(0, y); mapCtx.lineTo(cw, y); mapCtx.stroke();
              }
              const zx = metrics.safeZoneX * cw;
              const zy = metrics.safeZoneY * ch;
              const zr = metrics.zoneRadius * (Math.min(cw, ch) * 0.5);
              mapCtx.beginPath();
              mapCtx.fillStyle = 'rgba(17,17,17,0.06)';
              mapCtx.strokeStyle = '#111111';
              mapCtx.lineWidth = 2;
              mapCtx.arc(zx, zy, zr, 0, Math.PI * 2);
              mapCtx.fill();
              mapCtx.stroke();

              mapCtx.beginPath();
              mapCtx.fillStyle = '#111111';
              mapCtx.arc(enemyX * cw, enemyY * ch, 6, 0, Math.PI * 2);
              mapCtx.fill();
              mapCtx.beginPath();
              mapCtx.fillStyle = '#666666';
              mapCtx.arc(playerX * cw, playerY * ch, 5, 0, Math.PI * 2);
              mapCtx.fill();
            };
            setInterval(() => {
              t += 0.14;
              const bridge = window.__botSmokeParallelBridge || null;
              const bridgeFresh = !!bridge && (Date.now() - Number(bridge.ts || 0)) <= 3200;
              if (bridgeFresh && Number.isFinite(Number(bridge.mana_current))) {
                const bridgeMaxRaw = Number(bridge.mana_max || 30);
                const bridgeMax = Number.isFinite(bridgeMaxRaw) && bridgeMaxRaw > 0 ? bridgeMaxRaw : 30;
                metrics.maxMana = clamp(bridgeMax, 1, 30);
                metrics.mana = clamp(Number(bridge.mana_current || 0), 0, metrics.maxMana);
                externalManaMode = true;
              } else {
                metrics.maxMana = 30;
                externalManaMode = false;
              }
              externalZoneMode = false;
              externalEnemyMode = false;
              if (bridgeFresh) {
                const zc = Number(bridge.zone_countdown_sec);
                if (Number.isFinite(zc) && zc >= 0) {
                  metrics.zoneCountdownSec = zc;
                  externalZoneMode = true;
                }
                const szx = Number(bridge.safe_zone_x);
                const szy = Number(bridge.safe_zone_y);
                const szr = Number(bridge.safe_zone_radius);
                if (Number.isFinite(szx) && Number.isFinite(szy)) {
                  metrics.safeZoneX = clamp(szx, 0.0, 1.0);
                  metrics.safeZoneY = clamp(szy, 0.0, 1.0);
                }
                if (Number.isFinite(szr) && szr > 0) {
                  metrics.zoneRadius = clamp(szr, 0.08, 0.95);
                }
                metrics.zoneToxic = Number(bridge.zone_toxic || 0) > 0 ? 1 : 0;
                metrics.zoneOutside = Number(bridge.zone_outside || 0) > 0 ? 1 : 0;
                metrics.zoneSignalSource = String(bridge.zone_signal_source || 'none');
                const mapName = String(bridge.map_name || '').trim();
                if (mapName) metrics.mapName = mapName.slice(0, 80);
                const enemyConf = Number(bridge.enemy_conf);
                const enemySeen = Number(bridge.enemy_detected || 0);
                if (Number.isFinite(enemyConf)) {
                  metrics.enemyThreat = clamp(enemyConf, 0.0, 1.0);
                  metrics.enemyVisible = enemySeen > 0 || enemyConf >= 0.08 ? 1 : 0;
                  externalEnemyMode = true;
                } else if (Number.isFinite(enemySeen)) {
                  metrics.enemyVisible = enemySeen > 0 ? 1 : 0;
                  externalEnemyMode = true;
                }
                const enemyDir = String(bridge.enemy_dir || '').trim().toUpperCase();
                if (enemyDir) metrics.enemyDir = enemyDir;
                const exr = Number(bridge.enemy_x_ratio);
                const eyr = Number(bridge.enemy_y_ratio);
                if (Number.isFinite(exr) && Number.isFinite(eyr)) {
                  enemyX = clamp(exr, 0.0, 1.0);
                  enemyY = clamp(eyr, 0.0, 1.0);
                  metrics.enemyXRatio = enemyX;
                  metrics.enemyYRatio = enemyY;
                  externalEnemyMode = true;
                }
                const bridgeState = String(bridge.bot_state || '').trim().toLowerCase();
                if (bridgeState === 'in_match' || bridgeState === 'lobby' || bridgeState === 'loading' || bridgeState === 'death') {
                  metrics.simState = bridgeState;
                }
              }
              metrics.zoneSource = externalZoneMode ? 'live' : 'sim';
              metrics.enemySource = externalEnemyMode ? 'live' : 'sim';
              const w = window.innerWidth || 1280;
              const h = window.innerHeight || 720;
              if (!externalEnemyMode) {
                const exSim = (w * 0.60) + (Math.cos(t * 1.7) * w * 0.18);
                const eySim = (h * 0.60) + (Math.sin(t * 1.2) * h * 0.16);
                enemyX = Math.max(0.05, Math.min(0.95, exSim / w));
                enemyY = Math.max(0.05, Math.min(0.95, eySim / h));
              }
              metrics.enemyXRatio = enemyX;
              metrics.enemyYRatio = enemyY;
              const ex = enemyX * w;
              const ey = enemyY * h;
              enemyEl.style.left = ex + 'px';
              enemyEl.style.top = ey + 'px';

              let bx = w * 0.5;
              let by = h * 0.5;
              const cursor = document.getElementById('__lmsBotCursor');
              if (cursor) {
                const x = parseFloat(cursor.style.left || '');
                const y = parseFloat(cursor.style.top || '');
                if (!Number.isNaN(x)) bx = x;
                if (!Number.isNaN(y)) by = y;
              }
              playerX = Math.max(0.02, Math.min(0.98, bx / w));
              playerY = Math.max(0.02, Math.min(0.98, by / h));

              let moveX = 0;
              let moveY = 0;
              if (keysDown['KeyW']) moveY -= 1;
              if (keysDown['KeyS']) moveY += 1;
              if (keysDown['KeyA']) moveX -= 1;
              if (keysDown['KeyD']) moveX += 1;
              const sprintActive = !!keysDown['ShiftLeft'] || !!keysDown['ShiftRight'] || (Date.now() < sprintMouseUntil);
              if (sprintActive) {
                metrics.sprintUses += 0.12;
                if (!externalManaMode) {
                  metrics.mana = Math.max(0, Number(metrics.mana || 0) - (dt * 4.2));
                }
              }
              const speed = sprintActive ? 0.022 : 0.013;
              if (moveX !== 0 || moveY !== 0) {
                const mag = Math.sqrt((moveX * moveX) + (moveY * moveY)) || 1;
                playerX = Math.max(0.02, Math.min(0.98, playerX + ((moveX / mag) * speed)));
                playerY = Math.max(0.02, Math.min(0.98, playerY + ((moveY / mag) * speed)));
              }

              const nowTs = Date.now();
              if ((keysDown['Digit1'] || keysDown['Numpad1']) && nowTs >= abilityLockUntil && useMana(22)) {
                metrics.ability1Uses += 1;
                metrics.lastAbility = 'Digit1';
                metrics.lastAbilityClass = 'offense';
                abilityLockUntil = nowTs + 900;
                enemyHp = Math.max(0, enemyHp - 18);
                metrics.damageDealt += 18;
              }
              if ((keysDown['Digit2'] || keysDown['Numpad2']) && nowTs >= abilityLockUntil && useMana(18)) {
                metrics.ability2Uses += 1;
                metrics.lastAbility = 'Digit2';
                metrics.lastAbilityClass = 'mobility';
                abilityLockUntil = nowTs + 900;
                sprintMouseUntil = nowTs + 900;
              }
              if ((keysDown['Digit3'] || keysDown['Numpad3']) && nowTs >= abilityLockUntil && useMana(26)) {
                metrics.ability3Uses += 1;
                metrics.lastAbility = 'Digit3';
                metrics.lastAbilityClass = 'defense';
                abilityLockUntil = nowTs + 900;
                wallUntil = nowTs + 1300;
              }
              if (keysDown['KeyR'] && nowTs >= abilityLockUntil && useMana(14)) {
                metrics.buildWallUses += 1;
                metrics.lastAbility = 'KeyR';
                metrics.lastAbilityClass = 'utility_wall';
                abilityLockUntil = nowTs + 1000;
                wallUntil = nowTs + 1600;
              }
              const attackPressed = !!keysDown['Space'];
              if (attackPressed && nowTs >= attackLockUntil) {
                attackLockUntil = nowTs + 280;
                const adx = enemyX - playerX;
                const ady = enemyY - playerY;
                const ad = Math.sqrt((adx * adx) + (ady * ady));
                if (ad <= 0.18) {
                  const dmg = 9 + (Math.random() * 6);
                  enemyHp = Math.max(0, enemyHp - dmg);
                  metrics.damageDealt += dmg;
                }
              }
              if (enemyHp <= 0) {
                metrics.kills = Number(metrics.kills || 0) + 1;
                metrics.lastKillAt = Date.now();
                metrics.lastKillBy = String(
                  metrics.lastAbilityClass ||
                  (attackPressed ? 'basic_attack' : 'unknown')
                );
                const killHistory = Array.isArray(metrics.killHistory) ? metrics.killHistory : [];
                killHistory.push({
                  ts: Number(metrics.lastKillAt || Date.now()),
                  by: String(metrics.lastKillBy || 'unknown'),
                  damage_out_total: Number(metrics.damageDealt || 0)
                });
                if (killHistory.length > 40) killHistory.splice(0, killHistory.length - 40);
                metrics.killHistory = killHistory;
                enemyHp = 140;
              }

              if (!externalZoneMode) {
                metrics.zoneCountdownSec = Number(metrics.zoneCountdownSec || 0) - dt;
                if (metrics.zoneCountdownSec <= 0) {
                  metrics.zoneCountdownSec = 12.0;
                  metrics.zoneRadius = Math.max(0.12, Number(metrics.zoneRadius || 0.42) * 0.90);
                  metrics.safeZoneX = Math.max(0.20, Math.min(0.80, 0.50 + (Math.sin(t * 0.7) * 0.20)));
                  metrics.safeZoneY = Math.max(0.20, Math.min(0.80, 0.50 + (Math.cos(t * 0.9) * 0.20)));
                }
              }

              const dx = ex - bx;
              const dy = ey - by;
              const dist = Math.sqrt((dx * dx) + (dy * dy));
              const threat = Math.max(0, 1 - (dist / 220));
              const moving = (moveX !== 0 || moveY !== 0);
              const enemyDamageTick = threat * (moving ? 1.0 : 2.6);
              const pzx = playerX - Number(metrics.safeZoneX || 0.5);
              const pzy = playerY - Number(metrics.safeZoneY || 0.5);
              const pzd = Math.sqrt((pzx * pzx) + (pzy * pzy));
              const outsideSafeNow = pzd > Number(metrics.zoneRadius || 0.42);
              metrics.zoneOutside = externalZoneMode
                ? (Number(metrics.zoneOutside || 0) > 0 ? 1 : 0)
                : (outsideSafeNow ? 1 : 0);
              const zoneDamageTick = pzd > Number(metrics.zoneRadius || 0.42)
                ? (0.9 + ((pzd - Number(metrics.zoneRadius || 0.42)) * 6.5))
                : 0.0;
              if (!externalZoneMode) {
                metrics.zoneToxic = zoneDamageTick > 0.45 ? 1 : 0;
              }
              const wallReduction = (Date.now() <= wallUntil) ? 0.55 : 1.0;
              const enemyContribution = enemyDamageTick * wallReduction;
              const zoneContribution = zoneDamageTick * wallReduction;
              const damageTick = (enemyDamageTick + zoneDamageTick) * wallReduction;
              if (damageTick > 0.05) {
                metrics.hp = Math.max(0, Number(metrics.hp || 0) - damageTick);
                metrics.damageReceived = Number(metrics.damageReceived || 0) + damageTick;
                if (!externalEnemyMode) {
                  metrics.enemyVisible = 1;
                  metrics.enemyThreat = threat;
                }
              } else {
                if (!externalEnemyMode) {
                  metrics.enemyVisible = 0;
                  metrics.enemyThreat = Math.max(0, threat * 0.5);
                }
              }
              if (metrics.hp <= 0) {
                let deathCause = 'unknown';
                if (zoneContribution > 0.05 && enemyContribution > 0.05) {
                  deathCause = 'mixed_zone_enemy';
                } else if (zoneContribution > 0.05) {
                  deathCause = Number(metrics.zoneToxic || 0) > 0 ? 'toxic_zone' : 'zone_outside_safe';
                } else if (enemyContribution > 0.05) {
                  deathCause = 'enemy';
                }
                metrics.lastDeathCause = deathCause;
                metrics.lastDeathSource = deathCause === 'unknown' ? 'sim_unknown' : 'sim_damage';
                metrics.lastDeathAt = Date.now();
                const deathHistory = Array.isArray(metrics.deathHistory) ? metrics.deathHistory : [];
                deathHistory.push({
                  ts: Number(metrics.lastDeathAt || Date.now()),
                  cause: String(metrics.lastDeathCause || 'unknown'),
                  source: String(metrics.lastDeathSource || 'sim_damage'),
                  zone: Number(zoneContribution || 0),
                  enemy: Number(enemyContribution || 0),
                  toxic: Number(metrics.zoneToxic || 0),
                  outside: Number(metrics.zoneOutside || 0)
                });
                if (deathHistory.length > 30) deathHistory.splice(0, deathHistory.length - 30);
                metrics.deathHistory = deathHistory;
                metrics.deaths = Number(metrics.deaths || 0) + 1;
                metrics.simState = 'death_respawn';
                metrics.hp = metrics.maxHp;
              } else if (metrics.buttonClicks > 0) {
                metrics.simState = 'in_match';
              }
              const absDx = Math.abs(dx);
              const absDy = Math.abs(dy);
              if (!externalEnemyMode) {
                if (absDx >= absDy) metrics.enemyDir = dx >= 0 ? 'RIGHT' : 'LEFT';
                else metrics.enemyDir = dy >= 0 ? 'DOWN' : 'UP';
              }

              const hpRatio = Math.max(0, Math.min(1, Number(metrics.hp || 0) / Number(metrics.maxHp || 100)));
              hpBar.style.transform = `scaleX(${hpRatio})`;
              const manaRatio = Math.max(0, Math.min(1, Number(metrics.mana || 0) / Number(metrics.maxMana || 30)));
              manaBar.style.transform = `scaleX(${manaRatio})`;
              if (!externalManaMode) {
                metrics.mana = Math.min(metrics.maxMana, Number(metrics.mana || 0) + (dt * 2.8));
              }
              hpText.textContent =
                'hp=' + Number(metrics.hp || 0).toFixed(1) +
                ' in=' + Number(metrics.damageReceived || 0).toFixed(1) +
                ' out=' + Number(metrics.damageDealt || 0).toFixed(1) +
                ' kills=' + Number(metrics.kills || 0).toFixed(0) +
                ' threat=' + Number(metrics.enemyThreat || 0).toFixed(2) +
                ' dir=' + String(metrics.enemyDir || 'CENTER');
              manaText.textContent =
                'mana=' + Number(metrics.mana || 0).toFixed(1) +
                '/' + Number(metrics.maxMana || 30).toFixed(0) +
                ' (' + Number(manaRatio * 100).toFixed(0) + '%)' +
                ' src=' + (externalManaMode ? 'live' : 'sim') +
                ' abilities=' + Number(metrics.ability1Uses || 0).toFixed(0) + '/' +
                Number(metrics.ability2Uses || 0).toFixed(0) + '/' +
                Number(metrics.ability3Uses || 0).toFixed(0) +
                ' wall=' + Number(metrics.buildWallUses || 0).toFixed(0) +
                ' sprint=' + Number(metrics.sprintUses || 0).toFixed(0);
              const safeDx = Number(metrics.safeZoneX || 0.5) - playerX;
              const safeDy = Number(metrics.safeZoneY || 0.5) - playerY;
              const safeDist = Math.sqrt((safeDx * safeDx) + (safeDy * safeDy));
              const outOfSafeZone = Number(metrics.zoneOutside || 0) > 0 || (safeDist > Number(metrics.zoneRadius || 0.42));
              const toxicActive = Number(metrics.zoneToxic || 0) > 0;
              const zoneCounter = Number(metrics.zoneCountdownSec || -1);
              const enemyVisibleNow = Number(metrics.enemyVisible || 0) > 0;
              let decision = 'SCOUT';
              if ((toxicActive || outOfSafeZone) && zoneCounter >= 0 && zoneCounter <= 9.5) {
                decision = enemyVisibleNow ? 'EVADE + ROTATE SAFE ZONE' : 'ROTATE SAFE ZONE';
              } else if (toxicActive) {
                decision = 'EXIT TOXIC CLOUD';
              } else if (enemyVisibleNow && Number(metrics.enemyThreat || 0) >= 0.62) {
                decision = outOfSafeZone ? 'RETREAT TO SAFE ZONE' : 'ENGAGE ENEMY';
              } else if (enemyVisibleNow) {
                decision = 'TRACK ENEMY';
              } else if (outOfSafeZone) {
                decision = 'MOVE TO SAFE ZONE';
              }
              metrics.decisionHint = decision;
              zoneText.textContent =
                'map=' + String(metrics.mapName || 'SmokeArena') +
                ' counter=' + Number(metrics.zoneCountdownSec || 0).toFixed(1) +
                ' safe=(' + Number(metrics.safeZoneX || 0).toFixed(2) + ',' +
                Number(metrics.safeZoneY || 0).toFixed(2) + ',r=' +
                Number(metrics.zoneRadius || 0).toFixed(2) + ')';
              if (zoneClock) {
                zoneClock.textContent =
                  'ZONE SHRINK IN: ' + Number(metrics.zoneCountdownSec || 0).toFixed(1) + 's' +
                  ' source=' + String(metrics.zoneSource || 'sim') +
                  ' toxic=' + (toxicActive ? '1' : '0');
              }
              if (decisionText) {
                decisionText.textContent =
                  'decision=' + String(metrics.decisionHint || 'SCOUT') +
                  ' confidence=' + Number(metrics.enemyThreat || 0).toFixed(2) +
                  ' outside=' + (outOfSafeZone ? '1' : '0') +
                  ' toxic=' + (toxicActive ? '1' : '0') +
                  ' kills=' + Number(metrics.kills || 0).toFixed(0) +
                  ' last_death=' + String(metrics.lastDeathCause || 'none') +
                  ' zone_src=' + String(metrics.zoneSource || 'sim') +
                  '/' + String(metrics.zoneSignalSource || 'none') +
                  ' enemy_src=' + String(metrics.enemySource || 'sim');
              }
              if (metrics.mapOpen) mapPanel.classList.add('open');
              else mapPanel.classList.remove('open');
              drawMap();
              drawRadar();
              status.textContent =
                'state=' + String(metrics.simState || 'lobby') +
                ' jugar=' + metrics.buttonClicks +
                ' hp=' + Number(metrics.hp || 0).toFixed(1) +
                ' kills=' + Number(metrics.kills || 0).toFixed(0) +
                ' death=' + String(metrics.lastDeathCause || 'none') +
                ' mana=' + Number(metrics.mana || 0).toFixed(1) + '/' + Number(metrics.maxMana || 30).toFixed(0) +
                ' counter=' + Number(metrics.zoneCountdownSec || 0).toFixed(1);
            }, 170);
          </script>
        </body>
        </html>
        """
        try:
            smoke_page = context_obj.new_page()
            try:
                smoke_page.set_viewport_size({"width": 980, "height": 640})
            except Exception:
                pass
            try:
                smoke_page.set_default_timeout(max(300, min(1500, int(args.bot_action_timeout_ms))))
            except Exception:
                pass
            smoke_page.set_content(smoke_html, wait_until="domcontentloaded")
            smoke_frame = smoke_page.main_frame
            install_click_probe(smoke_frame)
            install_input_feedback_probe(smoke_frame)
            ensure_bot_cursor_overlay(smoke_frame, transition_ms=args.bot_cursor_transition_ms)
            if args.bot_debug_hud:
                ensure_bot_debug_hud(smoke_frame)
            print("[BOT][PARALLEL_SMOKE] Monitor listo (tab auxiliar).")
            return {
                "page": smoke_page,
                "frame": smoke_frame,
                "step": 0,
                "last_tick": 0.0,
                "last_log": 0.0,
                "key_index": 0,
                "key_pattern": parse_move_pattern_csv(args.bot_smoke_move_pattern),
                "degraded_ticks": 0,
                "degraded_restarts": 0,
            }
        except Exception as exc:
            print(f"[BOT][PARALLEL_SMOKE][WARN] No se pudo inicializar monitor: {exc}")
            return None

    def tick_parallel_smoke_monitor(state: Optional[Dict[str, Any]], now_mono: float) -> None:
        if not state:
            return
        smoke_page = state.get("page")
        smoke_frame = state.get("frame")
        if smoke_page is None or smoke_frame is None:
            return
        try:
            if smoke_page.is_closed():
                raise RuntimeError("parallel smoke page closed")
        except Exception:
            restarted = setup_parallel_smoke_monitor(context)
            if restarted:
                state.clear()
                state.update(restarted)
                print("[BOT][PARALLEL_SMOKE] Monitor reiniciado (pagina cerrada).")
            return

        tick_interval = max(0.08, float(args.bot_smoke_step_ms) / 1000.0)
        if (now_mono - float(state.get("last_tick", 0.0))) < tick_interval:
            return

        step_idx = int(state.get("step", 0))
        state["step"] = step_idx + 1
        state["last_tick"] = now_mono
        pattern = state.get("key_pattern") or parse_move_pattern_csv(args.bot_smoke_move_pattern)
        key_index = int(state.get("key_index", 0))
        key_to_press = pattern[key_index % len(pattern)]
        state["key_index"] = key_index + 1

        ratios = [
            (0.24, 0.28),
            (0.72, 0.28),
            (0.72, 0.72),
            (0.24, 0.72),
            (0.52, 0.52),
            (0.36, 0.48),
        ]
        x_ratio, y_ratio = ratios[step_idx % len(ratios)]
        target = get_canvas_target(
            page_obj=smoke_page,
            frame_obj=smoke_frame,
            iframe_selector=args.bot_iframe_selector,
            x_ratio=x_ratio,
            y_ratio=y_ratio,
        )
        if target is None:
            viewport = smoke_page.viewport_size or {"width": 980, "height": 640}
            target = {
                "frame_x": float(viewport["width"]) * x_ratio,
                "frame_y": float(viewport["height"]) * y_ratio,
                "page_x": float(viewport["width"]) * x_ratio,
                "page_y": float(viewport["height"]) * y_ratio,
                "source": "parallel_smoke_viewport",
            }

        try:
            play_target = build_target_from_locator(
                page_obj=smoke_page,
                frame_obj=smoke_frame,
                iframe_selector=args.bot_iframe_selector,
                locator=smoke_frame.locator("#play").first,
                source="parallel_smoke_button",
            )
        except Exception:
            play_target = None

        do_button_click = (step_idx % 4) == 0 and play_target is not None
        click_target = play_target if do_button_click else target
        step_active_keys: List[str] = [key_to_press, "Space"]
        if (step_idx % 3) == 0:
            step_active_keys.extend(["Shift", "MouseRight"])
        if (step_idx % 4) == 0:
            step_active_keys.append(["Digit1", "Digit2", "Digit3"][(step_idx // 4) % 3])
        if (step_idx % 5) == 0:
            step_active_keys.append("KeyR")
        if (step_idx % 6) == 0:
            step_active_keys.append("KeyC")

        move_bot_cursor_overlay(
            smoke_frame,
            frame_x=float(click_target["frame_x"]),
            frame_y=float(click_target["frame_y"]),
            source=str(click_target.get("source", "parallel_smoke_move")),
        )
        try:
            try:
                smoke_frame.focus("body", timeout=600)
            except Exception:
                pass
            if (step_idx % 3) == 0:
                smoke_page.keyboard.down("Shift")
                try:
                    smoke_page.mouse.down(button="right")
                    smoke_page.mouse.up(button="right")
                except Exception:
                    pass
            smoke_page.keyboard.down(key_to_press)
            time.sleep(max(0.02, min(0.09, tick_interval * 0.45)))
            smoke_page.keyboard.up(key_to_press)
            if (step_idx % 3) == 0:
                smoke_page.keyboard.up("Shift")
            smoke_page.keyboard.down("Space")
            smoke_page.keyboard.up("Space")
            if (step_idx % 4) == 0:
                ability_key = ["Digit1", "Digit2", "Digit3"][(step_idx // 4) % 3]
                smoke_page.keyboard.down(ability_key)
                smoke_page.keyboard.up(ability_key)
            if (step_idx % 5) == 0:
                smoke_page.keyboard.down("KeyR")
                smoke_page.keyboard.up("KeyR")
            if (step_idx % 6) == 0:
                smoke_page.keyboard.down("KeyC")
                smoke_page.keyboard.up("KeyC")
            smoke_page.mouse.move(
                float(click_target["page_x"]),
                float(click_target["page_y"]),
                steps=max(1, int(args.bot_mouse_move_steps)),
            )
        except Exception:
            pass

        if (step_idx % 2) == 0:
            try:
                smoke_page.mouse.click(
                    float(click_target["page_x"]),
                    float(click_target["page_y"]),
                    button="left",
                )
            except Exception:
                pass
            dispatch_dom_left_click(
                smoke_frame,
                frame_x=float(click_target["frame_x"]),
                frame_y=float(click_target["frame_y"]),
            )

        bridge_mana_current = bot_event_signals.get("mana_current")
        bridge_mana_max = bot_event_signals.get("mana_max")
        bridge_zone_countdown = bot_event_signals.get("zone_countdown_sec")
        bridge_safe_zone_x = bot_event_signals.get("safe_zone_x")
        bridge_safe_zone_y = bot_event_signals.get("safe_zone_y")
        bridge_safe_zone_radius = bot_event_signals.get("safe_zone_radius")
        bridge_zone_toxic = 1 if bool(bot_event_signals.get("zone_toxic_detected", False)) else 0
        bridge_zone_outside = 1 if bool(bot_event_signals.get("zone_outside_safe", False)) else 0
        bridge_zone_signal_source = str(bot_event_signals.get("zone_signal_source", "none") or "none")
        bridge_map_name = str(bot_event_signals.get("map_name", "") or "")
        bridge_bot_state = str(last_state or "unknown")
        bridge_enemy_conf = None
        bridge_enemy_detected = 0
        bridge_enemy_dir = "CENTER"
        bridge_enemy_x_ratio = None
        bridge_enemy_y_ratio = None
        try:
            if isinstance(enemy_signal_cache, dict):
                bridge_enemy_conf = float(enemy_signal_cache.get("confidence", 0.0) or 0.0)
                bridge_enemy_detected = 1 if bool(enemy_signal_cache.get("recent", enemy_signal_cache.get("detected", False))) else 0
                bridge_enemy_dir = str(enemy_signal_cache.get("dir", "CENTER") or "CENTER")
                exr = enemy_signal_cache.get("x_ratio")
                eyr = enemy_signal_cache.get("y_ratio")
                if exr is not None:
                    bridge_enemy_x_ratio = float(exr)
                if eyr is not None:
                    bridge_enemy_y_ratio = float(eyr)
        except Exception:
            bridge_enemy_conf = None
            bridge_enemy_detected = 0
            bridge_enemy_dir = "CENTER"
            bridge_enemy_x_ratio = None
            bridge_enemy_y_ratio = None
        try:
            if bridge_mana_current is not None:
                bridge_mana_current = float(bridge_mana_current)
        except Exception:
            bridge_mana_current = None
        try:
            if bridge_mana_max is not None:
                bridge_mana_max = float(bridge_mana_max)
        except Exception:
            bridge_mana_max = None
        try:
            if bridge_zone_countdown is not None:
                bridge_zone_countdown = float(bridge_zone_countdown)
                if bridge_zone_countdown < 0:
                    bridge_zone_countdown = None
        except Exception:
            bridge_zone_countdown = None
        try:
            if bridge_safe_zone_x is not None:
                bridge_safe_zone_x = float(bridge_safe_zone_x)
        except Exception:
            bridge_safe_zone_x = None
        try:
            if bridge_safe_zone_y is not None:
                bridge_safe_zone_y = float(bridge_safe_zone_y)
        except Exception:
            bridge_safe_zone_y = None
        try:
            if bridge_safe_zone_radius is not None:
                bridge_safe_zone_radius = float(bridge_safe_zone_radius)
        except Exception:
            bridge_safe_zone_radius = None
        try:
            smoke_frame.evaluate(
                """
                (payload) => {
                  window.__botSmokeParallelBridge = {
                    ts: Number(payload.ts || Date.now()),
                    mana_current: payload.mana_current,
                    mana_max: payload.mana_max,
                    zone_countdown_sec: payload.zone_countdown_sec,
                    safe_zone_x: payload.safe_zone_x,
                    safe_zone_y: payload.safe_zone_y,
                    safe_zone_radius: payload.safe_zone_radius,
                    zone_toxic: payload.zone_toxic,
                    zone_outside: payload.zone_outside,
                    zone_signal_source: payload.zone_signal_source,
                    map_name: payload.map_name,
                    bot_state: payload.bot_state,
                    enemy_detected: payload.enemy_detected,
                    enemy_conf: payload.enemy_conf,
                    enemy_dir: payload.enemy_dir,
                    enemy_x_ratio: payload.enemy_x_ratio,
                    enemy_y_ratio: payload.enemy_y_ratio
                  };
                }
                """,
                {
                    "ts": int(time.time() * 1000),
                    "mana_current": bridge_mana_current,
                    "mana_max": bridge_mana_max,
                    "zone_countdown_sec": bridge_zone_countdown,
                    "safe_zone_x": bridge_safe_zone_x,
                    "safe_zone_y": bridge_safe_zone_y,
                    "safe_zone_radius": bridge_safe_zone_radius,
                    "zone_toxic": bridge_zone_toxic,
                    "zone_outside": bridge_zone_outside,
                    "zone_signal_source": bridge_zone_signal_source,
                    "map_name": bridge_map_name,
                    "bot_state": bridge_bot_state,
                    "enemy_detected": bridge_enemy_detected,
                    "enemy_conf": bridge_enemy_conf,
                    "enemy_dir": bridge_enemy_dir,
                    "enemy_x_ratio": bridge_enemy_x_ratio,
                    "enemy_y_ratio": bridge_enemy_y_ratio,
                },
            )
        except Exception:
            pass

        log_interval = max(0.8, float(args.bot_cursor_log_interval_sec))
        if (now_mono - float(state.get("last_log", 0.0))) < log_interval:
            return
        state["last_log"] = now_mono
        cursor_probe = read_bot_cursor_probe(smoke_frame)
        click_probe = read_click_probe(smoke_frame)
        input_probe = read_input_feedback_probe(smoke_frame)
        try:
            smoke_metrics = smoke_frame.evaluate("() => window.__botSmokeParallel || {}")
        except Exception:
            smoke_metrics = {}
        if not isinstance(smoke_metrics, dict):
            smoke_metrics = {}
        has_metrics = bool(smoke_metrics) and ("hp" in smoke_metrics or "buttonClicks" in smoke_metrics)
        if not has_metrics:
            state["degraded_ticks"] = int(state.get("degraded_ticks", 0)) + 1
        else:
            state["degraded_ticks"] = 0
        if int(state.get("degraded_ticks", 0)) >= 3:
            restarted = setup_parallel_smoke_monitor(context)
            if restarted:
                restarted["degraded_restarts"] = int(state.get("degraded_restarts", 0)) + 1
                state.clear()
                state.update(restarted)
                print("[BOT][PARALLEL_SMOKE] Monitor reiniciado (telemetria degradada).")
            return
        if args.bot_debug_hud:
            update_bot_debug_hud(
                smoke_frame,
                {
                    "state": f"parallel_smoke:{str(smoke_metrics.get('simState', 'lobby') or 'lobby')}",
                    "reason": "tick",
                    "action": f"step={step_idx + 1} key={key_to_press}",
                    "key_down": int(input_probe.get("keyDown", 0)),
                    "key_up": int(input_probe.get("keyUp", 0)),
                    "last_key": str(input_probe.get("lastKeyDown", "") or "-"),
                    "pointer_down": int(input_probe.get("pointerDown", 0)),
                    "pointer_up": int(input_probe.get("pointerUp", 0)),
                    "pointer_move": int(input_probe.get("pointerMove", 0)),
                    "cursor_moves": int(cursor_probe.get("moves", 0)),
                    "cursor_source": str(cursor_probe.get("lastSource", "") or "-"),
                    "click0": int(click_probe.get("click0", 0)),
                    "click_target": str(click_probe.get("lastTarget", "") or "-"),
                    "enemy_seen": int(smoke_metrics.get("enemyVisible", 0)),
                    "enemy_conf": float(smoke_metrics.get("enemyThreat", 0.0)),
                    "enemy_dir": str(smoke_metrics.get("enemyDir", "") or "-"),
                    "damage_done": float(smoke_metrics.get("damageDealt", 0.0)),
                    "damage_taken": float(smoke_metrics.get("damageReceived", 0.0)),
                    "kills": int(smoke_metrics.get("kills", 0)),
                    "last_death": str(smoke_metrics.get("lastDeathCause", "none") or "none"),
                    "mana": float(smoke_metrics.get("mana", 0.0)),
                    "zone_counter": f"{float(smoke_metrics.get('zoneCountdownSec', 0.0)):.1f}s",
                    "zone_source": str(smoke_metrics.get("zoneSource", "") or "sim"),
                    "death_cause": str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                    "death_conf": float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                    "death_source": str(bot_event_signals.get("death_cause_source", "none") or "none"),
                    "safe_zone": (
                        f"({float(smoke_metrics.get('safeZoneX', 0.0)):.2f},"
                        f"{float(smoke_metrics.get('safeZoneY', 0.0)):.2f},"
                        f"r={float(smoke_metrics.get('zoneRadius', 0.0)):.2f})"
                    ),
                    "decision": str(smoke_metrics.get("decisionHint", "") or "-"),
                    "ability_last": str(smoke_metrics.get("lastAbility", "") or "-"),
                    "ability_class": str(smoke_metrics.get("lastAbilityClass", "") or "-"),
                    "dash_cd": (
                        float((bot_event_signals.get("ability_cooldown_sec", {}) or {}).get("Shift") or 0.0)
                        if isinstance(bot_event_signals.get("ability_cooldown_sec"), dict)
                        else 0.0
                    ),
                    "active_keys": step_active_keys,
                    "feed_line": (
                        f"ps{step_idx + 1} st={str(smoke_metrics.get('simState', 'lobby') or 'lobby')} "
                        f"key={key_to_press} kd={int(input_probe.get('keyDown', 0))} "
                        f"click0={int(click_probe.get('click0', 0))} "
                        f"in={float(smoke_metrics.get('damageReceived', 0.0)):.1f} "
                        f"out={float(smoke_metrics.get('damageDealt', 0.0)):.1f}"
                    ),
                },
            )
        print(
            "[BOT][PARALLEL_SMOKE] "
            f"step={step_idx + 1} "
            f"moves={int(cursor_probe.get('moves', 0))} "
            f"click0={int(click_probe.get('click0', 0))} "
            f"jugarClicks={int(smoke_metrics.get('buttonClicks', 0))} "
            f"pointerDown={int(smoke_metrics.get('pointerDown', 0))} "
            f"state={str(smoke_metrics.get('simState', 'lobby') or 'lobby')} "
            f"damageOut={float(smoke_metrics.get('damageDealt', 0.0)):.1f} "
            f"damage={float(smoke_metrics.get('damageReceived', 0.0)):.1f} "
            f"kills={int(smoke_metrics.get('kills', 0))} "
            f"lastDeath={str(smoke_metrics.get('lastDeathCause', 'none') or 'none')} "
            f"hp={float(smoke_metrics.get('hp', 0.0)):.1f} "
            f"mana={float(smoke_metrics.get('mana', 0.0)):.1f} "
            f"counter={float(smoke_metrics.get('zoneCountdownSec', 0.0)):.1f} "
            f"zoneSrc={str(smoke_metrics.get('zoneSource', 'sim') or 'sim')} "
            f"enemySrc={str(smoke_metrics.get('enemySource', 'sim') or 'sim')} "
            f"toxic={int(smoke_metrics.get('zoneToxic', 0) or 0)} "
            f"outside={int(smoke_metrics.get('zoneOutside', 0) or 0)} "
            f"decision={str(smoke_metrics.get('decisionHint', '-') or '-')}"
        )


    profile_dir = args.profile_dir.strip()
    cdp_url = args.cdp_url.strip()
    context = None
    browser = None
    page = None
    close_context_on_exit = True
    close_browser_on_exit = False
    route_pattern = "**/*"
    route_installed = False
    try:
        with sync_playwright() as p:
            if cdp_url:
                try:
                    browser, context, page, launch_mode = attach_over_cdp(
                        p,
                        cdp_url=cdp_url,
                        game_url=args.game_url,
                        open_new_page=args.cdp_new_page,
                    )
                    close_context_on_exit = False
                    close_browser_on_exit = True
                except PlaywrightError as exc:
                    raise SystemExit(
                        "No se pudo conectar por CDP.\n"
                        "Arranca Chrome asi:\n"
                        "chrome.exe --remote-debugging-port=9222\n"
                        "Y usa --cdp-url http://127.0.0.1:9222\n"
                        f"Error original: {exc}"
                    ) from exc
            else:
                try:
                    context, page, launch_mode = launch_context_with_fallback(
                        p, args, profile_dir
                    )
                except PlaywrightError as exc:
                    raise SystemExit(
                        "No se pudo abrir el navegador con Playwright.\n"
                        "Prueba:\n"
                        "1) Cerrar todas las ventanas de Chrome/Edge.\n"
                        "2) Ejecutar con --no-persistent.\n"
                        "3) Cambiar a --channel chromium.\n"
                        f"Error original: {exc}"
                    ) from exc

            if args.scan_websocket:
                try:
                    cdp_session = context.new_cdp_session(page)
                    cdp_session.send("Network.enable")

                    def on_ws_created(params: Dict[str, Any]) -> None:
                        request_id = str(params.get("requestId", ""))
                        ws_url = str(params.get("url", ""))
                        if request_id and ws_url:
                            ws_url_by_request_id[request_id] = ws_url

                    def on_ws_received(params: Dict[str, Any]) -> None:
                        request_id = str(params.get("requestId", ""))
                        response = params.get("response", {})
                        if isinstance(response, dict):
                            capture_ws_frame("recv", request_id, response)

                    def on_ws_sent(params: Dict[str, Any]) -> None:
                        request_id = str(params.get("requestId", ""))
                        response = params.get("response", {})
                        if isinstance(response, dict):
                            capture_ws_frame("sent", request_id, response)

                    cdp_session.on("Network.webSocketCreated", on_ws_created)
                    cdp_session.on("Network.webSocketFrameReceived", on_ws_received)
                    cdp_session.on("Network.webSocketFrameSent", on_ws_sent)
                except Exception as exc:
                    print(f"[WARN] No se pudo activar captura websocket CDP: {exc}")

            if args.capture_all_post:
                route_pattern = "**/*"
            else:
                endpoint_hint = str(args.endpoint_substring or "").strip("/")
                route_pattern = f"**/*{endpoint_hint}*" if endpoint_hint else "**/*"
            context.route(route_pattern, route_handler)
            route_installed = True
            if (not cdp_url) or args.cdp_new_page:
                page.goto(args.game_url, wait_until="domcontentloaded")

            run_play_game = bool(args.play_game)
            if args.bot_auto_phases:
                print("[BOT][AUTO] Fase 1/2: smoke test.")
                smoke_ok = run_bot_smoke_test(page)
                print(f"[BOT][AUTO] Smoke test result: {'PASS' if smoke_ok else 'WARN'}")
                if (not smoke_ok) and args.bot_auto_stop_on_smoke_fail:
                    print("[BOT][AUTO] Detenido por smoke fail (configurado para stop).")
                    return
                print("[BOT][AUTO] Fase 2/2: play-game.")
                run_play_game = True
                try:
                    page.goto(args.game_url, wait_until="domcontentloaded")
                except Exception as exc:
                    print(f"[BOT][AUTO][WARN] No se pudo volver a game_url tras smoke: {exc}")

            elif args.bot_smoke_test:
                run_bot_smoke_test(page)
                return

            if run_play_game and bool(args.bot_google_login):
                google_email, google_password = resolve_google_credentials(args)
                login_ok = ensure_google_login_pre_game(
                    page_obj=page,
                    email=google_email,
                    password=google_password,
                    timeout_sec=float(args.bot_google_login_timeout_sec),
                )
                if not login_ok:
                    print("[BOT][GOOGLE_LOGIN][WARN] Continuando sin confirmar login Google.")
                try:
                    page.goto(args.game_url, wait_until="domcontentloaded")
                except Exception as exc:
                    print(f"[BOT][GOOGLE_LOGIN][WARN] No se pudo volver a game_url tras login: {exc}")

            if run_play_game:
                print("[BOT] Bot activado. Intentando jugar...")
                iframe_selector = args.bot_iframe_selector
                game_actual_frame: Optional[Frame] = None
                try:
                    game_actual_frame = resolve_game_frame(
                        page,
                        iframe_selector=iframe_selector,
                        timeout_ms=args.bot_frame_timeout_ms,
                    )
                    if game_actual_frame is None:
                        raise RuntimeError("No se pudo obtener el objeto Frame del juego.")
                except Exception as exc:
                    print(f"[BOT][ERROR] No se pudo resolver frame de juego: {exc}")
                    # If frame cannot be resolved, bot cannot proceed
                    return
                runtime_timeout_ms = max(300, min(1500, int(args.bot_action_timeout_ms)))
                try:
                    page.set_default_timeout(runtime_timeout_ms)
                except Exception:
                    pass
                try:
                    game_actual_frame.set_default_timeout(runtime_timeout_ms)
                except Exception:
                    pass

                install_click_probe(game_actual_frame)
                install_input_feedback_probe(game_actual_frame)
                runtime_feedback_session = create_feedback_session(
                    base_dir=args.bot_feedback_dir,
                    mode="play_runtime",
                    explicit_jsonl=args.bot_feedback_jsonl,
                )
                if runtime_feedback_session:
                    print(
                        "[BOT][FEEDBACK] Runtime feedback en: "
                        f"{runtime_feedback_session.get('jsonl_path')}"
                    )
                    append_feedback_event(
                        runtime_feedback_session,
                        {
                            "event": "session_start",
                            "ts": time.time(),
                            "mode": "play_runtime",
                            "game_url": args.game_url,
                            "channel": args.channel,
                            "parallel_smoke": bool(args.bot_parallel_smoke),
                            "cursor_transition_ms": int(args.bot_cursor_transition_ms),
                            "ui_poll_ms": int(args.bot_ui_poll_ms),
                        },
                    )
                cursor_overlay_ready = False
                cursor_overlay_last_retry_at = 0.0
                cursor_probe_last_log_at = 0.0
                cursor_probe_last_moves = 0
                active_keys_runtime: Set[str] = set()
                last_action_label = "init"
                last_action_ok = True
                parallel_smoke_state: Optional[Dict[str, Any]] = None
                if args.bot_parallel_smoke:
                    parallel_smoke_state = setup_parallel_smoke_monitor(context)
                    try:
                        page.bring_to_front()
                    except Exception:
                        pass
                if args.bot_visual_cursor:
                    cursor_overlay_ready = ensure_bot_cursor_overlay(
                        game_actual_frame,
                        transition_ms=args.bot_cursor_transition_ms,
                    )
                    cursor_overlay_last_retry_at = time.monotonic()
                    if cursor_overlay_ready:
                        bootstrap_target = get_attack_target(
                            page,
                            game_actual_frame,
                            iframe_selector,
                        )
                        if bootstrap_target is not None:
                            move_bot_cursor_overlay(
                                game_actual_frame,
                                frame_x=float(bootstrap_target["frame_x"]),
                                frame_y=float(bootstrap_target["frame_y"]),
                                source="cursor_bootstrap",
                            )
                        print("[BOT][CURSOR] Overlay inicializado.")
                    else:
                        print("[BOT][CURSOR][WARN] No se pudo inicializar overlay en primer intento.")
                if args.bot_debug_hud:
                    ensure_bot_debug_hud(game_actual_frame)

                import random

                keys = [
                    "KeyW",
                    "KeyA",
                    "KeyS",
                    "KeyD",
                ]
                smoke_move_keys = parse_move_pattern_csv(args.bot_smoke_move_pattern)
                print(f"[BOT][MOVE_SMOKE] pattern={','.join(smoke_move_keys)}")
                smoke_move_index = 0
                move_escape_remaining = 0
                move_low_motion_streak = 0
                move_last_motion_score: Optional[float] = None
                enemy_signal_cache: Dict[str, Any] = {
                    "detected": False,
                    "confidence": 0.0,
                    "near": False,
                    "x_ratio": 0.5,
                    "y_ratio": 0.5,
                    "direction_key": "KeyW",
                    "dir": "CENTER",
                    "red_ratio": 0.0,
                    "area_ratio": 0.0,
                }
                enemy_scan_last_at = 0.0
                enemy_last_seen_at = 0.0
                last_state = "unknown"
                play_transition_started_at: Optional[float] = None
                last_play_attempt_at = 0.0
                cached_play_target: Optional[Dict[str, float]] = None
                last_play_target_scan_at = 0.0
                run_started_at = time.monotonic()
                knowledge_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
                runtime_probe_lobby_done = False
                runtime_probe_match_done = False
                runtime_probe_paths: List[str] = []
                knowledge_step_idx = 0
                knowledge_reward_sum = 0.0
                knowledge_motion_sum = 0.0
                knowledge_motion_samples = 0
                knowledge_enemy_seen_steps = 0
                knowledge_zone_observed_steps = 0
                knowledge_prev_damage_done = float(bot_event_signals.get("damage_done_total", 0.0) or 0.0)
                knowledge_prev_damage_taken = float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0)
                knowledge_prev_move_action = ""
                knowledge_prev_move_context = ""
                knowledge_prev_ability_action = ""
                knowledge_prev_ability_context = ""
                if bool(args.bot_runtime_probe):
                    try:
                        probe_snapshot = collect_runtime_variable_probe(
                            game_actual_frame,
                            max_keys=max(20, int(args.bot_runtime_probe_max_keys)),
                        )
                        probe_path = write_runtime_probe_snapshot(
                            base_dir=args.bot_runtime_probe_dir,
                            phase="lobby_bootstrap",
                            run_id=knowledge_run_id,
                            snapshot=probe_snapshot,
                        )
                        if probe_path is not None:
                            runtime_probe_lobby_done = True
                            runtime_probe_paths.append(str(probe_path))
                            print(f"[BOT][RUNTIME_PROBE] snapshot=lobby_bootstrap path={probe_path}")
                    except Exception as exc:
                        print(f"[BOT][RUNTIME_PROBE][WARN] fallo lobby_bootstrap: {exc}")
                last_in_match_entered_at = 0.0
                ability_state: Dict[str, Any] = {
                    "mana": 100.0,
                    "max_mana": 100.0,
                    "last_used_key": "",
                    "last_used_class": "",
                    "usage": {"Digit1": 0, "Digit2": 0, "Digit3": 0, "KeyR": 0, "Shift": 0, "MouseRight": 0},
                    "last_ability_at": 0.0,
                    "next_ability_at": 0.0,
                    "next_wall_at": 0.0,
                    "next_sprint_at": 0.0,
                    "blocked_until": {"Digit1": 0.0, "Digit2": 0.0, "Digit3": 0.0, "KeyR": 0.0},
                    "pending_outcomes": [],
                }
                bot_decision_backend = str(getattr(args, "bot_decision_backend", "legacy") or "legacy").strip().lower()
                lms_re_policy = None
                lms_re_last_error_at = 0.0
                if bot_decision_backend == "lms_re":
                    if LMSReverseEngineeredBot is None:
                        print("[BOT][DECISION][WARN] LMSReverseEngineeredBot no disponible; fallback a legacy.")
                        bot_decision_backend = "legacy"
                    else:
                        try:
                            lms_re_seed = int(time.time() * 1000) % 2_147_483_647
                            lms_re_policy = LMSReverseEngineeredBot(
                                player_id="live_runner",
                                mode_name=str(getattr(args, "bot_lmsre_mode_name", "royale_mode") or "royale_mode"),
                                seed=lms_re_seed,
                            )
                            print(
                                "[BOT][DECISION] backend=lms_re "
                                f"mode={str(getattr(args, 'bot_lmsre_mode_name', 'royale_mode') or 'royale_mode')} "
                                f"seed={lms_re_seed}"
                            )
                        except Exception as exc:
                            print(f"[BOT][DECISION][WARN] fallo init lms_re: {exc}. Fallback a legacy.")
                            lms_re_policy = None
                            bot_decision_backend = "legacy"
                else:
                    print("[BOT][DECISION] backend=legacy")
                movement_held_keys: Set[str] = set()
                if knowledge_conn is not None:
                    try:
                        knowledge_conn.execute(
                            """
                            INSERT OR REPLACE INTO bot_match_runs(
                                run_id, started_at_ms, mode, stop_reason, map_name,
                                damage_done, damage_taken, steps, avg_motion,
                                enemy_seen_steps, zone_observed_steps
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                knowledge_run_id,
                                int(time.time() * 1000),
                                "play_game",
                                "",
                                "",
                                0.0,
                                0.0,
                                0,
                                0.0,
                                0,
                                0,
                            ),
                        )
                        knowledge_pending_writes += 1
                    except Exception as exc:
                        print(f"[BOT][KNOWLEDGE][WARN] No se pudo iniciar run en DB: {exc}")
                last_map_toggle_at = 0.0
                visual_ocr_state: Dict[str, Any] = {
                    "last_ocr_at": 0.0,
                    "frames": 0,
                    "disabled": False,
                    "last_result": {},
                    "death_streak": 0,
                }
                run_stop_reason = ""
                match_opening_until = 0.0
                opening_move_index = 0
                collision_streak = 0
                last_salon_attempt_at = 0.0
                death_salon_attempted = False
                death_pending_since = 0.0
                last_recorded_death_ts = 0.0
                last_recorded_loot_ts = 0.0
                last_guardian_obs_sig = ""
                forced_move_queue: Deque[List[str]] = deque()
                last_move_action_runtime = ""
                repeat_move_action_streak = 0
                stuck_signal_streak = 0
                stuck_recovery_attempt = 0
                stuck_event_total = 0
                runtime_stuck_stats: Dict[str, Dict[str, float]] = {}
                runtime_move_context_penalties: Dict[Tuple[str, str], float] = {}
                runtime_move_action_penalties: Dict[str, float] = {}
                runtime_move_map_action_penalties: Dict[Tuple[str, str], float] = {}
                while True:
                    now_mono = time.monotonic()
                    zone_signal_ts_now = float(bot_event_signals.get("zone_signal_ts", 0.0) or 0.0)
                    if zone_signal_ts_now > 0.0 and (now_mono - zone_signal_ts_now) > 14.0:
                        bot_event_signals["zone_signal_source"] = "none"
                    if bool(bot_event_signals.get("zone_toxic_detected", False)):
                        if zone_signal_ts_now <= 0.0 or (now_mono - zone_signal_ts_now) > 6.0:
                            bot_event_signals["zone_toxic_detected"] = False
                            bot_event_signals["zone_toxic_confidence"] = max(
                                0.0,
                                float(bot_event_signals.get("zone_toxic_confidence", 0.0) or 0.0) * 0.6,
                            )
                    if knowledge_conn is not None and knowledge_pending_writes >= max(1, int(args.bot_knowledge_flush_every)):
                        try:
                            knowledge_conn.commit()
                            knowledge_pending_writes = 0
                        except Exception as exc:
                            print(f"[BOT][KNOWLEDGE][WARN] commit parcial fallo: {exc}")
                    move_combo_for_feedback: List[str] = []
                    move_escape_for_feedback = False
                    move_motion_score_for_feedback: Optional[float] = move_last_motion_score
                    enemy_for_feedback: Dict[str, Any] = dict(enemy_signal_cache)
                    ability_used_now = ""
                    ability_context_key = ""
                    move_stuck_now = False
                    zone_escape_mode = False
                    zone_outside_safe_now = bool(bot_event_signals.get("zone_outside_safe", False))
                    zone_signal_source_now = str(bot_event_signals.get("zone_signal_source", "none") or "none")
                    lms_re_move_combo: List[str] = []
                    lms_re_ability_key = ""
                    lms_re_action_desc = ""
                    visual_feedback_for_event: Dict[str, Any] = (
                        dict(visual_ocr_state.get("last_result", {}))
                        if isinstance(visual_ocr_state.get("last_result"), dict)
                        else {}
                    )
                    if parallel_smoke_state is not None:
                        tick_parallel_smoke_monitor(parallel_smoke_state, now_mono)
                        try:
                            page.bring_to_front()
                        except Exception:
                            pass
                    if args.bot_visual_cursor and (
                        (not cursor_overlay_ready) or ((now_mono - cursor_overlay_last_retry_at) >= 1.0)
                    ):
                        cursor_overlay_ready = ensure_bot_cursor_overlay(
                            game_actual_frame,
                            transition_ms=args.bot_cursor_transition_ms,
                        )
                        cursor_overlay_last_retry_at = now_mono
                        if cursor_overlay_ready:
                            print("[BOT][CURSOR] Overlay listo.")

                    ui_state = collect_bot_ui_state(game_actual_frame)
                    if ui_state.get("canvas_visible") and (
                        (now_mono - last_play_target_scan_at) >= 1.2
                        or play_transition_started_at is not None
                    ):
                        cached_play_target = detect_play_button_target(
                            page_obj=page,
                            frame_obj=game_actual_frame,
                            iframe_selector=iframe_selector,
                        )
                        last_play_target_scan_at = now_mono
                    elif not ui_state.get("canvas_visible"):
                        cached_play_target = None

                    visual_hint_for_state = "unknown"
                    visual_conf_for_state = 0.0
                    if is_recent_signal(bot_event_signals.get("visual_ocr_ts", 0.0), 12.0, now_mono):
                        visual_hint_for_state = str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown")
                        visual_conf_for_state = float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0)
                    state_info = detect_bot_game_state(
                        ui_state=ui_state,
                        play_target=cached_play_target,
                        now_mono=now_mono,
                        play_transition_started_at=play_transition_started_at,
                        visual_state_hint=visual_hint_for_state,
                        visual_state_confidence=visual_conf_for_state,
                    )
                    bot_state = str(state_info.get("state", "loading"))
                    state_reason = str(state_info.get("reason", "unknown"))
                    if bot_state != last_state:
                        print(f"[BOT][STATE] {last_state} -> {bot_state} ({state_reason})")
                        if bot_state == "in_match":
                            last_in_match_entered_at = now_mono
                            match_opening_until = now_mono + max(0.0, float(args.bot_opening_move_sec))
                            opening_move_index = 0
                            collision_streak = 0
                            stuck_signal_streak = 0
                            repeat_move_action_streak = 0
                            forced_move_queue.clear()
                            death_salon_attempted = False
                            death_pending_since = 0.0
                            if bool(args.bot_runtime_probe) and (not runtime_probe_match_done):
                                try:
                                    probe_snapshot = collect_runtime_variable_probe(
                                        game_actual_frame,
                                        max_keys=max(20, int(args.bot_runtime_probe_max_keys)),
                                    )
                                    probe_path = write_runtime_probe_snapshot(
                                        base_dir=args.bot_runtime_probe_dir,
                                        phase="in_match_entry",
                                        run_id=knowledge_run_id,
                                        snapshot=probe_snapshot,
                                    )
                                    if probe_path is not None:
                                        runtime_probe_match_done = True
                                        runtime_probe_paths.append(str(probe_path))
                                        print(f"[BOT][RUNTIME_PROBE] snapshot=in_match_entry path={probe_path}")
                                except Exception as exc:
                                    print(f"[BOT][RUNTIME_PROBE][WARN] fallo in_match_entry: {exc}")
                        if last_state == "in_match" and bot_state in ("lobby", "loading"):
                            last_event_lower = str(bot_event_signals.get("last_event_name", "") or "").lower()
                            death_like = any(
                                hint in last_event_lower
                                for hint in (
                                    "round_lost",
                                    "match_lost",
                                    "player_death",
                                    "you_died",
                                    "eliminated",
                                    "defeat",
                                    "match_end",
                                )
                            )
                            rank_val = bot_event_signals.get("last_rank")
                            try:
                                rank_num = float(rank_val) if rank_val is not None else None
                            except Exception:
                                rank_num = None
                            play_source_now = str(
                                cached_play_target.get("source", "")
                                if isinstance(cached_play_target, dict)
                                else ""
                            )
                            play_conf_now = float(
                                cached_play_target.get("confidence", 0.0) or 0.0
                            ) if isinstance(cached_play_target, dict) else 0.0
                            lobby_ui_now = bool(
                                ui_state.get("character_visible")
                                or ui_state.get("select_visible")
                                or ui_state.get("play_visible")
                            )
                            play_lobby_evidence = (
                                lobby_ui_now
                                or play_source_now == "dom_text_play"
                                or (
                                    play_source_now == "vision_yellow_button"
                                    and play_conf_now >= 0.68
                                )
                            )
                            match_elapsed = (
                                now_mono - float(last_in_match_entered_at)
                                if float(last_in_match_entered_at) > 0.0
                                else 0.0
                            )
                            match_end_recent = is_recent_signal(
                                bot_event_signals.get("match_end_ts", 0.0),
                                18.0,
                                now_mono,
                            )
                            visual_death_recent = (
                                is_recent_signal(bot_event_signals.get("visual_ocr_ts", 0.0), 10.0, now_mono)
                                and str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown").lower() == "death"
                                and float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0) >= 0.58
                            )
                            not_respawn_transition = (
                                "round_spawn" not in last_event_lower
                                and "waiting_area" not in last_event_lower
                                and "choosing_spawn" not in last_event_lower
                            )
                            death_by_transition = (
                                (bot_state == "lobby")
                                and play_lobby_evidence
                                and (match_elapsed >= 20.0)
                                and not_respawn_transition
                                and (
                                    match_end_recent
                                    or visual_death_recent
                                    or (
                                        (rank_num is not None)
                                        and (rank_num > 1.0)
                                        and is_recent_signal(bot_event_signals.get("match_end_ts", 0.0), 45.0, now_mono)
                                    )
                                )
                            )
                            if death_like or death_by_transition:
                                bot_event_signals["death_ts"] = now_mono
                                refresh_death_cause_context(
                                    now_mono=now_mono,
                                    enemy_recent=bool(enemy_signal_cache.get("detected", False)),
                                    zone_outside_safe=bool(bot_event_signals.get("zone_outside_safe", False)),
                                    zone_toxic_detected=bool(bot_event_signals.get("zone_toxic_detected", False)),
                                    visual_feedback=(
                                        dict(visual_ocr_state.get("last_result", {}))
                                        if isinstance(visual_ocr_state.get("last_result"), dict)
                                        else {}
                                    ),
                                )
                                print(
                                    "[BOT][STATE] Muerte inferida por transicion "
                                    f"in_match->{bot_state} event={bot_event_signals.get('last_event_name', '')} "
                                    f"rank={rank_num} play_src={play_source_now}:{play_conf_now:.2f}"
                                )
                        last_state = bot_state
                    if bot_state != "in_match" and movement_held_keys:
                        try:
                            for held_key in list(movement_held_keys):
                                try:
                                    page.keyboard.up(held_key)
                                except Exception:
                                    pass
                            movement_held_keys.clear()
                        except Exception:
                            movement_held_keys.clear()
                    if bot_state != "in_match":
                        forced_move_queue.clear()
                    if args.bot_log_ui_state:
                        print(f"[BOT][UI] state={bot_state} reason={state_reason} ui={ui_state}")
                    if args.bot_visual_cursor and (
                        (now_mono - cursor_probe_last_log_at) >= max(0.5, float(args.bot_cursor_log_interval_sec))
                    ):
                        probe = read_bot_cursor_probe(game_actual_frame)
                        moves_now = int(probe.get("moves", 0))
                        delta_moves = moves_now - int(cursor_probe_last_moves)
                        print(
                            "[BOT][CURSOR] "
                            f"visible={probe.get('visible', False)} "
                            f"moves={moves_now} "
                            f"(+{delta_moves}) "
                            f"pos=({float(probe.get('lastX', 0.0)):.1f},{float(probe.get('lastY', 0.0)):.1f}) "
                            f"source={probe.get('lastSource', '')}"
                        )
                        cursor_probe_last_moves = moves_now
                        cursor_probe_last_log_at = now_mono

                    input_probe_now = read_input_feedback_probe(game_actual_frame)
                    click_probe_now = read_click_probe(game_actual_frame)
                    cursor_probe_now = read_bot_cursor_probe(game_actual_frame)
                    if args.bot_debug_hud:
                        update_bot_debug_hud(
                            game_actual_frame,
                            {
                                "state": bot_state,
                                "reason": state_reason,
                                "action": last_action_label,
                                "key_down": int(input_probe_now.get("keyDown", 0)),
                                "key_up": int(input_probe_now.get("keyUp", 0)),
                                "last_key": str(input_probe_now.get("lastKeyDown", "") or "-"),
                                "pointer_down": int(input_probe_now.get("pointerDown", 0)),
                                "pointer_up": int(input_probe_now.get("pointerUp", 0)),
                                "pointer_move": int(input_probe_now.get("pointerMove", 0)),
                                "cursor_moves": int(cursor_probe_now.get("moves", 0)),
                                "cursor_source": str(cursor_probe_now.get("lastSource", "") or "-"),
                                "click0": int(click_probe_now.get("click0", 0)),
                                "click_target": str(click_probe_now.get("lastTarget", "") or "-"),
                                "enemy_seen": 1 if enemy_for_feedback.get("detected") else 0,
                                "enemy_conf": float(enemy_for_feedback.get("confidence", 0.0)),
                                "enemy_dir": str(enemy_for_feedback.get("dir", "") or "-"),
                                "visual_state": str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown"),
                                "visual_conf": float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0),
                                "damage_done": float(bot_event_signals.get("damage_done_total", 0.0) or 0.0),
                                "damage_taken": float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0),
                                "mana": float(ability_state.get("mana", 0.0) or 0.0),
                                "zone_counter": (
                                    "-"
                                    if float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0) < 0
                                    else f"{float(bot_event_signals.get('zone_countdown_sec', 0.0)):.1f}s"
                                ),
                                "death_cause": str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                "death_conf": float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                "death_source": str(bot_event_signals.get("death_cause_source", "none") or "none"),
                                "safe_zone": (
                                    f"({float(bot_event_signals.get('safe_zone_x') or 0.0):.1f},"
                                    f"{float(bot_event_signals.get('safe_zone_y') or 0.0):.1f},"
                                    f"r={float(bot_event_signals.get('safe_zone_radius') or 0.0):.1f})"
                                    if (bot_event_signals.get("safe_zone_x") is not None and bot_event_signals.get("safe_zone_y") is not None)
                                    else "(-,-,-)"
                                ),
                                "ability_last": str(ability_state.get("last_used_key", "") or "-"),
                                "ability_class": str(ability_state.get("last_used_class", "") or "-"),
                                "dash_cd": max(0.0, float(ability_state.get("next_sprint_at", 0.0) or 0.0) - now_mono),
                                "active_keys": sorted(active_keys_runtime),
                                "feed_line": (
                                    f"state={bot_state} action={last_action_label} "
                                    f"ok={1 if last_action_ok else 0} "
                                    f"motion={('NA' if move_motion_score_for_feedback is None else f'{move_motion_score_for_feedback:.2f}')}"
                                ),
                            },
                        )
                    active_keys_runtime.clear()
                    last_action_label = f"{bot_state}:observe"
                    last_action_ok = True

                    if args.bot_run_until_end:
                        elapsed_run = now_mono - run_started_at
                        max_run_sec = max(0, int(args.bot_run_max_sec))
                        death_ts_now = float(bot_event_signals.get("death_ts", 0.0) or 0.0)
                        if death_ts_now > 0.0:
                            refresh_death_cause_context(
                                now_mono=now_mono,
                                enemy_recent=bool(enemy_for_feedback.get("recent", enemy_for_feedback.get("detected", False))),
                                zone_outside_safe=bool(bot_event_signals.get("zone_outside_safe", False)),
                                zone_toxic_detected=bool(bot_event_signals.get("zone_toxic_detected", False)),
                                visual_feedback=visual_feedback_for_event,
                            )
                            if death_pending_since <= 0.0:
                                death_pending_since = now_mono
                            salon_clicked = False
                            if (now_mono - last_salon_attempt_at) >= 0.8:
                                death_salon_attempted = True
                                salon_clicked = click_first_visible_in_frame(
                                    game_actual_frame,
                                    selectors=[
                                        "button:has-text('SALON')",
                                        "button:has-text('SALÓN')",
                                        "[role='button']:has-text('SALON')",
                                        "[role='button']:has-text('SALÓN')",
                                        "text=/SAL[OÓ]N/i",
                                        "button:has-text('Lobby')",
                                    ],
                                    label="salon_on_death",
                                    timeout_ms=max(500, int(args.bot_action_timeout_ms)),
                                    page_obj=page,
                                    iframe_selector=iframe_selector,
                                    click_mode=args.bot_click_mode,
                                    mouse_move_steps=args.bot_mouse_move_steps,
                                    visual_cursor=args.bot_visual_cursor,
                                )
                                if not salon_clicked:
                                    for fallback_y_ratio in (0.84, 0.76):
                                        fallback_target = get_canvas_target(
                                            page_obj=page,
                                            frame_obj=game_actual_frame,
                                            iframe_selector=iframe_selector,
                                            x_ratio=0.50,
                                            y_ratio=fallback_y_ratio,
                                        )
                                        if fallback_target is None:
                                            continue
                                        salon_clicked, used_mode = perform_attack_click(
                                            page_obj=page,
                                            frame_obj=game_actual_frame,
                                            target=fallback_target,
                                            click_mode=args.bot_click_mode,
                                            mouse_move_steps=args.bot_mouse_move_steps,
                                            visual_cursor=args.bot_visual_cursor,
                                            allow_unverified_mouse=True,
                                        )
                                        if salon_clicked:
                                            print(
                                                "[BOT][DEATH] Click fallback para SALON "
                                                f"(y_ratio={fallback_y_ratio:.2f}) via {used_mode}."
                                            )
                                            break
                                last_salon_attempt_at = now_mono
                                if salon_clicked:
                                    print("[BOT][DEATH] Click en SALON confirmado antes de cerrar corrida.")
                            death_grace_sec = max(
                                3.0,
                                min(12.0, (float(args.bot_action_timeout_ms) / 1000.0) * 8.0),
                            )
                            if salon_clicked or ((now_mono - death_pending_since) >= death_grace_sec):
                                if (not salon_clicked) and death_salon_attempted:
                                    print("[BOT][DEATH][WARN] No se pudo clicar SALON antes de cerrar corrida.")
                                run_stop_reason = "death_event"
                                print(
                                    "[BOT][RUN] Fin detectado por muerte del bot. "
                                    f"cause={str(bot_event_signals.get('death_cause', 'unknown') or 'unknown')} "
                                    f"conf={float(bot_event_signals.get('death_cause_confidence', 0.0) or 0.0):.2f}"
                                )
                                break
                        else:
                            death_pending_since = 0.0
                        if (
                            (not bool(args.bot_run_stop_on_death_only))
                            and float(bot_event_signals.get("match_end_ts", 0.0) or 0.0) > 0.0
                        ):
                            run_stop_reason = "match_end_event"
                            print("[BOT][RUN] Fin detectado por match_end.")
                            break
                        if max_run_sec > 0 and elapsed_run >= float(max_run_sec):
                            run_stop_reason = "max_runtime_reached"
                            print(
                                "[BOT][RUN][WARN] Fin por max runtime "
                                f"({int(elapsed_run)}s)."
                            )
                            break

                    if bot_state == "lobby":
                        if play_transition_started_at is not None and (now_mono - play_transition_started_at) > 12.0:
                            print("[BOT][WARN] Click en JUGAR sin transicion; reintentando.")
                            play_transition_started_at = None

                        did_ui_action = False
                        death_visual_active = (
                            is_recent_signal(bot_event_signals.get("visual_ocr_ts", 0.0), 16.0, now_mono)
                            and str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown").lower() == "death"
                            and float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0) >= 0.62
                        )
                        death_signal_active = bool(
                            float(bot_event_signals.get("death_ts", 0.0) or 0.0) > 0.0 or death_visual_active
                        )
                        if (
                            (not did_ui_action)
                            and death_signal_active
                            and ((now_mono - last_salon_attempt_at) >= 0.8)
                        ):
                            last_action_label = "lobby:salon_click"
                            did_ui_action = click_first_visible_in_frame(
                                game_actual_frame,
                                selectors=[
                                    "button:has-text('SALON')",
                                    "button:has-text('SALÓN')",
                                    "[role='button']:has-text('SALON')",
                                    "[role='button']:has-text('SALÓN')",
                                    "text=/SAL[OÓ]N/i",
                                    "button:has-text('Lobby')",
                                ],
                                label="salon",
                                timeout_ms=max(450, int(args.bot_action_timeout_ms)),
                                page_obj=page,
                                iframe_selector=iframe_selector,
                                click_mode=args.bot_click_mode,
                                mouse_move_steps=args.bot_mouse_move_steps,
                                visual_cursor=args.bot_visual_cursor,
                            )
                            if not did_ui_action:
                                for fallback_y_ratio in (0.84, 0.76):
                                    fallback_target = get_canvas_target(
                                        page_obj=page,
                                        frame_obj=game_actual_frame,
                                        iframe_selector=iframe_selector,
                                        x_ratio=0.50,
                                        y_ratio=fallback_y_ratio,
                                    )
                                    if fallback_target is None:
                                        continue
                                    did_ui_action, used_mode = perform_attack_click(
                                        page_obj=page,
                                        frame_obj=game_actual_frame,
                                        target=fallback_target,
                                        click_mode=args.bot_click_mode,
                                        mouse_move_steps=args.bot_mouse_move_steps,
                                        visual_cursor=args.bot_visual_cursor,
                                        allow_unverified_mouse=True,
                                    )
                                    if did_ui_action:
                                        print(
                                            "[BOT] SALON fallback click "
                                            f"(y_ratio={fallback_y_ratio:.2f}) via {used_mode}."
                                        )
                                        break
                            last_action_ok = bool(did_ui_action)
                            last_salon_attempt_at = now_mono
                            if did_ui_action:
                                bot_event_signals["death_ts"] = 0.0
                                print("[BOT] SALON detectado y clickeado para volver al lobby.")

                        if (not did_ui_action) and ui_state.get("character_visible"):
                            last_action_label = "lobby:character_select"
                            did_ui_action = click_first_visible_in_frame(
                                game_actual_frame,
                                selectors=[
                                    "div.character-select-view div.character-container",
                                    "div.character-container",
                                ],
                                label="character",
                                timeout_ms=args.bot_action_timeout_ms,
                                page_obj=page,
                                iframe_selector=iframe_selector,
                                click_mode=args.bot_click_mode,
                                mouse_move_steps=args.bot_mouse_move_steps,
                                visual_cursor=args.bot_visual_cursor,
                            )
                            last_action_ok = bool(did_ui_action)

                        if (not did_ui_action) and ui_state.get("select_visible"):
                            last_action_label = "lobby:confirm_select"
                            did_ui_action = click_first_visible_in_frame(
                                game_actual_frame,
                                selectors=[
                                    "div.character-select-view div.select-button",
                                    "button:has-text('Select')",
                                ],
                                label="select",
                                timeout_ms=args.bot_action_timeout_ms,
                                page_obj=page,
                                iframe_selector=iframe_selector,
                                click_mode=args.bot_click_mode,
                                mouse_move_steps=args.bot_mouse_move_steps,
                                visual_cursor=args.bot_visual_cursor,
                            )
                            last_action_ok = bool(did_ui_action)

                        if (not did_ui_action) and args.bot_visual_cursor:
                            pre_click_target = add_idle_wiggle_target(
                                cached_play_target,
                                now_mono=now_mono,
                                amplitude_px=args.bot_cursor_idle_amplitude,
                            )
                            if pre_click_target is None:
                                pre_click_target = build_cursor_patrol_target(
                                    page_obj=page,
                                    frame_obj=game_actual_frame,
                                    iframe_selector=iframe_selector,
                                    now_mono=now_mono,
                                )
                            if pre_click_target is not None:
                                move_bot_cursor_overlay(
                                    game_actual_frame,
                                    frame_x=float(pre_click_target["frame_x"]),
                                    frame_y=float(pre_click_target["frame_y"]),
                                    source=str(pre_click_target.get("source", "lobby_pre_click")),
                                )

                        if (not did_ui_action) and (now_mono - last_play_attempt_at) >= 1.0:
                            clicked_play = False
                            last_action_label = "lobby:play_click_dom"
                            clicked_play = click_first_visible_in_frame(
                                game_actual_frame,
                                selectors=[
                                    "div.game-view div.play-button",
                                    "button:has-text('Play')",
                                    "button:has-text('JUGAR')",
                                    "div.play-button",
                                ],
                                label="play",
                                timeout_ms=args.bot_action_timeout_ms,
                                page_obj=page,
                                iframe_selector=iframe_selector,
                                click_mode=args.bot_click_mode,
                                mouse_move_steps=args.bot_mouse_move_steps,
                                visual_cursor=args.bot_visual_cursor,
                            )
                            last_action_ok = bool(clicked_play)
                            if (not clicked_play) and cached_play_target is not None:
                                print(
                                    "[BOT] Fallback JUGAR target "
                                    f"source={cached_play_target.get('source')} "
                                    f"page=({cached_play_target['page_x']:.1f},{cached_play_target['page_y']:.1f}) "
                                    f"frame=({cached_play_target['frame_x']:.1f},{cached_play_target['frame_y']:.1f})"
                                )
                                clicked_play, play_mode = perform_attack_click(
                                    page_obj=page,
                                    frame_obj=game_actual_frame,
                                    target=cached_play_target,
                                    click_mode=args.bot_click_mode,
                                    mouse_move_steps=args.bot_mouse_move_steps,
                                    visual_cursor=args.bot_visual_cursor,
                                    allow_unverified_mouse=True,
                                )
                                if clicked_play:
                                    last_action_label = f"lobby:play_click_fallback:{play_mode}"
                                    last_action_ok = True
                                    print(
                                        f"[BOT] JUGAR detectado/clic via {play_mode} "
                                        f"(source={cached_play_target.get('source', 'unknown')})"
                                    )
                                else:
                                    last_action_label = "lobby:play_click_fallback_failed"
                                    last_action_ok = False
                            if clicked_play:
                                play_transition_started_at = time.monotonic()
                                print("[BOT] Click JUGAR enviado; esperando transicion a loading/in_match.")
                            last_play_attempt_at = now_mono
                            if (not clicked_play) and args.bot_visual_cursor:
                                idle_target = add_idle_wiggle_target(
                                    cached_play_target,
                                    now_mono=now_mono,
                                    amplitude_px=args.bot_cursor_idle_amplitude,
                                )
                                if idle_target is None:
                                    idle_target = build_cursor_patrol_target(
                                        page_obj=page,
                                        frame_obj=game_actual_frame,
                                        iframe_selector=iframe_selector,
                                        now_mono=now_mono,
                                    )
                                if idle_target is not None:
                                    move_bot_cursor_overlay(
                                        game_actual_frame,
                                        frame_x=float(idle_target["frame_x"]),
                                        frame_y=float(idle_target["frame_y"]),
                                        source=str(idle_target.get("source", "lobby_idle")),
                                    )
                        elif (not did_ui_action) and args.bot_visual_cursor:
                            idle_target = add_idle_wiggle_target(
                                cached_play_target,
                                now_mono=now_mono,
                                amplitude_px=args.bot_cursor_idle_amplitude,
                            )
                            if idle_target is None:
                                idle_target = build_cursor_patrol_target(
                                    page_obj=page,
                                    frame_obj=game_actual_frame,
                                    iframe_selector=iframe_selector,
                                    now_mono=now_mono,
                                )
                            if idle_target is not None:
                                move_bot_cursor_overlay(
                                    game_actual_frame,
                                    frame_x=float(idle_target["frame_x"]),
                                    frame_y=float(idle_target["frame_y"]),
                                    source=str(idle_target.get("source", "lobby_idle")),
                                )

                    elif bot_state == "loading":
                        last_action_label = "loading:wait"
                        last_action_ok = True
                        if play_transition_started_at is not None and (now_mono - play_transition_started_at) > 55.0:
                            print("[BOT][WARN] Loading prolongado; reiniciando intento de JUGAR.")
                            play_transition_started_at = None
                        can_retry_play = (
                            (play_transition_started_at is None)
                            and ((now_mono - last_play_attempt_at) >= 1.1)
                        )
                        if can_retry_play:
                            loading_ui_play = bool(
                                ui_state.get("play_visible")
                                or ui_state.get("select_visible")
                                or ui_state.get("character_visible")
                            )
                            loading_play_source = str(
                                cached_play_target.get("source", "")
                                if isinstance(cached_play_target, dict)
                                else ""
                            )
                            loading_play_conf = float(
                                cached_play_target.get("confidence", 0.0) or 0.0
                            ) if isinstance(cached_play_target, dict) else 0.0
                            loading_has_play_target = bool(
                                loading_ui_play
                                or loading_play_source == "dom_text_play"
                                or (
                                    loading_play_source == "vision_yellow_button"
                                    and loading_play_conf >= 0.66
                                )
                            )
                            if loading_has_play_target:
                                clicked_play = click_first_visible_in_frame(
                                    game_actual_frame,
                                    selectors=[
                                        "div.game-view div.play-button",
                                        "button:has-text('Play')",
                                        "button:has-text('JUGAR')",
                                        "div.play-button",
                                    ],
                                    label="play_loading_recovery",
                                    timeout_ms=args.bot_action_timeout_ms,
                                    page_obj=page,
                                    iframe_selector=iframe_selector,
                                    click_mode=args.bot_click_mode,
                                    mouse_move_steps=args.bot_mouse_move_steps,
                                    visual_cursor=args.bot_visual_cursor,
                                )
                                play_mode = "dom"
                                if (not clicked_play) and isinstance(cached_play_target, dict):
                                    clicked_play, play_mode = perform_attack_click(
                                        page_obj=page,
                                        frame_obj=game_actual_frame,
                                        target=cached_play_target,
                                        click_mode=args.bot_click_mode,
                                        mouse_move_steps=args.bot_mouse_move_steps,
                                        visual_cursor=args.bot_visual_cursor,
                                        allow_unverified_mouse=True,
                                    )
                                if clicked_play:
                                    play_transition_started_at = time.monotonic()
                                    last_action_label = f"loading:play_recovery:{play_mode}"
                                    last_action_ok = True
                                    print(
                                        "[BOT][LOADING] Click JUGAR recovery "
                                        f"source={loading_play_source}:{loading_play_conf:.2f}"
                                    )
                                else:
                                    last_action_label = "loading:play_recovery_failed"
                                    last_action_ok = False
                                last_play_attempt_at = now_mono
                        if args.bot_visual_cursor:
                            loading_target = build_cursor_patrol_target(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                                now_mono=now_mono,
                            )
                            loading_target = add_idle_wiggle_target(
                                loading_target,
                                now_mono=now_mono,
                                amplitude_px=max(2.0, float(args.bot_cursor_idle_amplitude) * 0.7),
                            )
                            if loading_target is not None:
                                move_bot_cursor_overlay(
                                    game_actual_frame,
                                    frame_x=float(loading_target["frame_x"]),
                                    frame_y=float(loading_target["frame_y"]),
                                    source=str(loading_target.get("source", "loading_idle")),
                                )
                        loading_sleep = max(
                            0.05 if bool(args.bot_realtime_mode) else 0.2,
                            float(args.bot_ui_poll_ms) / 3000.0,
                        )
                        time.sleep(loading_sleep)
                        generate_report_if_needed(page)
                        continue

                    else:
                        if play_transition_started_at is not None:
                            print("[BOT] Transicion confirmada: in_match.")
                            play_transition_started_at = None

                        escape_mode = (move_escape_remaining > 0) or (len(forced_move_queue) > 0)
                        if args.bot_parallel_smoke:
                            key_to_press = smoke_move_keys[smoke_move_index % len(smoke_move_keys)]
                        else:
                            key_to_press = random.choice(keys)
                        smoke_move_index += 1

                        enemy_scan_interval_sec = max(
                            0.12, float(args.bot_enemy_vision_interval_ms) / 1000.0
                        )
                        if bool(args.bot_enemy_vision) and (
                            (now_mono - enemy_scan_last_at) >= enemy_scan_interval_sec
                        ):
                            enemy_signal_cache = detect_enemy_signal_from_canvas(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                                red_ratio_threshold=float(args.bot_enemy_red_ratio_threshold),
                                min_contour_area=float(args.bot_enemy_min_area),
                            )
                            enemy_scan_last_at = now_mono
                            if bool(enemy_signal_cache.get("detected", False)):
                                enemy_last_seen_at = now_mono
                        enemy_recent = bool(enemy_signal_cache.get("detected", False)) or (
                            (enemy_last_seen_at > 0.0) and ((now_mono - enemy_last_seen_at) <= 1.4)
                        )
                        enemy_conf_raw = float(enemy_signal_cache.get("confidence", 0.0) or 0.0)
                        enemy_conf_ema_prev = float(enemy_signal_cache.get("_ema_conf", enemy_conf_raw) or enemy_conf_raw)
                        enemy_conf_ema = (0.62 * enemy_conf_ema_prev) + (0.38 * enemy_conf_raw)
                        enemy_signal_cache["_ema_conf"] = enemy_conf_ema
                        if (not bool(enemy_signal_cache.get("detected", False))) and enemy_conf_ema < 0.18:
                            enemy_recent = False
                        elif bool(enemy_signal_cache.get("detected", False)) and enemy_conf_ema < 0.12:
                            enemy_recent = False
                        enemy_for_feedback = dict(enemy_signal_cache)
                        enemy_for_feedback["recent"] = enemy_recent

                        configured_hold_ms = max(
                            int(args.bot_move_base_hold_ms),
                            int(args.bot_smoke_move_hold_ms) if args.bot_parallel_smoke else int(args.bot_move_base_hold_ms),
                        )
                        base_hold_seconds = max(0.08, float(configured_hold_ms) / 1000.0)
                        zone_countdown_now = float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0)
                        safe_zone_known = (
                            bot_event_signals.get("safe_zone_x") is not None
                            and bot_event_signals.get("safe_zone_y") is not None
                        )
                        if safe_zone_known:
                            knowledge_zone_observed_steps += 1
                        zone_pressure_high = zone_countdown_now >= 0.0 and zone_countdown_now <= 12.0
                        current_damage_taken_total_step = float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0)
                        damage_taken_delta_step = max(0.0, current_damage_taken_total_step - float(knowledge_prev_damage_taken))
                        visual_toxic_hits = int(visual_feedback_for_event.get("toxic_zone_hits", 0) or 0)
                        visual_safe_hits = int(visual_feedback_for_event.get("safe_zone_hits", 0) or 0)
                        visual_toxic_ratio = float(visual_feedback_for_event.get("toxic_color_ratio", 0.0) or 0.0)
                        visual_toxic_top_ratio = float(visual_feedback_for_event.get("toxic_top_ratio", 0.0) or 0.0)
                        visual_toxic_bottom_ratio = float(visual_feedback_for_event.get("toxic_bottom_ratio", 0.0) or 0.0)
                        visual_toxic_left_ratio = float(visual_feedback_for_event.get("toxic_left_ratio", 0.0) or 0.0)
                        visual_toxic_right_ratio = float(visual_feedback_for_event.get("toxic_right_ratio", 0.0) or 0.0)

                        zone_signal_source_now = str(bot_event_signals.get("zone_signal_source", "none") or "none")
                        player_pos_known = (
                            bot_event_signals.get("player_pos_x") is not None
                            and bot_event_signals.get("player_pos_y") is not None
                        )
                        zone_distance = None
                        zone_outside_safe_now = bool(bot_event_signals.get("zone_outside_safe", False))
                        zone_move_candidates: List[List[str]] = []
                        if safe_zone_known and player_pos_known:
                            try:
                                safe_x = float(bot_event_signals.get("safe_zone_x"))
                                safe_y = float(bot_event_signals.get("safe_zone_y"))
                                player_x = float(bot_event_signals.get("player_pos_x"))
                                player_y = float(bot_event_signals.get("player_pos_y"))
                                if max(abs(safe_x), abs(safe_y), abs(player_x), abs(player_y)) <= 200000.0:
                                    zone_distance = math.sqrt(((safe_x - player_x) ** 2.0) + ((safe_y - player_y) ** 2.0))
                                    radius_now_raw = bot_event_signals.get("safe_zone_radius")
                                    if radius_now_raw is not None:
                                        try:
                                            radius_now = float(radius_now_raw)
                                            if radius_now > 0.0 and zone_distance is not None:
                                                zone_outside_safe_now = bool(zone_distance > (radius_now * 1.03))
                                        except Exception:
                                            pass
                                    zone_move_candidates = build_zone_recovery_move_candidates(
                                        player_x=player_x,
                                        player_y=player_y,
                                        safe_x=safe_x,
                                        safe_y=safe_y,
                                    )
                            except Exception:
                                pass

                        zone_toxic_proxy = (
                            (damage_taken_delta_step >= 0.9)
                            and (not enemy_recent)
                            and (zone_pressure_high or safe_zone_known)
                        )
                        zone_toxic_visual = visual_toxic_hits > 0 or visual_toxic_ratio >= 0.028
                        visual_zone_move_candidates: List[List[str]] = []
                        if zone_toxic_visual and (not zone_move_candidates):
                            visual_zone_move_candidates = build_visual_toxic_recovery_move_candidates(
                                toxic_top_ratio=visual_toxic_top_ratio,
                                toxic_bottom_ratio=visual_toxic_bottom_ratio,
                                toxic_left_ratio=visual_toxic_left_ratio,
                                toxic_right_ratio=visual_toxic_right_ratio,
                            )
                            if visual_zone_move_candidates:
                                zone_move_candidates.extend(visual_zone_move_candidates[:4])
                        zone_toxic_active_now = (
                            zone_toxic_proxy
                            or zone_toxic_visual
                            or zone_outside_safe_now
                            or bool(bot_event_signals.get("zone_toxic_detected", False))
                        )
                        if zone_toxic_active_now:
                            bot_event_signals["zone_toxic_detected"] = True
                            bot_event_signals["zone_toxic_confidence"] = max(
                                float(bot_event_signals.get("zone_toxic_confidence", 0.0) or 0.0),
                                min(1.0, (0.32 if zone_toxic_proxy else 0.0) + (0.26 if zone_outside_safe_now else 0.0) + (0.16 * visual_toxic_hits) + (1.25 * visual_toxic_ratio)),
                            )
                            bot_event_signals["zone_signal_ts"] = now_mono
                            if zone_toxic_visual:
                                zone_signal_source_now = "vision"
                            elif zone_outside_safe_now:
                                zone_signal_source_now = "event"
                            elif visual_zone_move_candidates:
                                zone_signal_source_now = "vision_border"
                            else:
                                zone_signal_source_now = zone_signal_source_now if zone_signal_source_now != "none" else "inferred"
                            bot_event_signals["zone_signal_source"] = zone_signal_source_now
                        elif visual_safe_hits > 0:
                            bot_event_signals["zone_signal_source"] = "vision"
                            bot_event_signals["zone_signal_ts"] = now_mono
                            zone_signal_source_now = "vision"
                        bot_event_signals["zone_outside_safe"] = bool(zone_outside_safe_now)
                        zone_escape_mode = bool(
                            zone_toxic_active_now
                            and (safe_zone_known or zone_pressure_high or zone_outside_safe_now or bool(visual_zone_move_candidates))
                        )
                        if bot_decision_backend == "lms_re" and lms_re_policy is not None:
                            lms_obs = build_lms_re_observation_from_live_signals(
                                tick_id=max(0, int(knowledge_step_idx) + int(smoke_move_index)),
                                bot_event_signals_ref=bot_event_signals,
                                enemy_signal_ref=enemy_signal_cache,
                                enemy_recent=bool(enemy_recent),
                                ability_state_ref=ability_state,
                                now_mono=now_mono,
                            )
                            if lms_obs is not None:
                                try:
                                    lms_action = lms_re_policy.act(lms_obs)
                                    lms_re_move_combo = lms_re_action_to_move_combo(lms_action)
                                    lms_re_ability_key = lms_re_action_to_ability_key(lms_action)
                                    lms_re_action_desc = (
                                        f"mv={'+'.join(lms_re_move_combo or ['-'])} "
                                        f"ab={lms_re_ability_key or '-'} "
                                        f"fire={1 if bool(getattr(lms_action, 'fire', False)) else 0}"
                                    )
                                except Exception as exc:
                                    if (now_mono - lms_re_last_error_at) >= 4.0:
                                        print(f"[BOT][DECISION][WARN] lms_re act fallo: {exc}")
                                        lms_re_last_error_at = now_mono

                        enemy_near_now = bool(enemy_signal_cache.get("near", False))
                        opening_phase_active = now_mono < float(match_opening_until)

                        policy_context_move = build_policy_context_key(
                            bot_state=bot_state,
                            enemy_detected=enemy_recent,
                            enemy_near=enemy_near_now,
                            enemy_dir=str(enemy_signal_cache.get("dir", "CENTER") or "CENTER"),
                            escape_mode=escape_mode,
                            zone_countdown_sec=zone_countdown_now,
                            visual_state_hint=str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown"),
                            zone_escape_mode=zone_escape_mode,
                            zone_outside_safe=zone_outside_safe_now,
                            zone_signal_source=zone_signal_source_now,
                        )
                        move_candidates: List[List[str]] = []
                        move_policy_score = 0.0
                        move_policy_samples = 0
                        if forced_move_queue:
                            key_combo = normalize_move_combo(list(forced_move_queue.popleft()))
                            escape_mode = True
                            move_policy_score = -0.025
                        elif opening_phase_active and (not enemy_recent) and (not escape_mode) and (bot_decision_backend != "lms_re"):
                            opening_pattern: List[List[str]] = [
                                ["KeyW", "KeyD"],
                                ["KeyW"],
                                ["KeyW", "KeyA"],
                                ["KeyW"],
                                ["KeyD"],
                                ["KeyA"],
                            ]
                            key_combo = normalize_move_combo(
                                opening_pattern[opening_move_index % len(opening_pattern)]
                            )
                            opening_move_index += 1
                            move_policy_score = 0.12
                        else:
                            if zone_escape_mode and zone_move_candidates:
                                move_candidates.extend(zone_move_candidates[:4])
                                move_policy_score = max(move_policy_score, 0.18)
                            if lms_re_move_combo:
                                lms_combo_norm = normalize_move_combo(list(lms_re_move_combo))
                                if zone_escape_mode:
                                    move_candidates.append(lms_combo_norm)
                                else:
                                    move_candidates.insert(0, lms_combo_norm)
                                move_policy_score = max(move_policy_score, 0.24)
                            if enemy_recent and (not zone_escape_mode):
                                enemy_primary = str(enemy_signal_cache.get("direction_key", "KeyW") or "KeyW")
                                move_candidates.append(
                                    choose_in_match_move_combo(
                                        primary_key=enemy_primary,
                                        step_index=smoke_move_index,
                                        use_diagonal=True,
                                        escape_mode=escape_mode,
                                    )
                                )
                                strafe_key = ORTHOGONAL_STRAFE.get(enemy_primary, "KeyD")
                                retreat_key = OPPOSITE_KEY.get(enemy_primary, enemy_primary)
                                move_candidates.append(normalize_move_combo([strafe_key, retreat_key]))
                                move_candidates.append(normalize_move_combo([retreat_key]))
                            elif enemy_recent and zone_escape_mode:
                                enemy_primary = str(enemy_signal_cache.get("direction_key", "KeyW") or "KeyW")
                                retreat_key = OPPOSITE_KEY.get(enemy_primary, enemy_primary)
                                move_candidates.append(normalize_move_combo([retreat_key]))
                                move_candidates.append(normalize_move_combo([ORTHOGONAL_STRAFE.get(retreat_key, "KeyA"), retreat_key]))
                            else:
                                move_candidates.append(
                                    choose_in_match_move_combo(
                                        primary_key=key_to_press,
                                        step_index=smoke_move_index,
                                        use_diagonal=bool(args.bot_move_diagonal),
                                        escape_mode=escape_mode,
                                    )
                                )
                                patrol_key = smoke_move_keys[(smoke_move_index + 1) % len(smoke_move_keys)]
                                move_candidates.append(
                                    choose_in_match_move_combo(
                                        primary_key=patrol_key,
                                        step_index=smoke_move_index + 1,
                                        use_diagonal=bool(args.bot_move_diagonal),
                                        escape_mode=escape_mode,
                                    )
                                )
                                if zone_pressure_high:
                                    move_candidates.append(normalize_move_combo(["KeyW", "KeyD"]))
                                    move_candidates.append(normalize_move_combo(["KeyW", "KeyA"]))
                            if zone_escape_mode:
                                move_candidates.append(normalize_move_combo(["KeyW", "KeyD"]))
                                move_candidates.append(normalize_move_combo(["KeyW", "KeyA"]))
                            move_candidate_keys = ["+".join(normalize_move_combo(c)) for c in move_candidates if c]
                            combined_ctx_penalties: Dict[Tuple[str, str], float] = {}
                            combined_action_penalties: Dict[str, float] = {}
                            combined_map_action_penalties: Dict[Tuple[str, str], float] = {}
                            map_name_now = str(bot_event_signals.get("map_name", "") or "").strip().lower()
                            for action_candidate in move_candidate_keys:
                                ctx_key = (policy_context_move, action_candidate)
                                combined_ctx_penalties[ctx_key] = (
                                    float(historical_move_context_penalties.get(ctx_key, 0.0) or 0.0)
                                    + float(runtime_move_context_penalties.get(ctx_key, 0.0) or 0.0)
                                )
                                combined_action_penalties[action_candidate] = (
                                    float(historical_move_action_penalties.get(action_candidate, 0.0) or 0.0)
                                    + float(runtime_move_action_penalties.get(action_candidate, 0.0) or 0.0)
                                )
                                if map_name_now:
                                    map_act_key = (map_name_now, action_candidate)
                                    combined_map_action_penalties[map_act_key] = (
                                        float(historical_move_map_action_penalties.get(map_act_key, 0.0) or 0.0)
                                        + float(runtime_move_map_action_penalties.get(map_act_key, 0.0) or 0.0)
                                    )
                            selected_move_key, move_policy_score, move_policy_samples = pick_action_from_policy(
                                candidates=move_candidate_keys,
                                context_key=policy_context_move,
                                policy_cache=knowledge_policy_cache,
                                min_samples=int(args.bot_knowledge_min_samples),
                                exploration=float(args.bot_knowledge_exploration),
                                context_action_penalties=combined_ctx_penalties,
                                action_penalties=combined_action_penalties,
                                map_action_penalties=combined_map_action_penalties,
                                map_name=map_name_now,
                                penalty_weight=max(0.0, float(args.bot_history_stuck_weight)),
                            )
                            if selected_move_key:
                                key_combo = normalize_move_combo(selected_move_key.split("+"))
                            elif move_candidates:
                                key_combo = normalize_move_combo(move_candidates[0])
                            else:
                                key_combo = normalize_move_combo(["KeyW"])
                        knowledge_prev_move_context = policy_context_move
                        knowledge_prev_move_action = "+".join(key_combo)
                        if knowledge_prev_move_action == last_move_action_runtime:
                            repeat_move_action_streak += 1
                        else:
                            repeat_move_action_streak = 1
                            last_move_action_runtime = knowledge_prev_move_action

                        move_combo_for_feedback = list(key_combo)
                        move_escape_for_feedback = bool(escape_mode or zone_escape_mode)
                        hold_seconds = float(base_hold_seconds)
                        if opening_phase_active and (not enemy_recent) and (not escape_mode):
                            hold_seconds = max(
                                hold_seconds * max(1.0, float(args.bot_opening_hold_multiplier)),
                                0.55,
                            )
                        if enemy_recent and (not escape_mode):
                            hold_seconds *= 1.35 if bool(enemy_signal_cache.get("near", False)) else 1.18
                        if zone_escape_mode:
                            hold_seconds = min(1.15, hold_seconds * 1.28)
                        if escape_mode:
                            hold_seconds = min(1.0, hold_seconds * 1.55)
                            move_escape_remaining = max(0, move_escape_remaining - 1)

                        print(
                            "[BOT][MOVE_SMOKE] "
                            f"keys={'+'.join(key_combo)} "
                            f"hold_ms={int(hold_seconds * 1000)} "
                            f"step={smoke_move_index} "
                            f"escape={'1' if escape_mode else '0'} "
                            f"zone_escape={'1' if zone_escape_mode else '0'} "
                            f"zone_out={'1' if zone_outside_safe_now else '0'} "
                            f"zone_src={zone_signal_source_now} "
                            f"repeat={repeat_move_action_streak} "
                            f"forcedq={len(forced_move_queue)} "
                            f"enemy={'1' if enemy_recent else '0'} "
                            f"conf={float(enemy_signal_cache.get('confidence', 0.0)):.2f} "
                            f"dir={enemy_signal_cache.get('dir', 'CENTER')} "
                            f"policy={move_policy_score:.3f}/{move_policy_samples} "
                            f"backend={bot_decision_backend} "
                            f"re={lms_re_action_desc or '-'}"
                        )

                        if enemy_recent:
                            tx = max(
                                0.10,
                                min(
                                    0.90,
                                    float(enemy_signal_cache.get("x_ratio", 0.50))
                                    + random.uniform(-0.03, 0.03),
                                ),
                            )
                            ty = max(
                                0.12,
                                min(
                                    0.90,
                                    float(enemy_signal_cache.get("y_ratio", 0.50))
                                    + random.uniform(-0.03, 0.03),
                                ),
                            )
                            aim_target = get_canvas_target(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                                x_ratio=tx,
                                y_ratio=ty,
                            )
                        else:
                            aim_target = get_canvas_target(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                                x_ratio=random.uniform(0.32, 0.68),
                                y_ratio=random.uniform(0.32, 0.68),
                            )

                        sample_every = max(1, int(args.bot_move_motion_sample_every))
                        do_motion_sample = (smoke_move_index % sample_every) == 0
                        pre_motion_frame = (
                            capture_canvas_motion_frame(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                            )
                            if do_motion_sample
                            else None
                        )

                        try:
                            focus_game_frame(page, game_actual_frame, iframe_selector)
                            if aim_target is not None:
                                if args.bot_visual_cursor:
                                    move_bot_cursor_overlay(
                                        game_actual_frame,
                                        frame_x=float(aim_target["frame_x"]),
                                        frame_y=float(aim_target["frame_y"]),
                                        source=str(aim_target.get("source", "aim_target")),
                                    )
                                page.mouse.move(
                                    float(aim_target["page_x"]),
                                    float(aim_target["page_y"]),
                                    steps=max(1, int(args.bot_mouse_move_steps)),
                                )
                            active_keys_runtime.clear()
                            sprint_engaged = False
                            if enemy_recent or escape_mode or zone_escape_mode:
                                if now_mono >= float(ability_state.get("next_sprint_at", 0.0) or 0.0):
                                    try:
                                        page.keyboard.down("Shift")
                                        active_keys_runtime.add("Shift")
                                        sprint_engaged = True
                                        ability_state["usage"]["Shift"] = int(ability_state["usage"].get("Shift", 0)) + 1
                                        ability_state["next_sprint_at"] = now_mono + 1.2
                                        ability_state["mana"] = max(
                                            0.0,
                                            float(ability_state.get("mana", 0.0) or 0.0) - 6.0,
                                        )
                                    except Exception:
                                        sprint_engaged = False
                                    try:
                                        page.mouse.down(button="right")
                                        page.mouse.up(button="right")
                                        active_keys_runtime.add("MouseRight")
                                        ability_state["usage"]["MouseRight"] = int(ability_state["usage"].get("MouseRight", 0)) + 1
                                    except Exception:
                                        pass
                            desired_keys = normalize_move_combo(key_combo)
                            desired_key_set = set(desired_keys)
                            for held_key in list(movement_held_keys):
                                if held_key not in desired_key_set:
                                    try:
                                        page.keyboard.up(held_key)
                                    except Exception:
                                        pass
                                    movement_held_keys.discard(held_key)
                            for desired_key in desired_keys:
                                if desired_key in movement_held_keys:
                                    continue
                                try:
                                    page.keyboard.down(desired_key)
                                    movement_held_keys.add(desired_key)
                                except Exception:
                                    pass
                            active_keys_runtime.update(movement_held_keys)
                            last_action_label = f"in_match:move_hold:{'+'.join(desired_keys)}"
                            time.sleep(hold_seconds)
                            if sprint_engaged:
                                try:
                                    page.keyboard.up("Shift")
                                except Exception:
                                    pass
                                active_keys_runtime.discard("Shift")
                            last_action_ok = True
                            print(f"[BOT] Moviendo en direccion: {'+'.join(desired_keys)}")
                        except Exception as exc:
                            for held_key in list(movement_held_keys):
                                try:
                                    page.keyboard.up(held_key)
                                except Exception:
                                    pass
                                movement_held_keys.discard(held_key)
                            active_keys_runtime.clear()
                            last_action_label = f"in_match:move_error:{'+'.join(key_combo)}"
                            last_action_ok = False
                            print(f"[BOT][WARN] Error en movimiento: {exc}")

                        if do_motion_sample:
                            post_motion_frame = capture_canvas_motion_frame(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                            )
                            motion_score = compute_motion_score(pre_motion_frame, post_motion_frame)
                            move_motion_score_for_feedback = motion_score
                            motion_threshold = max(0.2, float(args.bot_move_motion_threshold))
                            if motion_score is None:
                                move_last_motion_score = None
                                print("[BOT][MOVE_FEEDBACK] motion_score=NA")
                            else:
                                move_last_motion_score = motion_score
                                if motion_score < motion_threshold:
                                    move_low_motion_streak += 1
                                    collision_streak += 1
                                else:
                                    move_low_motion_streak = 0
                                    collision_streak = max(0, collision_streak - 1)
                                if move_low_motion_streak >= max(1, int(args.bot_move_stuck_streak)):
                                    move_escape_remaining = max(
                                        move_escape_remaining,
                                        max(1, int(args.bot_move_escape_steps)),
                                    )
                                    move_low_motion_streak = 0
                                    print(
                                        "[BOT][MOVE_FEEDBACK][WARN] Bajo movimiento detectado; "
                                        f"activando escape_steps={move_escape_remaining}"
                                    )
                                if collision_streak >= max(1, int(args.bot_collision_streak_threshold)) and (not enemy_recent):
                                    move_escape_remaining = max(
                                        move_escape_remaining,
                                        max(
                                            2,
                                            int(args.bot_move_escape_steps)
                                            + max(0, int(args.bot_collision_escape_extra_steps)),
                                        ),
                                    )
                                    print(
                                        "[BOT][COLLISION] Colision probable detectada; "
                                        f"collision_streak={collision_streak} "
                                        f"escape_steps={move_escape_remaining}"
                                    )
                                low_motion_for_stuck = motion_score < (
                                    motion_threshold * max(0.55, float(args.bot_stuck_motion_factor))
                                )
                                recent_dd_now = float(bot_event_signals.get("damage_done_total", 0.0) or 0.0)
                                recent_dt_now = float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0)
                                recent_dd_delta = recent_dd_now - float(knowledge_prev_damage_done)
                                recent_dt_delta = recent_dt_now - float(knowledge_prev_damage_taken)
                                repeat_signal = repeat_move_action_streak >= max(
                                    1, int(args.bot_stuck_repeat_action_streak)
                                )
                                no_progress_signal = (recent_dd_delta <= 0.0) and (recent_dt_delta <= 0.2)
                                escape_ineffective_signal = bool(escape_mode) and low_motion_for_stuck
                                signal_count = int(low_motion_for_stuck) + int(repeat_signal) + int(escape_ineffective_signal)
                                if signal_count >= 2 and no_progress_signal and (not enemy_recent):
                                    stuck_signal_streak += 1
                                else:
                                    stuck_signal_streak = max(0, stuck_signal_streak - 1)
                                if stuck_signal_streak >= max(1, int(args.bot_stuck_confirm_streak)):
                                    move_stuck_now = True
                                    stuck_signal_streak = 0
                                    stuck_event_total += 1
                                    stuck_recovery_attempt += 1
                                    recovery_steps_target = max(3, int(args.bot_stuck_recovery_steps))
                                    recovery_plan = build_stuck_recovery_plan(
                                        previous_combo=key_combo,
                                        attempt_index=stuck_recovery_attempt,
                                    )
                                    while len(recovery_plan) < recovery_steps_target:
                                        recovery_plan.extend(
                                            build_stuck_recovery_plan(
                                                previous_combo=recovery_plan[-1] if recovery_plan else key_combo,
                                                attempt_index=stuck_recovery_attempt + len(recovery_plan),
                                            )
                                        )
                                    forced_move_queue.clear()
                                    for recovery_combo in recovery_plan[:recovery_steps_target]:
                                        forced_move_queue.append(normalize_move_combo(recovery_combo))
                                    move_escape_remaining = max(move_escape_remaining, len(forced_move_queue))
                                    collision_streak = max(collision_streak, int(args.bot_collision_streak_threshold))
                                    if movement_held_keys:
                                        for held_key in list(movement_held_keys):
                                            try:
                                                page.keyboard.up(held_key)
                                            except Exception:
                                                pass
                                            movement_held_keys.discard(held_key)
                                        time.sleep(0.12)
                                    print(
                                        "[BOT][STUCK] Atasco confirmado; "
                                        f"signals={signal_count} repeat={repeat_move_action_streak} "
                                        f"motion={motion_score:.2f} dd={recent_dd_delta:.1f} dt={recent_dt_delta:.1f} "
                                        f"recovery_steps={len(forced_move_queue)} total={stuck_event_total}"
                                    )
                                update_runtime_move_penalties(
                                    stats_dict=runtime_stuck_stats,
                                    penalty_context_map=runtime_move_context_penalties,
                                    penalty_action_map=runtime_move_action_penalties,
                                    penalty_map_action_map=runtime_move_map_action_penalties,
                                    context_key=str(knowledge_prev_move_context or ""),
                                    action_key=str(knowledge_prev_move_action or ""),
                                    map_name=str(bot_event_signals.get("map_name", "") or ""),
                                    is_stuck=bool(move_stuck_now or (low_motion_for_stuck and repeat_signal)),
                                    motion_score=float(motion_score),
                                    motion_threshold=motion_threshold,
                                )
                                print(
                                    "[BOT][MOVE_FEEDBACK] "
                                    f"motion_score={motion_score:.2f} "
                                    f"threshold={motion_threshold:.2f} "
                                    f"escape_remaining={move_escape_remaining} "
                                    f"collision_streak={collision_streak}"
                                )
                        else:
                            move_motion_score_for_feedback = move_last_motion_score
                            print("[BOT][MOVE_FEEDBACK] motion_score=SKIP(sample)")

                        # Evaluate pending ability outcomes to infer real availability.
                        pending_outcomes = ability_state.get("pending_outcomes", [])
                        if isinstance(pending_outcomes, list):
                            still_pending: List[Dict[str, Any]] = []
                            for pending in pending_outcomes:
                                try:
                                    eval_at = float(pending.get("eval_at", 0.0) or 0.0)
                                except Exception:
                                    eval_at = 0.0
                                if now_mono < eval_at:
                                    still_pending.append(pending)
                                    continue
                                ability_key_eval = str(pending.get("key", "") or "")
                                dd_now = float(bot_event_signals.get("damage_done_total", 0.0) or 0.0)
                                dt_now = float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0)
                                dd_delta = dd_now - float(pending.get("damage_done_base", 0.0) or 0.0)
                                dt_delta = dt_now - float(pending.get("damage_taken_base", 0.0) or 0.0)
                                motion_now = float(move_motion_score_for_feedback or 0.0)
                                motion_threshold_eval = max(0.2, float(args.bot_move_motion_threshold))
                                if ability_key_eval == "Digit2":
                                    success_eval = motion_now >= (motion_threshold_eval * 1.15)
                                elif ability_key_eval == "KeyR":
                                    success_eval = dt_delta <= 0.2
                                else:
                                    success_eval = dd_delta >= 2.0
                                reward_eval = (dd_delta * 0.12) - (dt_delta * 0.16)
                                if ability_key_eval == "Digit2":
                                    reward_eval += min(1.6, motion_now * 0.05)
                                if not success_eval:
                                    blocked_map = ability_state.get("blocked_until", {})
                                    if isinstance(blocked_map, dict):
                                        blocked_map[ability_key_eval] = max(
                                            float(blocked_map.get(ability_key_eval, 0.0) or 0.0),
                                            now_mono + 1.35,
                                        )
                                pending_context = str(pending.get("context_key", "") or "")
                                if knowledge_conn is not None and pending_context and ability_key_eval:
                                    try:
                                        update_policy_cache_and_store(
                                            conn=knowledge_conn,
                                            policy_cache=knowledge_policy_cache,
                                            context_key=pending_context,
                                            action_key=ability_key_eval,
                                            reward=float(reward_eval),
                                            success=bool(success_eval),
                                            now_ms=int(time.time() * 1000),
                                        )
                                        knowledge_pending_writes += 1
                                    except Exception:
                                        pass
                            ability_state["pending_outcomes"] = still_pending

                        # mana + skill usage with availability signals
                        signal_mana_max = bot_event_signals.get("mana_max")
                        signal_mana_current = bot_event_signals.get("mana_current")
                        try:
                            if signal_mana_max is not None and float(signal_mana_max) > 0:
                                ability_state["max_mana"] = float(signal_mana_max)
                        except Exception:
                            pass
                        if signal_mana_current is not None:
                            try:
                                ability_state["mana"] = max(
                                    0.0,
                                    min(
                                        float(ability_state.get("max_mana", 100.0) or 100.0),
                                        float(signal_mana_current),
                                    ),
                                )
                            except Exception:
                                pass
                        else:
                            ability_state["mana"] = min(
                                float(ability_state.get("max_mana", 100.0) or 100.0),
                                float(ability_state.get("mana", 0.0) or 0.0) + 2.6,
                            )
                        dash_cd_local = max(0.0, float(ability_state.get("next_sprint_at", 0.0) or 0.0) - now_mono)
                        if isinstance(bot_event_signals.get("ability_cooldown_sec"), dict):
                            bot_event_signals["ability_cooldown_sec"]["Shift"] = float(dash_cd_local)
                        if isinstance(bot_event_signals.get("ability_ready"), dict):
                            bot_event_signals["ability_ready"]["Shift"] = bool(dash_cd_local <= 0.05)
                        bot_event_signals["ability_signal_ts"] = now_mono
                        ability_used_now = ""
                        ability_class_now = ""
                        ability_costs = {"Digit1": 20.0, "Digit2": 18.0, "Digit3": 24.0, "KeyR": 14.0}
                        ability_signal_recent = is_recent_signal(
                            float(bot_event_signals.get("ability_signal_ts", 0.0) or 0.0),
                            8.0,
                            now_mono,
                        )
                        ready_signals = bot_event_signals.get("ability_ready", {})
                        cooldown_signals = bot_event_signals.get("ability_cooldown_sec", {})
                        blocked_until = ability_state.get("blocked_until", {})
                        ability_context_key = build_policy_context_key(
                            bot_state=bot_state,
                            enemy_detected=enemy_recent,
                            enemy_near=bool(enemy_signal_cache.get("near", False)),
                            enemy_dir=str(enemy_signal_cache.get("dir", "CENTER") or "CENTER"),
                            escape_mode=escape_mode,
                            zone_countdown_sec=float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0),
                            visual_state_hint=str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown"),
                            zone_escape_mode=bool(zone_escape_mode),
                            zone_outside_safe=bool(bot_event_signals.get("zone_outside_safe", False)),
                            zone_signal_source=str(bot_event_signals.get("zone_signal_source", "none") or "none"),
                        )
                        mana_now_for_ctx = float(ability_state.get("mana", 0.0) or 0.0)
                        mana_max_for_ctx = max(1.0, float(ability_state.get("max_mana", 100.0) or 100.0))
                        mana_band = int(max(0.0, min(0.99, mana_now_for_ctx / mana_max_for_ctx)) * 4.0)
                        ability_context_key = f"{ability_context_key}|mana_band={mana_band}"

                        def can_use_ability_now(ability_key: str) -> Tuple[bool, str]:
                            mana_cost = float(ability_costs.get(ability_key, 20.0))
                            mana_now = float(ability_state.get("mana", 0.0) or 0.0)
                            if mana_now < mana_cost:
                                return False, "mana_low"
                            if isinstance(blocked_until, dict):
                                blocked_ts = float(blocked_until.get(ability_key, 0.0) or 0.0)
                                if now_mono < blocked_ts:
                                    return False, "blocked_backoff"
                            if ability_signal_recent and isinstance(ready_signals, dict):
                                ready_flag = ready_signals.get(ability_key)
                                if ready_flag is False:
                                    return False, "signal_not_ready"
                            if ability_signal_recent and isinstance(cooldown_signals, dict):
                                cooldown_val = cooldown_signals.get(ability_key)
                                try:
                                    cd_num = float(cooldown_val) if cooldown_val is not None else None
                                except Exception:
                                    cd_num = None
                                if cd_num is not None and cd_num > 0.35:
                                    return False, f"cd={cd_num:.2f}"
                            if ability_key == "KeyR":
                                if now_mono < float(ability_state.get("next_wall_at", 0.0) or 0.0):
                                    return False, "local_wall_cd"
                                return True, "ok"
                            if now_mono < float(ability_state.get("next_ability_at", 0.0) or 0.0):
                                return False, "local_skill_cd"
                            return True, "ok"

                        if escape_mode or zone_escape_mode:
                            base_ability_priority = ["Digit2", "KeyR", "Digit3", "Digit1"]
                        elif enemy_recent and bool(enemy_signal_cache.get("near", False)):
                            base_ability_priority = ["Digit3", "Digit1", "Digit2", "KeyR"]
                        elif enemy_recent:
                            base_ability_priority = ["Digit1", "Digit3", "Digit2", "KeyR"]
                        else:
                            base_ability_priority = ["Digit2", "Digit1", "Digit3"]
                        if (
                            bot_decision_backend == "lms_re"
                            and lms_re_ability_key in ("Digit1", "Digit2", "Digit3", "KeyR")
                        ):
                            base_ability_priority = [lms_re_ability_key] + [
                                key for key in base_ability_priority if key != lms_re_ability_key
                            ]
                        ability_pick, ability_policy_score, ability_policy_samples = pick_action_from_policy(
                            candidates=base_ability_priority,
                            context_key=ability_context_key,
                            policy_cache=knowledge_policy_cache,
                            min_samples=int(args.bot_knowledge_min_samples),
                            exploration=float(args.bot_knowledge_exploration),
                        )
                        ability_priority: List[str] = []
                        if ability_pick:
                            ability_priority.append(ability_pick)
                        ability_priority.extend(base_ability_priority)
                        dedup_priority: List[str] = []
                        for ability_key in ability_priority:
                            if ability_key not in dedup_priority:
                                dedup_priority.append(ability_key)
                        ability_priority = dedup_priority

                        ability_due = (
                            now_mono >= float(ability_state.get("next_ability_at", 0.0) or 0.0)
                            or escape_mode
                            or ((smoke_move_index % 4) == 0)
                        )
                        if ability_due:
                            skip_reason = ""
                            for candidate_key in ability_priority:
                                usable, reason = can_use_ability_now(candidate_key)
                                if not usable:
                                    if not skip_reason:
                                        skip_reason = f"{candidate_key}:{reason}"
                                    continue
                                mana_cost = float(ability_costs.get(candidate_key, 20.0))
                                try:
                                    page.keyboard.down(candidate_key)
                                    page.keyboard.up(candidate_key)
                                    ability_state["mana"] = max(
                                        0.0,
                                        float(ability_state.get("mana", 0.0) or 0.0) - mana_cost,
                                    )
                                    ability_state["usage"][candidate_key] = int(ability_state["usage"].get(candidate_key, 0)) + 1
                                    ability_state["last_used_key"] = candidate_key
                                    ability_state["last_used_class"] = (
                                        "utility_wall" if candidate_key == "KeyR" else ABILITY_CLASSIFICATION.get(candidate_key, "unknown")
                                    )
                                    ability_state["last_ability_at"] = now_mono
                                    if candidate_key == "KeyR":
                                        ability_state["next_wall_at"] = now_mono + 2.8
                                        ability_state["next_ability_at"] = max(
                                            float(ability_state.get("next_ability_at", 0.0) or 0.0),
                                            now_mono + 0.8,
                                        )
                                    elif candidate_key == "Digit2":
                                        ability_state["next_ability_at"] = now_mono + max(0.8, float(args.bot_ability_every_sec) * 0.7)
                                    else:
                                        ability_state["next_ability_at"] = now_mono + max(0.8, float(args.bot_ability_every_sec))
                                    ability_used_now = candidate_key
                                    ability_class_now = str(ability_state.get("last_used_class", "unknown") or "unknown")
                                    active_keys_runtime.add(candidate_key)
                                    if isinstance(ability_state.get("pending_outcomes"), list):
                                        ability_state["pending_outcomes"].append(
                                            {
                                                "key": candidate_key,
                                                "context_key": ability_context_key,
                                                "eval_at": now_mono + 1.25,
                                                "damage_done_base": float(
                                                    bot_event_signals.get("damage_done_total", 0.0) or 0.0
                                                ),
                                                "damage_taken_base": float(
                                                    bot_event_signals.get("damage_taken_total", 0.0) or 0.0
                                                ),
                                            }
                                        )
                                    print(
                                        "[BOT][ABILITY] "
                                        f"used={candidate_key} class={ability_class_now} "
                                        f"mana={float(ability_state.get('mana', 0.0)):.1f} "
                                        f"reason={reason} "
                                        f"policy={ability_policy_score:.3f}/{ability_policy_samples}"
                                    )
                                    break
                                except Exception:
                                    continue
                            if (not ability_used_now) and skip_reason and ((smoke_move_index % 3) == 0):
                                print(f"[BOT][ABILITY] skipped={skip_reason} mana={float(ability_state.get('mana', 0.0)):.1f}")
                        map_interval = float(args.bot_open_map_every_sec)
                        if map_interval > 0 and (now_mono - last_map_toggle_at) >= map_interval:
                            try:
                                page.keyboard.down("KeyC")
                                page.keyboard.up("KeyC")
                                active_keys_runtime.add("KeyC")
                                last_map_toggle_at = now_mono
                                print("[BOT][MAP] Toggle map (KeyC).")
                            except Exception:
                                pass

                        install_click_probe(game_actual_frame)
                        attack_every_steps = max(1, int(args.bot_move_click_every_steps))
                        should_attack = (
                            (enemy_recent and (not zone_escape_mode))
                            or escape_mode
                            or (smoke_move_index <= 1)
                            or ((smoke_move_index % attack_every_steps) == 0)
                        )
                        if zone_escape_mode and (not enemy_recent):
                            should_attack = False
                        attack_target = None
                        if should_attack and enemy_recent:
                            tx = max(
                                0.08,
                                min(
                                    0.92,
                                    float(enemy_signal_cache.get("x_ratio", 0.50))
                                    + random.uniform(-0.05, 0.05),
                                ),
                            )
                            ty = max(
                                0.10,
                                min(
                                    0.92,
                                    float(enemy_signal_cache.get("y_ratio", 0.50))
                                    + random.uniform(-0.05, 0.05),
                                ),
                            )
                            attack_target = get_canvas_target(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                iframe_selector=iframe_selector,
                                x_ratio=tx,
                                y_ratio=ty,
                            )
                        if should_attack and attack_target is None:
                            attack_target = get_attack_target(page, game_actual_frame, iframe_selector)

                        if should_attack and (attack_target is not None):
                            try:
                                page.keyboard.down("Space")
                                time.sleep(0.015)
                                page.keyboard.up("Space")
                                active_keys_runtime.add("Space")
                            except Exception:
                                pass
                            ok_click, used_mode = perform_attack_click(
                                page_obj=page,
                                frame_obj=game_actual_frame,
                                target=attack_target,
                                click_mode=args.bot_click_mode,
                                mouse_move_steps=args.bot_mouse_move_steps,
                                visual_cursor=args.bot_visual_cursor,
                            )
                            probe = read_click_probe(game_actual_frame)
                            if ok_click:
                                last_action_label = f"in_match:attack_click:{used_mode}"
                                last_action_ok = True
                                print(
                                    f"[BOT] Left click ok via {used_mode} "
                                    f"(click0={probe.get('click0')}, target={probe.get('lastTarget')})"
                                )
                            else:
                                last_action_label = f"in_match:attack_click_failed:{args.bot_click_mode}"
                                last_action_ok = False
                                print(
                                    f"[BOT][WARN] Left click no detectado "
                                    f"(mode={args.bot_click_mode}, click0={probe.get('click0')})"
                                )
                        elif should_attack and (attack_target is None):
                            last_action_label = "in_match:no_attack_target"
                            last_action_ok = False
                            print("[BOT][WARN] Sin coordenadas de ataque.")
                        else:
                            last_action_label = f"in_match:move_only:{'+'.join(key_combo)}"
                            last_action_ok = True
                            print(
                                "[BOT] Ataque omitido para priorizar movimiento "
                                f"(cadence={attack_every_steps})"
                            )
                    post_input_probe = read_input_feedback_probe(game_actual_frame)
                    post_click_probe = read_click_probe(game_actual_frame)
                    post_cursor_probe = read_bot_cursor_probe(game_actual_frame)
                    shot_path = maybe_capture_feedback_screenshot(
                        session=runtime_feedback_session,
                        page_obj=page,
                        now_mono=now_mono,
                        every_sec=float(args.bot_feedback_screenshot_every_sec),
                        max_screenshots=int(args.bot_feedback_max_screenshots),
                        label=f"{bot_state}_{last_action_label}",
                    )
                    if args.bot_visual_ocr and (not bool(visual_ocr_state.get("disabled", False))) and shot_path:
                        ocr_interval = max(0.3, float(args.bot_visual_ocr_every_sec))
                        ocr_max_frames = max(1, int(args.bot_visual_ocr_max_frames))
                        can_run_ocr = (
                            (now_mono - float(visual_ocr_state.get("last_ocr_at", 0.0))) >= ocr_interval
                            and int(visual_ocr_state.get("frames", 0)) < ocr_max_frames
                        )
                        if can_run_ocr:
                            visual_ocr_now = extract_visual_feedback_from_screenshot(
                                image_path=shot_path,
                                max_names=int(args.bot_visual_ocr_max_names),
                            )
                            visual_ocr_state["last_ocr_at"] = now_mono
                            visual_ocr_state["frames"] = int(visual_ocr_state.get("frames", 0)) + 1
                            visual_ocr_state["last_result"] = dict(visual_ocr_now)
                            visual_feedback_for_event = dict(visual_ocr_now)
                            if bool(visual_ocr_now.get("ok", False)):
                                bot_event_signals["visual_state_hint"] = str(visual_ocr_now.get("state_hint", "unknown") or "unknown")
                                bot_event_signals["visual_state_confidence"] = float(
                                    visual_ocr_now.get("state_confidence", 0.0) or 0.0
                                )
                                bot_event_signals["visual_ocr_ts"] = now_mono
                                visual_names = [
                                    str(name).strip()
                                    for name in (visual_ocr_now.get("names", []) or [])
                                    if str(name).strip()
                                ]
                                if visual_names:
                                    bot_event_signals["visual_names"] = visual_names
                                    process_detected_entities(
                                        set(visual_names),
                                        set(),
                                        source="vision_runtime_ocr",
                                    )
                                dmg_nums = [
                                    float(v)
                                    for v in (visual_ocr_now.get("damage_numbers", []) or [])
                                    if isinstance(v, (int, float))
                                ]
                                if dmg_nums:
                                    bot_event_signals["visual_damage_hint"] = max(dmg_nums)
                                safe_zone_hits_now = int(visual_ocr_now.get("safe_zone_hits", 0) or 0)
                                toxic_zone_hits_now = int(visual_ocr_now.get("toxic_zone_hits", 0) or 0)
                                toxic_color_ratio_now = float(visual_ocr_now.get("toxic_color_ratio", 0.0) or 0.0)
                                if safe_zone_hits_now > 0 and bot_state == "in_match":
                                    bot_event_signals["zone_signal_source"] = "vision"
                                    bot_event_signals["zone_signal_ts"] = now_mono
                                if bot_state == "in_match" and (toxic_zone_hits_now > 0 or toxic_color_ratio_now >= 0.028):
                                    bot_event_signals["zone_toxic_detected"] = True
                                    bot_event_signals["zone_toxic_confidence"] = max(
                                        float(bot_event_signals.get("zone_toxic_confidence", 0.0) or 0.0),
                                        min(1.0, 0.38 + (0.18 * float(toxic_zone_hits_now)) + (1.35 * toxic_color_ratio_now)),
                                    )
                                    bot_event_signals["zone_signal_source"] = "vision"
                                    bot_event_signals["zone_signal_ts"] = now_mono
                                visual_state_now = str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown").lower()
                                visual_conf_now = float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0)
                                death_hits_now = int(visual_ocr_now.get("death_hits", 0) or 0)
                                death_streak_now = int(visual_ocr_state.get("death_streak", 0) or 0)
                                if (
                                    visual_state_now == "death"
                                    and visual_conf_now >= 0.72
                                    and death_hits_now >= 2
                                ):
                                    death_streak_now += 1
                                else:
                                    death_streak_now = 0
                                visual_ocr_state["death_streak"] = death_streak_now
                                if death_streak_now >= 2:
                                    last_event_for_death = str(bot_event_signals.get("last_event_name", "") or "").lower()
                                    looks_like_respawn = any(
                                        token in last_event_for_death
                                        for token in ("round_spawn", "round_start", "waiting_area", "choosing_spawn")
                                    )
                                    if not looks_like_respawn:
                                        bot_event_signals["death_ts"] = now_mono
                                        refresh_death_cause_context(
                                            now_mono=now_mono,
                                            enemy_recent=bool(enemy_for_feedback.get("recent", enemy_for_feedback.get("detected", False))),
                                            zone_outside_safe=bool(bot_event_signals.get("zone_outside_safe", False)),
                                            zone_toxic_detected=bool(bot_event_signals.get("zone_toxic_detected", False)),
                                            visual_feedback=visual_ocr_now,
                                        )
                                        print(
                                            "[BOT][VISION] Muerte detectada visualmente "
                                            f"(streak={death_streak_now}, conf={visual_conf_now:.2f}, hits={death_hits_now})."
                                        )
                                print(
                                    "[BOT][VISION] "
                                    f"state_hint={bot_event_signals['visual_state_hint']} "
                                    f"conf={float(bot_event_signals.get('visual_state_confidence', 0.0)):.2f} "
                                    f"names={len(visual_names)} "
                                    f"damage_hint={float(bot_event_signals.get('visual_damage_hint', 0.0) or 0.0):.1f}"
                                )
                            else:
                                err_txt = str(visual_ocr_now.get("error", "") or "")
                                if err_txt and ("tesseract" in err_txt.lower() or "not found" in err_txt.lower()):
                                    visual_ocr_state["disabled"] = True
                                    print(
                                        "[BOT][VISION][WARN] OCR desactivado en runtime "
                                        f"por error: {err_txt}"
                                    )
                        else:
                            visual_feedback_for_event = (
                                dict(visual_ocr_state.get("last_result", {}))
                                if isinstance(visual_ocr_state.get("last_result"), dict)
                                else {}
                            )
                    elif isinstance(visual_ocr_state.get("last_result"), dict):
                        visual_feedback_for_event = dict(visual_ocr_state.get("last_result", {}))
                    current_damage_done_total = float(bot_event_signals.get("damage_done_total", 0.0) or 0.0)
                    current_damage_taken_total = float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0)
                    damage_done_delta = current_damage_done_total - float(knowledge_prev_damage_done)
                    damage_taken_delta = current_damage_taken_total - float(knowledge_prev_damage_taken)
                    motion_eval = (
                        float(move_motion_score_for_feedback)
                        if move_motion_score_for_feedback is not None
                        else 0.0
                    )
                    reward_now = (
                        (damage_done_delta * 0.16)
                        - (damage_taken_delta * 0.20)
                        + (motion_eval * 0.018)
                        + (0.06 if bool(last_action_ok) else -0.05)
                    )
                    if bool(zone_escape_mode):
                        reward_now += 0.05
                    if bool(zone_outside_safe_now):
                        reward_now -= 0.10
                    if bool(bot_event_signals.get("zone_toxic_detected", False)) and damage_taken_delta > 0.45:
                        reward_now -= 0.12
                    if bool(move_stuck_now):
                        reward_now -= 0.34
                    if repeat_move_action_streak >= max(3, int(args.bot_stuck_repeat_action_streak) + 1):
                        reward_now -= 0.06
                    if bool(enemy_for_feedback.get("recent")):
                        reward_now += 0.04
                    if (
                        float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0) >= 0.0
                        and bot_event_signals.get("safe_zone_x") is not None
                        and bot_event_signals.get("safe_zone_y") is not None
                    ):
                        reward_now += 0.03
                    reward_now = max(-3.5, min(3.5, reward_now))
                    knowledge_reward_sum += reward_now
                    knowledge_step_idx += 1
                    if move_motion_score_for_feedback is not None:
                        knowledge_motion_sum += motion_eval
                        knowledge_motion_samples += 1
                    if bool(enemy_for_feedback.get("recent")):
                        knowledge_enemy_seen_steps += 1
                    refresh_death_cause_context(
                        now_mono=now_mono,
                        enemy_recent=bool(enemy_for_feedback.get("recent", enemy_for_feedback.get("detected", False))),
                        zone_outside_safe=bool(bot_event_signals.get("zone_outside_safe", False)),
                        zone_toxic_detected=bool(bot_event_signals.get("zone_toxic_detected", False)),
                        visual_feedback=visual_feedback_for_event,
                    )
                    dash_cd_remaining = max(
                        0.0,
                        float(ability_state.get("next_sprint_at", 0.0) or 0.0) - now_mono,
                    )
                    dash_ready_now = bool(
                        (dash_cd_remaining <= 0.05)
                        and ((bot_event_signals.get("ability_ready", {}) or {}).get("Shift") is not False)
                    )
                    loot_context_snapshot = {
                        "type": str(bot_event_signals.get("loot_last_type", "") or ""),
                        "name": str(bot_event_signals.get("loot_last_name", "") or ""),
                        "count": int(bot_event_signals.get("loot_count", 0) or 0),
                        "ts": float(bot_event_signals.get("loot_ts", 0.0) or 0.0),
                    }

                    if (
                        knowledge_conn is not None
                        and bot_state == "in_match"
                        and knowledge_prev_move_context
                        and knowledge_prev_move_action
                    ):
                        motion_threshold_now = max(0.2, float(args.bot_move_motion_threshold))
                        move_success = (
                            (move_motion_score_for_feedback is not None and motion_eval >= motion_threshold_now)
                            or (damage_done_delta > 0.0)
                            or (damage_taken_delta <= 0.2 and bool(last_action_ok))
                        ) and (not bool(move_stuck_now))
                        try:
                            update_policy_cache_and_store(
                                conn=knowledge_conn,
                                policy_cache=knowledge_policy_cache,
                                context_key=knowledge_prev_move_context,
                                action_key=knowledge_prev_move_action,
                                reward=float(reward_now),
                                success=bool(move_success),
                                now_ms=int(time.time() * 1000),
                            )
                            knowledge_pending_writes += 1
                        except Exception:
                            pass

                    if knowledge_conn is not None:
                        try:
                            knowledge_conn.execute(
                                """
                                INSERT INTO bot_step_feedback(
                                    run_id, ts_ms, bot_state, context_key, action_kind, action_key, action_label,
                                    motion_score, enemy_seen, enemy_conf, enemy_dir,
                                    zone_countdown_sec, safe_zone_x, safe_zone_y, safe_zone_radius,
                                    ability_key, ability_used, ability_ready_snapshot,
                                    damage_done_total, damage_taken_total, reward,
                                    death_cause, death_cause_conf, death_attacker_name, death_attacker_is_bot,
                                    own_guardian, enemy_guardian, dash_cooldown_sec, loot_context
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    knowledge_run_id,
                                    int(time.time() * 1000),
                                    str(bot_state),
                                    str(knowledge_prev_move_context or ""),
                                    "move",
                                    str(knowledge_prev_move_action or ""),
                                    str(last_action_label),
                                    (None if move_motion_score_for_feedback is None else motion_eval),
                                    1 if bool(enemy_for_feedback.get("recent")) else 0,
                                    float(enemy_for_feedback.get("confidence", 0.0) or 0.0),
                                    str(enemy_for_feedback.get("dir", "") or ""),
                                    float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0),
                                    (
                                        None
                                        if bot_event_signals.get("safe_zone_x") is None
                                        else float(bot_event_signals.get("safe_zone_x"))
                                    ),
                                    (
                                        None
                                        if bot_event_signals.get("safe_zone_y") is None
                                        else float(bot_event_signals.get("safe_zone_y"))
                                    ),
                                    (
                                        None
                                        if bot_event_signals.get("safe_zone_radius") is None
                                        else float(bot_event_signals.get("safe_zone_radius"))
                                    ),
                                    str(ability_used_now or ""),
                                    1 if bool(ability_used_now) else 0,
                                    json.dumps(bot_event_signals.get("ability_ready", {}), ensure_ascii=True),
                                    current_damage_done_total,
                                    current_damage_taken_total,
                                    float(reward_now),
                                    str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                    float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                    str(bot_event_signals.get("death_attacker_name", "") or ""),
                                    (
                                        None
                                        if bot_event_signals.get("death_attacker_is_bot") is None
                                        else (1 if bool(bot_event_signals.get("death_attacker_is_bot")) else 0)
                                    ),
                                    str(bot_event_signals.get("own_guardian", "") or ""),
                                    str(bot_event_signals.get("enemy_guardian", "") or ""),
                                    float(dash_cd_remaining),
                                    json.dumps(loot_context_snapshot, ensure_ascii=True),
                                ),
                            )
                            knowledge_pending_writes += 1
                        except Exception:
                            pass
                        try:
                            now_ms = int(time.time() * 1000)
                            own_guardian_now = str(bot_event_signals.get("own_guardian", "") or "").strip()
                            enemy_guardian_now = str(bot_event_signals.get("enemy_guardian", "") or "").strip()
                            if own_guardian_now:
                                knowledge_conn.execute(
                                    """
                                    INSERT INTO bot_guardian_catalog(guardian_name, sightings, first_seen_ms, last_seen_ms, source)
                                    VALUES (?, 1, ?, ?, ?)
                                    ON CONFLICT(guardian_name) DO UPDATE SET
                                        sightings = bot_guardian_catalog.sightings + 1,
                                        last_seen_ms = excluded.last_seen_ms,
                                        source = excluded.source
                                    """,
                                    (own_guardian_now, now_ms, now_ms, "self_signal"),
                                )
                                knowledge_pending_writes += 1
                            if enemy_guardian_now:
                                knowledge_conn.execute(
                                    """
                                    INSERT INTO bot_guardian_catalog(guardian_name, sightings, first_seen_ms, last_seen_ms, source)
                                    VALUES (?, 1, ?, ?, ?)
                                    ON CONFLICT(guardian_name) DO UPDATE SET
                                        sightings = bot_guardian_catalog.sightings + 1,
                                        last_seen_ms = excluded.last_seen_ms,
                                        source = excluded.source
                                    """,
                                    (enemy_guardian_now, now_ms, now_ms, "enemy_signal"),
                                )
                                knowledge_pending_writes += 1

                            cooldown_shift = (
                                (bot_event_signals.get("ability_cooldown_sec", {}) or {}).get("Shift")
                                if isinstance(bot_event_signals.get("ability_cooldown_sec"), dict)
                                else None
                            )
                            guardian_obs_sig = (
                                f"{own_guardian_now}|{enemy_guardian_now}|{ability_used_now}|"
                                f"{str(cooldown_shift)}|{dash_cd_remaining:.2f}|{int(dash_ready_now)}"
                            )
                            if guardian_obs_sig != last_guardian_obs_sig:
                                if ability_used_now:
                                    knowledge_conn.execute(
                                        """
                                        INSERT INTO bot_guardian_ability_observations(
                                            run_id, ts_ms, owner_type, guardian_name, ability_key, ability_class,
                                            cooldown_sec, ready, source, event_path
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            knowledge_run_id,
                                            now_ms,
                                            "self",
                                            own_guardian_now,
                                            str(ability_used_now),
                                            str(ability_class_now or ABILITY_CLASSIFICATION.get(str(ability_used_now), "unknown")),
                                            None,
                                            1,
                                            "runtime_use",
                                            "",
                                        ),
                                    )
                                    knowledge_pending_writes += 1
                                if cooldown_shift is not None:
                                    knowledge_conn.execute(
                                        """
                                        INSERT INTO bot_guardian_ability_observations(
                                            run_id, ts_ms, owner_type, guardian_name, ability_key, ability_class,
                                            cooldown_sec, ready, source, event_path
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            knowledge_run_id,
                                            now_ms,
                                            "self",
                                            own_guardian_now,
                                            "Shift",
                                            ABILITY_CLASSIFICATION.get("Shift", "dash"),
                                            float(cooldown_shift),
                                            1 if dash_ready_now else 0,
                                            "event_signal",
                                            "dash/sprint",
                                        ),
                                    )
                                    knowledge_pending_writes += 1
                                last_guardian_obs_sig = guardian_obs_sig

                            death_ts_now_step = float(bot_event_signals.get("death_ts", 0.0) or 0.0)
                            if death_ts_now_step > 0.0 and death_ts_now_step > (last_recorded_death_ts + 0.001):
                                knowledge_conn.execute(
                                    """
                                    INSERT INTO bot_death_events(
                                        run_id, ts_ms, cause, cause_conf, attacker_name, attacker_is_bot,
                                        zone_toxic, zone_outside, enemy_recent, map_name, event_name, visual_state, visual_conf
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        knowledge_run_id,
                                        now_ms,
                                        str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                        float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                        str(bot_event_signals.get("death_attacker_name", "") or ""),
                                        (
                                            None
                                            if bot_event_signals.get("death_attacker_is_bot") is None
                                            else (1 if bool(bot_event_signals.get("death_attacker_is_bot")) else 0)
                                        ),
                                        1 if bool(bot_event_signals.get("zone_toxic_detected", False)) else 0,
                                        1 if bool(bot_event_signals.get("zone_outside_safe", False)) else 0,
                                        1 if bool(enemy_for_feedback.get("recent", enemy_for_feedback.get("detected", False))) else 0,
                                        str(bot_event_signals.get("map_name", "") or ""),
                                        str(bot_event_signals.get("last_event_name", "") or ""),
                                        str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown"),
                                        float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0),
                                    ),
                                )
                                knowledge_pending_writes += 1
                                last_recorded_death_ts = death_ts_now_step

                            loot_ts_now_step = float(bot_event_signals.get("loot_ts", 0.0) or 0.0)
                            if loot_ts_now_step > 0.0 and loot_ts_now_step > (last_recorded_loot_ts + 0.001):
                                knowledge_conn.execute(
                                    """
                                    INSERT INTO bot_loot_observations(
                                        run_id, ts_ms, loot_type, loot_name, source, context_json
                                    ) VALUES (?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        knowledge_run_id,
                                        now_ms,
                                        str(bot_event_signals.get("loot_last_type", "") or "generic_loot"),
                                        str(bot_event_signals.get("loot_last_name", "") or ""),
                                        "event_or_ocr",
                                        json.dumps(
                                            {
                                                "count": int(bot_event_signals.get("loot_count", 0) or 0),
                                                "map": str(bot_event_signals.get("map_name", "") or ""),
                                                "event": str(bot_event_signals.get("last_event_name", "") or ""),
                                                "guardian": own_guardian_now,
                                            },
                                            ensure_ascii=True,
                                        ),
                                    ),
                                )
                                knowledge_pending_writes += 1
                                last_recorded_loot_ts = loot_ts_now_step
                        except Exception:
                            pass

                    knowledge_prev_damage_done = current_damage_done_total
                    knowledge_prev_damage_taken = current_damage_taken_total
                    append_feedback_event(
                        runtime_feedback_session,
                        {
                            "ts": time.time(),
                            "mono": now_mono,
                            "bot_state": bot_state,
                            "state_reason": state_reason,
                            "ui_state": ui_state,
                            "action": last_action_label,
                            "action_ok": bool(last_action_ok),
                            "active_keys": sorted(active_keys_runtime),
                            "move_keys": move_combo_for_feedback,
                            "move_escape_mode": bool(move_escape_for_feedback),
                            "move_motion_score": move_motion_score_for_feedback,
                            "move_repeat_streak": int(repeat_move_action_streak),
                            "move_stuck_now": bool(move_stuck_now),
                            "move_stuck_total": int(stuck_event_total),
                            "move_forced_queue_len": int(len(forced_move_queue)),
                            "enemy_signal": enemy_for_feedback,
                            "damage_done_total": current_damage_done_total,
                            "damage_taken_total": current_damage_taken_total,
                            "map_name": str(bot_event_signals.get("map_name", "") or ""),
                            "zone_countdown_sec": float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0),
                            "zone_signal_source": str(bot_event_signals.get("zone_signal_source", "none") or "none"),
                            "zone_outside_safe": bool(bot_event_signals.get("zone_outside_safe", False)),
                            "zone_toxic_detected": bool(bot_event_signals.get("zone_toxic_detected", False)),
                            "zone_toxic_confidence": float(bot_event_signals.get("zone_toxic_confidence", 0.0) or 0.0),
                            "zone_escape_mode": bool(zone_escape_mode),
                            "death": {
                                "active": bool(float(bot_event_signals.get("death_ts", 0.0) or 0.0) > 0.0),
                                "cause": str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                "cause_confidence": float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                "cause_source": str(bot_event_signals.get("death_cause_source", "none") or "none"),
                                "attacker_name": str(bot_event_signals.get("death_attacker_name", "") or ""),
                                "attacker_is_bot": bot_event_signals.get("death_attacker_is_bot"),
                            },
                            "guardian": {
                                "self": str(bot_event_signals.get("own_guardian", "") or ""),
                                "enemy": str(bot_event_signals.get("enemy_guardian", "") or ""),
                            },
                            "safe_zone": {
                                "x": bot_event_signals.get("safe_zone_x"),
                                "y": bot_event_signals.get("safe_zone_y"),
                                "radius": bot_event_signals.get("safe_zone_radius"),
                            },
                            "mana": float(ability_state.get("mana", 0.0) or 0.0),
                            "ability_last": str(ability_state.get("last_used_key", "") or ""),
                            "ability_class": str(ability_state.get("last_used_class", "") or ""),
                            "ability_usage": dict(ability_state.get("usage", {})),
                            "cooldowns": {
                                "abilities": dict(bot_event_signals.get("ability_cooldown_sec", {}) or {}),
                                "dash_remaining_sec": float(dash_cd_remaining),
                                "dash_ready": bool(dash_ready_now),
                            },
                            "loot": {
                                "last_type": str(bot_event_signals.get("loot_last_type", "") or ""),
                                "last_name": str(bot_event_signals.get("loot_last_name", "") or ""),
                                "count": int(bot_event_signals.get("loot_count", 0) or 0),
                            },
                            "visual_ocr": visual_feedback_for_event,
                            "cursor_probe": post_cursor_probe,
                            "click_probe": post_click_probe,
                            "input_probe": post_input_probe,
                            "last_event_signal": str(bot_event_signals.get("last_event_name", "")),
                            "feedback_screenshot": shot_path or "",
                            "knowledge_reward": float(reward_now),
                            "knowledge_context": str(knowledge_prev_move_context or ""),
                            "knowledge_move_action": str(knowledge_prev_move_action or ""),
                        },
                    )
                    if args.bot_debug_hud:
                        update_bot_debug_hud(
                            game_actual_frame,
                            {
                                "state": bot_state,
                                "reason": state_reason,
                                "action": last_action_label,
                                "key_down": int(post_input_probe.get("keyDown", 0)),
                                "key_up": int(post_input_probe.get("keyUp", 0)),
                                "last_key": str(post_input_probe.get("lastKeyDown", "") or "-"),
                                "pointer_down": int(post_input_probe.get("pointerDown", 0)),
                                "pointer_up": int(post_input_probe.get("pointerUp", 0)),
                                "pointer_move": int(post_input_probe.get("pointerMove", 0)),
                                "cursor_moves": int(post_cursor_probe.get("moves", 0)),
                                "cursor_source": str(post_cursor_probe.get("lastSource", "") or "-"),
                                "click0": int(post_click_probe.get("click0", 0)),
                                "click_target": str(post_click_probe.get("lastTarget", "") or "-"),
                                "enemy_seen": 1 if enemy_for_feedback.get("detected") else 0,
                                "enemy_conf": float(enemy_for_feedback.get("confidence", 0.0)),
                                "enemy_dir": str(enemy_for_feedback.get("dir", "") or "-"),
                                "move_stuck": 1 if move_stuck_now else 0,
                                "move_repeat": int(repeat_move_action_streak),
                                "move_q": int(len(forced_move_queue)),
                                "visual_state": str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown"),
                                "visual_conf": float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0),
                                "damage_done": float(bot_event_signals.get("damage_done_total", 0.0) or 0.0),
                                "damage_taken": float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0),
                                "mana": float(ability_state.get("mana", 0.0) or 0.0),
                                "zone_counter": (
                                    "-"
                                    if float(bot_event_signals.get("zone_countdown_sec", -1.0) or -1.0) < 0
                                    else f"{float(bot_event_signals.get('zone_countdown_sec', 0.0)):.1f}s"
                                ),
                                "zone_source": str(bot_event_signals.get("zone_signal_source", "none") or "none"),
                                "zone_toxic": 1 if bool(bot_event_signals.get("zone_toxic_detected", False)) else 0,
                                "zone_outside": 1 if bool(bot_event_signals.get("zone_outside_safe", False)) else 0,
                                "death_cause": str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                "death_conf": float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                "death_source": str(bot_event_signals.get("death_cause_source", "none") or "none"),
                                "safe_zone": (
                                    f"({float(bot_event_signals.get('safe_zone_x') or 0.0):.1f},"
                                    f"{float(bot_event_signals.get('safe_zone_y') or 0.0):.1f},"
                                    f"r={float(bot_event_signals.get('safe_zone_radius') or 0.0):.1f})"
                                    if (bot_event_signals.get("safe_zone_x") is not None and bot_event_signals.get("safe_zone_y") is not None)
                                    else "(-,-,-)"
                                ),
                                "ability_last": str(ability_state.get("last_used_key", "") or "-"),
                                "ability_class": str(ability_state.get("last_used_class", "") or "-"),
                                "dash_cd": max(0.0, float(ability_state.get("next_sprint_at", 0.0) or 0.0) - now_mono),
                                "active_keys": sorted(active_keys_runtime),
                                "feed_line": (
                                    f"{bot_state} {last_action_label} "
                                    f"ok={1 if last_action_ok else 0} "
                                    f"motion={('NA' if move_motion_score_for_feedback is None else f'{move_motion_score_for_feedback:.2f}')} "
                                    f"stuck={1 if move_stuck_now else 0} rep={repeat_move_action_streak}"
                                ),
                            },
                        )
                    min_poll_sleep = 0.04 if bool(args.bot_realtime_mode) else 0.1
                    time.sleep(max(min_poll_sleep, float(args.bot_ui_poll_ms) / 1000.0))
                    generate_report_if_needed(page)
                if knowledge_conn is not None:
                    try:
                        avg_motion = (
                            float(knowledge_motion_sum) / float(max(1, knowledge_motion_samples))
                            if knowledge_motion_samples > 0
                            else 0.0
                        )
                        knowledge_conn.execute(
                            """
                            UPDATE bot_match_runs
                            SET ended_at_ms = ?,
                                stop_reason = ?,
                                map_name = ?,
                                damage_done = ?,
                                damage_taken = ?,
                                steps = ?,
                                avg_motion = ?,
                                enemy_seen_steps = ?,
                                zone_observed_steps = ?,
                                death_cause = ?,
                                death_cause_conf = ?,
                                own_guardian = ?,
                                death_attacker_name = ?
                            WHERE run_id = ?
                            """,
                            (
                                int(time.time() * 1000),
                                str(run_stop_reason or "loop_exit"),
                                str(bot_event_signals.get("map_name", "") or ""),
                                float(bot_event_signals.get("damage_done_total", 0.0) or 0.0),
                                float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0),
                                int(knowledge_step_idx),
                                float(avg_motion),
                                int(knowledge_enemy_seen_steps),
                                int(knowledge_zone_observed_steps),
                                str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                str(bot_event_signals.get("own_guardian", "") or ""),
                                str(bot_event_signals.get("death_attacker_name", "") or ""),
                                knowledge_run_id,
                            ),
                        )
                        knowledge_pending_writes += 1
                        knowledge_conn.commit()
                        knowledge_pending_writes = 0
                        print(
                            "[BOT][KNOWLEDGE] run="
                            f"{knowledge_run_id} steps={knowledge_step_idx} "
                            f"avg_motion={avg_motion:.2f} reward_sum={knowledge_reward_sum:.2f}"
                        )
                    except Exception as exc:
                        print(f"[BOT][KNOWLEDGE][WARN] No se pudo cerrar run en DB: {exc}")
                if runtime_feedback_session:
                    append_feedback_event(
                        runtime_feedback_session,
                        {
                            "event": "session_end",
                            "ts": time.time(),
                            "run_stop_reason": run_stop_reason or "loop_exit",
                            "damage_done_total": float(bot_event_signals.get("damage_done_total", 0.0) or 0.0),
                            "damage_taken_total": float(bot_event_signals.get("damage_taken_total", 0.0) or 0.0),
                            "death": {
                                "active": bool(float(bot_event_signals.get("death_ts", 0.0) or 0.0) > 0.0),
                                "cause": str(bot_event_signals.get("death_cause", "unknown") or "unknown"),
                                "cause_confidence": float(bot_event_signals.get("death_cause_confidence", 0.0) or 0.0),
                                "cause_source": str(bot_event_signals.get("death_cause_source", "none") or "none"),
                                "attacker_name": str(bot_event_signals.get("death_attacker_name", "") or ""),
                                "attacker_is_bot": bot_event_signals.get("death_attacker_is_bot"),
                            },
                            "guardian": {
                                "self": str(bot_event_signals.get("own_guardian", "") or ""),
                                "enemy": str(bot_event_signals.get("enemy_guardian", "") or ""),
                            },
                            "cooldowns": {
                                "abilities": dict(bot_event_signals.get("ability_cooldown_sec", {}) or {}),
                                "dash_remaining_sec": max(0.0, float(ability_state.get("next_sprint_at", 0.0) or 0.0) - time.monotonic()),
                            },
                            "loot": {
                                "last_type": str(bot_event_signals.get("loot_last_type", "") or ""),
                                "last_name": str(bot_event_signals.get("loot_last_name", "") or ""),
                                "count": int(bot_event_signals.get("loot_count", 0) or 0),
                            },
                            "visual_state_hint": str(bot_event_signals.get("visual_state_hint", "unknown") or "unknown"),
                            "visual_state_confidence": float(bot_event_signals.get("visual_state_confidence", 0.0) or 0.0),
                            "visual_names": list(bot_event_signals.get("visual_names", []) or []),
                            "visual_damage_hint": float(bot_event_signals.get("visual_damage_hint", 0.0) or 0.0),
                            "visual_ocr_frames": int(visual_ocr_state.get("frames", 0)),
                            "runtime_probe_paths": list(runtime_probe_paths),
                        },
                    )
                    if args.bot_feedback_render_video:
                        video_path = render_feedback_video(
                            session=runtime_feedback_session,
                            fps=float(args.bot_feedback_video_fps),
                        )
                        if video_path:
                            append_feedback_event(
                                runtime_feedback_session,
                                {
                                    "event": "feedback_video_rendered",
                                    "ts": time.time(),
                                    "video_path": video_path,
                                },
                            )
                            print(f"[BOT][FEEDBACK] Video generado: {video_path}")
                        else:
                            print("[BOT][FEEDBACK][WARN] No se pudo generar video de feedback.")
                if args.bot_run_until_end:
                    reason = run_stop_reason or "loop_exit"
                    if runtime_probe_paths:
                        print(
                            "[BOT][RUNTIME_PROBE] snapshots="
                            f"{len(runtime_probe_paths)} last={runtime_probe_paths[-1]}"
                        )
                    print(
                        "[BOT][RUN] Ejecucion completa finalizada. "
                        f"reason={reason} "
                        f"damage_done={float(bot_event_signals.get('damage_done_total', 0.0) or 0.0):.1f} "
                        f"damage_taken={float(bot_event_signals.get('damage_taken_total', 0.0) or 0.0):.1f} "
                        f"death={str(bot_event_signals.get('death_cause', 'unknown') or 'unknown')}:"
                        f"{float(bot_event_signals.get('death_cause_confidence', 0.0) or 0.0):.2f} "
                        f"attacker={str(bot_event_signals.get('death_attacker_name', '') or '-')} "
                        f"guardian={str(bot_event_signals.get('own_guardian', '') or '-')} "
                        f"enemy_guardian={str(bot_event_signals.get('enemy_guardian', '') or '-')} "
                        f"loot={str(bot_event_signals.get('loot_last_type', '') or '-')} "
                        f"map={str(bot_event_signals.get('map_name', '') or '-')} "
                        f"visual={str(bot_event_signals.get('visual_state_hint', 'unknown') or 'unknown')}:"
                        f"{float(bot_event_signals.get('visual_state_confidence', 0.0) or 0.0):.2f}"
                    )
            if not args.play_game: # This block handles the original reporting loop when not playing the game
                while True:
                    active_page = pick_active_page(page)
                    if active_page is not None:
                        try:
                            active_page.wait_for_timeout(1000)
                        except PlaywrightError as exc:
                            print(f"[WARN] wait_for_timeout fallo: {exc}")
                            time.sleep(1)
                    else:
                        print("[WARN] No hay pagina activa; esperando...")
                        time.sleep(1)
                    generate_report_if_needed(page) # Call the new function

    except KeyboardInterrupt:
        total_in_db = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        print("\nCaptura detenida por usuario.")
        print(
            f"requests_seen={stats['requests_seen']} "
            f"events_saved={stats['events_saved']} "
            f"no_payload={stats['no_payload']} "
            f"no_events={stats['no_events']} "
            f"payload_errors={stats['payload_errors']} "
            f"raw_posts_saved={stats['raw_posts_saved']} "
            f"raw_events_saved={stats['raw_events_saved']} "
            f"raw_posts_skipped={stats['raw_posts_skipped']} "
            f"ws_frames={stats['ws_frames']} "
            f"ws_keyword_hits={stats['ws_keyword_hits']} "
            f"ws_saved={stats['ws_saved']} "
            f"ocr_queued={stats['ocr_queued']} "
            f"ocr_runs={stats['ocr_runs']} "
            f"ocr_saved={stats['ocr_saved']} "
            f"ocr_errors={stats['ocr_errors']} "
            f"db_events={total_in_db}"
        )
        print_top_events(conn, limit=10)
        print_detected_entities("Personajes detectados", seen_characters)
        print_detected_entities("Jugadores detectados (sin bots)", seen_players)
        print("\nUltimas partidas (match_end):")
        has_match = print_match_report(conn, limit=args.report_limit)
        if not has_match:
            print("\nActividad de ronda (fallback):")
            print_round_activity_report(conn, limit=args.report_limit)
    finally:
        try:
            drain_ocr_queue(page_obj=page, force=True)
        except Exception:
            pass
        commit_ws_if_needed(force=True)
        try:
            if close_context_on_exit and context is not None:
                try:
                    if route_installed:
                        context.unroute(route_pattern, route_handler)
                except Exception:
                    pass
                context.close()
        except Exception:
            pass
        try:
            if close_browser_on_exit and browser is not None:
                browser.close()
        except Exception:
            pass
        try:
            if knowledge_conn is not None:
                knowledge_conn.commit()
                knowledge_conn.close()
        except Exception:
            pass
        conn.close()


if __name__ == "__main__":
    main()



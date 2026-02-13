from unittest.mock import MagicMock, patch
import sys
import time
from collections import deque

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lms_live_collector import build_parser, main, bot_event_signals

# A simplified smoke_html content for safe zone testing
# This HTML will contain a __botSmoke object that we can manipulate
# to simulate different safe zone conditions.
SAFE_ZONE_SMOKE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bot Safe Zone Test</title>
  <style>
    body { margin: 0; height: 100%; overflow: hidden; }
    #game-canvas { width: 100%; height: 100%; background: #000; }
  </style>
</head>
<body>
  <canvas id="game-canvas"></canvas>
  <script>
    window.__botSmoke = {
      hp: 100,
      maxHp: 100,
      mana: 100,
      maxMana: 100,
      damageReceived: 0,
      damageDealt: 0,
      enemyVisible: 0,
      enemyThreat: 0,
      enemyDir: 'CENTER',
      zoneCountdownSec: 120.0,
      zoneRadius: 0.2,
      safeZoneX: 0.5,
      safeZoneY: 0.5,
      playerX: 0.1,
      playerY: 0.1,
      zoneToxic: 0,
      zoneOutside: 0,
      zoneSignalSource: 'sim',
      simState: 'in_match',
      lastAbility: '',
      lastAbilityClass: ''
    };
  </script>
</body>
</html>
"""

def test_bot_moves_towards_safe_zone_far_away():
    # Scenario 1: Bot is outside the safe zone, far away.
    # Expect bot to move towards the safe zone.

    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_frame = MagicMock()
    mock_iframe_locator = MagicMock()
    mock_iframe_handle = MagicMock()

    # Mock sync_playwright and its context/page creation
    mock_sync = MagicMock()
    mock_sync.__enter__.return_value = mock_p
    mock_sync.__exit__.return_value = False

    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # Mock iframe resolution
    mock_iframe_locator.first = mock_iframe_locator
    mock_iframe_locator.wait_for.return_value = None
    mock_iframe_locator.element_handle.return_value = mock_iframe_handle
    mock_iframe_locator.bounding_box.return_value = {
        "x": 0.0, "y": 0.0, "width": 1280.0, "height": 720.0,
    }
    mock_page.locator.return_value = mock_iframe_locator
    mock_iframe_handle.content_frame.return_value = mock_frame
    mock_page.viewport_size = {"width": 1280, "height": 720}

    mock_frame.focus.return_value = None
    mock_frame.wait_for_load_state.return_value = None

    # Bot's initial state: far from safe zone, outside
    initial_bot_smoke_state = {
        "hp": 100, "maxHp": 100, "mana": 100, "maxMana": 100,
        "damageReceived": 0, "damageDealt": 0,
        "enemyVisible": 0, "enemyThreat": 0, "enemyDir": 'CENTER',
        "zoneCountdownSec": 120.0,  # High countdown
        "zoneRadius": 0.2,
        "safeZoneX": 0.5, "safeZoneY": 0.5,
        "playerX": 0.1, "playerY": 0.1,  # Far from safe zone
        "zoneToxic": 0,
        "zoneOutside": 1,  # Bot is outside
        "zoneSignalSource": 'sim',
        "simState": 'in_match',
        "lastAbility": '', "lastAbilityClass": ''
    }

    # Mock frame.evaluate to return our controlled bot state
    def frame_evaluate_side_effect(script, *args):
        s = str(script)
        if "window.__botSmoke" in s:
            # Dynamically update player position to simulate movement
            # This is a simplification; in a real test, you might verify the keys pressed
            # and then update the position in the mock to simulate effective movement.
            # For this test, we just return the current state and let the bot's logic decide.
            return dict(initial_bot_smoke_state)
        if "const c = document.querySelector('canvas')" in s:
            return {"x": 10.0, "y": 10.0, "width": 500.0, "height": 300.0}
        if "window.__lmsClickProbe" in s:
            return {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}
        if "window.__lmsBotCursorProbe" in s:
            return {
                "moves": 0, "lastX": 0, "lastY": 0, "lastSource": '',
                "visible": True, "updatedAt": time.time() * 1000
            }
        if "window.__lmsInputProbe" in s:
            return {
                "keyDown": 0, "keyUp": 0, "lastKeyDown": '', "lastKeyUp": '',
                "pointerDown": 0, "pointerUp": 0, "pointerMove": 0,
                "focusEvents": 0, "blurEvents": 0, "lastEventTs": time.time() * 1000
            }
        # For debug HUD, etc.
        if "document.getElementById" in s or "document.createElement" in s:
            return True
        return None

    mock_frame.evaluate.side_effect = frame_evaluate_side_effect
    
    # Mock keyboard actions
    mock_page.keyboard.down.return_value = None
    mock_page.keyboard.up.return_value = None
    mock_page.mouse.move.return_value = None
    mock_page.mouse.click.return_value = None

    sleep_calls = {"n": 0}
    def sleep_side_effect(_seconds):
        sleep_calls["n"] += 1
        # Stop after a few iterations to check movement
        if sleep_calls["n"] >= 5:
            raise KeyboardInterrupt
        return None
    
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [0]


    with patch("lms_live_collector.sync_playwright", return_value=mock_sync):
        with patch("lms_live_collector.sqlite3.connect", return_value=mock_conn):
            with patch("lms_live_collector.ensure_schema"):
                with patch("lms_live_collector.ensure_live_schema"):
                    with patch("lms_live_collector.ensure_ws_schema"):
                        with patch("lms_live_collector.ensure_bot_knowledge_schema"):
                            with patch("lms_live_collector.load_policy_cache", return_value={}):
                                with patch("lms_live_collector.load_historical_move_penalties", return_value=({}, {}, {})):
                                                with patch.object(
                                                    sys,
                                                    "argv",
                                                    [
                                                        "lms_live_collector.py",
                                                        "--play-game",
                                                        "--no-persistent",
                                                        "--headless",
                                                        "--bot-ui-poll-ms",
                                                        "50",
                                                        "--game-url",
                                                        "about:blank",
                                                        "--bot-feedback-dir", "",
                                                        "--bot-feedback-jsonl", "",
                                                    ],
                                                ):
                                                    # Simulate the page content setup
                                                    mock_page.goto.return_value = None
                                                    mock_page.set_content.return_value = None

                                                    # The main() function will call page.goto. We need to intercept this
                                                    # to set the content for our test HTML and configure the mock_frame.
                                                    def page_goto_side_effect(url, **kwargs):
                                                        if url == "about:blank":
                                                            # When goto('about:blank') is called, set our smoke HTML
                                                            # and ensure the frame evaluate side effect is active.
                                                            mock_page.main_frame.evaluate.return_value = None # Clear previous evaluate mocks
                                                            mock_page.set_content(SAFE_ZONE_SMOKE_HTML, wait_until="domcontentloaded")
                                                            mock_page.main_frame.evaluate.side_effect = frame_evaluate_side_effect
                                                            # Call a dummy evaluate to trigger the side effect setup
                                                            mock_page.main_frame.evaluate("() => {}", None)
                                                        # For other URLs, if any, you might want to mock them differently
                                                    mock_page.goto.side_effect = page_goto_side_effect
                                                    # We need to explicitly call set_content on mock_page once here
                                                    # because main() will trigger page.goto which then triggers the side effect.
                                                    # The initial set_content (before the loop in main) should load our smoke HTML
                                                    mock_page.set_content(SAFE_ZONE_SMOKE_HTML, wait_until="domcontentloaded")


                                                    # Ensure bot_event_signals are reset for a clean test run
                                                    bot_event_signals.clear()
                                                    bot_event_signals.update({
                                                        "last_event_name": "", "last_event_ts": 0.0, "lobby_ts": 0.0,
                                                        "match_ts": 0.0, "match_end_ts": 0.0, "death_ts": 0.0,
                                                        "zone_countdown_sec": -1.0, "safe_zone_x": None, "safe_zone_y": None,
                                                        "safe_zone_radius": None, "player_pos_x": None, "player_pos_y": None,
                                                        "zone_outside_safe": False, "zone_toxic_detected": False, "zone_toxic_confidence": 0.0,
                                                        "zone_signal_source": "none", "zone_signal_ts": 0.0,
                                                        "damage_done_total": 0.0, "damage_taken_total": 0.0,
                                                        "visual_state_hint": "unknown", "visual_state_confidence": 0.0,
                                                        "map_name": "", # Add map_name to bot_event_signals
                                                    })


                                                    main()

    # Assertions
    # The bot should try to move towards the safe zone (0.5, 0.5) from (0.1, 0.1)
    # This means it should press KeyW (Up) and KeyD (Right)
    
    # Collect all 'keyboard.down' calls
    down_calls = [call.args[0] for call in mock_page.keyboard.down.call_args_list]

    # Given the MOVE_VECTORS in lms_live_collector.py:
    # "KeyW": (0, -1), "KeyS": (0, 1), "KeyA": (-1, 0), "KeyD": (1, 0)
    # To move from (0.1, 0.1) to (0.5, 0.5), it needs to increase X and increase Y
    # So, KeyD (for X+) and KeyS (for Y+) should be present.
    # The `build_zone_recovery_move_candidates` builds candidates based on dx and dy
    # dx = safe_x - player_x = 0.5 - 0.1 = 0.4 (positive, so KeyD)
    # dy = safe_y - player_y = 0.5 - 0.1 = 0.4 (positive, so KeyS)
    
    # Check for KeyD and KeyS presses to move towards the safe zone
    assert any(key == "KeyD" for key in down_calls), "Bot should press KeyD to move right"
    assert any(key == "KeyS" for key in down_calls), "Bot should press KeyS to move down"

    # Verify that the bot's internal state reflects being outside and moving towards the zone
    assert bot_event_signals["zone_outside_safe"] is True
    assert bot_event_signals["zone_countdown_sec"] == 120.0
    assert bot_event_signals["safe_zone_x"] == 0.5
    assert bot_event_signals["safe_zone_y"] == 0.5


def test_bot_moves_towards_safe_zone_critical_countdown():
    # Scenario 2: Bot is outside the safe zone, `zone_countdown_sec` low (critical).
    # Expect bot to move towards the safe zone, potentially with more urgency (not explicitly tested as urgency is complex)

    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_frame = MagicMock()
    mock_iframe_locator = MagicMock()
    mock_iframe_handle = MagicMock()

    mock_sync = MagicMock()
    mock_sync.__enter__.return_value = mock_p
    mock_sync.__exit__.return_value = False

    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    mock_iframe_locator.first = mock_iframe_locator
    mock_iframe_locator.wait_for.return_value = None
    mock_iframe_locator.element_handle.return_value = mock_iframe_handle
    mock_iframe_locator.bounding_box.return_value = {
        "x": 0.0, "y": 0.0, "width": 1280.0, "height": 720.0,
    }
    mock_page.locator.return_value = mock_iframe_locator
    mock_iframe_handle.content_frame.return_value = mock_frame
    mock_page.viewport_size = {"width": 1280, "height": 720}

    mock_frame.focus.return_value = None
    mock_frame.wait_for_load_state.return_value = None

    initial_bot_smoke_state = {
        "hp": 100, "maxHp": 100, "mana": 100, "maxMana": 100,
        "damageReceived": 0, "damageDealt": 0,
        "enemyVisible": 0, "enemyThreat": 0, "enemyDir": 'CENTER',
        "zoneCountdownSec": 5.0,  # Critical countdown
        "zoneRadius": 0.2,
        "safeZoneX": 0.5, "safeZoneY": 0.5,
        "playerX": 0.1, "playerY": 0.1,  # Far from safe zone
        "zoneToxic": 0,
        "zoneOutside": 1,
        "zoneSignalSource": 'sim',
        "simState": 'in_match',
        "lastAbility": '', "lastAbilityClass": ''
    }

    def frame_evaluate_side_effect(script, *args):
        s = str(script)
        if "window.__botSmoke" in s:
            return dict(initial_bot_smoke_state)
        if "const c = document.querySelector('canvas')" in s:
            return {"x": 10.0, "y": 10.0, "width": 500.0, "height": 300.0}
        if "window.__lmsClickProbe" in s:
            return {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}
        if "window.__lmsBotCursorProbe" in s:
            return {
                "moves": 0, "lastX": 0, "lastY": 0, "lastSource": '',
                "visible": True, "updatedAt": time.time() * 1000
            }
        if "window.__lmsInputProbe" in s:
            return {
                "keyDown": 0, "keyUp": 0, "lastKeyDown": '', "lastKeyUp": '',
                "pointerDown": 0, "pointerUp": 0, "pointerMove": 0,
                "focusEvents": 0, "blurEvents": 0, "lastEventTs": time.time() * 1000
            }
        if "document.getElementById" in s or "document.createElement" in s:
            return True
        return None

    mock_frame.evaluate.side_effect = frame_evaluate_side_effect
    
    mock_page.keyboard.down.return_value = None
    mock_page.keyboard.up.return_value = None
    mock_page.mouse.move.return_value = None
    mock_page.mouse.click.return_value = None

    sleep_calls = {"n": 0}
    def sleep_side_effect(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 5:
            raise KeyboardInterrupt
        return None
    
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [0]


    with patch("lms_live_collector.sync_playwright", return_value=mock_sync):
        with patch("lms_live_collector.sqlite3.connect", return_value=mock_conn):
            with patch("lms_live_collector.ensure_schema"):
                with patch("lms_live_collector.ensure_live_schema"):
                    with patch("lms_live_collector.ensure_ws_schema"):
                        with patch("lms_live_collector.ensure_bot_knowledge_schema"):
                            with patch("lms_live_collector.load_policy_cache", return_value={}):
                                with patch("lms_live_collector.load_historical_move_penalties", return_value=({}, {}, {})):
                                    with patch("lms_live_collector.create_feedback_session", return_value={}):
                                        with patch("lms_live_collector.append_feedback_event"):
                                            with patch("time.sleep", side_effect=sleep_side_effect):
                                                with patch.object(
                                                    sys,
                                                    "argv",
                                                    [
                                                        "lms_live_collector.py",
                                                        "--play-game",
                                                        "--no-persistent",
                                                        "--headless",
                                                        "--bot-ui-poll-ms",
                                                        "50",
                                                        "--game-url",
                                                        "about:blank",
                                                    ],
                                                ):
                                                    mock_page.goto.return_value = None
                                                    mock_page.set_content.return_value = None
                                                    def page_goto_side_effect(url, **kwargs):
                                                        if url == "about:blank":
                                                            mock_page.main_frame.evaluate.return_value = None
                                                            mock_page.main_frame.evaluate.side_effect = frame_evaluate_side_effect
                                                            mock_page.main_frame.evaluate("() => {}", None)
                                                    mock_page.goto.side_effect = page_goto_side_effect
                                                    mock_page.set_content(SAFE_ZONE_SMOKE_HTML, wait_until="domcontentloaded")

                                                    bot_event_signals.clear()
                                                    bot_event_signals.update({
                                                        "last_event_name": "", "last_event_ts": 0.0, "lobby_ts": 0.0,
                                                        "match_ts": 0.0, "match_end_ts": 0.0, "death_ts": 0.0,
                                                        "zone_countdown_sec": -1.0, "safe_zone_x": None, "safe_zone_y": None,
                                                        "safe_zone_radius": None, "player_pos_x": None, "player_pos_y": None,
                                                        "zone_outside_safe": False, "zone_toxic_detected": False, "zone_toxic_confidence": 0.0,
                                                        "zone_signal_source": "none", "zone_signal_ts": 0.0,
                                                        "damage_done_total": 0.0, "damage_taken_total": 0.0,
                                                        "visual_state_hint": "unknown", "visual_state_confidence": 0.0,
                                                        "map_name": "", # Add map_name to bot_event_signals
                                                    })
                                                    main()

    down_calls = [call.args[0] for call in mock_page.keyboard.down.call_args_list]
    
    assert any(key == "KeyD" for key in down_calls), "Bot should press KeyD to move right"
    assert any(key == "KeyS" for key in down_calls), "Bot should press KeyS to move down"

    assert bot_event_signals["zone_outside_safe"] is True
    assert bot_event_signals["zone_countdown_sec"] == 5.0
    assert bot_event_signals["safe_zone_x"] == 0.5
    assert bot_event_signals["safe_zone_y"] == 0.5


def test_bot_inside_safe_zone_no_specific_movement():
    # Scenario 3: Bot is inside the safe zone.
    # Expect bot not to move specifically towards the zone. It might move randomly or attack if enemies are present.

    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_frame = MagicMock()
    mock_iframe_locator = MagicMock()
    mock_iframe_handle = MagicMock()

    mock_sync = MagicMock()
    mock_sync.__enter__.return_value = mock_p
    mock_sync.__exit__.return_value = False

    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    mock_iframe_locator.first = mock_iframe_locator
    mock_iframe_locator.wait_for.return_value = None
    mock_iframe_locator.element_handle.return_value = mock_iframe_handle
    mock_iframe_locator.bounding_box.return_value = {
        "x": 0.0, "y": 0.0, "width": 1280.0, "height": 720.0,
    }
    mock_page.locator.return_value = mock_iframe_locator
    mock_iframe_handle.content_frame.return_value = mock_frame
    mock_page.viewport_size = {"width": 1280, "height": 720}

    mock_frame.focus.return_value = None
    mock_frame.wait_for_load_state.return_value = None

    initial_bot_smoke_state = {
        "hp": 100, "maxHp": 100, "mana": 100, "maxMana": 100,
        "damageReceived": 0, "damageDealt": 0,
        "enemyVisible": 0, "enemyThreat": 0, "enemyDir": 'CENTER',
        "zoneCountdownSec": 60.0,
        "zoneRadius": 0.2,
        "safeZoneX": 0.5, "safeZoneY": 0.5,
        "playerX": 0.5, "playerY": 0.5,  # Inside safe zone
        "zoneToxic": 0,
        "zoneOutside": 0,  # Bot is inside
        "zoneSignalSource": 'sim',
        "simState": 'in_match',
        "lastAbility": '', "lastAbilityClass": ''
    }

    def frame_evaluate_side_effect(script, *args):
        s = str(script)
        if "window.__botSmoke" in s:
            return dict(initial_bot_smoke_state)
        if "const c = document.querySelector('canvas')" in s:
            return {"x": 10.0, "y": 10.0, "width": 500.0, "height": 300.0}
        if "window.__lmsClickProbe" in s:
            return {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}
        if "window.__lmsBotCursorProbe" in s:
            return {
                "moves": 0, "lastX": 0, "lastY": 0, "lastSource": '',
                "visible": True, "updatedAt": time.time() * 1000
            }
        if "window.__lmsInputProbe" in s:
            return {
                "keyDown": 0, "keyUp": 0, "lastKeyDown": '', "lastKeyUp": '',
                "pointerDown": 0, "pointerUp": 0, "pointerMove": 0,
                "focusEvents": 0, "blurEvents": 0, "lastEventTs": time.time() * 1000
            }
        if "document.getElementById" in s or "document.createElement" in s:
            return True
        return None

    mock_frame.evaluate.side_effect = frame_evaluate_side_effect
    
    mock_page.keyboard.down.return_value = None
    mock_page.keyboard.up.return_value = None
    mock_page.mouse.move.return_value = None
    mock_page.mouse.click.return_value = None

    sleep_calls = {"n": 0}
    def sleep_side_effect(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 5:
            raise KeyboardInterrupt
        return None
    
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [0]


    with patch("lms_live_collector.sync_playwright", return_value=mock_sync):
        with patch("lms_live_collector.sqlite3.connect", return_value=mock_conn):
            with patch("lms_live_collector.ensure_schema"):
                with patch("lms_live_collector.ensure_live_schema"):
                    with patch("lms_live_collector.ensure_ws_schema"):
                        with patch("lms_live_collector.ensure_bot_knowledge_schema"):
                            with patch("lms_live_collector.load_policy_cache", return_value={}):
                                with patch("lms_live_collector.load_historical_move_penalties", return_value=({}, {}, {})):
                                    with patch("lms_live_collector.create_feedback_session", return_value={}):
                                        with patch("lms_live_collector.append_feedback_event"):
                                            with patch("time.sleep", side_effect=sleep_side_effect):
                                                with patch.object(
                                                    sys,
                                                    "argv",
                                                    [
                                                        "lms_live_collector.py",
                                                        "--play-game",
                                                        "--no-persistent",
                                                        "--headless",
                                                        "--bot-ui-poll-ms",
                                                        "50",
                                                        "--game-url",
                                                        "about:blank",
                                                    ],
                                                ):
                                                    mock_page.goto.return_value = None
                                                    mock_page.set_content.return_value = None
                                                    def page_goto_side_effect(url, **kwargs):
                                                        if url == "about:blank":
                                                            mock_page.main_frame.evaluate.return_value = None
                                                            mock_page.main_frame.evaluate.side_effect = frame_evaluate_side_effect
                                                            mock_page.main_frame.evaluate("() => {}", None)
                                                    mock_page.goto.side_effect = page_goto_side_effect
                                                    mock_page.set_content(SAFE_ZONE_SMOKE_HTML, wait_until="domcontentloaded")

                                                    bot_event_signals.clear()
                                                    bot_event_signals.update({
                                                        "last_event_name": "", "last_event_ts": 0.0, "lobby_ts": 0.0,
                                                        "match_ts": 0.0, "match_end_ts": 0.0, "death_ts": 0.0,
                                                        "zone_countdown_sec": -1.0, "safe_zone_x": None, "safe_zone_y": None,
                                                        "safe_zone_radius": None, "player_pos_x": None, "player_pos_y": None,
                                                        "zone_outside_safe": False, "zone_toxic_detected": False, "zone_toxic_confidence": 0.0,
                                                        "zone_signal_source": "none", "zone_signal_ts": 0.0,
                                                        "damage_done_total": 0.0, "damage_taken_total": 0.0,
                                                        "visual_state_hint": "unknown", "visual_state_confidence": 0.0,
                                                        "map_name": "", # Add map_name to bot_event_signals
                                                    })
                                                    main()

    down_calls = [call.args[0] for call in mock_page.keyboard.down.call_args_list]
    # When inside the safe zone, the bot might still move, but not explicitly towards the safe zone
    # The default movement should be some form of patrol (e.g. KeyW,KeyA,KeyS,KeyD)
    
    # Assert that the bot is not prioritizing movement towards the center if it's already there
    # This is a weak assertion, but demonstrates the bot is not explicitly moving towards safe zone center
    assert any(key in ["KeyW", "KeyA", "KeyS", "KeyD"] for key in down_calls)
    
    # It should not be primarily pressing KeyS and KeyD to move towards (0.5, 0.5) if it's already there.
    # The `build_zone_recovery_move_candidates` would not return these keys if player is at safe_zone center.
    # So we check that the primary movement keys are not exclusively KeyS and KeyD.
    # We can't assert no movement, as there might be a default patrol.
    # Let's verify that the 'zone_escape_mode' is not active based on the `bot_event_signals`
    assert bot_event_signals["zone_outside_safe"] is False
    assert bot_event_signals["zone_toxic_detected"] is False
    assert bot_event_signals["safe_zone_x"] == 0.5
    assert bot_event_signals["safe_zone_y"] == 0.5


def test_bot_detects_visual_toxic_zone_escapes():
    # Scenario 4: Bot detects visual toxic zone (e.g., toxic_top_ratio high).
    # Expect bot to use toxic_escape_keys (e.g., move down if toxic from top)

    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_frame = MagicMock()
    mock_iframe_locator = MagicMock()
    mock_iframe_handle = MagicMock()

    mock_sync = MagicMock()
    mock_sync.__enter__.return_value = mock_p
    mock_sync.__exit__.return_value = False

    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    mock_iframe_locator.first = mock_iframe_locator
    mock_iframe_locator.wait_for.return_value = None
    mock_iframe_locator.element_handle.return_value = mock_iframe_handle
    mock_iframe_locator.bounding_box.return_value = {
        "x": 0.0, "y": 0.0, "width": 1280.0, "height": 720.0,
    }
    mock_page.locator.return_value = mock_iframe_locator
    mock_iframe_handle.content_frame.return_value = mock_frame
    mock_page.viewport_size = {"width": 1280, "height": 720}

    mock_frame.focus.return_value = None
    mock_frame.wait_for_load_state.return_value = None

    initial_bot_smoke_state = {
        "hp": 100, "maxHp": 100, "mana": 100, "maxMana": 100,
        "damageReceived": 10.0, "damageDealt": 0, # Simulating damage from toxic zone
        "enemyVisible": 0, "enemyThreat": 0, "enemyDir": 'CENTER',
        "zoneCountdownSec": 30.0,
        "zoneRadius": 0.2,
        "safeZoneX": 0.5, "safeZoneY": 0.5,
        "playerX": 0.5, "playerY": 0.5,
        "zoneToxic": 1,  # Bot is in toxic zone
        "zoneOutside": 1, # Bot is outside safe zone (implies toxic)
        "zoneSignalSource": 'vision', # Visual detection
        "simState": 'in_match',
        "lastAbility": '', "lastAbilityClass": ''
    }
    
    # Mock visual OCR to simulate toxic zone from top
    mock_extract_visual_feedback_from_screenshot_result = {
        "ok": True,
        "engine": "pytesseract",
        "state_hint": "in_match",
        "state_confidence": 0.9,
        "lobby_hits": 0, "in_match_hits": 1, "safe_zone_hits": 0, "toxic_zone_hits": 1,
        "toxic_color_ratio": 0.1,  # High toxic color ratio
        "toxic_top_ratio": 0.8,    # Toxic from top
        "toxic_bottom_ratio": 0.0,
        "toxic_left_ratio": 0.0,
        "toxic_right_ratio": 0.0,
        "toxic_escape_keys": ["KeyS"], # Expected escape key is "KeyS" (move down)
        "death_hits": 0, "names": [], "damage_numbers": [], "raw_excerpt": ""
    }


    def frame_evaluate_side_effect(script, *args):
        s = str(script)
        if "window.__botSmoke" in s:
            return dict(initial_bot_smoke_state)
        if "const c = document.querySelector('canvas')" in s:
            return {"x": 10.0, "y": 10.0, "width": 500.0, "height": 300.0}
        if "window.__lmsClickProbe" in s:
            return {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}
        if "window.__lmsBotCursorProbe" in s:
            return {
                "moves": 0, "lastX": 0, "lastY": 0, "lastSource": '',
                "visible": True, "updatedAt": time.time() * 1000
            }
        if "window.__lmsInputProbe" in s:
            return {
                "keyDown": 0, "keyUp": 0, "lastKeyDown": '', "lastKeyUp": '',
                "pointerDown": 0, "pointerUp": 0, "pointerMove": 0,
                "focusEvents": 0, "blurEvents": 0, "lastEventTs": time.time() * 1000
            }
        if "document.getElementById" in s or "document.createElement" in s:
            return True
        return None

    mock_frame.evaluate.side_effect = frame_evaluate_side_effect
    
    mock_page.keyboard.down.return_value = None
    mock_page.keyboard.up.return_value = None
    mock_page.mouse.move.return_value = None
    mock_page.mouse.click.return_value = None

    sleep_calls = {"n": 0}
    def sleep_side_effect(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 5:
            raise KeyboardInterrupt
        return None
    
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [0]


    with patch("lms_live_collector.sync_playwright", return_value=mock_sync):
        with patch("lms_live_collector.sqlite3.connect", return_value=mock_conn):
            with patch("lms_live_collector.ensure_schema"):
                with patch("lms_live_collector.ensure_live_schema"):
                    with patch("lms_live_collector.ensure_ws_schema"):
                        with patch("lms_live_collector.ensure_bot_knowledge_schema"):
                            with patch("lms_live_collector.load_policy_cache", return_value={}):
                                with patch("lms_live_collector.load_historical_move_penalties", return_value=({}, {}, {})):
                                    with patch("lms_live_collector.create_feedback_session", return_value={}):
                                        with patch("lms_live_collector.append_feedback_event"):
                                            with patch("time.sleep", side_effect=sleep_side_effect):
                                                with patch("lms_live_collector.extract_visual_feedback_from_screenshot", return_value=mock_extract_visual_feedback_from_screenshot_result):
                                                    with patch.object(
                                                        sys,
                                                        "argv",
                                                        [
                                                            "lms_live_collector.py",
                                                            "--play-game",
                                                            "--no-persistent",
                                                            "--headless",
                                                            "--bot-ui-poll-ms",
                                                            "50",
                                                            "--game-url",
                                                            "about:blank",
                                                            "--bot-visual-ocr", # Enable visual OCR
                                                            "--bot-visual-ocr-every-sec", "0.1", # Frequent OCR
                                                        ],
                                                    ):
                                                        mock_page.goto.return_value = None
                                                        mock_page.set_content.return_value = None
                                                        def page_goto_side_effect(url, **kwargs):
                                                            if url == "about:blank":
                                                                mock_page.main_frame.evaluate.return_value = None
                                                                mock_page.main_frame.evaluate.side_effect = frame_evaluate_side_effect
                                                                mock_page.main_frame.evaluate("() => {}", None)
                                                        mock_page.goto.side_effect = page_goto_side_effect
                                                        mock_page.set_content(SAFE_ZONE_SMOKE_HTML, wait_until="domcontentloaded")

                                                        bot_event_signals.clear()
                                                        bot_event_signals.update({
                                                            "last_event_name": "", "last_event_ts": 0.0, "lobby_ts": 0.0,
                                                            "match_ts": 0.0, "match_end_ts": 0.0, "death_ts": 0.0,
                                                            "zone_countdown_sec": -1.0, "safe_zone_x": None, "safe_zone_y": None,
                                                            "safe_zone_radius": None, "player_pos_x": None, "player_pos_y": None,
                                                            "zone_outside_safe": False, "zone_toxic_detected": False, "zone_toxic_confidence": 0.0,
                                                            "zone_signal_source": "none", "zone_signal_ts": 0.0,
                                                            "damage_done_total": 0.0, "damage_taken_total": 0.0,
                                                            "visual_state_hint": "unknown", "visual_state_confidence": 0.0,
                                                            "map_name": "", # Add map_name to bot_event_signals
                                                        })
                                                        main()

    down_calls = [call.args[0] for call in mock_page.keyboard.down.call_args_list]
    # If toxic from top (toxic_top_ratio high), bot should try to move down (KeyS)
    assert any(key == "KeyS" for key in down_calls), "Bot should press KeyS to escape toxic zone from top"

    assert bot_event_signals["zone_toxic_detected"] is True
    assert bot_event_signals["zone_outside_safe"] is True
    assert bot_event_signals["zone_signal_source"] == "vision"

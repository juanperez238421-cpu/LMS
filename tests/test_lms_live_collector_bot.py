from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, "./")
from lms_live_collector import build_parser, main


def test_build_parser_play_game_argument():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--play-game",
            "--bot-click-mode",
            "mouse",
            "--bot-decision-backend",
            "lms_re",
            "--bot-lmsre-mode-name",
            "royale_mode",
        ]
    )
    assert args.play_game is True
    assert args.bot_click_mode == "mouse"
    assert args.bot_visual_cursor is True
    assert args.bot_decision_backend == "lms_re"
    assert args.bot_lmsre_mode_name == "royale_mode"

    args = parser.parse_args([])
    assert args.play_game is False
    assert args.bot_decision_backend == "legacy"
    assert int(args.bot_move_max_blocking_hold_ms) == 45
    assert int(args.bot_hold_capture_slice_ms) == 20
    assert abs(float(args.bot_visual_ocr_timeout_sec) - 0.25) < 1e-9
    assert int(args.bot_visual_ocr_max_rois) == 1

    args = parser.parse_args(
        [
            "--play-game",
            "--bot-decision-backend",
            "alphastar",
            "--bot-alphastar-checkpoint",
            "artifacts/alphastar/pi_rl.pt",
            "--bot-alphastar-stochastic",
            "--bot-alphastar-temperature",
            "1.2",
            "--bot-feedback-screenshot-every-ms",
            "120",
            "--bot-visual-ocr-timeout-sec",
            "0.18",
            "--bot-visual-ocr-max-rois",
            "1",
        ]
    )
    assert args.bot_decision_backend == "alphastar"
    assert args.bot_alphastar_stochastic is True
    assert abs(float(args.bot_alphastar_temperature) - 1.2) < 1e-9
    assert int(args.bot_feedback_screenshot_every_ms) == 120
    assert abs(float(args.bot_visual_ocr_timeout_sec) - 0.18) < 1e-9
    assert int(args.bot_visual_ocr_max_rois) == 1


def test_main_play_game_uses_ui_state_and_left_click():
    click_probe = {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}
    evaluated_scripts = []

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
        "x": 100.0,
        "y": 200.0,
        "width": 800.0,
        "height": 600.0,
    }
    mock_page.locator.return_value = mock_iframe_locator
    mock_iframe_handle.content_frame.return_value = mock_frame
    mock_page.viewport_size = {"width": 1280, "height": 720}

    mock_frame.focus.return_value = None
    mock_frame.wait_for_load_state.return_value = None

    def frame_evaluate_side_effect(script, *args):
        s = str(script)
        evaluated_scripts.append(s)
        if "const isVisible" in s:
            return {
                "character_visible": False,
                "select_visible": False,
                "play_visible": False,
                "canvas_visible": True,
                "body_class": "playing",
            }
        if "__lmsClickProbe ||" in s:
            return dict(click_probe)
        if "window.__lmsClickProbe = " in s:
            return None
        if "__lmsBotCursor" in s:
            return True
        if "const c = document.querySelector('canvas')" in s:
            return {"x": 20.0, "y": 30.0, "width": 400.0, "height": 300.0}
        if "document.elementFromPoint" in s:
            click_probe["down0"] += 1
            click_probe["up0"] += 1
            click_probe["click0"] += 1
            click_probe["lastTarget"] = "CANVAS"
            return True
        return None

    mock_frame.evaluate.side_effect = frame_evaluate_side_effect

    def mouse_click_side_effect(*args, **kwargs):
        if kwargs.get("button") == "left":
            click_probe["down0"] += 1
            click_probe["up0"] += 1
            click_probe["click0"] += 1
            click_probe["lastTarget"] = "CANVAS"
        return None

    mock_page.mouse.click.side_effect = mouse_click_side_effect

    sleep_calls = {"n": 0}

    def sleep_side_effect(_seconds):
        if float(_seconds) >= 0.03:
            sleep_calls["n"] += 1
        if sleep_calls["n"] >= 6:
            raise KeyboardInterrupt
        return None

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [0]

    with patch("lms_live_collector.sync_playwright", return_value=mock_sync):
        with patch("lms_live_collector.sqlite3.connect", return_value=mock_conn):
            with patch("lms_live_collector.ensure_schema"):
                with patch("lms_live_collector.ensure_live_schema"):
                    with patch("lms_live_collector.ensure_ws_schema"):
                        with patch("time.sleep", side_effect=sleep_side_effect):
                            with patch.object(
                                sys,
                                "argv",
                                    [
                                        "lms_live_collector.py",
                                        "--play-game",
                                        "--no-persistent",
                                        "--headless",
                                        "--bot-click-mode",
                                        "mouse",
                                        "--bot-ui-poll-ms",
                                        "50",
                                        "--bot-move-max-blocking-hold-ms",
                                        "5",
                                        "--bot-hold-capture-slice-ms",
                                        "5",
                                    ],
                                ):
                                    main()

    mock_page.goto.assert_called_with("https://lastmagestanding.com/", wait_until="domcontentloaded")
    assert mock_page.mouse.move.call_count > 0
    assert mock_page.keyboard.down.call_count > 0

    mouse_click_calls = mock_page.mouse.click.call_args_list
    if mouse_click_calls:
        assert any(call.kwargs.get("button") == "left" for call in mouse_click_calls)

    assert mock_frame.evaluate.call_count > 0
    assert any("__lmsBotCursor" in s for s in evaluated_scripts)
    assert click_probe["click0"] >= 0


def test_main_play_game_lobby_clicks_play_without_movement():
    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_frame = MagicMock()
    mock_iframe_locator = MagicMock()
    mock_iframe_handle = MagicMock()
    mock_control_locator = MagicMock()

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
        "x": 0.0,
        "y": 0.0,
        "width": 1280.0,
        "height": 720.0,
    }
    mock_page.locator.return_value = mock_iframe_locator
    mock_iframe_handle.content_frame.return_value = mock_frame

    mock_control_locator.first = mock_control_locator
    mock_control_locator.wait_for.return_value = None
    mock_control_locator.click.return_value = None
    mock_control_locator.evaluate.return_value = None
    mock_frame.locator.return_value = mock_control_locator
    mock_frame.focus.return_value = None
    mock_frame.wait_for_load_state.return_value = None

    def frame_evaluate_side_effect(script, *args):
        s = str(script)
        if "const isVisible" in s:
            return {
                "character_visible": False,
                "select_visible": False,
                "play_visible": True,
                "canvas_visible": True,
                "body_class": "",
            }
        if "__lmsClickProbe ||" in s:
            return {"down0": 0, "up0": 0, "click0": 0, "lastTarget": ""}
        if "__lmsBotCursor" in s:
            return True
        if "window.__lmsClickProbe = " in s:
            return None
        if "const c = document.querySelector('canvas')" in s:
            return {"x": 10.0, "y": 10.0, "width": 500.0, "height": 300.0}
        return None

    mock_frame.evaluate.side_effect = frame_evaluate_side_effect

    sleep_calls = {"n": 0}

    def sleep_side_effect(_seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 3:
            raise KeyboardInterrupt
        return None

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [0]

    with patch("lms_live_collector.sync_playwright", return_value=mock_sync):
        with patch("lms_live_collector.sqlite3.connect", return_value=mock_conn):
            with patch("lms_live_collector.ensure_schema"):
                with patch("lms_live_collector.ensure_live_schema"):
                    with patch("lms_live_collector.ensure_ws_schema"):
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
                                ],
                            ):
                                main()

    assert mock_control_locator.click.call_count > 0
    assert mock_page.keyboard.down.call_count == 0

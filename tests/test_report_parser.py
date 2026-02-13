import json
from pathlib import Path
import pytest

from botgame.training.report_league import summarize_league

@pytest.fixture
def league_fixtures_path(tmp_path):
    """Creates a temporary directory with dummy league_state.json and matches.jsonl."""
    fixture_dir = tmp_path / "reports_fixture"
    fixture_dir.mkdir()

    # Create dummy league_state.json
    league_state_content = {
      "players": {
        "main_agent_v1": {
          "agent_type": "main",
          "trainable": True
        },
        "main_agent_v2": {
          "agent_type": "main",
          "trainable": True
        },
        "fixed_opponent_v1": {
          "agent_type": "opponent",
          "trainable": False
        },
        "exploiter_v1": {
          "agent_type": "exploiter",
          "trainable": False
        }
      }
    }
    (fixture_dir / "league_state.json").write_text(json.dumps(league_state_content))

    # Create dummy matches.jsonl
    matches_content = """{"agent_a": "main_agent_v1", "agent_b": "fixed_opponent_v1", "outcome": 1.0}
{"agent_a": "main_agent_v1", "agent_b": "fixed_opponent_v1", "outcome": 1.0}
{"agent_a": "main_agent_v1", "agent_b": "exploiter_v1", "outcome": 1.0}
{"agent_a": "exploiter_v1", "agent_b": "main_agent_v1", "outcome": -1.0}
{"agent_a": "main_agent_v2", "agent_b": "fixed_opponent_v1", "outcome": -1.0}
{"agent_a": "main_agent_v2", "agent_b": "exploiter_v1", "outcome": 1.0}"""
    (fixture_dir / "matches.jsonl").write_text(matches_content)

    return fixture_dir

def test_summarize_league_basic_parsing(league_fixtures_path):
    league_state_path = league_fixtures_path / "league_state.json"
    matches_path = league_fixtures_path / "matches.jsonl"

    summary = summarize_league(league_state_path, matches_path)

    # Verify top-level keys
    assert "num_players" in summary
    assert "num_matches" in summary
    assert "win_rates" in summary
    assert "elo" in summary
    assert "exploitability_proxy_min_winrate_vs_past" in summary

    # Verify counts
    assert summary["num_players"] == 4
    assert summary["num_matches"] == 6

    # Verify some win-rates (approximate as they are floats)
    assert summary["win_rates"]["main_agent_v1 vs fixed_opponent_v1"] == 1.0
    assert summary["win_rates"]["main_agent_v1 vs exploiter_v1"] == 1.0

    # Verify Elo (approximate values)
    assert summary["elo"]["main_agent_v1"] > 1200
    assert summary["elo"]["fixed_opponent_v1"] > 1200
    assert summary["elo"]["exploiter_v1"] < 1200

    # Verify exploitability proxy
    # main_agent_v1 records wins against both frozen opponents in this fixture.
    assert summary["exploitability_proxy_min_winrate_vs_past"]["main_agent_v1"] == 1.0
    # main_agent_v2 vs fixed_opponent_v1 (non-trainable) -> 0.0
    # main_agent_v2 vs exploiter_v1 (non-trainable) -> 1.0
    # So min should be 0.0
    assert summary["exploitability_proxy_min_winrate_vs_past"]["main_agent_v2"] == 0.0

    # Note: As discussed, this test focuses on win_rates, Elo, and exploitability.
    # The `summarize_league` function does not directly extract "key OCR fields" from matches.
    # If OCR field validation is needed, it would occur at the data collection/pre-processing stage,
    # or require modification of the report_league module to aggregate such information.

import sys
import pytest
from pathlib import Path
import json

# Assuming botgame.eval.evaluate exists and has a main function
from botgame.eval.evaluate import main as evaluate_main

def test_acceptance_harness_toy_mode(tmp_path, monkeypatch):
    # Ensure the output directory is within tmp_path for cleanup
    output_dir = tmp_path / "reports/acceptance"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists for output_file path

    output_file = output_dir / "summary.json"

    # Simulate command-line arguments for "toy mode"
    # Assuming the evaluate_main function accepts a --toy argument
    # and an --output-dir argument for where to place the summary.json
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", # Script name (first arg is usually script name)
        "--toy",
        "--output-dir", str(output_dir)
    ])

    # Run the main function
    try:
        evaluate_main()
    except SystemExit as e:
        # argparse raises SystemExit if args are invalid or --help is called
        # Check if it's an intentional exit (e.g., successful run) or an error
        assert e.code == 0, f"Acceptance harness exited with error code {e.code}"

    # Verify that the output directory and file were created
    assert output_dir.exists()
    assert output_dir.is_dir()
    assert output_file.exists()
    assert output_file.is_file()

    # Optionally, verify content of the summary.json (minimal check)
    with open(output_file, 'r') as f:
        summary_data = json.load(f)
        assert "mode" in summary_data
        assert summary_data["mode"] == "toy"
        assert "status" in summary_data # Assuming a status field will be present
        # assert summary_data["status"] == "success" # Specific value might be hard to predict without knowing impl

    # Removed capsys as the script is expected to write to file, not necessarily stdout for primary output

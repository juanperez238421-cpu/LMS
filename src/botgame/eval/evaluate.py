from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acceptance evaluation entrypoint.")
    parser.add_argument("--toy", action="store_true", help="Run toy acceptance mode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/acceptance"),
        help="Directory where summary.json is written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.toy:
        raise SystemExit("Only --toy mode is currently implemented.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "toy",
        "status": "ok",
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

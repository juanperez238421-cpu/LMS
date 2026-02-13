from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import orjson


def tail_jsonl_dicts(path: str, sleep_s: float = 0.05) -> Iterator[Dict[str, Any]]:
    """
    Tail -f for JSONL with rotation/truncation support:
    - waits until file exists
    - seeks to end on first open
    - if file rotates or is recreated, reopens and continues from start of new file
    - if file truncates in place, seeks to beginning
    - yields dict frames.
    """
    p = Path(path)
    f = None
    first_open = True

    while True:
        if f is None:
            if not p.exists():
                time.sleep(0.2)
                continue
            try:
                f = p.open("rb")
                if first_open:
                    f.seek(0, 2)  # EOF on initial attach
                    first_open = False
                else:
                    f.seek(0, 0)  # new/rotated file: read from start
            except Exception:
                f = None
                time.sleep(0.2)
                continue

        try:
            line = f.readline()
        except Exception:
            try:
                f.close()
            except Exception:
                pass
            f = None
            time.sleep(sleep_s)
            continue

        if line:
            try:
                obj = orjson.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                pass
            continue

        # No new line yet; check truncation/rotation/recreate.
        try:
            if not p.exists():
                try:
                    f.close()
                except Exception:
                    pass
                f = None
                time.sleep(0.2)
                continue

            path_stat = p.stat()
            file_stat = os.fstat(f.fileno())

            rotated = (path_stat.st_ino != file_stat.st_ino) or (path_stat.st_dev != file_stat.st_dev)
            truncated = path_stat.st_size < f.tell()

            if rotated:
                try:
                    f.close()
                except Exception:
                    pass
                f = None
                continue

            if truncated:
                f.seek(0, 0)
                continue
        except Exception:
            try:
                f.close()
            except Exception:
                pass
            f = None
            time.sleep(sleep_s)
            continue

        time.sleep(sleep_s)

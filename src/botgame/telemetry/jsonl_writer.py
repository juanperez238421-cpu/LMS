from __future__ import annotations

import os
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional

import orjson

from .schema import TelemetryFrame


@dataclass
class TelemetryWriterConfig:
    path: str
    rate_hz: float = 10.0              # max publish rate from bot loop
    flush_every_s: float = 0.5         # flush interval (writer thread)
    queue_max: int = 256               # frames buffer; drop if full
    enabled: bool = True


class JsonlTelemetryWriter:
    """
    Ultra-light publisher:
      - call maybe_emit(frame) from bot loop
      - actual IO happens in background thread
      - drop-on-overload (never blocks bot)
    """
    def __init__(self, cfg: TelemetryWriterConfig):
        self.cfg = cfg
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=cfg.queue_max)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_emit_t = 0.0
        self._emit_period = 1.0 / max(cfg.rate_hz, 1e-6)

        os.makedirs(os.path.dirname(cfg.path), exist_ok=True)

        self._fh = None
        self._dropped = 0

    @property
    def dropped(self) -> int:
        return self._dropped

    def start(self) -> "JsonlTelemetryWriter":
        if not self.cfg.enabled:
            return self
        self._fh = open(self.cfg.path, "ab", buffering=1024 * 1024)
        self._thread = threading.Thread(target=self._run, name="telemetry-jsonl-writer", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None

    def maybe_emit(self, frame: TelemetryFrame) -> bool:
        """
        Rate-limited. Returns True if queued, False if skipped/dropped.
        """
        if not self.cfg.enabled:
            return False

        now = time.perf_counter()
        if (now - self._last_emit_t) < self._emit_period:
            return False
        self._last_emit_t = now

        try:
            b = orjson.dumps(frame.model_dump(exclude_none=True)) + b"\n"
            self._q.put_nowait(b)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def _run(self):
        assert self._fh is not None
        last_flush = time.perf_counter()

        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                chunk = None

            if chunk is not None:
                self._fh.write(chunk)

            now = time.perf_counter()
            if (now - last_flush) >= self.cfg.flush_every_s:
                self._fh.flush()
                last_flush = now

        # drain best-effort
        while True:
            try:
                chunk = self._q.get_nowait()
            except queue.Empty:
                break
            self._fh.write(chunk)
        self._fh.flush()

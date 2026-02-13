from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .jsonl_tail import tail_jsonl_dicts


app = FastAPI(title="LMS Telemetry Web GUI")

_clients: Set[WebSocket] = set()
_last_frame: Optional[Dict[str, Any]] = None
_last_frame_version = 0

_history: list[Dict[str, Any]] = []
_HISTORY_MAX = int(os.environ.get("LMS_TELEMETRY_HISTORY", "1200"))  # e.g. 120s @ 10Hz

# Backpressure for broadcaster -> WS
_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=int(os.environ.get("LMS_TELEMETRY_QUEUE", "64")))
_TELEMETRY_PATH = os.environ.get("LMS_TELEMETRY_PATH", "reports/runtime/telemetry_live.jsonl")
_WS_MAX_HZ = float(os.environ.get("LMS_TELEMETRY_WS_HZ", "10"))

_stats: Dict[str, float] = {
    "frames_seen": 0.0,
    "frames_enqueued": 0.0,
    "frames_dropped": 0.0,
    "broadcast_frames": 0.0,
    "broadcast_hz": 0.0,
}
_broadcast_window_start = time.perf_counter()
_broadcast_window_count = 0.0


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _push_history(frame: Dict[str, Any]) -> None:
    global _history
    _history.append(frame)
    if len(_history) > _HISTORY_MAX:
        _history = _history[-_HISTORY_MAX:]


@app.get("/health")
def health():
    return {
        "ok": True,
        "clients": len(_clients),
        "history": len(_history),
        "stats": dict(_stats),
        "config": {
            "telemetry_path": _TELEMETRY_PATH,
            "ws_hz": _WS_MAX_HZ,
            "history": _HISTORY_MAX,
            "queue": _queue.maxsize,
        },
    }


@app.get("/config")
def config():
    return JSONResponse(
        {
            "telemetry_path": _TELEMETRY_PATH,
            "ws_hz": _WS_MAX_HZ,
            "history": _HISTORY_MAX,
            "queue": _queue.maxsize,
        }
    )


@app.get("/last")
def last():
    return JSONResponse(_last_frame or {})


@app.get("/history")
def history(n: int = 600):
    n = max(1, min(int(n), _HISTORY_MAX))
    return JSONResponse(_history[-n:])


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    try:
        if _last_frame:
            await ws.send_json(_last_frame)

        # Keep alive (browser WS does not require ping frames, but this helps proxies)
        while True:
            await asyncio.sleep(30)
            try:
                await ws.send_json({"_type": "ping", "ts": int(time.time()*1000)})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


async def producer(telemetry_path: str, max_hz: float) -> None:
    """
    Tails JSONL and stores latest frame in memory/history.
    """
    global _last_frame, _last_frame_version

    tail_iter = tail_jsonl_dicts(telemetry_path)
    while True:
        frame = await asyncio.to_thread(next, tail_iter)
        _last_frame = frame
        _last_frame_version = int(_last_frame_version) + 1
        _stats["frames_seen"] = float(_stats.get("frames_seen", 0.0)) + 1.0
        _push_history(frame)
        await asyncio.sleep(0)


async def publisher(max_hz: float) -> None:
    """
    Publishes the latest frame at most max_hz.
    This avoids burst under-sampling when JSONL flushes in chunks.
    """
    global _last_frame_version
    period = 1.0 / max(float(max_hz), 1e-6)
    last_sent_version = -1
    while True:
        await asyncio.sleep(period)
        frame = _last_frame
        version = int(_last_frame_version)
        if frame is None or version <= last_sent_version:
            continue
        last_sent_version = version
        # bounded queue: drop oldest to keep most recent
        if _queue.full():
            try:
                _queue.get_nowait()
                _stats["frames_dropped"] = float(_stats.get("frames_dropped", 0.0)) + 1.0
            except Exception:
                pass
        try:
            _queue.put_nowait(frame)
            _stats["frames_enqueued"] = float(_stats.get("frames_enqueued", 0.0)) + 1.0
        except Exception:
            pass
        await asyncio.sleep(0)


async def consumer() -> None:
    """
    Broadcast frames from queue to all clients.
    """
    while True:
        global _broadcast_window_start, _broadcast_window_count
        frame = await _queue.get()
        _stats["broadcast_frames"] = float(_stats.get("broadcast_frames", 0.0)) + 1.0
        _broadcast_window_count += 1.0
        now = time.perf_counter()
        dt = now - _broadcast_window_start
        if dt >= 1.0:
            _stats["broadcast_hz"] = float(_broadcast_window_count / dt)
            _broadcast_window_count = 0.0
            _broadcast_window_start = now

        dead = []
        for c in list(_clients):
            try:
                await c.send_json(frame)
            except Exception:
                dead.append(c)
        for d in dead:
            _clients.discard(d)
        _queue.task_done()


@app.on_event("startup")
async def on_startup():
    global _TELEMETRY_PATH, _WS_MAX_HZ
    telemetry_path = os.environ.get("LMS_TELEMETRY_PATH", _TELEMETRY_PATH)
    max_hz = _safe_float(os.environ.get("LMS_TELEMETRY_WS_HZ", _WS_MAX_HZ), _WS_MAX_HZ)
    _TELEMETRY_PATH = str(telemetry_path)
    _WS_MAX_HZ = max(0.5, float(max_hz))

    Path(telemetry_path).parent.mkdir(parents=True, exist_ok=True)

    asyncio.create_task(producer(_TELEMETRY_PATH, _WS_MAX_HZ))
    asyncio.create_task(publisher(_WS_MAX_HZ))
    asyncio.create_task(consumer())


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(frontend_dir / "index.html"))

#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_URL = (
    "https://us-central1-last-mage-production.cloudfunctions.net/"
    "GameFunction/LogEvents"
)
DEFAULT_MAX_BODY_BYTES = 2_000_000
DEFAULT_DB_PATH = "data/processed/sqlite/lms_events.db"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_timestamp INTEGER,
            platform TEXT,
            source TEXT,
            raw_event TEXT NOT NULL,
            ingested_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS event_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value_text TEXT,
            value_num REAL,
            value_type TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_name ON events(event_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_params_event_id ON event_params(event_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_params_key ON event_params(key)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            url TEXT NOT NULL,
            method TEXT NOT NULL,
            content_type TEXT,
            body_text TEXT,
            body_hash TEXT,
            body_size INTEGER,
            source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_request_id INTEGER,
            captured_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            url TEXT NOT NULL,
            json_obj TEXT NOT NULL,
            source TEXT,
            FOREIGN KEY (raw_request_id) REFERENCES raw_requests(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_requests_url ON raw_requests(url)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_requests_hash ON raw_requests(body_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_events_req_id ON raw_events(raw_request_id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL UNIQUE,
            captured_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
            damage_done REAL,
            damage_taken REAL,
            kills REAL,
            crafted REAL,
            items_crafted REAL,
            ocr_confidence REAL,
            screenshot_path TEXT,
            source TEXT,
            raw_json TEXT,
            FOREIGN KEY (event_id) REFERENCES events(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_match_stats_event_id ON match_stats(event_id)"
    )
    conn.commit()


def flatten_values(value: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_values(nested_value, next_prefix)
        return

    if isinstance(value, list):
        for idx, nested_value in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from flatten_values(nested_value, next_prefix)
        return

    yield (prefix, value)


def to_param_row(param_key: str, value: Any) -> Tuple[str, Optional[str], Optional[float], str]:
    if value is None:
        return (param_key, None, None, "null")

    if isinstance(value, bool):
        return (param_key, str(value).lower(), 1.0 if value else 0.0, "bool")

    if isinstance(value, (int, float)):
        return (param_key, str(value), float(value), "number")

    return (param_key, str(value), None, "string")


def cap_text_bytes(text: str, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> str:
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def body_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def insert_raw_request(
    conn: sqlite3.Connection,
    url: str,
    method: str,
    content_type: Optional[str],
    body_text: Optional[str],
    source: str,
    captured_at: Optional[int] = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> int:
    stored_text: Optional[str] = None
    if isinstance(body_text, str):
        stored_text = cap_text_bytes(body_text, max_bytes=max_body_bytes)
    raw_bytes = stored_text.encode("utf-8", errors="ignore") if stored_text else b""
    cur = conn.execute(
        """
        INSERT INTO raw_requests (
            captured_at, url, method, content_type, body_text, body_hash, body_size, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            captured_at if captured_at is not None else int(time.time()),
            url,
            method,
            content_type,
            stored_text,
            body_sha256(stored_text) if stored_text is not None else None,
            len(raw_bytes) if stored_text is not None else 0,
            source,
        ),
    )
    return int(cur.lastrowid)


def insert_raw_event(
    conn: sqlite3.Connection,
    raw_request_id: Optional[int],
    url: str,
    obj: Any,
    source: str,
    captured_at: Optional[int] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO raw_events (raw_request_id, captured_at, url, json_obj, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            raw_request_id,
            captured_at if captured_at is not None else int(time.time()),
            url,
            json.dumps(obj, ensure_ascii=True),
            source,
        ),
    )
    return int(cur.lastrowid)


def iter_json_objects_from_text(body_text: str) -> Iterable[Any]:
    text = body_text.strip()
    if not text:
        return

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    else:
        if isinstance(parsed, (dict, list)):
            yield parsed
            return

    for line in body_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed_line = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_line, (dict, list)):
            yield parsed_line


def payload_candidates_from_obj(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        return

    if isinstance(obj, list):
        dict_items = [item for item in obj if isinstance(item, dict)]
        if dict_items:
            yield {"events": dict_items}


def extract_har_content_type(request: Dict[str, Any], post_data: Dict[str, Any]) -> Optional[str]:
    mime = post_data.get("mimeType")
    if isinstance(mime, str) and mime.strip():
        return mime.strip()

    headers = request.get("headers", [])
    if isinstance(headers, list):
        for header in headers:
            if not isinstance(header, dict):
                continue
            name = str(header.get("name", "")).lower()
            if name == "content-type":
                value = header.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return None


def insert_event(
    conn: sqlite3.Connection,
    event: Dict[str, Any],
    platform: Optional[str],
    source: str,
) -> int:
    event_name = event.get("eventName") or "unknown"
    event_ts = event.get("eventTimestamp")
    raw_event = json.dumps(event, ensure_ascii=True)
    cur = conn.execute(
        """
        INSERT INTO events (event_name, event_timestamp, platform, source, raw_event)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event_name, event_ts, platform, source, raw_event),
    )
    event_id = int(cur.lastrowid)

    params = event.get("eventParameters", {})
    if isinstance(params, dict):
        rows: List[Tuple[int, str, Optional[str], Optional[float], str]] = []
        for key, value in flatten_values(params):
            if not key:
                continue
            p_key, p_text, p_num, p_type = to_param_row(key, value)
            rows.append((event_id, p_key, p_text, p_num, p_type))
        if rows:
            conn.executemany(
                """
                INSERT INTO event_params (event_id, key, value_text, value_num, value_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    return event_id


def parse_payload(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    platform = None
    events: List[Dict[str, Any]] = []

    data_obj = payload.get("data")
    if isinstance(data_obj, str):
        try:
            data_obj = json.loads(data_obj)
        except json.JSONDecodeError:
            data_obj = None

    if isinstance(data_obj, dict):
        data = data_obj
        platform = data.get("platform")
        maybe_events = data.get("events")
        if isinstance(maybe_events, list):
            events = [e for e in maybe_events if isinstance(e, dict)]
            return events, platform
        maybe_event = data.get("event")
        if isinstance(maybe_event, dict):
            return [maybe_event], platform

    if isinstance(payload.get("events"), list):
        events = [e for e in payload["events"] if isinstance(e, dict)]
        platform = payload.get("platform")
        return events, platform

    maybe_event = payload.get("event")
    if isinstance(maybe_event, dict):
        platform = payload.get("platform")
        return [maybe_event], platform

    if payload.get("eventName"):
        platform = payload.get("platform")
        return [payload], platform

    raise ValueError("No se encontraron eventos en payload (esperado data.events o events).")


def ingest_payload(
    conn: sqlite3.Connection,
    payload: Dict[str, Any],
    source: str,
) -> int:
    return len(ingest_payload_records(conn, payload, source))


def ingest_payload_records(
    conn: sqlite3.Connection,
    payload: Dict[str, Any],
    source: str,
) -> List[Tuple[int, Dict[str, Any]]]:
    events, platform = parse_payload(payload)
    inserted: List[Tuple[int, Dict[str, Any]]] = []
    for event in events:
        event_id = insert_event(conn, event, platform=platform, source=source)
        inserted.append((event_id, event))
    conn.commit()
    return inserted


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} no contiene un JSON objeto.")
    return data


def ingest_har(
    conn: sqlite3.Connection,
    har_path: Path,
    endpoint_substring: str,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> int:
    with har_path.open("r", encoding="utf-8-sig") as f:
        har = json.load(f)

    entries = (
        har.get("log", {}).get("entries", [])
        if isinstance(har, dict)
        else []
    )
    if not isinstance(entries, list):
        raise ValueError("HAR inválido: log.entries no es una lista.")

    ingested = 0
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        request = entry.get("request", {})
        if not isinstance(request, dict):
            continue

        url = str(request.get("url", ""))
        method = str(request.get("method", "")).upper()
        if method != "POST":
            continue

        post_data = request.get("postData", {})
        if not isinstance(post_data, dict):
            continue
        body = post_data.get("text")
        if not isinstance(body, str):
            continue

        source = f"{har_path.name}#entry_{idx}"
        content_type = extract_har_content_type(request, post_data)
        raw_request_id = insert_raw_request(
            conn=conn,
            url=url,
            method=method,
            content_type=content_type,
            body_text=body,
            source=source,
            max_body_bytes=max_body_bytes,
        )

        parsed_objects = list(iter_json_objects_from_text(body))
        for obj in parsed_objects:
            insert_raw_event(
                conn=conn,
                raw_request_id=raw_request_id,
                url=url,
                obj=obj,
                source=source,
            )

        if endpoint_substring not in url:
            continue

        for obj in parsed_objects:
            for payload in payload_candidates_from_obj(obj):
                try:
                    ingested += ingest_payload(conn, payload, source=source)
                except ValueError:
                    continue

    conn.commit()

    return ingested


def maybe_post_payload(
    payload: Dict[str, Any],
    url: str,
    token: str,
) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Falta 'requests'. Instala con: pip install requests"
        ) from exc

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, json=payload, timeout=30)


def print_match_report(conn: sqlite3.Connection, limit: int) -> bool:
    query = """
    SELECT
        e.id,
        e.event_timestamp,
        e.platform,
        MAX(CASE WHEN p.key='damage_done' THEN p.value_num END) AS damage_done,
        MAX(CASE WHEN p.key='damage_taken' THEN p.value_num END) AS damage_taken,
        MAX(CASE WHEN p.key='leaderboard_rank' THEN p.value_num END) AS leaderboard_rank,
        MAX(CASE WHEN p.key='character_selected' THEN p.value_text END) AS character_selected,
        MAX(CASE WHEN p.key='map_name' THEN p.value_text END) AS map_name
    FROM events e
    LEFT JOIN event_params p ON p.event_id = e.id
    WHERE e.event_name='games_played/match_end'
    GROUP BY e.id, e.event_timestamp, e.platform
    ORDER BY e.event_timestamp DESC, e.id DESC
    LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()
    if not rows:
        print("No hay eventos games_played/match_end en la base.")
        return False

    print(
        "event_id | timestamp_ms | platform | damage_done | damage_taken | rank | character | map"
    )
    for row in rows:
        print(
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]}"
        )

    print("\nParametros detectados para match_end:")
    key_rows = conn.execute(
        """
        SELECT p.key, COUNT(*) AS n
        FROM event_params p
        JOIN events e ON e.id = p.event_id
        WHERE e.event_name='games_played/match_end'
        GROUP BY p.key
        ORDER BY n DESC, p.key
        """
    ).fetchall()
    for key, count in key_rows:
        print(f"- {key}: {count}")
    return True


def print_round_activity_report(conn: sqlite3.Connection, limit: int) -> bool:
    query = """
    SELECT
        e.id,
        e.event_name,
        e.event_timestamp,
        MAX(CASE WHEN p.key='character_selected' THEN p.value_text END) AS character_selected,
        MAX(CASE WHEN p.key='map_name' THEN p.value_text END) AS map_name,
        MAX(CASE WHEN p.key='num_humans' THEN p.value_num END) AS num_humans,
        MAX(CASE WHEN p.key='num_bots' THEN p.value_num END) AS num_bots,
        MAX(CASE WHEN p.key='num_humans_at_start' THEN p.value_num END) AS num_humans_at_start,
        MAX(CASE WHEN p.key='num_bots_at_start' THEN p.value_num END) AS num_bots_at_start,
        MAX(CASE WHEN p.key='game_time' THEN p.value_num END) AS game_time,
        MAX(CASE WHEN p.key='team_size' THEN p.value_num END) AS team_size,
        MAX(CASE WHEN p.key='matchmaking_tier' THEN p.value_text END) AS matchmaking_tier,
        MAX(CASE WHEN p.key='stage_id' THEN p.value_num END) AS stage_id
    FROM events e
    LEFT JOIN event_params p ON p.event_id = e.id
    WHERE e.event_name IN (
        'games_played/round_start',
        'games_played/round_spawn',
        'games_played/round_after_spawn',
        'frame_rate/playing'
    )
    GROUP BY e.id, e.event_name, e.event_timestamp
    ORDER BY e.id DESC
    LIMIT ?
    """
    rows = conn.execute(query, (limit,)).fetchall()
    if not rows:
        print("No hay eventos de ronda para mostrar.")
        return False

    print(
        "event_id | event_name | timestamp_ms | character | map | humans | bots | humans_start | bots_start | game_time | team_size | tier | stage_id"
    )
    for row in rows:
        print(
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} | {row[10]} | {row[11]} | {row[12]}"
        )
    return True


def print_metric_availability(conn: sqlite3.Connection) -> None:
    wanted_keys = [
        "damage_done",
        "damage_taken",
        "kills",
        "eliminations",
        "crafted_items",
        "items_crafted",
        "player_name",
        "enemy_name",
    ]
    rows = conn.execute(
        """
        SELECT DISTINCT p.key
        FROM event_params p
        JOIN events e ON e.id = p.event_id
        WHERE e.event_name LIKE 'games_played/%'
        """
    ).fetchall()
    available = {row[0] for row in rows}
    missing = [key for key in wanted_keys if key not in available]
    present = [key for key in wanted_keys if key in available]

    print("\nDisponibilidad de metricas objetivo:")
    print(f"- presentes: {', '.join(present) if present else 'ninguna'}")
    print(f"- faltantes: {', '.join(missing) if missing else 'ninguna'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingesta y reporte de eventos de Last Mage en SQLite."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Ruta de SQLite.")
    parser.add_argument(
        "--payload-json",
        help="Archivo JSON con payload de eventos (data.events o events).",
    )
    parser.add_argument(
        "--har",
        help="Archivo HAR exportado desde DevTools para extraer requests a LogEvents.",
    )
    parser.add_argument(
        "--endpoint-substring",
        default="/GameFunction/LogEvents",
        help="Texto para identificar requests objetivo en HAR.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=DEFAULT_MAX_BODY_BYTES,
        help="Limite de bytes para guardar body_text en raw_requests.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Enviar payload-json al endpoint remoto antes de guardar.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint LogEvents.")
    parser.add_argument(
        "--token-env",
        default="LMS_BEARER_TOKEN",
        help="Nombre de variable de entorno con el Bearer token.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Mostrar reporte de partidas (match_end).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Límite de filas para --report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    total_ingested = 0

    if args.payload_json:
        payload_path = Path(args.payload_json)
        payload = load_json_file(payload_path)
        if args.post:
            token = os.getenv(args.token_env)
            if not token:
                raise SystemExit(
                    f"Falta token en variable de entorno {args.token_env}."
                )
            response = maybe_post_payload(payload, url=args.url, token=token)
            print(f"POST status: {response.status_code}")
            body_preview = response.text[:500]
            print(f"POST response (primeros 500 chars): {body_preview}")

        total_ingested += ingest_payload(conn, payload, source=payload_path.name)

    if args.har:
        har_path = Path(args.har)
        total_ingested += ingest_har(
            conn,
            har_path=har_path,
            endpoint_substring=args.endpoint_substring,
            max_body_bytes=max(1, int(args.max_body_bytes)),
        )

    if args.payload_json or args.har:
        print(f"Eventos guardados: {total_ingested}")

    if args.report:
        has_match = print_match_report(conn, args.limit)
        if not has_match:
            print("\nActividad de ronda (fallback):")
            print_round_activity_report(conn, args.limit)
        print_metric_availability(conn)

    conn.close()


if __name__ == "__main__":
    main()

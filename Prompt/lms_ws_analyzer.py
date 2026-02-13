#!/usr/bin/env python3
import argparse
import base64
import json
import sqlite3
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analiza frames websocket guardados en SQLite."
    )
    parser.add_argument("--db", default="data/processed/sqlite/lms_events.db", help="Ruta SQLite.")
    parser.add_argument(
        "--url-like",
        default="",
        help="Filtro SQL para ws_url (ej: robotoserver, firebaseio).",
    )
    parser.add_argument(
        "--contains",
        default="",
        help="Texto a buscar en text_preview (lowercase).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limite de filas detalle.",
    )
    parser.add_argument(
        "--export-samples",
        default="data/raw/ws/ws_samples",
        help="Directorio para exportar payload_b64 como binario con metadatos.",
    )
    parser.add_argument(
        "--export-count",
        type=int,
        default=20,
        help="Cantidad maxima de frames a exportar cuando se usa --export-samples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    conn = sqlite3.connect(Path(args.db))
    c = conn.cursor()

    where = []
    params = []
    if args.url_like:
        where.append("coalesce(ws_url,'') like ?")
        params.append(f"%{args.url_like}%")
    if args.contains:
        where.append("lower(coalesce(text_preview,'')) like ?")
        params.append(f"%{args.contains.lower()}%")

    where_sql = ""
    if where:
        where_sql = "WHERE " + " AND ".join(where)

    print("Resumen por socket:")
    rows = c.execute(
        f"""
        SELECT coalesce(ws_url,'(null)') AS ws_url, COUNT(*) AS n
        FROM ws_frames
        {where_sql}
        GROUP BY ws_url
        ORDER BY n DESC, ws_url
        """,
        params,
    ).fetchall()
    for ws_url, n in rows:
        print(f"- {ws_url}: {n}")

    print("\nResumen por tipo:")
    rows = c.execute(
        f"""
        SELECT opcode, decoded_kind, COUNT(*) AS n
        FROM ws_frames
        {where_sql}
        GROUP BY opcode, decoded_kind
        ORDER BY n DESC, opcode
        """,
        params,
    ).fetchall()
    for opcode, decoded_kind, n in rows:
        print(f"- opcode={opcode} kind={decoded_kind}: {n}")

    print("\nTop keyword_hit:")
    rows = c.execute(
        f"""
        SELECT keyword_hit, COUNT(*) AS n
        FROM ws_frames
        {where_sql}
        GROUP BY keyword_hit
        ORDER BY n DESC
        """,
        params,
    ).fetchall()
    for keyword_hit, n in rows:
        print(f"- {keyword_hit}: {n}")

    print("\nMuestras:")
    rows = c.execute(
        f"""
        SELECT id, ws_url, direction, opcode, payload_len, decoded_kind, keyword_hit,
               substr(coalesce(text_preview,''), 1, 220), payload_b64
        FROM ws_frames
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
        """,
        [*params, args.limit],
    ).fetchall()

    for row in rows:
        frame_id, ws_url, direction, opcode, payload_len, decoded_kind, keyword_hit, text_preview, payload_b64 = row
        head = f"{frame_id} | {direction} | opcode={opcode} | len={payload_len} | {decoded_kind} | {keyword_hit}"
        print(head)
        if text_preview:
            print(f"  text: {text_preview}")
        if payload_b64:
            try:
                raw = base64.b64decode(payload_b64)
                print(f"  b64(raw first 24 bytes hex): {raw[:24].hex()}")
            except Exception:
                print("  b64(raw): decode_error")

    if args.export_samples:
        export_dir = Path(args.export_samples)
        export_dir.mkdir(parents=True, exist_ok=True)

        export_where = list(where)
        export_where.append("payload_b64 IS NOT NULL")
        export_where_sql = "WHERE " + " AND ".join(export_where)
        export_rows = c.execute(
            f"""
            SELECT id, ws_url, direction, opcode, payload_len, decoded_kind, keyword_hit, payload_b64
            FROM ws_frames
            {export_where_sql}
            ORDER BY id DESC
            LIMIT ?
            """,
            [*params, max(1, int(args.export_count))],
        ).fetchall()

        exported = []
        for frame_id, ws_url, direction, opcode, payload_len, decoded_kind, keyword_hit, payload_b64 in export_rows:
            try:
                raw = base64.b64decode(payload_b64)
            except Exception:
                continue

            safe_direction = (direction or "unk").replace("/", "_")
            filename = f"frame_{frame_id}_{safe_direction}_op{opcode}.bin"
            file_path = export_dir / filename
            file_path.write_bytes(raw)
            exported.append(
                {
                    "id": frame_id,
                    "file": filename,
                    "ws_url": ws_url,
                    "direction": direction,
                    "opcode": opcode,
                    "payload_len": payload_len,
                    "decoded_kind": decoded_kind,
                    "keyword_hit": keyword_hit,
                }
            )

        (export_dir / "metadata.json").write_text(
            json.dumps({"count": len(exported), "frames": exported}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        print(f"\nExportados {len(exported)} frames binarios en: {export_dir}")

    conn.close()


if __name__ == "__main__":
    main()

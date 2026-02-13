#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from lms_events_collector import iter_json_objects_from_text


DEFAULT_KEYWORDS = "damage,dmg,kill,kills,elim,frag,craft,item,create"
DEFAULT_DB_PATH = "data/processed/sqlite/lms_events.db"


def parse_keywords(raw: str) -> List[str]:
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def build_like_clause(column: str, keywords: Sequence[str]) -> Tuple[str, List[str]]:
    clause = " OR ".join([f"lower({column}) LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords]
    return clause, params


def clamp_snippet(text: str, keyword: str, window: int) -> str:
    lower = text.lower()
    idx = lower.find(keyword.lower())
    if idx < 0:
        return text[: max(40, window * 2)].replace("\n", " ").replace("\r", " ")
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    return text[start:end].replace("\n", " ").replace("\r", " ")


def collect_keyword_paths(
    node: Any,
    keywords: Sequence[str],
    path: str = "$",
    hits: List[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    if hits is None:
        hits = []

    if isinstance(node, dict):
        for key, value in node.items():
            key_s = str(key)
            key_l = key_s.lower()
            child_path = f"{path}.{key_s}"
            for kw in keywords:
                if kw in key_l:
                    hits.append({"keyword": kw, "path": child_path, "kind": "key"})
            collect_keyword_paths(value, keywords, child_path, hits)
        return hits

    if isinstance(node, list):
        for idx, item in enumerate(node):
            collect_keyword_paths(item, keywords, f"{path}[{idx}]", hits)
        return hits

    scalar_text = str(node).lower()
    for kw in keywords:
        if kw in scalar_text:
            hits.append({"keyword": kw, "path": path, "kind": "value"})
    return hits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mineria offline de raw_requests/raw_events en SQLite."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Ruta SQLite.")
    parser.add_argument(
        "--keywords",
        default=DEFAULT_KEYWORDS,
        help="CSV de keywords a buscar.",
    )
    parser.add_argument(
        "--url-limit",
        type=int,
        default=30,
        help="Cantidad de URLs candidatas a mostrar.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=40,
        help="Cantidad de muestras (requests) a mostrar.",
    )
    parser.add_argument(
        "--snippet-window",
        type=int,
        default=120,
        help="Contexto a izquierda/derecha para snippets.",
    )
    parser.add_argument(
        "--json-path-limit",
        type=int,
        default=30,
        help="Cantidad maxima de paths JSON a mostrar.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Ruta opcional para guardar reporte JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    keywords = parse_keywords(args.keywords)
    if not keywords:
        raise SystemExit("Debes pasar al menos una keyword en --keywords.")

    conn = sqlite3.connect(Path(args.db))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    clause, clause_params = build_like_clause("body_text", keywords)

    url_rows = c.execute(
        f"""
        SELECT url, COUNT(*) AS hits
        FROM raw_requests
        WHERE body_text IS NOT NULL AND ({clause})
        GROUP BY url
        ORDER BY hits DESC, url
        LIMIT ?
        """,
        [*clause_params, max(1, int(args.url_limit))],
    ).fetchall()

    sample_rows = c.execute(
        f"""
        SELECT id, captured_at, url, source, body_text
        FROM raw_requests
        WHERE body_text IS NOT NULL AND ({clause})
        ORDER BY id DESC
        LIMIT ?
        """,
        [*clause_params, max(1, int(args.sample_limit))],
    ).fetchall()

    print("URLs candidatas (raw_requests.body_text):")
    if not url_rows:
        print("- sin hits")
    else:
        for row in url_rows:
            print(f"- {row['url']} -> {row['hits']}")

    print("\nMuestras con snippet:")
    snippets_out: List[Dict[str, Any]] = []
    json_path_hits: List[Dict[str, Any]] = []
    for row in sample_rows:
        body_text = row["body_text"] or ""
        matched_kw = next((kw for kw in keywords if kw in body_text.lower()), "")
        snippet = clamp_snippet(body_text, matched_kw or keywords[0], int(args.snippet_window))
        print(f"- req_id={row['id']} kw={matched_kw} url={row['url']}")
        print(f"  snippet: {snippet}")
        snippets_out.append(
            {
                "request_id": row["id"],
                "captured_at": row["captured_at"],
                "url": row["url"],
                "source": row["source"],
                "keyword": matched_kw,
                "snippet": snippet,
            }
        )

        for obj in iter_json_objects_from_text(body_text):
            for hit in collect_keyword_paths(obj, keywords):
                json_path_hits.append(
                    {
                        "request_id": row["id"],
                        "url": row["url"],
                        "source": row["source"],
                        "keyword": hit["keyword"],
                        "path": hit["path"],
                        "kind": hit["kind"],
                    }
                )

    print("\nJSON path examples:")
    if not json_path_hits:
        print("- sin paths JSON detectados")
    else:
        seen = set()
        shown = 0
        for hit in json_path_hits:
            key = (hit["url"], hit["keyword"], hit["path"], hit["kind"])
            if key in seen:
                continue
            seen.add(key)
            print(
                f"- kw={hit['keyword']} kind={hit['kind']} path={hit['path']} "
                f"url={hit['url']}"
            )
            shown += 1
            if shown >= max(1, int(args.json_path_limit)):
                break

    if args.report_json:
        out = {
            "keywords": keywords,
            "url_hits": [dict(row) for row in url_rows],
            "samples": snippets_out,
            "json_path_hits": json_path_hits,
        }
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"\nReporte JSON guardado en: {out_path}")

    conn.close()


if __name__ == "__main__":
    main()

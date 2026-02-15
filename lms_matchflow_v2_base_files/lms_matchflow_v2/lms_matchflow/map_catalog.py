from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(frozen=True)
class MapDef:
    map_id: str
    display_name: str
    filename: str
    size_class: str  # "large" | "small"


DEFAULT_MAPS: List[MapDef] = [
    MapDef("castillos_deserticos", "CASTILLOS DESÉRTICOS", "CASTILLOS DESÉRTICOS.png", "large"),
    MapDef("la_finca", "LA FINCA", "LA FINCA.png", "large"),
    MapDef("asentamiento_desertico", "ASENTAMIENTO DESÉRTICO", "ASENTAMIENTO DESÉRTICO.png", "small"),
    MapDef("fortaleza_de_lava", "FORTALEZA DE LAVA", "FORTALEZA DE LAVA.png", "small"),
    MapDef("jardines_congelados", "JARDINES CONGELADOS", "JARDINES CONGELADOS.png", "small"),
]


def match_map_name_from_ocr(ocr_text: str, maps: List[MapDef] = DEFAULT_MAPS) -> Optional[MapDef]:
    """Match a map by OCR text (preferred if the UI shows a map name)."""
    t = _norm(ocr_text)
    if not t:
        return None

    for m in maps:
        if _norm(m.display_name) in t:
            return m

    tset = set(t.split())
    best: Tuple[int, Optional[MapDef]] = (0, None)
    for m in maps:
        mset = set(_norm(m.display_name).split())
        score = len(tset.intersection(mset))
        if score > best[0]:
            best = (score, m)

    return best[1] if best[0] >= 2 else None

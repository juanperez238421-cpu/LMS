from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(frozen=True)
class MapDef:
    map_id: str
    display_name: str
    relative_path: str
    size_class: str  # "big" | "small"

    @property
    def filename(self) -> str:
        """Backward compatibility for callers that still use `filename`."""
        return self.relative_path


DEFAULT_MAPS: List[MapDef] = [
    MapDef("castillos_deserticos", "CASTILLOS DESÉRTICOS", "Big/CASTILLOS DESÉRTICOS.png", "big"),
    MapDef("la_finca", "LA FINCA", "Big/LA FINCA.png", "big"),
    MapDef("asentamiento_desertico", "ASENTAMIENTO DESÉRTICO", "Small/ASENTAMIENTO DESÉRTICO.png", "small"),
    MapDef("fortaleza_de_lava", "FORTALEZA DE LAVA", "Small/FORTALEZA DE LAVA.png", "small"),
    MapDef("jardines_congelados", "JARDINES CONGELADOS", "Small/JARDINES CONGELADOS.png", "small"),
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

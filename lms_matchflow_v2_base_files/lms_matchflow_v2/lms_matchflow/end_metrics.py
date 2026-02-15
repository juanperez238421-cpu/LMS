from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import re


@dataclass(frozen=True)
class EndOfCombatMetrics:
    time_alive_sec: Optional[int] = None
    eliminations: Optional[int] = None
    objects_built: Optional[int] = None
    raw: Tuple[str, ...] = ()


_MMSS = re.compile(r"\b(?P<m>\d{1,2}):(?P<s>\d{2})\b")
_INT = re.compile(r"(?<!\d)(?P<n>\d{1,4})(?!\d)")


def parse_mmss_to_seconds(text: str) -> Optional[int]:
    m = _MMSS.search(text or "")
    if not m:
        return None
    mm = int(m.group("m"))
    ss = int(m.group("s"))
    if ss >= 60:
        return None
    return mm * 60 + ss


def parse_first_int(text: str) -> Optional[int]:
    m = _INT.search(text or "")
    if not m:
        return None
    return int(m.group("n"))


def extract_end_of_combat_metrics(ocr_by_field: Dict[str, str]) -> EndOfCombatMetrics:
    time_txt = ocr_by_field.get("end_time_alive", "")
    elim_txt = ocr_by_field.get("end_eliminations", "")
    built_txt = ocr_by_field.get("end_objects_built", "")

    return EndOfCombatMetrics(
        time_alive_sec=parse_mmss_to_seconds(time_txt),
        eliminations=parse_first_int(elim_txt),
        objects_built=parse_first_int(built_txt),
        raw=(time_txt, elim_txt, built_txt),
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from PIL import Image, ImageOps

from .map_catalog import MapDef, DEFAULT_MAPS


def _to_gray(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def dhash64(img: Image.Image, size: int = 9) -> int:
    """Compute 64-bit dHash (difference hash)."""
    g = _to_gray(img).resize((size, size - 1), Image.BILINEAR)
    pixels = list(g.getdata())
    w, h = g.size
    bits = 0
    for y in range(h):
        row = pixels[y * w:(y + 1) * w]
        for x in range(w - 1):
            bits = (bits << 1) | (1 if row[x] > row[x + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _crop_regions(img: Image.Image) -> List[Image.Image]:
    """Multiple regions improve robustness against dynamic icons."""
    w, h = img.size
    return [
        img.crop((w * 0.20, h * 0.20, w * 0.80, h * 0.80)),
        img.crop((w * 0.35, h * 0.10, w * 0.65, h * 0.90)),
        img.crop((w * 0.10, h * 0.35, w * 0.90, h * 0.65)),
    ]


@dataclass
class MapFingerprintDB:
    """Loads reference map images and compares against a minimap crop."""

    ref_dirs: Union[Path, Sequence[Path], None] = None
    maps: Optional[List[MapDef]] = None
    ref_hashes: Optional[Dict[str, Tuple[int, int, int]]] = None

    def __post_init__(self) -> None:
        if self.maps is None:
            self.maps = DEFAULT_MAPS
        self.ref_hashes = {}
        self._normalized_ref_dirs = self._normalize_ref_dirs(self.ref_dirs)
        self._load()

    @staticmethod
    def _normalize_ref_dirs(ref_dirs: Union[Path, Sequence[Path], None]) -> Tuple[Path, ...]:
        if ref_dirs is None:
            return (Path("assets/Mapas/Big"), Path("assets/Mapas/Small"))
        if isinstance(ref_dirs, Path):
            return (ref_dirs,)
        return tuple(Path(p) for p in ref_dirs)

    @staticmethod
    def _size_subdir(size_class: str) -> str:
        size_norm = (size_class or "").strip().lower()
        return "Big" if size_norm in {"big", "large"} else "Small"

    def _candidate_paths_for_map(self, m: MapDef) -> Iterable[Path]:
        rel = Path(str(m.relative_path).replace("\\", "/"))
        rel_parts = rel.parts
        has_subdir = len(rel_parts) >= 2 and rel_parts[0].lower() in {"big", "small"}

        for base in self._normalized_ref_dirs:
            base_name = str(base.name).lower()
            if has_subdir:
                rel_sub = rel_parts[0].lower()
                rel_tail = Path(*rel_parts[1:])
                if base_name == rel_sub:
                    # Base already points to Big or Small.
                    yield base / rel_tail
                else:
                    # Base may point to assets/Mapas root.
                    yield base / rel
                    # Base may point to sibling folder (Big/Small), then parent is root.
                    yield base.parent / rel
            else:
                # If base already points to size folder, load directly from it.
                if base_name in {"big", "small"}:
                    yield base / rel
                else:
                    yield base / self._size_subdir(m.size_class) / rel

            # Backward compatibility: old map defs using bare filename.
            yield base / m.filename

    def _load(self) -> None:
        for m in self.maps or []:
            selected: Optional[Path] = None
            for p in self._candidate_paths_for_map(m):
                if p.exists():
                    selected = p
                    break
            if selected is None:
                continue

            img = Image.open(selected).convert("RGBA")
            regs = _crop_regions(img)
            self.ref_hashes[m.map_id] = tuple(dhash64(r) for r in regs)

    def is_ready(self) -> bool:
        return len(self.ref_hashes or {}) >= 3

    def match_minimap(self, minimap_img: Image.Image, *, max_dist: int = 18) -> Optional[MapDef]:
        regs = _crop_regions(minimap_img.convert("RGBA"))
        q = tuple(dhash64(r) for r in regs)

        best_id = None
        best_score = 1e9
        for map_id, ref in (self.ref_hashes or {}).items():
            d = (hamming(q[0], ref[0]) + hamming(q[1], ref[1]) + hamming(q[2], ref[2])) / 3.0
            if d < best_score:
                best_score = d
                best_id = map_id

        if best_id is None or best_score > max_dist:
            return None

        for m in self.maps or []:
            if m.map_id == best_id:
                return m
        return None

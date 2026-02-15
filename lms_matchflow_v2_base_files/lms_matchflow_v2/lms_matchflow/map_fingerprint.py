from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
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
        img.crop((w*0.20, h*0.20, w*0.80, h*0.80)),  # center
        img.crop((w*0.35, h*0.10, w*0.65, h*0.90)),  # vertical mid strip
        img.crop((w*0.10, h*0.35, w*0.90, h*0.65)),  # horizontal mid strip
    ]


@dataclass
class MapFingerprintDB:
    """Loads reference map images and compares against a minimap crop."""
    ref_dir: Path
    maps: List[MapDef] = None
    ref_hashes: Dict[str, Tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        if self.maps is None:
            self.maps = DEFAULT_MAPS
        self.ref_hashes = {}
        self._load()

    def _load(self) -> None:
        for m in self.maps:
            p = self.ref_dir / m.filename
            if not p.exists():
                continue
            img = Image.open(p).convert("RGBA")
            regs = _crop_regions(img)
            self.ref_hashes[m.map_id] = tuple(dhash64(r) for r in regs)

    def is_ready(self) -> bool:
        return len(self.ref_hashes) >= 3

    def match_minimap(self, minimap_img: Image.Image, *, max_dist: int = 18) -> Optional[MapDef]:
        regs = _crop_regions(minimap_img.convert("RGBA"))
        q = tuple(dhash64(r) for r in regs)

        best_id = None
        best_score = 1e9
        for map_id, ref in self.ref_hashes.items():
            d = (hamming(q[0], ref[0]) + hamming(q[1], ref[1]) + hamming(q[2], ref[2])) / 3.0
            if d < best_score:
                best_score = d
                best_id = map_id

        if best_id is None or best_score > max_dist:
            return None

        for m in self.maps:
            if m.map_id == best_id:
                return m
        return None

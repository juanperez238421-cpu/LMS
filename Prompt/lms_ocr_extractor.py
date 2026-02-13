#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Faltan dependencias OCR. Instala con:\n"
        "  pip install opencv-python numpy"
    ) from exc


DEFAULT_ROI_CONFIG: Dict[str, Any] = {
    "default_profile": "1920x1080",
    "profiles": {
        "1920x1080": {
            "resolution": [1920, 1080],
            "metrics": {
                "damage_done": [1500, 120, 300, 70],
                "damage_taken": [1500, 200, 300, 70],
                "kills": [1500, 280, 300, 70],
                "items_crafted": [1500, 360, 300, 70],
            },
        }
    },
    "ranges": {
        "damage_done": [0, 200000],
        "damage_taken": [0, 200000],
        "kills": [0, 200],
        "items_crafted": [0, 2000],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae metricas visibles por OCR desde screenshots del juego."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Ruta a imagen PNG/JPG. Repetible.",
    )
    parser.add_argument(
        "--input-glob",
        default="",
        help="Glob de imagenes (ej: shots/*.png).",
    )
    parser.add_argument(
        "--config",
        default="",
        help="JSON de ROIs por resolucion.",
    )
    parser.add_argument(
        "--dpr",
        type=float,
        default=1.0,
        help="DevicePixelRatio para escalar ROIs.",
    )
    parser.add_argument(
        "--resize-factor",
        type=int,
        default=3,
        help="Factor de resize antes de OCR (2 o 3 recomendado).",
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "pytesseract", "easyocr"],
        help="Motor OCR.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
        help="Confianza minima para aceptar valor.",
    )
    parser.add_argument(
        "--playwright-url",
        default="",
        help="Si se define, captura screenshot del canvas desde esta URL.",
    )
    parser.add_argument(
        "--playwright-selector",
        default="canvas",
        help="Selector CSS del canvas para screenshot Playwright.",
    )
    parser.add_argument(
        "--playwright-output",
        default="data/processed/ocr/ocr_capture.png",
        help="Ruta para guardar screenshot con Playwright.",
    )
    parser.add_argument(
        "--playwright-wait-ms",
        type=int,
        default=6000,
        help="Espera previa al screenshot Playwright.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Imprime solo JSON final.",
    )
    return parser


def load_roi_config(path: str) -> Dict[str, Any]:
    if not path:
        return DEFAULT_ROI_CONFIG
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config OCR invalida: se esperaba objeto JSON.")
    return data


def capture_with_playwright(
    url: str,
    selector: str,
    out_path: Path,
    wait_ms: int,
) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Playwright no esta instalado. Usa: pip install playwright"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(max(0, wait_ms))
        clip = None
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=5000)
            box = locator.bounding_box()
            if box and box.get("width", 0) > 1 and box.get("height", 0) > 1:
                clip = {
                    "x": box["x"],
                    "y": box["y"],
                    "width": box["width"],
                    "height": box["height"],
                }
        except Exception:
            clip = None

        if clip:
            page.screenshot(path=str(out_path), clip=clip)
        else:
            page.screenshot(path=str(out_path), full_page=True)
        context.close()
        browser.close()
    return out_path


def pick_profile(config: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Config OCR sin profiles.")

    exact_key = f"{width}x{height}"
    if exact_key in profiles and isinstance(profiles[exact_key], dict):
        return profiles[exact_key]

    default_name = config.get("default_profile")
    if isinstance(default_name, str):
        default_profile = profiles.get(default_name)
        if isinstance(default_profile, dict):
            return default_profile

    first = next(iter(profiles.values()))
    if not isinstance(first, dict):
        raise ValueError("Config OCR invalida: profile no es objeto.")
    return first


def scale_roi(
    roi: Sequence[int],
    img_width: int,
    img_height: int,
    profile_resolution: Sequence[int],
    dpr: float,
) -> Tuple[int, int, int, int]:
    base_w = int(profile_resolution[0]) if len(profile_resolution) >= 2 else img_width
    base_h = int(profile_resolution[1]) if len(profile_resolution) >= 2 else img_height

    sx = (img_width / max(1, base_w)) * max(dpr, 0.01)
    sy = (img_height / max(1, base_h)) * max(dpr, 0.01)

    x, y, w, h = [int(v) for v in roi]
    sx_roi = int(x * sx)
    sy_roi = int(y * sy)
    sw_roi = int(w * sx)
    sh_roi = int(h * sy)

    sx_roi = max(0, min(img_width - 1, sx_roi))
    sy_roi = max(0, min(img_height - 1, sy_roi))
    sw_roi = max(1, min(img_width - sx_roi, sw_roi))
    sh_roi = max(1, min(img_height - sy_roi, sh_roi))
    return sx_roi, sy_roi, sw_roi, sh_roi


def preprocess_for_ocr(crop: np.ndarray, resize_factor: int) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray,
        None,
        fx=max(1, resize_factor),
        fy=max(1, resize_factor),
        interpolation=cv2.INTER_CUBIC,
    )
    thresholded = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def ocr_with_pytesseract(image: np.ndarray) -> Tuple[str, float]:
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pytesseract no instalado.") from exc

    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,"
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config=config,
    )
    tokens: List[str] = []
    confs: List[float] = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        txt_s = str(txt).strip()
        if txt_s:
            tokens.append(txt_s)
        try:
            conf_v = float(conf)
            if conf_v >= 0:
                confs.append(conf_v / 100.0)
        except Exception:
            pass
    full_text = " ".join(tokens).strip()
    confidence = sum(confs) / len(confs) if confs else 0.0
    return full_text, confidence


def ocr_with_easyocr(image: np.ndarray) -> Tuple[str, float]:
    try:
        import easyocr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("easyocr no instalado.") from exc

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(image)
    texts: List[str] = []
    confs: List[float] = []
    for _bbox, text, conf in results:
        text_s = str(text).strip()
        if text_s:
            texts.append(text_s)
        try:
            confs.append(float(conf))
        except Exception:
            pass
    full_text = " ".join(texts).strip()
    confidence = sum(confs) / len(confs) if confs else 0.0
    return full_text, confidence


def parse_number(raw_text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:[.,]\d+)?)", raw_text)
    if not match:
        return None
    token = match.group(1).replace(",", ".")
    try:
        num = float(token)
    except ValueError:
        return None
    if num.is_integer():
        return float(int(num))
    return num


def run_ocr_on_image(
    image_path: Path,
    config: Dict[str, Any],
    dpr: float,
    resize_factor: int,
    engine: str,
    min_confidence: float,
) -> Dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo abrir imagen: {image_path}")

    h, w = image.shape[:2]
    profile = pick_profile(config, w, h)
    metrics = profile.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Config OCR: metrics vacio o invalido.")
    profile_resolution = profile.get("resolution", [w, h])
    if not isinstance(profile_resolution, list) or len(profile_resolution) < 2:
        profile_resolution = [w, h]

    ranges = config.get("ranges", {})
    if not isinstance(ranges, dict):
        ranges = {}

    results: Dict[str, Any] = {}
    logs: List[str] = []

    for metric_name, roi in metrics.items():
        if not isinstance(roi, list) or len(roi) != 4:
            logs.append(f"ROI invalido para {metric_name}: {roi}")
            continue

        x, y, rw, rh = scale_roi(
            roi=roi,
            img_width=w,
            img_height=h,
            profile_resolution=profile_resolution,
            dpr=dpr,
        )
        crop = image[y : y + rh, x : x + rw]
        prep = preprocess_for_ocr(crop, resize_factor)

        raw_text = ""
        confidence = 0.0
        ocr_errors: List[str] = []
        if engine == "auto":
            attempt_order = ["pytesseract", "easyocr"]
        elif engine == "pytesseract":
            attempt_order = ["pytesseract"]
        else:
            attempt_order = ["easyocr"]

        for attempt_engine in attempt_order:
            try:
                if attempt_engine == "pytesseract":
                    raw_text, confidence = ocr_with_pytesseract(prep)
                else:
                    raw_text, confidence = ocr_with_easyocr(prep)
            except Exception as exc:
                ocr_errors.append(f"{attempt_engine}: {exc}")
                continue

            if raw_text.strip():
                break

        value = parse_number(raw_text)
        valid = True

        metric_range = ranges.get(metric_name)
        if isinstance(metric_range, list) and len(metric_range) == 2 and value is not None:
            min_v = float(metric_range[0])
            max_v = float(metric_range[1])
            if value < min_v or value > max_v:
                valid = False

        if confidence < min_confidence:
            valid = False

        if not valid:
            value = None

        results[metric_name] = {
            "value": value,
            "confidence": round(float(confidence), 4),
            "raw_text": raw_text,
            "valid": valid,
            "roi": [x, y, rw, rh],
            "ocr_errors": ocr_errors,
        }

    if "crafted" not in results and "items_crafted" in results:
        results["crafted"] = dict(results["items_crafted"])

    confidence_values = [
        v.get("confidence", 0.0)
        for v in results.values()
        if isinstance(v, dict) and isinstance(v.get("confidence"), (int, float))
    ]
    mean_conf = (
        sum(float(v) for v in confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return {
        "timestamp": int(time.time()),
        "image_path": str(image_path),
        "image_size": {"width": w, "height": h},
        "engine": engine,
        "metrics": results,
        "mean_confidence": round(float(mean_conf), 4),
        "logs": logs,
    }


def collect_input_images(args: argparse.Namespace) -> List[Path]:
    inputs: List[Path] = []
    for raw in args.input:
        p = Path(raw)
        if p.exists():
            inputs.append(p)
    if args.input_glob:
        inputs.extend(sorted(Path(".").glob(args.input_glob)))
    return inputs


def main() -> None:
    args = build_parser().parse_args()
    config = load_roi_config(args.config)

    if args.playwright_url:
        out_path = Path(args.playwright_output)
        capture_with_playwright(
            url=args.playwright_url,
            selector=args.playwright_selector,
            out_path=out_path,
            wait_ms=args.playwright_wait_ms,
        )
        args.input.append(str(out_path))

    images = collect_input_images(args)
    if not images:
        raise SystemExit("No hay imagenes para OCR. Usa --input/--input-glob o --playwright-url.")

    results = []
    for image_path in images:
        results.append(
            run_ocr_on_image(
                image_path=image_path,
                config=config,
                dpr=float(args.dpr),
                resize_factor=max(1, int(args.resize_factor)),
                engine=args.engine,
                min_confidence=float(args.min_confidence),
            )
        )

    output: Dict[str, Any]
    if len(results) == 1:
        output = results[0]
    else:
        output = {"timestamp": int(time.time()), "results": results}

    if args.json_only:
        print(json.dumps(output, ensure_ascii=True))
        return

    print("OCR completado.")
    if len(results) == 1:
        metric_items = results[0].get("metrics", {})
        for metric_name, metric in metric_items.items():
            if not isinstance(metric, dict):
                continue
            print(
                f"- {metric_name}: value={metric.get('value')} "
                f"conf={metric.get('confidence')} raw='{metric.get('raw_text')}'"
            )
    print("\nJSON:")
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

"""Shared GPU cost constants and helpers for all Modal OCR apps."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

GPU_SECOND_RATES_USD = {
    # Modal GPU Tasks pricing, verified against modal.com/pricing 2026-07-25.
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
}


def calculate_modal_cost(gpu_seconds: float, gpu_type: str = "T4") -> float:
    """Convert GPU wall-clock seconds to USD cost."""
    rate = GPU_SECOND_RATES_USD.get(gpu_type.upper(), GPU_SECOND_RATES_USD["T4"])
    return rate * gpu_seconds


def write_normalized_image(image_bytes: bytes, output_path: str | Path, max_side: int | None = None) -> None:
    """Write image bytes as an RGB PNG with EXIF orientation applied."""
    import io

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(output_path, format="PNG")


def read_text_artifacts(output_dir: str | Path) -> str:
    """Return text from the common OCR artifact formats written by model helpers."""
    output_path = Path(output_dir)
    candidates = []
    for pattern in ("*.mmd", "*.md", "*.txt"):
        candidates.extend(sorted(output_path.rglob(pattern)))

    parts = []
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)

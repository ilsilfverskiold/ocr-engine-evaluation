"""Apple Vision OCR (VNRecognizeTextRequest) — local, macOS only, OPT-IN.

VNRecognizeTextRequest runs on-device (Apple Neural Engine / GPU / CPU as the
OS decides). The project rule is no heavy model inference on the user's
machine, so this engine is registered default=False and must be enabled
explicitly; whether it may run at all is the user's decision.

How it runs: runners/apple_vision/recognize.swift is compiled once with
`swiftc -O` into ~/.cache/ocr-benchmark/apple_vision/<source-sha256[:16]>/,
then invoked per image as a subprocess with a timeout. The helper prints JSON:
lines (text, confidence, bbox, poly), words (text, bbox, poly from
VNRecognizedText.boundingBox(for:) over String.enumerateSubstrings(.byWords)
ranges — "word precision" at recognitionLevel .accurate), image size, EXIF
orientation, request revision, OS version/build and hardware model.
Coordinates arrive already flipped to a top-left origin and normalised 0-1
against the EXIF-oriented image.

Configuration (fixed): recognitionLevel .accurate, usesLanguageCorrection
true, recognitionLanguages ["en-US"], automaticallyDetectsLanguage false
(macOS 13+), request revision = the OS default (recorded, not pinned).

Word text comes from .byWords segmentation, which excludes punctuation
("Total:" -> "Total", "$12.00" -> "12.00"); the box is Vision's word box.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from runners.base import OCRResult, OCRRunner, OCRTask, provider_payload

SWIFT_SOURCE = Path(__file__).with_name("apple_vision") / "recognize.swift"
CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) \
    / "ocr-benchmark" / "apple_vision"
COMPILE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 120


class AppleVisionRunner(OCRRunner):
    name = "apple_vision"
    supports_pdf = False
    accepts_task = False   # fixed request; no prompt interface

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("apple_vision requires macOS (Vision framework)")
        self._binary = self._ensure_binary()

    # ---------------------------------------------------------------- compile
    @staticmethod
    def _ensure_binary() -> Path:
        """Compile the helper once per source hash; reuse it afterwards."""
        source = SWIFT_SOURCE.read_bytes()
        digest = hashlib.sha256(source).hexdigest()[:16]
        out_dir = CACHE_ROOT / digest
        binary = out_dir / "recognize"
        if binary.exists() and os.access(binary, os.X_OK):
            return binary
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"recognize.tmp{os.getpid()}"
        proc = subprocess.run(
            ["swiftc", "-O", str(SWIFT_SOURCE), "-o", str(tmp)],
            capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
        )
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"swiftc failed ({proc.returncode}): {proc.stderr[-2000:]}")
        os.replace(tmp, binary)  # atomic: never leave a half-written binary
        return binary

    # -------------------------------------------------------------------- run
    def run(self, document_path: Path, task: OCRTask | None = None) -> OCRResult:
        if task is not None:
            raise ValueError(f"{self.name} takes no task (got leg {task.leg!r})")
        if document_path.suffix.lower() == ".pdf":
            raise ValueError(f"{self.name}: PDF input not supported")

        t0 = self._timer()
        proc = subprocess.run(
            [str(self._binary), str(document_path)],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        )
        elapsed = self._elapsed(t0)
        try:
            reply = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"helper returned non-JSON (exit {proc.returncode}): "
                f"{proc.stdout[:300]!r} {proc.stderr[-500:]!r}") from exc
        if proc.returncode != 0 or reply.get("error"):
            raise RuntimeError(f"helper failed (exit {proc.returncode}): {reply.get('error')}")

        lines = [l for l in reply.get("lines", []) if str(l.get("text", "")).strip()]
        words = [w for w in reply.get("words", []) if str(w.get("text", "")).strip()]
        structured: dict = {
            "lines": lines,
            "image_size": reply.get("image_size"),
            "exif_orientation": reply.get("exif_orientation"),
            "word_range_failures": reply.get("word_range_failures", 0),
            "request": reply.get("request"),
        }
        if words:
            structured["words"] = words
            structured["geometry_granularity"] = "word"
            structured["geometry_coordinates"] = "normalized_xyxy"
        elif lines:
            structured["text_regions"] = [
                {"text": l["text"], "bbox": l["bbox"], "poly": l.get("poly")} for l in lines]
            structured["geometry_granularity"] = "line"
            structured["geometry_coordinates"] = "normalized_xyxy"

        request = reply.get("request") or {}
        model_version = (
            f"apple-vision VNRecognizeTextRequest rev{request.get('revision')} "
            f"accurate en-US lc=on macOS {reply.get('os_version')} "
            f"({reply.get('os_build')})"
        )
        return OCRResult(
            engine_name=self.name,
            document_path=str(document_path),
            # Observation order as Vision returns it, one line each.
            raw_text="\n".join(l["text"] for l in lines),
            structured_output=structured,
            latency_seconds=elapsed,
            cost_per_page=0.0,  # no API charge; host hardware time not priced
            page_count=1,
            model_version=model_version,
            metadata={
                "helper_binary": str(self._binary),
                "helper_source_sha256": hashlib.sha256(SWIFT_SOURCE.read_bytes()).hexdigest(),
                "inference_seconds": reply.get("inference_seconds"),
                "process_seconds": elapsed,
                "os_version": reply.get("os_version_string"),
                "os_build": reply.get("os_build"),
                "hardware_model": reply.get("hardware_model"),
                "cpu": reply.get("cpu"),
            },
            raw_response=provider_payload(reply),
        )

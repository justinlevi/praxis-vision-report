"""Service layer for vision_analyze_batch task.

Handles image resolution, transcript alignment, base64 encoding,
and concurrent vision API calls via Claude Code CLI. Each method is
independently testable but not exposed as a separate task.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from pathlib import Path

from praxis.task.core.transform.claude_inference.service import ClaudeInferenceService
from praxis_vision_report.tasks.vision_analyze_batch.models import (
    SlideAnalysis,
    VisionAnalyzeBatchConfig,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class VisionAnalyzeBatchService:
    """Encapsulates vision batch analysis logic."""

    # ------------------------------------------------------------------ #
    # Image resolution
    # ------------------------------------------------------------------ #

    def resolve_images(
        self,
        image_paths: list[str] | None,
        images_dir: str | None,
    ) -> list[Path]:
        """Return a sorted list of image file paths.

        Raises ValueError if neither argument is provided.
        Raises FileNotFoundError if a specified path or directory is missing.
        """
        if image_paths is not None:
            resolved: list[Path] = []
            for p in image_paths:
                path = Path(p)
                if not path.is_file():
                    raise FileNotFoundError(f"Image file not found: {p}")
                resolved.append(path)
            return resolved

        if images_dir is not None:
            dir_path = Path(images_dir)
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            files = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS)
            return files

        raise ValueError("Either image_paths or images_dir must be provided")

    # ------------------------------------------------------------------ #
    # Manifest & segments
    # ------------------------------------------------------------------ #

    def load_slide_manifest(self, images_dir: str | None) -> list[dict[str, object]] | None:
        """Read slide_manifest.json from images_dir if it exists.

        Format: [{"filename": "x.png", "timecode_seconds": 42.0,
                   "timecode_display": "00:00:42", "frame_number": 1260}]

        Returns None if images_dir is None or the manifest file is missing.
        """
        if images_dir is None:
            return None
        manifest_path = Path(images_dir) / "slide_manifest.json"
        if not manifest_path.is_file():
            return None
        data = json.loads(manifest_path.read_text())
        # The manifest may be a dict with a "slides" key or a bare list
        if isinstance(data, list):
            return data  # type: ignore[return-value]
        if isinstance(data, dict) and "slides" in data:
            return data["slides"]  # type: ignore[return-value]
        return None

    def load_segments(self, segments_path: str) -> list[dict[str, object]]:
        """Parse a .segments.json file -- [{start, end, text}].

        Raises FileNotFoundError if the file is missing.
        Raises json.JSONDecodeError on invalid JSON.
        """
        path = Path(segments_path)
        if not path.is_file():
            raise FileNotFoundError(f"Segments file not found: {segments_path}")
        return json.loads(path.read_text())  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Transcript alignment
    # ------------------------------------------------------------------ #

    def build_accumulated_text(
        self,
        segments: list[dict[str, object]],
        timecode_seconds: float,
    ) -> str:
        """Return all transcript text from t=0 up to timecode_seconds (inclusive end)."""
        parts: list[str] = []
        for seg in segments:
            start = float(seg.get("start", 0))  # type: ignore[arg-type]
            if start <= timecode_seconds:
                text = str(seg.get("text", "")).strip()
                if text:
                    parts.append(text)
        return " ".join(parts)

    def distribute_transcript_evenly(self, transcript: str, n: int) -> list[str]:
        """Split a raw transcript into n roughly equal chunks.

        Edge cases: n <= 0 returns [], n == 1 returns [transcript].
        """
        if n <= 0:
            return []
        if n == 1:
            return [transcript]

        words = transcript.split()
        total = len(words)
        if total == 0:
            return [""] * n

        chunk_size = math.ceil(total / n)
        chunks: list[str] = []
        for i in range(n):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

        # If we ended early (fewer chunks than n), pad with empty strings
        while len(chunks) < n:
            chunks.append("")

        return chunks

    # ------------------------------------------------------------------ #
    # Image encoding
    # ------------------------------------------------------------------ #

    def encode_image_base64(self, image_path: Path) -> str:
        """Read an image file and return its base64-encoded content."""
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # ------------------------------------------------------------------ #
    # Slide filtering
    # ------------------------------------------------------------------ #

    def filter_stage_shots(
        self,
        slides: list[dict[str, object]],
        images_dir: Path,
    ) -> list[dict[str, object]]:
        """Remove 'presenter on stage' shots using an OpenCV brightness heuristic.

        Stage shots (wide-angle audience view) have a bright top half (stage
        lighting / backdrop) AND a lit bottom half (illuminated audience).
        Content slides from a laptop have a dark bottom strip (~macOS dock/taskbar).

        Heuristic thresholds (empirically validated on GTC footage):
          - top_mean  > 175  (bright backdrop / presenter lit)
          - bot_mean  > 80   (audience rows illuminated from stage)

        Requires opencv-python-headless.
        Returns the filtered list; silently keeps slides whose image is missing.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("opencv-python-headless not installed — skipping stage-shot filter")
            return slides

        kept: list[dict[str, object]] = []
        removed = 0
        for entry in slides:
            filename = str(entry.get("filename", ""))
            img_path = images_dir / filename
            if not img_path.is_file():
                kept.append(entry)
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                kept.append(entry)
                continue

            h = img.shape[0]
            top_mean = float(np.mean(img[: h // 2, :]))
            bot_mean = float(np.mean(img[h // 2 :, :]))

            if top_mean > 175 and bot_mean > 80:
                removed += 1
                logger.debug(
                    "Stage shot removed",
                    extra={"file": filename, "top_mean": top_mean, "bot_mean": bot_mean},
                )
            else:
                kept.append(entry)

        logger.info(
            "Stage-shot filter complete",
            extra={"total": len(slides), "removed": removed, "kept": len(kept)},
        )
        return kept

    def filter_by_min_interval(
        self,
        slides: list[dict[str, object]],
        min_seconds: float,
    ) -> list[dict[str, object]]:
        """Keep only slides that are at least min_seconds apart.

        Walks slides in their existing order (assumed sorted by timecode).
        Keeps the first slide unconditionally, then keeps subsequent slides
        only if they are >= min_seconds after the last kept slide.

        Slides without a timecode_seconds value are always kept.
        """
        if min_seconds <= 0.0:
            return slides

        kept: list[dict[str, object]] = []
        last_kept_seconds: float | None = None
        removed = 0

        for entry in slides:
            raw_ts = entry.get("timecode_seconds")
            if raw_ts is None:
                kept.append(entry)
                continue

            ts = float(raw_ts)  # type: ignore[arg-type]
            if last_kept_seconds is None or (ts - last_kept_seconds) >= min_seconds:
                kept.append(entry)
                last_kept_seconds = ts
            else:
                removed += 1

        logger.info(
            "Min-interval filter complete",
            extra={
                "total": len(slides),
                "removed": removed,
                "kept": len(kept),
                "min_seconds": min_seconds,
            },
        )
        return kept

    # ------------------------------------------------------------------ #
    # Key-point extraction
    # ------------------------------------------------------------------ #

    def parse_key_points(self, text: str) -> list[str]:
        """Extract bullet points from a 'KEY POINTS:' section in the response.

        Returns an empty list if no KEY POINTS section is found.
        """
        marker = "KEY POINTS:"
        idx = text.upper().find(marker.upper())
        if idx == -1:
            return []

        after = text[idx + len(marker) :]
        points: list[str] = []
        for line in after.splitlines():
            stripped = line.strip()
            if stripped.startswith("-") or stripped.startswith("*"):
                point = stripped[1:].strip()
                if point:
                    points.append(point)

        return points

    # ------------------------------------------------------------------ #
    # Single-image analysis
    # ------------------------------------------------------------------ #

    async def analyze_image(
        self,
        image_path: Path,
        context_text: str,
        system_prompt: str,
        global_context: str | None,
        config: VisionAnalyzeBatchConfig,
        timecode: str | None = None,
        timecode_seconds: float | None = None,
        max_retries: int = 3,
    ) -> SlideAnalysis:
        """Analyze a single image via Claude Code CLI and return a SlideAnalysis.

        Uses ClaudeInferenceService to pass the image path directly to Claude,
        which reads the file natively via its Read tool.

        Retries up to max_retries times with exponential backoff.
        """
        system_content = system_prompt
        if global_context:
            system_content = f"{system_prompt}\n\nContext: {global_context}"

        user_text = (
            f"Transcript up to this point:\n{context_text}\n\n"
            "Analyze this slide. End your response with:\n"
            "KEY POINTS:\n- point 1\n- point 2"
        )

        claude_svc = ClaudeInferenceService()
        full_prompt = claude_svc.build_full_prompt(
            prompt=user_text,
            system_prompt=system_content,
            image_paths=[str(image_path)],
            file_paths=None,
            context=None,
        )

        raw, _attempt_count = await claude_svc.execute_with_retry(
            full_prompt=full_prompt,
            model=config.model,
            output_format="text",
            timeout=300,
            max_retries=max_retries,
            retry_delay=2.0,
        )

        analysis_text = raw
        key_points = self.parse_key_points(analysis_text)

        logger.info(
            "Analyzed image",
            extra={
                "image": str(image_path),
                "key_points": len(key_points),
            },
        )

        return SlideAnalysis(
            image_path=str(image_path),
            timecode=timecode,
            timecode_seconds=timecode_seconds,
            analysis=analysis_text,
            key_points=key_points,
            token_usage={},
        )

    # ------------------------------------------------------------------ #
    # Batch analysis (multiple slides per claude call)                    #
    # ------------------------------------------------------------------ #

    async def analyze_image_batch(
        self,
        items: list[tuple[Path, str, str | None, float | None]],
        system_prompt: str,
        global_context: str | None,
        config: VisionAnalyzeBatchConfig,
        max_retries: int = 3,
    ) -> list[SlideAnalysis]:
        """Analyze a batch of slides in a single Claude Code CLI call.

        Passes all image paths together so Claude reads them sequentially,
        then returns a JSON array — one ``{analysis, key_points}`` object
        per slide.  Falls back to empty analysis on JSON parse failure so
        the overall batch never raises.

        Args:
            items: Ordered list of ``(image_path, context_text, timecode, timecode_seconds)``
            system_prompt: Per-session analysis instructions.
            global_context: Optional overall session context.
            config: Task configuration (model, etc.)
            max_retries: Retry attempts on subprocess failure.

        Returns:
            One ``SlideAnalysis`` per input item, in the same order.
        """
        n = len(items)
        system_content = system_prompt
        if global_context:
            system_content = f"{system_prompt}\n\nContext: {global_context}"

        # Build the per-slide context block
        slide_blocks: list[str] = []
        image_paths: list[str] = []
        for i, (image_path, context_text, timecode, _) in enumerate(items):
            image_paths.append(str(image_path))
            tc_str = f" [{timecode}]" if timecode else ""
            slide_blocks.append(
                f"SLIDE {i + 1}{tc_str} — {image_path.name}\n"
                f"Transcript so far: {context_text or '(none)'}"
            )

        slides_context = "\n\n".join(slide_blocks)

        user_text = (
            f"Analyze the following {n} conference slides. "
            f"The images are provided in order and you must read each one.\n\n"
            f"{slides_context}\n\n"
            f"Return ONLY a JSON array with exactly {n} objects — one per slide, "
            f"in the same order — using this schema:\n"
            f'[{{"analysis": "detailed technical analysis", "key_points": ["point 1", "point 2"]}}, ...]\n\n'
            f"No preamble. No explanation. Start directly with `[`."
        )

        claude_svc = ClaudeInferenceService()
        full_prompt = claude_svc.build_full_prompt(
            prompt=user_text,
            system_prompt=system_content,
            image_paths=image_paths,
            file_paths=None,
            context=None,
        )

        raw, _attempts = await claude_svc.execute_with_retry(
            full_prompt=full_prompt,
            model=config.model,
            output_format="text",
            timeout=max(300, min(60 * n, 1800)),  # scale with batch size, cap at 30 min
            max_retries=max_retries,
            retry_delay=2.0,
        )

        # Parse the JSON array from the response
        import re as _re

        parsed: list[dict[str, object]] | None = None
        stripped = raw.strip()
        # Strip optional markdown code fence
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            stripped = "\n".join(inner).strip()
        try:
            candidate = json.loads(stripped)
            if isinstance(candidate, list):
                parsed = candidate  # type: ignore[assignment]
        except (ValueError, json.JSONDecodeError):
            # Try to extract just the array portion
            match = _re.search(r"\[\s*\{.*\}\s*\]", stripped, _re.DOTALL)
            if match:
                try:
                    candidate = json.loads(match.group())
                    if isinstance(candidate, list):
                        parsed = candidate  # type: ignore[assignment]
                except (ValueError, json.JSONDecodeError):
                    pass

        logger.info(
            "Analyzed image batch",
            extra={
                "batch_size": n,
                "parsed_ok": parsed is not None,
                "parsed_count": len(parsed) if parsed else 0,
            },
        )

        results: list[SlideAnalysis] = []
        for i, (image_path, _, timecode, timecode_seconds) in enumerate(items):
            if parsed and i < len(parsed) and isinstance(parsed[i], dict):
                entry = parsed[i]
                analysis_text = str(entry.get("analysis", ""))
                raw_kp = entry.get("key_points", [])
                key_points = [str(p) for p in raw_kp] if isinstance(raw_kp, list) else []
            else:
                # JSON parsing failed for this slide — log degradation and use empty analysis
                logger.warning(
                    "Batch JSON parse failed for slide %d/%d — using empty analysis",
                    i + 1,
                    n,
                    extra={"image": str(image_path)},
                )
                analysis_text = ""
                key_points = []

            results.append(
                SlideAnalysis(
                    image_path=str(image_path),
                    timecode=timecode,
                    timecode_seconds=timecode_seconds,
                    analysis=analysis_text,
                    key_points=key_points,
                    token_usage={},
                )
            )

        return results

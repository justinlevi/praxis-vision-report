"""Service layer for vision_analyze_batch task.

Handles image resolution, transcript alignment, base64 encoding,
and concurrent OpenAI vision API calls. Each method is independently
testable but not exposed as a separate task.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from praxis_vision_report.tasks.vision_analyze_batch.models import (
    SlideAnalysis,
    VisionAnalyzeBatchConfig,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

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
        """Parse a .segments.json file — [{start, end, text}].

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
            if stripped.startswith("-") or stripped.startswith("•"):
                point = stripped[1:].strip()
                if point:
                    points.append(point)

        return points

    # ------------------------------------------------------------------ #
    # Single-image analysis
    # ------------------------------------------------------------------ #

    async def analyze_image(
        self,
        client: AsyncOpenAI,
        image_path: Path,
        context_text: str,
        system_prompt: str,
        global_context: str | None,
        config: VisionAnalyzeBatchConfig,
        timecode: str | None = None,
        timecode_seconds: float | None = None,
    ) -> SlideAnalysis:
        """Call the OpenAI vision API for a single image and return a SlideAnalysis."""
        b64 = self.encode_image_base64(image_path)

        system_content = system_prompt
        if global_context:
            system_content = f"{system_prompt}\n\nContext: {global_context}"

        user_text = (
            f"Transcript up to this point:\n{context_text}\n\n"
            "Analyze this slide. End your response with:\n"
            "KEY POINTS:\n- point 1\n- point 2"
        )

        response = await client.chat.completions.create(
            model=config.model,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": system_content},
                {  # pyright: ignore[reportArgumentType]
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": config.detail,
                            },
                        },
                    ],
                },
            ],
        )

        choice = response.choices[0]
        analysis_text = choice.message.content or ""

        usage = response.usage
        token_usage: dict[str, int] = {}
        if usage is not None:
            token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        key_points = self.parse_key_points(analysis_text)

        logger.info(
            "Analyzed image",
            extra={
                "image": str(image_path),
                "tokens": token_usage.get("total_tokens", 0),
                "key_points": len(key_points),
            },
        )

        return SlideAnalysis(
            image_path=str(image_path),
            timecode=timecode,
            timecode_seconds=timecode_seconds,
            analysis=analysis_text,
            key_points=key_points,
            token_usage=token_usage,
        )

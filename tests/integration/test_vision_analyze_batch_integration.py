"""Integration tests for vision_analyze_batch task.

These tests make real OpenAI API calls and require OPENAI_API_KEY to be set.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import pytest
from praxis_vision_report.tasks.vision_analyze_batch.models import (
    VisionAnalyzeBatchConfig,
    VisionAnalyzeBatchInput,
    VisionAnalyzeBatchOutput,
)
from praxis_vision_report.tasks.vision_analyze_batch.task import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY"))


def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image in memory (stdlib only)."""

    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    raw_rows = b""
    for _ in range(height):
        row = b"\x00" + b"\x80\x80\x80" * width
        raw_rows += row
    compressed = zlib.compress(raw_rows, 9)
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_API_KEY, reason="OPENAI_API_KEY not set")
class TestVisionAnalyzeBatchIntegration:
    @pytest.mark.asyncio
    async def test_real_api_call_single_image(self, tmp_path: Path) -> None:
        """Real API call with one tiny PNG returns non-empty analysis."""
        img_dir = tmp_path / "slides"
        img_dir.mkdir()
        (img_dir / "slide_0001.png").write_bytes(_make_png_bytes(100, 100))

        inp = VisionAnalyzeBatchInput(
            images_dir=str(img_dir),
            system_prompt=(
                "You are analyzing presentation slides. "
                "Describe what you see and extract key points."
            ),
            global_context="This is a test image for integration testing.",
        )
        config = VisionAnalyzeBatchConfig(
            model="gpt-4o",  # Use gpt-4o for integration (widely available)
            detail="low",
            max_tokens=512,
            concurrency=1,
        )

        result = await run(inp, config)

        assert isinstance(result, VisionAnalyzeBatchOutput)
        assert result.status == "success"
        assert result.total_slides == 1
        assert len(result.analyses) == 1

        analysis = result.analyses[0]
        assert len(analysis.analysis) > 0, "Analysis text should not be empty"
        # The model should return some content — even for a gray image
        assert analysis.token_usage.get("total_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_real_api_call_with_transcript(self, tmp_path: Path) -> None:
        """Real API call uses transcript context in the request."""
        img_dir = tmp_path / "slides2"
        img_dir.mkdir()
        (img_dir / "slide_0001.png").write_bytes(_make_png_bytes(50, 50))

        segments = [
            {"start": 0.0, "end": 30.0, "text": "Welcome to this demonstration."},
            {"start": 30.0, "end": 60.0, "text": "We are testing the vision API."},
        ]
        seg_path = tmp_path / "test.segments.json"
        seg_path.write_text(json.dumps(segments))

        # Create a manifest so timecode alignment works
        manifest = [
            {
                "filename": "slide_0001.png",
                "timecode_seconds": 45.0,
                "timecode_display": "00:00:45",
                "frame_number": 1350,
            }
        ]
        (img_dir / "slide_manifest.json").write_text(json.dumps(manifest))

        inp = VisionAnalyzeBatchInput(
            images_dir=str(img_dir),
            segments_path=str(seg_path),
            system_prompt="Analyze this slide in the context of the transcript provided.",
        )
        config = VisionAnalyzeBatchConfig(
            model="gpt-4o",
            detail="low",
            max_tokens=512,
            concurrency=1,
        )

        result = await run(inp, config)

        assert result.status == "success"
        assert result.total_slides == 1
        assert len(result.analyses[0].analysis) > 0
        assert result.analyses[0].timecode == "00:00:45"
        assert result.analyses[0].timecode_seconds == 45.0

    @pytest.mark.asyncio
    async def test_real_api_call_multiple_images(self, tmp_path: Path) -> None:
        """Real API call with 2 images both get analyzed."""
        img_dir = tmp_path / "slides3"
        img_dir.mkdir()
        (img_dir / "slide_0001.png").write_bytes(_make_png_bytes(50, 50))
        (img_dir / "slide_0002.png").write_bytes(_make_png_bytes(50, 50))

        inp = VisionAnalyzeBatchInput(
            images_dir=str(img_dir),
            system_prompt="Briefly describe what you see in each slide.",
            transcript="This is a short test transcript for two slides.",
        )
        config = VisionAnalyzeBatchConfig(
            model="gpt-4o",
            detail="low",
            max_tokens=256,
            concurrency=2,
        )

        result = await run(inp, config)

        assert result.status == "success"
        assert result.total_slides == 2
        assert len(result.analyses) == 2
        for analysis in result.analyses:
            assert len(analysis.analysis) > 0
        assert result.total_tokens > 0

"""Shared fixtures for praxis-vision-report tests."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Minimal valid PNG helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid PNG image in memory (no Pillow required).

    Generates a solid-colour (gray) PNG using raw deflate-compressed IDAT.
    """

    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    # IHDR: width, height, bit depth=8, colour type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: one filter byte (0) + RGB pixels per row
    raw_rows = b""
    for _ in range(height):
        row = b"\x00" + b"\x80\x80\x80" * width  # filter=None, gray pixels
        raw_rows += row
    compressed = zlib.compress(raw_rows, 9)
    idat = _chunk(b"IDAT", compressed)

    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    """Write a tiny 100x100 PNG to a temp file and return its path."""
    path = tmp_path / "test_image.png"
    path.write_bytes(_make_png_bytes(100, 100))
    return path


@pytest.fixture
def images_dir_with_pngs(tmp_path: Path) -> Path:
    """Create a temporary directory containing three small PNGs."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(1, 4):
        (img_dir / f"slide_{i:04d}.png").write_bytes(_make_png_bytes(10, 10))
    return img_dir


@pytest.fixture
def images_dir_with_manifest(tmp_path: Path) -> Path:
    """Directory with three PNGs and a slide_manifest.json."""
    img_dir = tmp_path / "images_with_manifest"
    img_dir.mkdir()
    slides = []
    for i in range(1, 4):
        fname = f"slide_{i:04d}.png"
        (img_dir / fname).write_bytes(_make_png_bytes(10, 10))
        slides.append(
            {
                "filename": fname,
                "timecode_seconds": float((i - 1) * 60),
                "timecode_display": f"00:{(i - 1):02d}:00",
                "frame_number": (i - 1) * 1800,
            }
        )
    manifest = {
        "slide_count": 3,
        "slides": slides,
    }
    (img_dir / "slide_manifest.json").write_text(json.dumps(manifest))
    return img_dir


@pytest.fixture
def sample_segments_json(tmp_path: Path) -> Path:
    """Write a sample .segments.json file and return its path."""
    segments = [
        {"start": 0.0, "end": 30.0, "text": "Welcome to the talk."},
        {"start": 30.0, "end": 60.0, "text": "Today we cover deep learning."},
        {"start": 60.0, "end": 90.0, "text": "Let us start with the basics."},
        {"start": 90.0, "end": 120.0, "text": "Neural networks are powerful."},
        {"start": 120.0, "end": 150.0, "text": "Thank you for attending."},
    ]
    path = tmp_path / "session.segments.json"
    path.write_text(json.dumps(segments))
    return path

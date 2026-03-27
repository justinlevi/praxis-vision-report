"""Integration tests for compile_vision_report task.

These tests make real OpenAI API calls and require OPENAI_API_KEY to be set.
Run with: pytest tests/integration/ -v -m integration
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from praxis_vision_report.tasks.compile_vision_report.models import (
    CompileVisionReportConfig,
    CompileVisionReportInput,
    CompileVisionReportOutput,
)
from praxis_vision_report.tasks.compile_vision_report.task import run

pytestmark = pytest.mark.integration


def _make_fake_png(path: Path) -> None:
    """Write a minimal valid PNG header to the given path."""
    path.write_bytes(
        bytes(
            [
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D,  # IHDR chunk length
                0x49, 0x48, 0x44, 0x52,  # IHDR
                0x00, 0x00, 0x00, 0x01,  # width: 1
                0x00, 0x00, 0x00, 0x01,  # height: 1
                0x08, 0x02,              # bit depth 8, color type RGB
                0x00, 0x00, 0x00,        # compression, filter, interlace
                0x90, 0x77, 0x53, 0xDE,  # CRC
            ]
        )
    )


@pytest.fixture()
def fake_slide_images(tmp_path: Path) -> list[Path]:
    """Create three fake PNG slide images in tmp_path."""
    images = []
    for i in range(1, 4):
        img_path = tmp_path / f"slide_{i:04d}.png"
        _make_fake_png(img_path)
        images.append(img_path)
    return images


@pytest.fixture()
def sample_analyses(fake_slide_images: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "image_path": str(fake_slide_images[0]),
            "slide_number": 1,
            "analysis": (
                "This slide introduces the NVIDIA Blackwell architecture. "
                "It shows the B200 GPU chip with 208 billion transistors and the new "
                "NVLink 5 interconnect enabling 900 GB/s bidirectional bandwidth."
            ),
            "key_points": [
                "B200 GPU: 208 billion transistors",
                "NVLink 5: 900 GB/s bidirectional bandwidth",
                "5th generation Transformer Engine",
                "FP4 precision support for 20 petaFLOPS",
            ],
            "title": "Blackwell Architecture Overview",
        },
        {
            "image_path": str(fake_slide_images[1]),
            "slide_number": 2,
            "analysis": (
                "Performance comparison between H100 and B200 GPUs on LLM inference. "
                "The chart shows B200 achieves 5x higher throughput on GPT-4 sized models "
                "while consuming 25% less energy per token."
            ),
            "key_points": [
                "5x inference throughput vs H100",
                "25% lower energy per token",
                "30x faster time-to-first-token",
                "Supported frameworks: TensorRT-LLM, vLLM, Triton",
            ],
            "title": "B200 vs H100 Performance Benchmarks",
        },
        {
            "image_path": str(fake_slide_images[2]),
            "slide_number": 3,
            "analysis": (
                "DGX B200 system specifications and rack configuration. "
                "Shows a full rack with 8 DGX nodes, 72 B200 GPUs total, "
                "connected via NVLink switches for 57.6 TB/s aggregate bandwidth."
            ),
            "key_points": [
                "DGX B200: 8 B200 GPUs per node",
                "Full rack: 72 GPUs, 57.6 TB/s aggregate bandwidth",
                "NVLink Switch for all-to-all GPU communication",
                "Liquid cooling standard",
                "Available Q2 2026",
            ],
            "title": "DGX B200 System Architecture",
        },
    ]


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real API call",
)
class TestCompileVisionReportIntegration:
    @pytest.mark.asyncio
    async def test_returns_non_empty_markdown(
        self, sample_analyses: list[dict[str, Any]]
    ) -> None:
        inp = CompileVisionReportInput(
            analyses=sample_analyses,
            title="GTC 2026 Blackwell Architecture Deep Dive",
            description=(
                "A technical deep dive into NVIDIA's Blackwell GPU architecture, "
                "performance characteristics, and system-level configurations "
                "presented at GTC 2026."
            ),
            metadata={
                "speaker": "Jensen Huang",
                "date": "2026-03-18",
                "level": "Technical",
                "topic": "GPU Architecture",
            },
        )
        config = CompileVisionReportConfig(
            output_formats=["markdown"],  # No HTML to keep integration test lightweight
            copy_images=False,
        )

        result = await run(inp, config)

        assert isinstance(result, CompileVisionReportOutput)
        assert result.status == "success"
        assert len(result.markdown_content) > 0
        assert result.word_count > 50

    @pytest.mark.asyncio
    async def test_markdown_contains_proper_headers(
        self, sample_analyses: list[dict[str, Any]]
    ) -> None:
        inp = CompileVisionReportInput(
            analyses=sample_analyses,
            title="GTC 2026 Blackwell Architecture Deep Dive",
        )
        config = CompileVisionReportConfig(
            output_formats=["markdown"],
            copy_images=False,
        )

        result = await run(inp, config)

        # Should have at least one markdown header
        assert "#" in result.markdown_content

    @pytest.mark.asyncio
    async def test_markdown_covers_key_content(
        self, sample_analyses: list[dict[str, Any]]
    ) -> None:
        """Verify the synthesized article covers key themes from the analyses."""
        inp = CompileVisionReportInput(
            analyses=sample_analyses,
            title="GTC 2026 Blackwell Architecture Deep Dive",
        )
        config = CompileVisionReportConfig(
            output_formats=["markdown"],
            copy_images=False,
        )

        result = await run(inp, config)

        content_lower = result.markdown_content.lower()
        # These are core topics from the analyses — the article should mention them
        assert any(
            term in content_lower
            for term in ["blackwell", "b200", "gpu", "nvidia"]
        ), f"Expected key GPU terms in report. Got:\n{result.markdown_content[:500]}"

    @pytest.mark.asyncio
    async def test_no_artifact_context_paths_are_none(
        self, sample_analyses: list[dict[str, Any]]
    ) -> None:
        """Without artifact context, paths should be None but content should still exist."""
        inp = CompileVisionReportInput(
            analyses=sample_analyses,
            title="GTC 2026 Integration Test",
        )
        config = CompileVisionReportConfig(
            output_formats=["markdown"],
            copy_images=False,
        )

        result = await run(inp, config)

        assert result.report_path is None
        assert result.html_path is None
        assert result.markdown_content  # non-empty

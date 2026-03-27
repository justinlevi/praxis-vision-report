"""Tests for compile_vision_report task run() function."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from praxis_vision_report.tasks.compile_vision_report.models import (
    CompileVisionReportConfig,
    CompileVisionReportInput,
    CompileVisionReportOutput,
)
from praxis_vision_report.tasks.compile_vision_report.task import run

FAKE_MARKDOWN = """# GTC 2026 Keynote

## Introduction

This keynote revealed NVIDIA's new Blackwell architecture.

## GPU Architecture

The new B200 GPU brings unprecedented performance.

## Conclusion

The future of AI infrastructure looks bright.
"""

SAMPLE_ANALYSES: list[dict[str, Any]] = [
    {
        "image_path": "/tmp/slide1.png",
        "analysis": "Introduces Blackwell GPU architecture",
        "key_points": ["B200 GPU", "5x performance"],
    },
    {
        "image_path": "/tmp/slide2.png",
        "analysis": "NVLink 5 interconnect details",
        "key_points": ["NVLink 5", "900 GB/s bandwidth"],
    },
]


def make_mock_artifact_context(tmp_path: Path) -> MagicMock:
    """Create a mock ArtifactContext that saves files to tmp_path."""
    mock_ctx = MagicMock()
    mock_ctx.artifact_dir = tmp_path

    def fake_save_artifact(filename: str, content: str) -> Path:
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")
        return path

    mock_ctx.save_artifact = MagicMock(side_effect=fake_save_artifact)
    return mock_ctx


class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_with_artifact_context(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026 Keynote")
        config = CompileVisionReportConfig()

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "Test prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026-keynote"
            mock_svc.copy_images_to_artifact.return_value = 2
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html><html><body>HTML</body></html>"

            result = await run(inp, config)

        assert isinstance(result, CompileVisionReportOutput)
        assert result.status == "success"
        assert result.markdown_content == FAKE_MARKDOWN
        assert result.report_path is not None
        assert "gtc-2026-keynote.md" in result.report_path
        assert result.html_path is not None
        assert "gtc-2026-keynote.html" in result.html_path
        assert result.image_count == 2

    @pytest.mark.asyncio
    async def test_word_count_computed_correctly(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026")
        config = CompileVisionReportConfig()

        known_content = "one two three four five"  # 5 words

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=known_content)
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.copy_images_to_artifact.return_value = 0
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp, config)

        assert result.word_count == 5


class TestRunNoArtifactContext:
    @pytest.mark.asyncio
    async def test_no_artifact_context_paths_are_none(self) -> None:
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026 Keynote")
        config = CompileVisionReportConfig()

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=None,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026-keynote"
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp, config)

        assert result.report_path is None
        assert result.html_path is None
        assert result.markdown_content == FAKE_MARKDOWN
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_no_artifact_context_image_count_zero(self) -> None:
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026 Keynote")
        config = CompileVisionReportConfig(copy_images=True)

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=None,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026-keynote"
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp, config)

        assert result.image_count == 0
        # copy_images_to_artifact should NOT be called when there's no artifact context
        mock_svc.copy_images_to_artifact.assert_not_called()


class TestRunCopyImagesDisabled:
    @pytest.mark.asyncio
    async def test_copy_images_false_skips_copy(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026")
        config = CompileVisionReportConfig(copy_images=False)

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp, config)

        mock_svc.copy_images_to_artifact.assert_not_called()
        assert result.image_count == 0


class TestRunOutputFormats:
    @pytest.mark.asyncio
    async def test_only_markdown_format_no_html(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026")
        config = CompileVisionReportConfig(output_formats=["markdown"])

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.copy_images_to_artifact.return_value = 0

            result = await run(inp, config)

        assert result.html_path is None
        assert result.html_content is None
        mock_svc.convert_to_html.assert_not_called()

    @pytest.mark.asyncio
    async def test_html_in_output_formats_triggers_conversion(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = CompileVisionReportInput(analyses=SAMPLE_ANALYSES, title="GTC 2026")
        config = CompileVisionReportConfig(output_formats=["markdown", "html"])

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.copy_images_to_artifact.return_value = 0
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html><html>...</html>"

            result = await run(inp, config)

        mock_svc.convert_to_html.assert_called_once()
        assert result.html_content == "<!DOCTYPE html><html>...</html>"
        assert result.html_path is not None


class TestRunMetadata:
    @pytest.mark.asyncio
    async def test_result_metadata_contains_expected_keys(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = CompileVisionReportInput(
            analyses=SAMPLE_ANALYSES,
            title="GTC 2026",
            metadata={"speaker": "Jensen Huang"},
        )
        config = CompileVisionReportConfig()

        with (
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.CompileVisionReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.compile_vision_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
            patch("praxis_vision_report.tasks.compile_vision_report.task.openai.AsyncOpenAI"),
        ):
            mock_svc = MockService.return_value
            mock_svc.build_synthesis_prompt.return_value = "prompt"
            mock_svc.synthesize_markdown = AsyncMock(return_value=FAKE_MARKDOWN)
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.copy_images_to_artifact.return_value = 2
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp, config)

        assert result.metadata["title"] == "GTC 2026"
        assert result.metadata["analysis_count"] == 2
        assert result.metadata["model"] == "gpt-5.4"
        assert "session_metadata" in result.metadata
        assert result.metadata["session_metadata"]["speaker"] == "Jensen Huang"

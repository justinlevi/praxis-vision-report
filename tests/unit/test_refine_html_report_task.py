"""Tests for refine_html_report task run() function."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from praxis_vision_report.tasks.refine_html_report.models import (
    RefineHtmlReportConfig,
    RefineHtmlReportInput,
    RefineHtmlReportOutput,
)
from praxis_vision_report.tasks.refine_html_report.task import run

SAMPLE_ANALYSES: list[dict[str, Any]] = [
    {"image_path": "/tmp/slide1.png", "key_points": ["GPU"]},
]

SAMPLE_CRITIQUE: dict[str, Any] = {
    "overall_score": 7,
    "priority_edits": ["Fix intro"],
    "image_decisions": [],
}


def make_mock_artifact_context(tmp_path: Path) -> MagicMock:
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
    async def test_returns_correct_output(self) -> None:
        inp = RefineHtmlReportInput(
            markdown_content="# Draft\n\nOriginal content.",
            analyses=SAMPLE_ANALYSES,
            critique=SAMPLE_CRITIQUE,
            title="GTC 2026",
        )
        config = RefineHtmlReportConfig()

        with (
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.RefineHtmlReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.get_current_artifact_context",
                return_value=None,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.rewrite = AsyncMock(
                return_value=("# Refined\n\nBetter.", "Full rewrite.", {"input_tokens": 10000, "output_tokens": 5000})
            )
            mock_svc.count_words.return_value = 3
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp, config)

        assert isinstance(result, RefineHtmlReportOutput)
        assert result.status == "success"
        assert result.refined_markdown == "# Refined\n\nBetter."
        assert result.changes_summary == "Full rewrite."
        assert result.word_count == 3
        assert result.refined_html == "<!DOCTYPE html>..."

    @pytest.mark.asyncio
    async def test_default_config(self) -> None:
        inp = RefineHtmlReportInput(
            markdown_content="# Draft",
            analyses=[],
            critique={},
            title="Test",
        )

        with (
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.RefineHtmlReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.get_current_artifact_context",
                return_value=None,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.rewrite = AsyncMock(return_value=("# Refined", "Rewrote.", {}))
            mock_svc.count_words.return_value = 2
            mock_svc.safe_filename.return_value = "test"
            mock_svc.convert_to_html.return_value = "<html>...</html>"

            result = await run(inp, None)

        assert result.status == "success"
        call_kwargs = mock_svc.rewrite.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4-mini"


class TestRunWithArtifactContext:
    @pytest.mark.asyncio
    async def test_saves_artifacts(self, tmp_path: Path) -> None:
        mock_ctx = make_mock_artifact_context(tmp_path)
        inp = RefineHtmlReportInput(
            markdown_content="# Draft",
            analyses=SAMPLE_ANALYSES,
            critique=SAMPLE_CRITIQUE,
            title="GTC 2026",
        )

        with (
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.RefineHtmlReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.rewrite = AsyncMock(return_value=("# Refined", "Rewrote.", {}))
            mock_svc.count_words.return_value = 2
            mock_svc.safe_filename.return_value = "gtc-2026"
            mock_svc.convert_to_html.return_value = "<!DOCTYPE html>..."

            result = await run(inp)

        assert result.refined_md_path is not None
        assert "gtc-2026-refined.md" in result.refined_md_path
        assert result.refined_html_path is not None
        assert "gtc-2026-refined.html" in result.refined_html_path
        assert mock_ctx.save_artifact.call_count == 2


class TestRunNoArtifactContext:
    @pytest.mark.asyncio
    async def test_paths_are_none(self) -> None:
        inp = RefineHtmlReportInput(
            markdown_content="# Draft",
            analyses=[],
            critique={},
            title="Test",
        )

        with (
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.RefineHtmlReportService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.refine_html_report.task.get_current_artifact_context",
                return_value=None,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.rewrite = AsyncMock(return_value=("# Refined", "Rewrote.", {}))
            mock_svc.count_words.return_value = 2
            mock_svc.safe_filename.return_value = "test"
            mock_svc.convert_to_html.return_value = "<html>...</html>"

            result = await run(inp, None)

        assert result.refined_md_path is None
        assert result.refined_html_path is None

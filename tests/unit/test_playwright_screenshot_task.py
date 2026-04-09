"""Tests for playwright_screenshot task run() function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from praxis_vision_report.tasks.playwright_screenshot.models import (
    PlaywrightScreenshotConfig,
    PlaywrightScreenshotInput,
    PlaywrightScreenshotOutput,
)
from praxis_vision_report.tasks.playwright_screenshot.task import run


class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_with_url_and_artifact_context(self, tmp_path: Path) -> None:
        mock_ctx = MagicMock()
        mock_ctx.artifact_dir = tmp_path

        inp = PlaywrightScreenshotInput(url="https://example.com")
        config = PlaywrightScreenshotConfig(section_screenshots=3, full_page=True)

        with (
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.PlaywrightScreenshotService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.resolve_url.return_value = "https://example.com"
            mock_svc.screenshot_page = AsyncMock(
                return_value=(
                    Path("/tmp/full_page.png"),
                    [Path("/tmp/s1.png"), Path("/tmp/s2.png"), Path("/tmp/s3.png")],
                    "Example",
                    "https://example.com",
                )
            )

            result = await run(inp, config)

        assert isinstance(result, PlaywrightScreenshotOutput)
        assert result.status == "success"
        assert result.full_page_path == "/tmp/full_page.png"
        assert len(result.section_paths) == 3
        assert result.page_title == "Example"
        assert result.page_url == "https://example.com"
        assert result.screenshot_count == 4  # 1 full + 3 sections

    @pytest.mark.asyncio
    async def test_no_full_page(self, tmp_path: Path) -> None:
        mock_ctx = MagicMock()
        mock_ctx.artifact_dir = tmp_path

        inp = PlaywrightScreenshotInput(url="https://example.com")
        config = PlaywrightScreenshotConfig(section_screenshots=2, full_page=False)

        with (
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.PlaywrightScreenshotService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.get_current_artifact_context",
                return_value=mock_ctx,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.resolve_url.return_value = "https://example.com"
            mock_svc.screenshot_page = AsyncMock(
                return_value=(
                    None,
                    [Path("/tmp/s1.png"), Path("/tmp/s2.png")],
                    "Example",
                    "https://example.com",
                )
            )

            result = await run(inp, config)

        assert result.full_page_path is None
        assert result.screenshot_count == 2


class TestRunNoArtifactContext:
    @pytest.mark.asyncio
    async def test_uses_tempdir(self) -> None:
        inp = PlaywrightScreenshotInput(url="https://example.com")

        with (
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.PlaywrightScreenshotService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.get_current_artifact_context",
                return_value=None,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.resolve_url.return_value = "https://example.com"
            mock_svc.screenshot_page = AsyncMock(
                return_value=(
                    None,
                    [Path("/tmp/s1.png")],
                    "Example",
                    "https://example.com",
                )
            )

            result = await run(inp)

        assert result.status == "success"
        assert result.screenshot_count == 1


class TestRunDefaultConfig:
    @pytest.mark.asyncio
    async def test_none_config_uses_defaults(self) -> None:
        inp = PlaywrightScreenshotInput(html_path="/tmp/report.html")

        with (
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.PlaywrightScreenshotService"
            ) as MockService,
            patch(
                "praxis_vision_report.tasks.playwright_screenshot.task.get_current_artifact_context",
                return_value=None,
            ),
        ):
            mock_svc = MockService.return_value
            mock_svc.resolve_url.return_value = "file:///tmp/report.html"
            mock_svc.screenshot_page = AsyncMock(
                return_value=(None, [], "Report", "file:///tmp/report.html")
            )

            result = await run(inp)

        assert result.metadata["viewport"] == "1280x900"
        assert result.metadata["url"] == "file:///tmp/report.html"

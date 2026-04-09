"""Tests for playwright_screenshot service."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from praxis_vision_report.tasks.playwright_screenshot.service import (
    PlaywrightScreenshotService,
)


class TestResolveUrl:
    def test_with_http_url(self) -> None:
        service = PlaywrightScreenshotService()
        result = service.resolve_url(None, "https://example.com")
        assert result == "https://example.com"

    def test_with_html_path(self) -> None:
        service = PlaywrightScreenshotService()
        result = service.resolve_url("/tmp/report.html", None)
        assert result == f"file://{Path('/tmp/report.html').resolve()}"

    def test_url_takes_precedence(self) -> None:
        service = PlaywrightScreenshotService()
        result = service.resolve_url("/tmp/report.html", "https://example.com")
        assert result == "https://example.com"

    def test_raises_with_neither(self) -> None:
        service = PlaywrightScreenshotService()
        with pytest.raises(ValueError, match="Either url or html_path required"):
            service.resolve_url(None, None)


class TestTakeSectionScreenshots:
    @pytest.mark.asyncio
    async def test_takes_n_screenshots(self, tmp_path: Path) -> None:
        service = PlaywrightScreenshotService()
        mock_page: Any = MagicMock()
        mock_page.evaluate = AsyncMock(return_value=3600)
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.screenshot = AsyncMock()

        paths = await service.take_section_screenshots(
            mock_page, tmp_path, n=3, image_format="png", viewport_height=900
        )
        assert len(paths) == 3
        assert mock_page.screenshot.call_count == 3
        assert all(p.name.startswith("section_") for p in paths)

    @pytest.mark.asyncio
    async def test_section_filenames_are_numbered(self, tmp_path: Path) -> None:
        service = PlaywrightScreenshotService()
        mock_page: Any = MagicMock()
        mock_page.evaluate = AsyncMock(return_value=1800)
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.screenshot = AsyncMock()

        paths = await service.take_section_screenshots(
            mock_page, tmp_path, n=2, image_format="png", viewport_height=900
        )
        assert paths[0].name == "section_01.png"
        assert paths[1].name == "section_02.png"

    @pytest.mark.asyncio
    async def test_zero_sections_returns_empty(self, tmp_path: Path) -> None:
        service = PlaywrightScreenshotService()
        mock_page: Any = MagicMock()
        paths = await service.take_section_screenshots(
            mock_page, tmp_path, n=0, image_format="png", viewport_height=900
        )
        assert paths == []
        mock_page.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_scroll_positions_span_full_height(self, tmp_path: Path) -> None:
        service = PlaywrightScreenshotService()
        mock_page: Any = MagicMock()
        mock_page.evaluate = AsyncMock(side_effect=[2700, None, None, None])
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.screenshot = AsyncMock()

        await service.take_section_screenshots(
            mock_page, tmp_path, n=3, image_format="png", viewport_height=900
        )
        # scroll positions: y=0 (top), y=900 (middle), y=1800 (near bottom)
        scroll_calls = [
            call.args[0]
            for call in mock_page.evaluate.call_args_list
            if "scrollTo" in str(call.args[0])
        ]
        assert scroll_calls[0] == "window.scrollTo(0, 0)"
        assert scroll_calls[-1] == "window.scrollTo(0, 1800)"


class TestTakeFullPageScreenshot:
    @pytest.mark.asyncio
    async def test_full_page_screenshot(self, tmp_path: Path) -> None:
        service = PlaywrightScreenshotService()
        mock_page: Any = MagicMock()
        mock_page.screenshot = AsyncMock()

        path = await service.take_full_page_screenshot(mock_page, tmp_path, "png")
        assert path == tmp_path / "full_page.png"
        mock_page.screenshot.assert_called_once_with(
            full_page=True, path=str(tmp_path / "full_page.png")
        )

    @pytest.mark.asyncio
    async def test_jpeg_extension(self, tmp_path: Path) -> None:
        service = PlaywrightScreenshotService()
        mock_page: Any = MagicMock()
        mock_page.screenshot = AsyncMock()

        path = await service.take_full_page_screenshot(mock_page, tmp_path, "jpeg")
        assert path.suffix == ".jpeg"

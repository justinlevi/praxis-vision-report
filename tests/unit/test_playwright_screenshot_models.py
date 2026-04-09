"""Tests for playwright_screenshot models."""

from __future__ import annotations

import pytest
from praxis_vision_report.tasks.playwright_screenshot.models import (
    PlaywrightScreenshotConfig,
    PlaywrightScreenshotInput,
    PlaywrightScreenshotOutput,
)
from pydantic import ValidationError


class TestPlaywrightScreenshotInput:
    def test_valid_with_url_only(self) -> None:
        inp = PlaywrightScreenshotInput(url="https://example.com")
        assert inp.url == "https://example.com"
        assert inp.html_path is None

    def test_valid_with_html_path_only(self) -> None:
        inp = PlaywrightScreenshotInput(html_path="/tmp/report.html")
        assert inp.html_path == "/tmp/report.html"
        assert inp.url is None

    def test_valid_with_both(self) -> None:
        inp = PlaywrightScreenshotInput(url="https://example.com", html_path="/tmp/report.html")
        assert inp.url == "https://example.com"
        assert inp.html_path == "/tmp/report.html"

    def test_raises_with_neither(self) -> None:
        with pytest.raises(ValidationError, match="Either url or html_path"):
            PlaywrightScreenshotInput()


class TestPlaywrightScreenshotConfig:
    def test_defaults(self) -> None:
        config = PlaywrightScreenshotConfig()
        assert config.viewport_width == 1280
        assert config.viewport_height == 900
        assert config.full_page is False
        assert config.section_screenshots == 12
        assert config.image_format == "png"
        assert config.wait_until == "networkidle"

    def test_custom_values(self) -> None:
        config = PlaywrightScreenshotConfig(
            viewport_width=1920,
            viewport_height=1080,
            full_page=True,
            section_screenshots=5,
            image_format="jpeg",
            wait_until="load",
        )
        assert config.viewport_width == 1920
        assert config.full_page is True
        assert config.section_screenshots == 5


class TestPlaywrightScreenshotOutput:
    def test_construction(self) -> None:
        output = PlaywrightScreenshotOutput(
            full_page_path="/tmp/full.png",
            section_paths=["/tmp/s1.png", "/tmp/s2.png"],
            page_title="Test Page",
            page_url="https://example.com",
            screenshot_count=3,
            status="success",
            metadata={"viewport": "1280x900"},
        )
        assert output.full_page_path == "/tmp/full.png"
        assert len(output.section_paths) == 2
        assert output.page_title == "Test Page"
        assert output.screenshot_count == 3

    def test_defaults(self) -> None:
        output = PlaywrightScreenshotOutput(page_url="https://example.com")
        assert output.full_page_path is None
        assert output.section_paths == []
        assert output.page_title is None
        assert output.screenshot_count == 0
        assert output.status == "success"
        assert output.metadata == {}

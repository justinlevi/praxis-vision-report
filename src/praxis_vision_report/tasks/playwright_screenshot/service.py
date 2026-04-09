"""Service layer for playwright_screenshot task."""

from __future__ import annotations

from pathlib import Path

import structlog
from playwright.async_api import Page, async_playwright

from praxis_vision_report.tasks.playwright_screenshot.models import (
    PlaywrightScreenshotConfig,
)

logger = structlog.get_logger(__name__)


class PlaywrightScreenshotService:
    """Encapsulates Playwright screenshot logic."""

    def resolve_url(self, html_path: str | None, url: str | None) -> str:
        """Return URL to load. Converts local html_path to file:// URL."""
        if url:
            return url
        if html_path:
            return f"file://{Path(html_path).resolve()}"
        raise ValueError("Either url or html_path required")

    async def take_full_page_screenshot(
        self, page: Page, output_dir: Path, image_format: str
    ) -> Path:
        """Take a full-page screenshot."""
        path = output_dir / f"full_page.{image_format}"
        await page.screenshot(full_page=True, path=str(path))
        logger.debug("full_page_screenshot_saved", path=str(path))
        return path

    async def take_section_screenshots(
        self,
        page: Page,
        output_dir: Path,
        n: int,
        image_format: str,
        viewport_height: int,
    ) -> list[Path]:
        """Take N evenly-spaced viewport screenshots along the page height."""
        if n == 0:
            return []
        scroll_height: int = await page.evaluate(
            "document.documentElement.scrollHeight"
        )
        positions = [
            int(i * (scroll_height - viewport_height) / max(n - 1, 1))
            for i in range(n)
        ]
        paths: list[Path] = []
        for i, y in enumerate(positions, 1):
            await page.evaluate(f"window.scrollTo(0, {y})")
            await page.wait_for_timeout(300)
            p = output_dir / f"section_{i:02d}.{image_format}"
            await page.screenshot(path=str(p))
            paths.append(p)
            logger.debug("section_screenshot_saved", section=i, y=y, path=str(p))
        return paths

    async def screenshot_page(
        self,
        url: str,
        config: PlaywrightScreenshotConfig,
        output_dir: Path,
    ) -> tuple[Path | None, list[Path], str | None, str]:
        """Launch browser, navigate, take screenshots.

        Returns (full_page_path, section_paths, title, final_url).
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_viewport_size(
                {"width": config.viewport_width, "height": config.viewport_height}
            )
            await page.goto(url, wait_until=config.wait_until, timeout=30000)
            title: str | None = await page.title()
            final_url: str = page.url

            full_page_path: Path | None = None
            if config.full_page:
                full_page_path = await self.take_full_page_screenshot(
                    page, output_dir, config.image_format
                )

            section_paths = await self.take_section_screenshots(
                page,
                output_dir,
                config.section_screenshots,
                config.image_format,
                config.viewport_height,
            )
            await browser.close()

        return full_page_path, section_paths, title, final_url

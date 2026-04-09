"""playwright_screenshot task — renders HTML and takes screenshots."""

from __future__ import annotations

import tempfile
from pathlib import Path

from praxis_vision_report.tasks.playwright_screenshot.models import (
    PlaywrightScreenshotConfig,
    PlaywrightScreenshotInput,
    PlaywrightScreenshotOutput,
)
from praxis_vision_report.tasks.playwright_screenshot.service import (
    PlaywrightScreenshotService,
)

try:
    from praxis.observability.artifacts import get_current_artifact_context  # type: ignore
except ImportError:

    def get_current_artifact_context() -> None:  # type: ignore
        return None


async def run(
    input: PlaywrightScreenshotInput,
    config: PlaywrightScreenshotConfig | None = None,
) -> PlaywrightScreenshotOutput:
    """Render a URL or local HTML file and take screenshots."""
    if config is None:
        config = PlaywrightScreenshotConfig()

    service = PlaywrightScreenshotService()
    url = service.resolve_url(input.html_path, input.url)

    art_ctx = get_current_artifact_context()
    if art_ctx is not None:
        output_dir = art_ctx.artifact_dir / "screenshots"
    else:
        output_dir = Path(tempfile.mkdtemp()) / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_page_path, section_paths, title, final_url = await service.screenshot_page(
        url, config, output_dir
    )

    return PlaywrightScreenshotOutput(
        full_page_path=str(full_page_path) if full_page_path else None,
        section_paths=[str(p) for p in section_paths],
        page_title=title,
        page_url=final_url,
        screenshot_count=(1 if full_page_path else 0) + len(section_paths),
        status="success",
        metadata={
            "viewport": f"{config.viewport_width}x{config.viewport_height}",
            "url": url,
        },
    )

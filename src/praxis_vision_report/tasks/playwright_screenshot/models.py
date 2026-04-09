"""Models for playwright_screenshot task."""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class PlaywrightScreenshotInput(BaseModel):
    """Input for playwright_screenshot task."""

    url: str | None = Field(None, description="HTTP(S) URL to screenshot")
    html_path: str | None = Field(
        None, description="Local .html file path (auto-converted to file:// URL)"
    )

    @model_validator(mode="after")
    def require_url_or_path(self) -> Self:
        if not self.url and not self.html_path:
            raise ValueError("Either url or html_path must be provided")
        return self


class PlaywrightScreenshotConfig(BaseModel):
    """Configuration for playwright_screenshot task."""

    viewport_width: int = Field(default=1280, ge=320, le=3840)
    viewport_height: int = Field(default=900, ge=240, le=2160)
    full_page: bool = Field(
        default=False,
        description="Also take one full-page screenshot (may be very tall)",
    )
    section_screenshots: int = Field(
        default=12,
        ge=0,
        le=30,
        description="Number of evenly-spaced viewport-height screenshots",
    )
    image_format: Literal["png", "jpeg"] = Field(
        default="png", description="Screenshot image format"
    )
    wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = Field(
        default="networkidle",
        description="Playwright navigation wait condition",
    )


class PlaywrightScreenshotOutput(BaseModel):
    """Output from playwright_screenshot task."""

    full_page_path: str | None = Field(None)
    section_paths: list[str] = Field(default_factory=list)
    page_title: str | None = Field(None)
    page_url: str = Field(...)
    screenshot_count: int = Field(0)
    status: str = Field(default="success")
    metadata: dict[str, Any] = Field(default_factory=dict)

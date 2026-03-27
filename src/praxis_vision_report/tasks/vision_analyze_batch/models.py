"""Models for vision_analyze_batch task."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class VisionAnalyzeBatchInput(BaseModel):
    """Input for batch vision analysis."""

    # Images — one required
    image_paths: list[str] | None = Field(
        None,
        description="Explicit ordered list of image file paths",
    )
    images_dir: str | None = Field(
        None,
        description="Directory of images (sorted by filename)",
    )

    # Transcript — optional but recommended
    segments_path: str | None = Field(
        None,
        description="Path to .segments.json [{start, end, text}]",
    )
    transcript: str | None = Field(
        None,
        description="Raw transcript text (distributed evenly across images)",
    )

    system_prompt: str = Field(
        ...,
        min_length=1,
        description="Instructions for what to analyze per image/section",
    )
    global_context: str | None = Field(
        None,
        description="Optional overall context (session brief, etc.)",
    )

    @model_validator(mode="after")
    def validate_image_source(self) -> "VisionAnalyzeBatchInput":
        if self.image_paths is None and self.images_dir is None:
            raise ValueError("Either image_paths or images_dir must be provided")
        return self

    class Config:
        json_schema_extra = {
            "semantic_type": "image_batch",
            "compatible_with": ["image_collection", "file_path", "vision"],
        }


class SlideAnalysis(BaseModel):
    """Analysis result for a single slide/image."""

    image_path: str = Field(..., description="Path to the analyzed image")
    timecode: str | None = Field(None, description="Timecode string, e.g. '00:03:42'")
    timecode_seconds: float | None = Field(None, description="Timecode in seconds")
    analysis: str = Field(..., description="Full analysis text from the vision LLM")
    key_points: list[str] = Field(
        default_factory=list, description="Extracted key bullet points"
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="Token usage from the API call"
    )


class VisionAnalyzeBatchOutput(BaseModel):
    """Output from batch vision analysis."""

    analyses: list[SlideAnalysis] = Field(
        default_factory=list, description="Per-slide analysis results"
    )
    total_slides: int = Field(..., description="Total number of slides analyzed")
    total_tokens: int = Field(..., description="Total tokens consumed across all calls")
    model_used: str = Field(..., description="Vision model used for analysis")
    status: str = Field(default="success", description="Task status")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )

    class Config:
        json_schema_extra = {
            "semantic_type": "vision_analysis_batch",
            "compatible_with": ["analysis_collection", "slides"],
        }


class VisionAnalyzeBatchConfig(BaseModel):
    """Configuration for batch vision analysis."""

    model: str = Field(
        default="gpt-5.4",
        description="Vision model to use",
        examples=["gpt-5.4", "gpt-4o", "gpt-4-vision-preview"],
    )
    detail: str = Field(
        default="high",
        description="Image detail level for the vision API",
        examples=["low", "high", "auto"],
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens per image analysis",
        ge=256,
        le=16384,
    )
    concurrency: int = Field(
        default=5,
        description="Maximum concurrent API calls",
        ge=1,
        le=20,
    )

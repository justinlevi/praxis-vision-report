"""Models for vision_analyze_batch task."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    model_config = ConfigDict(
        json_schema_extra={
            "semantic_type": "image_batch",
            "compatible_with": ["image_collection", "file_path", "vision"],
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "semantic_type": "vision_analysis_batch",
            "compatible_with": ["analysis_collection", "slides"],
        }
    )


class VisionAnalyzeBatchConfig(BaseModel):
    """Configuration for batch vision analysis."""

    model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model to use for vision analysis",
        examples=["claude-sonnet-4-6", "claude-opus-4-6"],
    )
    concurrency: int = Field(
        default=5,
        description="Maximum concurrent claude -p calls",
        ge=1,
        le=20,
    )
    batch_size: int = Field(
        default=1,
        description=(
            "Number of slides to analyze per claude -p call. "
            "batch_size=1 sends one image per call (original behavior). "
            "batch_size=10 groups 10 slides into a single call, reducing subprocess overhead."
        ),
        ge=1,
        le=20,
    )
    min_slide_interval_seconds: float = Field(
        default=0.0,
        description=(
            "Minimum seconds between consecutive slides. "
            "Slides closer together than this are dropped (keeps the first). "
            "0.0 disables the filter. Recommended: 15.0 for conference talks."
        ),
        ge=0.0,
    )
    filter_stage_shots: bool = Field(
        default=False,
        description=(
            "When True, removes 'presenter on stage' shots before analysis. "
            "Detects wide-angle audience-view frames (bright backdrop, lit audience bottom) "
            "using OpenCV. Requires opencv-python-headless."
        ),
    )

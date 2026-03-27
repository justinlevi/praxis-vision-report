"""Models for compile_vision_report task."""

from typing import Any

from pydantic import BaseModel, Field


class CompileVisionReportInput(BaseModel):
    """Input for compile_vision_report task."""

    analyses: list[dict[str, Any]] = Field(
        ..., description="List of SlideAnalysis dicts from vision_analyze_batch"
    )
    title: str = Field(..., min_length=1, description="Report title, e.g. 'GTC 2026 Keynote'")
    description: str | None = Field(
        None, description="Optional session brief/abstract to include in the intro"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata: speaker, date, level, topic, etc."
    )

    class Config:
        json_schema_extra = {
            "semantic_type": "vision_analyses",
            "compatible_with": ["analysis_collection", "report_input"],
        }


class CompileVisionReportOutput(BaseModel):
    """Output from compile_vision_report task."""

    markdown_content: str = Field(..., description="Full markdown report content")
    html_content: str | None = Field(None, description="HTML version of the report")
    report_path: str | None = Field(
        None, description="Absolute path where the .md file was saved (None in standalone mode)"
    )
    html_path: str | None = Field(
        None, description="Absolute path where the .html file was saved (None if not generated)"
    )
    image_count: int = Field(0, description="Number of slide images copied to artifact directory")
    word_count: int = Field(0, description="Approximate word count of the markdown content")
    status: str = Field(default="success", description="Task status")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )

    class Config:
        json_schema_extra = {
            "semantic_type": "vision_report",
            "compatible_with": ["report", "markdown", "html"],
        }


class CompileVisionReportConfig(BaseModel):
    """Configuration for compile_vision_report task."""

    model: str = Field(
        default="gpt-5.4",
        description="OpenAI model to use for synthesis",
    )
    output_formats: list[str] = Field(
        default_factory=lambda: ["markdown", "html"],
        description="Output formats to generate: 'markdown' and/or 'html'",
    )
    copy_images: bool = Field(
        default=True,
        description="Copy slide images into the artifact directory for self-contained output",
    )
    images_subdir: str = Field(
        default="slides",
        description="Subdirectory name within the artifact dir for copied images",
    )

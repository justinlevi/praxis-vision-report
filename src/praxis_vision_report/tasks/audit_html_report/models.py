"""Models for audit_html_report task."""

from typing import Any

from pydantic import BaseModel, Field


class ImageDecision(BaseModel):
    """Decision about a specific image in the report."""

    image_filename: str = Field(..., description="Filename like 'frame_0042.png'")
    action: str = Field(..., description="'keep', 'remove', or 'move'")
    reason: str = Field(..., description="Why this image should be kept, removed, or moved")
    text_context: str | None = Field(None, description="What surrounding text says (or should say)")


class SectionIssue(BaseModel):
    """An issue with a specific section."""

    section_heading: str = Field(..., description="Heading or description of the section")
    problem: str = Field(..., description="What's wrong")
    suggestion: str = Field(..., description="How to fix it")


class AuditCritique(BaseModel):
    """Structured editorial critique of the HTML report."""

    overall_score: int = Field(..., ge=1, le=10, description="Overall quality score 1-10")
    executive_summary: str = Field(..., description="2-3 sentence summary of main issues")
    narrative_coherence_score: int = Field(
        ..., ge=1, le=10, description="Narrative coherence score 1-10"
    )
    narrative_issues: list[str] = Field(
        default_factory=list, description="Specific narrative problems"
    )
    image_decisions: list[ImageDecision] = Field(
        default_factory=list,
        description="Keep/remove/move decisions for each image",
    )
    section_issues: list[SectionIssue] = Field(default_factory=list)
    missing_technical_depth: list[str] = Field(
        default_factory=list,
        description="Topics from the slides not covered adequately",
    )
    redundant_content: list[str] = Field(
        default_factory=list, description="Sections or paragraphs to cut"
    )
    priority_edits: list[str] = Field(..., description="Top 5 highest-impact changes, in order")


class AuditHtmlReportInput(BaseModel):
    """Input for audit_html_report task."""

    screenshot_paths: list[str] = Field(
        default_factory=list,
        description="Paths to viewport screenshots from playwright_screenshot (optional)",
    )
    markdown_content: str = Field(
        ...,
        description="Full markdown content of the report (with image references)",
    )
    analyses: list[dict[str, Any]] = Field(
        ..., description="All SlideAnalysis dicts from vision_analyze_batch"
    )
    title: str = Field(..., description="Report title")
    description: str | None = Field(None, description="Session abstract/brief for context")


class AuditHtmlReportConfig(BaseModel):
    """Config for audit_html_report task."""

    model: str = Field(default="claude-sonnet-4-6", description="Claude model to use")
    max_screenshots: int = Field(
        default=12,
        ge=0,
        le=30,
        description="Max screenshots to include (trims from end if over limit). 0 = no screenshots.",
    )


class AuditHtmlReportOutput(BaseModel):
    """Output from audit_html_report task."""

    critique: AuditCritique = Field(..., description="Structured editorial critique")
    status: str = Field(default="success", description="Task status")
    model_used: str = Field(..., description="Model used for the audit")
    token_usage: dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")

"""Models for refine_html_report task."""

from typing import Any

from pydantic import BaseModel, Field


class RefineHtmlReportInput(BaseModel):
    """Input for refine_html_report task."""

    markdown_content: str = Field(
        ..., description="Original markdown content from compile_vision_report"
    )
    analyses: list[dict[str, Any]] = Field(
        ..., description="All SlideAnalysis dicts — full context for every slide"
    )
    critique: dict[str, Any] = Field(..., description="AuditCritique dict from audit_html_report")
    title: str = Field(..., description="Report title")
    description: str | None = Field(None, description="Session abstract/brief")
    speaker: str | None = Field(
        None, description="Speaker name(s), e.g. 'Aman Khan, Cursor'. Rendered as subtitle."
    )


class RefineHtmlReportConfig(BaseModel):
    """Config for refine_html_report task."""

    model: str = Field(default="gpt-5.4-mini", description="OpenAI model to use")


class RefineHtmlReportOutput(BaseModel):
    """Output from refine_html_report task."""

    refined_markdown: str = Field(..., description="The rewritten markdown content")
    refined_html: str | None = Field(None, description="HTML version of refined report")
    refined_md_path: str | None = Field(None, description="Path to saved *-refined.md")
    refined_html_path: str | None = Field(None, description="Path to saved *-refined.html")
    changes_summary: str = Field(..., description="Brief summary of major changes made")
    word_count: int = Field(0)
    status: str = Field(default="success")
    token_usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

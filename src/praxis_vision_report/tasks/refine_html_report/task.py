"""refine_html_report task."""

from __future__ import annotations

from praxis_vision_report.tasks.refine_html_report.models import (
    RefineHtmlReportConfig,
    RefineHtmlReportInput,
    RefineHtmlReportOutput,
)
from praxis_vision_report.tasks.refine_html_report.service import (
    RefineHtmlReportService,
)

try:
    from praxis.observability.artifacts import get_current_artifact_context  # type: ignore
except ImportError:

    def get_current_artifact_context() -> None:  # type: ignore
        return None


async def run(
    input: RefineHtmlReportInput,
    config: RefineHtmlReportConfig | None = None,
) -> RefineHtmlReportOutput:
    if config is None:
        config = RefineHtmlReportConfig()

    service = RefineHtmlReportService()

    refined_markdown, changes_summary, token_usage = await service.rewrite(
        markdown_content=input.markdown_content,
        analyses=input.analyses,
        critique=input.critique,
        title=input.title,
        description=input.description,
        model=config.model,
    )

    if input.speaker:
        refined_markdown = service.inject_subtitle(refined_markdown, input.speaker)

    word_count = service.count_words(refined_markdown)
    safe_title = service.safe_filename(input.title)

    refined_html: str | None = None
    refined_md_path: str | None = None
    refined_html_path: str | None = None

    art_ctx = get_current_artifact_context()

    # Convert to HTML
    try:
        refined_html = service.convert_to_html(refined_markdown, input.title)
    except Exception:
        refined_html = None

    # Save via artifact context
    if art_ctx is not None:
        md_path = art_ctx.save_artifact(f"{safe_title}-refined.md", refined_markdown)
        refined_md_path = str(md_path)
        if refined_html is not None:
            html_path = art_ctx.save_artifact(f"{safe_title}-refined.html", refined_html)
            refined_html_path = str(html_path)

    return RefineHtmlReportOutput(
        refined_markdown=refined_markdown,
        refined_html=refined_html,
        refined_md_path=refined_md_path,
        refined_html_path=refined_html_path,
        changes_summary=changes_summary,
        word_count=word_count,
        status="success",
        token_usage=token_usage,
        metadata={"model": config.model},
    )

"""compile_vision_report task."""

from __future__ import annotations

import logging
from typing import Any

import openai

from praxis_vision_report.tasks.compile_vision_report.models import (
    CompileVisionReportConfig,
    CompileVisionReportInput,
    CompileVisionReportOutput,
)
from praxis_vision_report.tasks.compile_vision_report.service import CompileVisionReportService

try:
    from praxis.observability.artifacts import (  # type: ignore[reportMissingImports]
        get_current_artifact_context,
    )
except ImportError:

    def get_current_artifact_context() -> None:  # type: ignore[misc]
        return None


logger = logging.getLogger(__name__)


async def run(
    input: CompileVisionReportInput,
    config: CompileVisionReportConfig,
) -> CompileVisionReportOutput:
    """Synthesize vision analyses into a cohesive blog-post markdown/HTML report."""
    logger.info("CompileVisionReport: Starting for '%s'", input.title)

    service = CompileVisionReportService()
    client = openai.AsyncOpenAI()

    # 1. Build synthesis prompt
    prompt = service.build_synthesis_prompt(
        title=input.title,
        description=input.description,
        analyses=input.analyses,
        metadata=input.metadata,
    )

    # 2. Call OpenAI to synthesize markdown
    logger.info(
        "CompileVisionReport: Calling %s to synthesize %d analyses",
        config.model,
        len(input.analyses),
    )
    markdown_content = await service.synthesize_markdown(
        client=client,
        prompt=prompt,
        model=config.model,
    )

    # 3. Compute derived values
    safe_title = service.safe_filename(input.title)
    word_count = len(markdown_content.split())

    # 4. Get artifact context (may be None in standalone mode)
    art_ctx = get_current_artifact_context()

    # 5. Copy images if configured and artifact context is available
    image_count = 0
    if config.copy_images and art_ctx is not None:
        image_count = service.copy_images_to_artifact(
            analyses=input.analyses,
            artifact_dir=str(art_ctx.artifact_dir),
            images_subdir=config.images_subdir,
        )

    # 6. Save markdown via artifact context
    report_path: str | None = None
    if art_ctx is not None:
        md_artifact_path = art_ctx.save_artifact(f"{safe_title}.md", markdown_content)
        report_path = str(md_artifact_path)
        logger.info("CompileVisionReport: Saved markdown to %s", report_path)

    # 7. Generate and save HTML if requested
    html_content: str | None = None
    html_path: str | None = None
    if "html" in config.output_formats:
        html_content = service.convert_to_html(markdown_content, input.title)
        if art_ctx is not None:
            html_artifact_path = art_ctx.save_artifact(f"{safe_title}.html", html_content)
            html_path = str(html_artifact_path)
            logger.info("CompileVisionReport: Saved HTML to %s", html_path)

    # 8. Build result metadata
    result_metadata: dict[str, Any] = {
        "title": input.title,
        "analysis_count": len(input.analyses),
        "model": config.model,
        "output_formats": config.output_formats,
        "copy_images": config.copy_images,
        "images_subdir": config.images_subdir,
    }
    if input.metadata:
        result_metadata["session_metadata"] = input.metadata

    logger.info(
        "CompileVisionReport: Completed — %d words, %d images, report_path=%s",
        word_count,
        image_count,
        report_path,
    )

    return CompileVisionReportOutput(
        markdown_content=markdown_content,
        html_content=html_content,
        report_path=report_path,
        html_path=html_path,
        image_count=image_count,
        word_count=word_count,
        status="success",
        metadata=result_metadata,
    )

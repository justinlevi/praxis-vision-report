"""audit_html_report task."""

from __future__ import annotations

from praxis_vision_report.tasks.audit_html_report.models import (
    AuditHtmlReportConfig,
    AuditHtmlReportInput,
    AuditHtmlReportOutput,
)
from praxis_vision_report.tasks.audit_html_report.service import (
    AuditHtmlReportService,
)


async def run(
    input: AuditHtmlReportInput,
    config: AuditHtmlReportConfig | None = None,
) -> AuditHtmlReportOutput:
    if config is None:
        config = AuditHtmlReportConfig()

    service = AuditHtmlReportService()
    critique, model_used, token_usage = await service.audit(
        screenshot_paths=input.screenshot_paths,
        markdown_content=input.markdown_content,
        analyses=input.analyses,
        title=input.title,
        description=input.description,
        model=config.model,
        max_screenshots=config.max_screenshots,
    )

    return AuditHtmlReportOutput(
        critique=critique,
        status="success",
        model_used=model_used,
        token_usage=token_usage,
        metadata={"screenshots_used": min(len(input.screenshot_paths), config.max_screenshots)},
    )

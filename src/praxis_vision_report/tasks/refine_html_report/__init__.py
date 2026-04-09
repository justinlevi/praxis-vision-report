"""refine_html_report task package.

Rewrite a blog-style report using editorial critique — full prose rewrite with
only value-adding images, saves *-refined.md and *-refined.html.
"""

from praxis_vision_report.tasks.refine_html_report.models import (
    RefineHtmlReportConfig,
    RefineHtmlReportInput,
    RefineHtmlReportOutput,
)
from praxis_vision_report.tasks.refine_html_report.task import run

__all__ = [
    "RefineHtmlReportConfig",
    "RefineHtmlReportInput",
    "RefineHtmlReportOutput",
    "run",
]

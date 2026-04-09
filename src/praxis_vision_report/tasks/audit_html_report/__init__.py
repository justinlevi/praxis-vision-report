"""audit_html_report task package.

Visually audit a rendered HTML report using Claude vision — evaluates image-text
correlation, narrative quality, and editorial polish.
"""

from praxis_vision_report.tasks.audit_html_report.models import (
    AuditCritique,
    AuditHtmlReportConfig,
    AuditHtmlReportInput,
    AuditHtmlReportOutput,
    ImageDecision,
    SectionIssue,
)
from praxis_vision_report.tasks.audit_html_report.task import run

__all__ = [
    "AuditCritique",
    "AuditHtmlReportConfig",
    "AuditHtmlReportInput",
    "AuditHtmlReportOutput",
    "ImageDecision",
    "SectionIssue",
    "run",
]

"""Tests for audit_html_report task run() function."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from praxis_vision_report.tasks.audit_html_report.models import (
    AuditCritique,
    AuditHtmlReportConfig,
    AuditHtmlReportInput,
    AuditHtmlReportOutput,
)
from praxis_vision_report.tasks.audit_html_report.task import run


def make_fake_critique() -> AuditCritique:
    return AuditCritique(
        overall_score=7,
        executive_summary="Solid draft.",
        narrative_coherence_score=6,
        priority_edits=["Remove title slide", "Expand GPU section"],
    )


SAMPLE_ANALYSES: list[dict[str, Any]] = [
    {"image_path": "/tmp/slide1.png", "key_points": ["GPU"]},
]


class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_returns_correct_output(self) -> None:
        fake_critique = make_fake_critique()
        fake_token_usage: dict[str, Any] = {"input_tokens": 5000, "output_tokens": 800}

        inp = AuditHtmlReportInput(
            screenshot_paths=["/tmp/shot1.png", "/tmp/shot2.png"],
            markdown_content="# Report\n\nContent.",
            analyses=SAMPLE_ANALYSES,
            title="GTC 2026",
        )
        config = AuditHtmlReportConfig()

        with patch(
            "praxis_vision_report.tasks.audit_html_report.task.AuditHtmlReportService"
        ) as MockService:
            mock_svc = MockService.return_value
            mock_svc.audit = AsyncMock(
                return_value=(fake_critique, "gpt-5.4-mini", fake_token_usage)
            )

            result = await run(inp, config)

        assert isinstance(result, AuditHtmlReportOutput)
        assert result.status == "success"
        assert result.critique.overall_score == 7
        assert result.model_used == "gpt-5.4-mini"
        assert result.token_usage["input_tokens"] == 5000
        assert result.metadata["screenshots_used"] == 2

    @pytest.mark.asyncio
    async def test_default_config(self) -> None:
        fake_critique = make_fake_critique()

        inp = AuditHtmlReportInput(
            screenshot_paths=["/tmp/shot.png"],
            markdown_content="# Report",
            analyses=[],
            title="Test",
        )

        with patch(
            "praxis_vision_report.tasks.audit_html_report.task.AuditHtmlReportService"
        ) as MockService:
            mock_svc = MockService.return_value
            mock_svc.audit = AsyncMock(return_value=(fake_critique, "gpt-5.4-mini", {}))

            result = await run(inp, None)

        assert result.status == "success"
        # Verify default config was used — audit called with default model
        call_kwargs = mock_svc.audit.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4-mini"
        assert call_kwargs["max_screenshots"] == 12

    @pytest.mark.asyncio
    async def test_screenshots_used_capped_by_max(self) -> None:
        fake_critique = make_fake_critique()

        inp = AuditHtmlReportInput(
            screenshot_paths=[f"/tmp/shot{i}.png" for i in range(20)],
            markdown_content="# Report",
            analyses=[],
            title="Test",
        )
        config = AuditHtmlReportConfig(max_screenshots=5)

        with patch(
            "praxis_vision_report.tasks.audit_html_report.task.AuditHtmlReportService"
        ) as MockService:
            mock_svc = MockService.return_value
            mock_svc.audit = AsyncMock(return_value=(fake_critique, "gpt-5.4-mini", {}))

            result = await run(inp, config)

        assert result.metadata["screenshots_used"] == 5

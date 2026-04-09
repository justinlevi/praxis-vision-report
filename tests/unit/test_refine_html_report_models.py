"""Tests for refine_html_report models."""

from __future__ import annotations

import pytest
from praxis_vision_report.tasks.refine_html_report.models import (
    RefineHtmlReportConfig,
    RefineHtmlReportInput,
    RefineHtmlReportOutput,
)
from pydantic import ValidationError


class TestRefineHtmlReportInput:
    def test_valid_construction(self) -> None:
        inp = RefineHtmlReportInput(
            markdown_content="# Report\n\nContent.",
            analyses=[{"image_path": "/tmp/slide.png", "key_points": ["GPU"]}],
            critique={"overall_score": 7, "priority_edits": ["Fix intro"]},
            title="GTC 2026",
        )
        assert inp.title == "GTC 2026"
        assert inp.description is None

    def test_with_description(self) -> None:
        inp = RefineHtmlReportInput(
            markdown_content="# Report",
            analyses=[],
            critique={},
            title="Test",
            description="A session about AI",
        )
        assert inp.description == "A session about AI"

    def test_requires_markdown_content(self) -> None:
        with pytest.raises(ValidationError):
            RefineHtmlReportInput(  # type: ignore[call-arg]
                analyses=[],
                critique={},
                title="Test",
            )

    def test_requires_critique(self) -> None:
        with pytest.raises(ValidationError):
            RefineHtmlReportInput(  # type: ignore[call-arg]
                markdown_content="# Report",
                analyses=[],
                title="Test",
            )


class TestRefineHtmlReportConfig:
    def test_defaults(self) -> None:
        cfg = RefineHtmlReportConfig()
        assert cfg.model == "gpt-5.4-mini"

    def test_custom(self) -> None:
        cfg = RefineHtmlReportConfig(model="gpt-5.4")
        assert cfg.model == "gpt-5.4"


class TestRefineHtmlReportOutput:
    def test_all_fields(self) -> None:
        out = RefineHtmlReportOutput(
            refined_markdown="# Refined\n\nBetter content.",
            refined_html="<html>...</html>",
            refined_md_path="/tmp/artifacts/report-refined.md",
            refined_html_path="/tmp/artifacts/report-refined.html",
            changes_summary="Full rewrite applying 5 priority edits.",
            word_count=42,
            status="success",
            token_usage={"input_tokens": 10000, "output_tokens": 5000},
            metadata={"model": "gpt-5.4-mini"},
        )
        assert out.word_count == 42
        assert out.refined_html is not None

    def test_defaults(self) -> None:
        out = RefineHtmlReportOutput(
            refined_markdown="# Report",
            changes_summary="Rewrote.",
        )
        assert out.refined_html is None
        assert out.refined_md_path is None
        assert out.refined_html_path is None
        assert out.word_count == 0
        assert out.status == "success"
        assert out.token_usage == {}
        assert out.metadata == {}

"""Tests for audit_html_report models."""

from __future__ import annotations

import pytest
from praxis_vision_report.tasks.audit_html_report.models import (
    AuditCritique,
    AuditHtmlReportConfig,
    AuditHtmlReportInput,
    AuditHtmlReportOutput,
    ImageDecision,
    SectionIssue,
)
from pydantic import ValidationError


class TestImageDecision:
    def test_keep_action(self) -> None:
        d = ImageDecision(
            image_filename="frame_0042.png", action="keep", reason="architecture diagram"
        )
        assert d.action == "keep"

    def test_remove_action(self) -> None:
        d = ImageDecision(image_filename="slide_001.png", action="remove", reason="title slide")
        assert d.action == "remove"

    def test_move_action_with_text_context(self) -> None:
        d = ImageDecision(
            image_filename="bench.png",
            action="move",
            reason="belongs after benchmarks section",
            text_context="The benchmarks show 5x speedup",
        )
        assert d.action == "move"
        assert d.text_context is not None

    def test_text_context_defaults_none(self) -> None:
        d = ImageDecision(image_filename="x.png", action="keep", reason="good")
        assert d.text_context is None


class TestSectionIssue:
    def test_valid_construction(self) -> None:
        s = SectionIssue(
            section_heading="Introduction",
            problem="Too vague",
            suggestion="Add specific architecture names",
        )
        assert s.section_heading == "Introduction"
        assert s.problem == "Too vague"


class TestAuditCritique:
    def test_valid_construction(self) -> None:
        c = AuditCritique(
            overall_score=7,
            executive_summary="Solid draft but too many decorative images.",
            narrative_coherence_score=6,
            priority_edits=["Remove title slide image", "Expand GPU section"],
        )
        assert c.overall_score == 7
        assert len(c.priority_edits) == 2
        assert c.image_decisions == []
        assert c.section_issues == []

    def test_score_must_be_1_to_10(self) -> None:
        with pytest.raises(ValidationError):
            AuditCritique(
                overall_score=0,
                executive_summary="Bad",
                narrative_coherence_score=5,
                priority_edits=["fix"],
            )

    def test_score_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            AuditCritique(
                overall_score=11,
                executive_summary="Bad",
                narrative_coherence_score=5,
                priority_edits=["fix"],
            )

    def test_narrative_coherence_score_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            AuditCritique(
                overall_score=5,
                executive_summary="Ok",
                narrative_coherence_score=0,
                priority_edits=["fix"],
            )

    def test_narrative_coherence_score_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            AuditCritique(
                overall_score=5,
                executive_summary="Ok",
                narrative_coherence_score=11,
                priority_edits=["fix"],
            )

    def test_with_nested_models(self) -> None:
        c = AuditCritique(
            overall_score=5,
            executive_summary="Needs work.",
            narrative_coherence_score=4,
            image_decisions=[
                ImageDecision(
                    image_filename="slide_001.png", action="remove", reason="title slide"
                ),
            ],
            section_issues=[
                SectionIssue(section_heading="Intro", problem="Vague", suggestion="Be specific"),
            ],
            priority_edits=["Fix intro"],
        )
        assert len(c.image_decisions) == 1
        assert len(c.section_issues) == 1


class TestAuditHtmlReportInput:
    def test_valid_construction(self) -> None:
        inp = AuditHtmlReportInput(
            screenshot_paths=["/tmp/shot1.png", "/tmp/shot2.png"],
            markdown_content="# Report\n\nContent here.",
            analyses=[{"image_path": "/tmp/slide.png", "key_points": ["GPU"]}],
            title="GTC 2026",
        )
        assert len(inp.screenshot_paths) == 2
        assert inp.description is None

    def test_requires_screenshot_paths(self) -> None:
        with pytest.raises(ValidationError):
            AuditHtmlReportInput(  # type: ignore[call-arg]
                markdown_content="# Report",
                analyses=[],
                title="Test",
            )

    def test_requires_markdown_content(self) -> None:
        with pytest.raises(ValidationError):
            AuditHtmlReportInput(  # type: ignore[call-arg]
                screenshot_paths=["/tmp/shot.png"],
                analyses=[],
                title="Test",
            )

    def test_requires_title(self) -> None:
        with pytest.raises(ValidationError):
            AuditHtmlReportInput(  # type: ignore[call-arg]
                screenshot_paths=["/tmp/shot.png"],
                markdown_content="# Report",
                analyses=[],
            )

    def test_with_description(self) -> None:
        inp = AuditHtmlReportInput(
            screenshot_paths=["/tmp/shot.png"],
            markdown_content="# Report",
            analyses=[],
            title="Test",
            description="A deep dive session",
        )
        assert inp.description == "A deep dive session"


class TestAuditHtmlReportConfig:
    def test_defaults(self) -> None:
        cfg = AuditHtmlReportConfig()
        assert cfg.model == "gpt-5.4-mini"
        assert cfg.max_screenshots == 12

    def test_custom_values(self) -> None:
        cfg = AuditHtmlReportConfig(model="claude-sonnet-4-20250514", max_screenshots=20)
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.max_screenshots == 20

    def test_max_screenshots_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            AuditHtmlReportConfig(max_screenshots=0)

    def test_max_screenshots_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            AuditHtmlReportConfig(max_screenshots=31)


class TestAuditHtmlReportOutput:
    def test_with_nested_critique(self) -> None:
        critique = AuditCritique(
            overall_score=8,
            executive_summary="Good report.",
            narrative_coherence_score=7,
            priority_edits=["Minor polish"],
        )
        out = AuditHtmlReportOutput(
            critique=critique,
            model_used="gpt-5.4-mini",
            token_usage={"input_tokens": 1000, "output_tokens": 500},
        )
        assert out.status == "success"
        assert out.critique.overall_score == 8
        assert out.model_used == "gpt-5.4-mini"

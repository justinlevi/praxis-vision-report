"""Tests for refine_html_report service methods."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from praxis_vision_report.tasks.refine_html_report.service import RefineHtmlReportService


@pytest.fixture()
def service() -> RefineHtmlReportService:
    return RefineHtmlReportService()


@pytest.fixture()
def sample_critique() -> dict[str, Any]:
    return {
        "overall_score": 6,
        "executive_summary": "Needs more technical depth and fewer decorative images.",
        "priority_edits": ["Remove title slide", "Expand GPU section", "Fix intro"],
        "image_decisions": [
            {"image_filename": "frame_0001.png", "action": "remove"},
            {"image_filename": "frame_0002.png", "action": "keep"},
            {"image_filename": "frame_0003.png", "action": "remove"},
        ],
        "section_issues": [
            {
                "section_heading": "Introduction",
                "problem": "Too generic",
                "suggestion": "Lead with the key claim",
            }
        ],
    }


def _make_mock_response(
    text: str, prompt_tokens: int = 100, completion_tokens: int = 100
) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = text
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    return mock_response


class TestBuildAnalysesContext:
    def test_formats_correctly(self, service: RefineHtmlReportService) -> None:
        analyses: list[dict[str, Any]] = [
            {
                "image_path": "/tmp/slides/frame_0001.png",
                "timecode": "00:01:00",
                "analysis": "Shows GPU cluster architecture",
                "key_points": ["Blackwell GPU", "NVLink 5"],
            },
        ]
        result = service.build_analyses_context(analyses)
        assert "frame_0001.png" in result
        assert "00:01:00" in result
        assert "GPU cluster architecture" in result
        assert "Blackwell GPU" in result

    def test_empty_list(self, service: RefineHtmlReportService) -> None:
        assert service.build_analyses_context([]) == ""

    def test_caps_analysis_text(self, service: RefineHtmlReportService) -> None:
        long_analysis = "x" * 500
        analyses: list[dict[str, Any]] = [
            {"image_path": "/tmp/slide.png", "analysis": long_analysis, "key_points": []},
        ]
        result = service.build_analyses_context(analyses)
        # analysis_text capped at 300 chars
        assert len(long_analysis) > 300
        assert "x" * 300 in result
        assert "x" * 301 not in result


class TestFormatCritiqueForPrompt:
    def test_contains_summary(
        self, service: RefineHtmlReportService, sample_critique: dict[str, Any]
    ) -> None:
        result = service.format_critique_for_prompt(sample_critique)
        assert "Needs more technical depth" in result

    def test_lists_removes(
        self, service: RefineHtmlReportService, sample_critique: dict[str, Any]
    ) -> None:
        result = service.format_critique_for_prompt(sample_critique)
        assert "frame_0001.png" in result
        assert "frame_0003.png" in result
        # Keep should NOT appear in removes
        assert "frame_0002.png" not in result.split("IMAGES TO REMOVE:")[1].split("\n")[0]

    def test_lists_priority_edits(
        self, service: RefineHtmlReportService, sample_critique: dict[str, Any]
    ) -> None:
        result = service.format_critique_for_prompt(sample_critique)
        assert "Remove title slide" in result
        assert "Expand GPU section" in result

    def test_lists_section_issues(
        self, service: RefineHtmlReportService, sample_critique: dict[str, Any]
    ) -> None:
        result = service.format_critique_for_prompt(sample_critique)
        assert "Introduction" in result
        assert "Too generic" in result

    def test_no_removes_shows_none(self, service: RefineHtmlReportService) -> None:
        critique: dict[str, Any] = {
            "executive_summary": "Good.",
            "priority_edits": [],
            "image_decisions": [{"image_filename": "x.png", "action": "keep"}],
            "section_issues": [],
        }
        result = service.format_critique_for_prompt(critique)
        assert "IMAGES TO REMOVE: none" in result


class TestSafeFilename:
    def test_basic(self, service: RefineHtmlReportService) -> None:
        assert service.safe_filename("GTC 2026 Keynote") == "gtc-2026-keynote"

    def test_special_chars_removed(self, service: RefineHtmlReportService) -> None:
        result = service.safe_filename("Title: With (Parens) & More!")
        assert "(" not in result
        assert ")" not in result
        assert "&" not in result

    def test_max_80_chars(self, service: RefineHtmlReportService) -> None:
        result = service.safe_filename("A" * 200)
        assert len(result) <= 80

    def test_collapses_whitespace_and_underscores(self, service: RefineHtmlReportService) -> None:
        result = service.safe_filename("hello   world__test")
        assert "--" not in result
        assert "__" not in result


class TestCountWords:
    def test_simple(self, service: RefineHtmlReportService) -> None:
        assert service.count_words("hello world foo") == 3

    def test_empty(self, service: RefineHtmlReportService) -> None:
        assert service.count_words("") == 0

    def test_multiline(self, service: RefineHtmlReportService) -> None:
        assert service.count_words("one two\nthree four\nfive") == 5


class TestConvertToHtml:
    def test_contains_h1(self, service: RefineHtmlReportService) -> None:
        html = service.convert_to_html("# Title\n\nParagraph.", "Test")
        assert "<h1" in html

    def test_contains_p(self, service: RefineHtmlReportService) -> None:
        html = service.convert_to_html("# Title\n\nParagraph.", "Test")
        assert "<p>" in html

    def test_contains_body(self, service: RefineHtmlReportService) -> None:
        html = service.convert_to_html("Content", "Title")
        assert "<body>" in html
        assert "</body>" in html

    def test_starts_with_doctype(self, service: RefineHtmlReportService) -> None:
        html = service.convert_to_html("Content", "Title")
        assert html.startswith("<!DOCTYPE html>")

    def test_title_in_head(self, service: RefineHtmlReportService) -> None:
        html = service.convert_to_html("Content", "My Report")
        assert "<title>My Report</title>" in html

    def test_has_refined_badge_css(self, service: RefineHtmlReportService) -> None:
        html = service.convert_to_html("Content", "Title")
        assert ".refined-badge" in html


class TestRewrite:
    @pytest.mark.asyncio
    async def test_returns_refined_markdown(self, service: RefineHtmlReportService) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(
                "# Refined Report\n\nBetter content here.", 10000, 5000
            )
        )
        service._client = mock_client

        refined, changes, token_usage = await service.rewrite(
            markdown_content="# Draft\n\nOriginal.",
            analyses=[{"image_path": "/tmp/slide.png", "key_points": ["GPU"]}],
            critique={"priority_edits": ["Fix intro"], "image_decisions": []},
            title="GTC 2026",
            description=None,
            model="gpt-5.4-mini",
        )

        assert "Refined Report" in refined
        assert token_usage["input_tokens"] == 10000
        assert "1 priority edits" in changes
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_strips_code_block_from_response(self, service: RefineHtmlReportService) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("```markdown\n# Report\n\nContent.\n```")
        )
        service._client = mock_client

        refined, _, _ = await service.rewrite(
            markdown_content="# Draft",
            analyses=[],
            critique={"priority_edits": [], "image_decisions": []},
            title="Test",
            description=None,
            model="gpt-5.4-mini",
        )

        assert not refined.startswith("```")
        assert not refined.endswith("```")
        assert "# Report" in refined

    @pytest.mark.asyncio
    async def test_changes_summary_counts_removes(self, service: RefineHtmlReportService) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("# Report")
        )
        service._client = mock_client

        _, changes, _ = await service.rewrite(
            markdown_content="# Draft",
            analyses=[],
            critique={
                "priority_edits": ["a", "b"],
                "image_decisions": [
                    {"image_filename": "a.png", "action": "remove"},
                    {"image_filename": "b.png", "action": "keep"},
                    {"image_filename": "c.png", "action": "remove"},
                ],
            },
            title="Test",
            description=None,
            model="gpt-5.4-mini",
        )

        assert "2 priority edits" in changes
        assert "2 decorative images" in changes

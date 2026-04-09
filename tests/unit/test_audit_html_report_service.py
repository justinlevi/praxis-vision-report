"""Tests for audit_html_report service methods."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from praxis_vision_report.tasks.audit_html_report.models import AuditCritique
from praxis_vision_report.tasks.audit_html_report.service import AuditHtmlReportService


@pytest.fixture()
def service() -> AuditHtmlReportService:
    return AuditHtmlReportService()


@pytest.fixture()
def sample_analyses() -> list[dict[str, Any]]:
    return [
        {
            "image_path": "/tmp/slides/frame_0001.png",
            "timecode": "00:01:00",
            "key_points": ["Blackwell GPU", "NVLink 5", "72 GPUs per rack"],
        },
        {
            "image_path": "/tmp/slides/frame_0002.png",
            "timecode": "00:03:30",
            "key_points": ["5x speedup"],
        },
    ]


VALID_CRITIQUE_JSON = json.dumps(
    {
        "overall_score": 7,
        "executive_summary": "Solid draft but images need pruning.",
        "narrative_coherence_score": 6,
        "narrative_issues": ["Intro is weak"],
        "image_decisions": [
            {
                "image_filename": "frame_0001.png",
                "action": "keep",
                "reason": "architecture diagram",
                "text_context": "paragraph discusses GPU layout",
            },
            {
                "image_filename": "frame_0002.png",
                "action": "remove",
                "reason": "title slide",
                "text_context": None,
            },
        ],
        "section_issues": [
            {
                "section_heading": "Introduction",
                "problem": "Too generic",
                "suggestion": "Lead with the key technical claim",
            }
        ],
        "missing_technical_depth": ["NVLink topology"],
        "redundant_content": ["Conclusion repeats intro"],
        "priority_edits": ["Remove title slide", "Expand GPU section", "Fix intro", "Add benchmarks", "Tighten prose"],
    }
)


class TestEncodeScreenshot:
    def test_returns_base64(self, service: AuditHtmlReportService, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n fake image data")
        result = service.encode_screenshot(str(img_path))
        decoded = base64.standard_b64decode(result)
        assert decoded == b"\x89PNG\r\n\x1a\n fake image data"

    def test_mock_file_open(self, service: AuditHtmlReportService) -> None:
        fake_data = b"fake png bytes"
        with patch("builtins.open", mock_open(read_data=fake_data)):
            result = service.encode_screenshot("/fake/path.png")
        expected = base64.standard_b64encode(fake_data).decode()
        assert result == expected


class TestBuildAnalysesSummary:
    def test_formats_correctly(
        self, service: AuditHtmlReportService, sample_analyses: list[dict[str, Any]]
    ) -> None:
        summary = service.build_analyses_summary(sample_analyses)
        assert "frame_0001.png" in summary
        assert "00:01:00" in summary
        assert "Blackwell GPU" in summary
        assert "frame_0002.png" in summary

    def test_no_key_points(self, service: AuditHtmlReportService) -> None:
        analyses: list[dict[str, Any]] = [{"image_path": "/tmp/x.png", "timecode": "00:00:00"}]
        summary = service.build_analyses_summary(analyses)
        assert "no key points" in summary

    def test_empty_list(self, service: AuditHtmlReportService) -> None:
        assert service.build_analyses_summary([]) == ""


class TestBuildPrompt:
    def test_contains_title(
        self, service: AuditHtmlReportService, sample_analyses: list[dict[str, Any]]
    ) -> None:
        prompt = service.build_prompt("# Report", sample_analyses, "GTC 2026", None)
        assert "GTC 2026" in prompt

    def test_contains_description(
        self, service: AuditHtmlReportService, sample_analyses: list[dict[str, Any]]
    ) -> None:
        prompt = service.build_prompt("# Report", sample_analyses, "GTC", "Deep dive into Blackwell")
        assert "Deep dive into Blackwell" in prompt
        assert "SESSION ABSTRACT" in prompt

    def test_no_description_section_when_none(
        self, service: AuditHtmlReportService, sample_analyses: list[dict[str, Any]]
    ) -> None:
        prompt = service.build_prompt("# Report", sample_analyses, "GTC", None)
        assert "SESSION ABSTRACT" not in prompt

    def test_contains_analyses(
        self, service: AuditHtmlReportService, sample_analyses: list[dict[str, Any]]
    ) -> None:
        prompt = service.build_prompt("# Report", sample_analyses, "GTC", None)
        assert "frame_0001.png" in prompt
        assert "Blackwell GPU" in prompt


class TestParseCritique:
    def test_valid_json(self, service: AuditHtmlReportService) -> None:
        critique = service.parse_critique(VALID_CRITIQUE_JSON)
        assert isinstance(critique, AuditCritique)
        assert critique.overall_score == 7
        assert len(critique.image_decisions) == 2
        assert critique.image_decisions[0].action == "keep"
        assert len(critique.section_issues) == 1
        assert len(critique.priority_edits) == 5

    def test_strips_code_block_wrapper(self, service: AuditHtmlReportService) -> None:
        wrapped = f"```json\n{VALID_CRITIQUE_JSON}\n```"
        critique = service.parse_critique(wrapped)
        assert isinstance(critique, AuditCritique)
        assert critique.overall_score == 7

    def test_strips_plain_code_block(self, service: AuditHtmlReportService) -> None:
        wrapped = f"```\n{VALID_CRITIQUE_JSON}\n```"
        critique = service.parse_critique(wrapped)
        assert critique.overall_score == 7

    def test_raises_on_malformed_json(self, service: AuditHtmlReportService) -> None:
        with pytest.raises(json.JSONDecodeError):
            service.parse_critique("not valid json at all")

    def test_raises_on_invalid_score(self, service: AuditHtmlReportService) -> None:
        bad = json.dumps({
            "overall_score": 0,
            "executive_summary": "Bad",
            "narrative_coherence_score": 5,
            "priority_edits": ["fix"],
        })
        with pytest.raises(Exception):  # ValidationError from pydantic
            service.parse_critique(bad)


def _make_mock_response(text: str, prompt_tokens: int = 5000, completion_tokens: int = 800) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = text
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    return mock_response


class TestAudit:
    @pytest.mark.asyncio
    async def test_audit_builds_content_and_parses(
        self, service: AuditHtmlReportService, tmp_path: Path
    ) -> None:
        shots = []
        for i in range(3):
            p = tmp_path / f"shot_{i}.png"
            p.write_bytes(b"fake png")
            shots.append(str(p))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(VALID_CRITIQUE_JSON, 5000, 800)
        )
        service._client = mock_client

        analyses: list[dict[str, Any]] = [
            {"image_path": "/tmp/slide.png", "timecode": "00:00:00", "key_points": ["GPU"]},
        ]

        critique, model_used, token_usage = await service.audit(
            screenshot_paths=shots,
            markdown_content="# Report\n\nContent.",
            analyses=analyses,
            title="GTC 2026",
            description=None,
            model="gpt-5.4-mini",
            max_screenshots=12,
        )

        assert isinstance(critique, AuditCritique)
        assert model_used == "gpt-5.4-mini"
        assert token_usage["input_tokens"] == 5000
        assert token_usage["output_tokens"] == 800
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_audit_respects_max_screenshots(
        self, service: AuditHtmlReportService, tmp_path: Path
    ) -> None:
        shots = []
        for i in range(10):
            p = tmp_path / f"shot_{i}.png"
            p.write_bytes(b"fake")
            shots.append(str(p))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(VALID_CRITIQUE_JSON, 100, 100)
        )
        service._client = mock_client

        await service.audit(
            screenshot_paths=shots,
            markdown_content="# Report",
            analyses=[],
            title="Test",
            description=None,
            model="gpt-5.4-mini",
            max_screenshots=3,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        # messages[0] = system, messages[1] = user with [3 images + 1 text] = 4 items
        assert len(messages[1]["content"]) == 4

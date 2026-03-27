"""Tests for compile_vision_report models."""

from __future__ import annotations

import pytest
from praxis_vision_report.tasks.compile_vision_report.models import (
    CompileVisionReportConfig,
    CompileVisionReportInput,
    CompileVisionReportOutput,
)
from pydantic import ValidationError


class TestCompileVisionReportInput:
    def test_valid_input_minimal(self) -> None:
        inp = CompileVisionReportInput(
            analyses=[{"analysis": "slide 1 content"}],
            title="GTC 2026 Keynote",
        )
        assert inp.title == "GTC 2026 Keynote"
        assert len(inp.analyses) == 1
        assert inp.description is None
        assert inp.metadata is None

    def test_valid_input_full(self) -> None:
        inp = CompileVisionReportInput(
            analyses=[{"analysis": "slide 1"}, {"analysis": "slide 2"}],
            title="GTC 2026 Keynote",
            description="A keynote about AI infrastructure",
            metadata={"speaker": "Jensen Huang", "date": "2026-03-18", "level": "keynote"},
        )
        assert inp.description == "A keynote about AI infrastructure"
        assert inp.metadata is not None
        assert inp.metadata["speaker"] == "Jensen Huang"

    def test_title_required(self) -> None:
        with pytest.raises(ValidationError):
            CompileVisionReportInput(  # type: ignore[call-arg]
                analyses=[{"analysis": "content"}],
            )

    def test_analyses_required(self) -> None:
        with pytest.raises(ValidationError):
            CompileVisionReportInput(  # type: ignore[call-arg]
                title="My Report",
            )

    def test_empty_analyses_list_is_valid(self) -> None:
        inp = CompileVisionReportInput(analyses=[], title="Empty")
        assert inp.analyses == []

    def test_analyses_contain_arbitrary_dicts(self) -> None:
        analyses = [
            {
                "image_path": "/tmp/slide1.png",
                "analysis": "This slide shows GPU architecture",
                "key_points": ["point 1", "point 2"],
                "slide_number": 1,
            }
        ]
        inp = CompileVisionReportInput(analyses=analyses, title="Test")
        assert inp.analyses[0]["image_path"] == "/tmp/slide1.png"

    def test_serialization_round_trip(self) -> None:
        inp = CompileVisionReportInput(
            analyses=[{"text": "hello"}],
            title="Round Trip Test",
            description="A test",
            metadata={"key": "value"},
        )
        data = inp.model_dump()
        restored = CompileVisionReportInput(**data)
        assert restored.title == inp.title
        assert restored.description == inp.description
        assert restored.metadata == inp.metadata
        assert restored.analyses == inp.analyses


class TestCompileVisionReportConfig:
    def test_default_model(self) -> None:
        config = CompileVisionReportConfig()
        assert config.model == "gpt-5.4"

    def test_default_output_formats(self) -> None:
        config = CompileVisionReportConfig()
        assert "markdown" in config.output_formats
        assert "html" in config.output_formats

    def test_default_copy_images(self) -> None:
        config = CompileVisionReportConfig()
        assert config.copy_images is True

    def test_default_images_subdir(self) -> None:
        config = CompileVisionReportConfig()
        assert config.images_subdir == "slides"

    def test_custom_values(self) -> None:
        config = CompileVisionReportConfig(
            model="gpt-4o",
            output_formats=["markdown"],
            copy_images=False,
            images_subdir="images",
        )
        assert config.model == "gpt-4o"
        assert config.output_formats == ["markdown"]
        assert config.copy_images is False
        assert config.images_subdir == "images"

    def test_output_formats_are_independent_instances(self) -> None:
        """Verify default_factory creates new list per instance."""
        c1 = CompileVisionReportConfig()
        c2 = CompileVisionReportConfig()
        assert c1.output_formats is not c2.output_formats


class TestCompileVisionReportOutput:
    def test_valid_output_minimal(self) -> None:
        out = CompileVisionReportOutput(
            markdown_content="# Hello\n\nWorld.",
            word_count=2,
            status="success",
            metadata={},
        )
        assert out.markdown_content == "# Hello\n\nWorld."
        assert out.html_content is None
        assert out.report_path is None
        assert out.html_path is None
        assert out.image_count == 0

    def test_valid_output_with_paths(self) -> None:
        out = CompileVisionReportOutput(
            markdown_content="# Report",
            html_content="<html>...</html>",
            report_path="/tmp/artifacts/report.md",
            html_path="/tmp/artifacts/report.html",
            image_count=5,
            word_count=100,
            status="success",
            metadata={"model": "gpt-5.4"},
        )
        assert out.report_path == "/tmp/artifacts/report.md"
        assert out.html_path == "/tmp/artifacts/report.html"
        assert out.image_count == 5

    def test_none_paths_explicit(self) -> None:
        out = CompileVisionReportOutput(
            markdown_content="# Content",
            html_content=None,
            report_path=None,
            html_path=None,
            word_count=10,
            status="success",
            metadata={},
        )
        assert out.report_path is None
        assert out.html_path is None
        assert out.html_content is None

    def test_serialization_round_trip(self) -> None:
        out = CompileVisionReportOutput(
            markdown_content="# My Report\n\nContent here.",
            html_content="<html><body>Content</body></html>",
            report_path="/artifacts/report.md",
            html_path="/artifacts/report.html",
            image_count=3,
            word_count=42,
            status="success",
            metadata={"model": "gpt-5.4", "analysis_count": 10},
        )
        data = out.model_dump()
        restored = CompileVisionReportOutput(**data)
        assert restored.markdown_content == out.markdown_content
        assert restored.report_path == out.report_path
        assert restored.image_count == out.image_count
        assert restored.word_count == out.word_count
        assert restored.metadata == out.metadata

    def test_image_count_defaults_to_zero(self) -> None:
        out = CompileVisionReportOutput(
            markdown_content="# Test",
            word_count=1,
            status="success",
            metadata={},
        )
        assert out.image_count == 0

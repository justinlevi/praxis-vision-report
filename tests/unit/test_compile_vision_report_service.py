"""Tests for compile_vision_report service methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from praxis_vision_report.tasks.compile_vision_report.service import CompileVisionReportService


@pytest.fixture()
def service() -> CompileVisionReportService:
    return CompileVisionReportService()


@pytest.fixture()
def sample_analyses() -> list[dict[str, Any]]:
    return [
        {
            "image_path": "/tmp/slide1.png",
            "analysis": "This slide introduces GPU cluster architecture",
            "key_points": ["Blackwell GPU", "NVLink 5", "72 GPUs per rack"],
            "slide_number": 1,
        },
        {
            "image_path": "/tmp/slide2.png",
            "analysis": "Performance benchmarks comparing H100 and B200",
            "key_points": ["5x speedup on training", "3x memory bandwidth"],
            "slide_number": 2,
        },
    ]


class TestBuildSynthesisPrompt:
    def test_contains_title(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026 Keynote",
            description=None,
            analyses=sample_analyses,
            metadata=None,
        )
        assert "GTC 2026 Keynote" in prompt

    def test_contains_analysis_text(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description=None,
            analyses=sample_analyses,
            metadata=None,
        )
        assert "GPU cluster architecture" in prompt
        assert "Performance benchmarks" in prompt

    def test_contains_image_refs(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description=None,
            analyses=sample_analyses,
            metadata=None,
        )
        assert "slide1.png" in prompt
        assert "slide2.png" in prompt

    def test_contains_description_when_provided(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description="A deep dive into NVIDIA Blackwell architecture",
            analyses=sample_analyses,
            metadata=None,
        )
        assert "A deep dive into NVIDIA Blackwell architecture" in prompt

    def test_no_description_section_when_none(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description=None,
            analyses=sample_analyses,
            metadata=None,
        )
        assert "Session Description" not in prompt

    def test_contains_metadata_when_provided(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description=None,
            analyses=sample_analyses,
            metadata={"speaker": "Jensen Huang", "date": "2026-03-18"},
        )
        assert "Jensen Huang" in prompt
        assert "2026-03-18" in prompt

    def test_instructs_not_slide_by_slide(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description=None,
            analyses=sample_analyses,
            metadata=None,
        )
        assert "unified" in prompt.lower() or "narrative" in prompt.lower()

    def test_instructs_image_embedding(
        self,
        service: CompileVisionReportService,
        sample_analyses: list[dict[str, Any]],
    ) -> None:
        prompt = service.build_synthesis_prompt(
            title="GTC 2026",
            description=None,
            analyses=sample_analyses,
            metadata=None,
        )
        assert "slides/" in prompt

    def test_handles_analyses_without_image_path(
        self,
        service: CompileVisionReportService,
    ) -> None:
        analyses: list[dict[str, Any]] = [
            {"analysis": "No image here", "key_points": ["point A"]}
        ]
        prompt = service.build_synthesis_prompt(
            title="Test",
            description=None,
            analyses=analyses,
            metadata=None,
        )
        assert "No image here" in prompt


class TestSynthesizeMarkdown:
    @pytest.mark.asyncio
    async def test_returns_string_content(
        self,
        service: CompileVisionReportService,
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "# GTC 2026 Keynote\n\nContent here."
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.synthesize_markdown(
            client=mock_client,
            prompt="Test prompt",
            model="gpt-5.4",
        )

        assert result == "# GTC 2026 Keynote\n\nContent here."

    @pytest.mark.asyncio
    async def test_passes_model_to_openai(
        self,
        service: CompileVisionReportService,
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Some content"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await service.synthesize_markdown(
            client=mock_client,
            prompt="My prompt",
            model="gpt-4o",
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_passes_prompt_as_user_message(
        self,
        service: CompileVisionReportService,
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await service.synthesize_markdown(
            client=mock_client,
            prompt="The actual prompt text",
            model="gpt-5.4",
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert any(
            m["role"] == "user" and "The actual prompt text" in m["content"]
            for m in messages
        )

    @pytest.mark.asyncio
    async def test_raises_on_none_content(
        self,
        service: CompileVisionReportService,
    ) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="empty content"):
            await service.synthesize_markdown(
                client=mock_client,
                prompt="prompt",
                model="gpt-5.4",
            )


class TestCopyImagesToArtifact:
    def test_copies_png_files(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        # Create fake source images
        src1 = tmp_path / "slide1.png"
        src2 = tmp_path / "slide2.png"
        src1.write_bytes(b"\x89PNG\r\n\x1a\n")
        src2.write_bytes(b"\x89PNG\r\n\x1a\n")

        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        analyses: list[dict[str, Any]] = [
            {"image_path": str(src1), "analysis": "slide 1"},
            {"image_path": str(src2), "analysis": "slide 2"},
        ]

        count = service.copy_images_to_artifact(
            analyses=analyses,
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )

        assert count == 2
        assert (artifact_dir / "slides" / "slide1.png").exists()
        assert (artifact_dir / "slides" / "slide2.png").exists()

    def test_creates_slides_subdir(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        src = tmp_path / "slide1.png"
        src.write_bytes(b"fake")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        service.copy_images_to_artifact(
            analyses=[{"image_path": str(src)}],
            artifact_dir=str(artifact_dir),
            images_subdir="images",
        )

        assert (artifact_dir / "images").is_dir()

    def test_returns_correct_count(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        images = []
        for i in range(3):
            img = tmp_path / f"slide{i}.png"
            img.write_bytes(b"fake")
            images.append(img)

        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        analyses: list[dict[str, Any]] = [{"image_path": str(img)} for img in images]

        count = service.copy_images_to_artifact(
            analyses=analyses,
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )
        assert count == 3

    def test_skips_missing_files_gracefully(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        src = tmp_path / "real.png"
        src.write_bytes(b"fake")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        analyses: list[dict[str, Any]] = [
            {"image_path": str(src)},
            {"image_path": "/nonexistent/missing.png"},
        ]

        # Should not raise — missing files are warned and skipped
        count = service.copy_images_to_artifact(
            analyses=analyses,
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )
        assert count == 1
        assert (artifact_dir / "slides" / "real.png").exists()

    def test_deduplicates_same_image_path(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        src = tmp_path / "slide1.png"
        src.write_bytes(b"fake")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        analyses: list[dict[str, Any]] = [
            {"image_path": str(src)},
            {"image_path": str(src)},  # same path twice
        ]

        count = service.copy_images_to_artifact(
            analyses=analyses,
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )
        # Only copied once
        assert count == 1

    def test_returns_zero_for_empty_analyses(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        count = service.copy_images_to_artifact(
            analyses=[],
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )
        assert count == 0

    def test_handles_analyses_without_image_path(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        analyses: list[dict[str, Any]] = [
            {"analysis": "no image here", "key_points": []},
        ]

        count = service.copy_images_to_artifact(
            analyses=analyses,
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )
        assert count == 0

    def test_uses_slide_path_fallback(
        self, service: CompileVisionReportService, tmp_path: Path
    ) -> None:
        src = tmp_path / "slide1.png"
        src.write_bytes(b"fake")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        analyses: list[dict[str, Any]] = [
            {"slide_path": str(src), "analysis": "uses slide_path key"},
        ]

        count = service.copy_images_to_artifact(
            analyses=analyses,
            artifact_dir=str(artifact_dir),
            images_subdir="slides",
        )
        assert count == 1


class TestConvertToHtml:
    def test_output_starts_with_doctype(
        self, service: CompileVisionReportService
    ) -> None:
        html = service.convert_to_html("# Hello\n\nWorld.", "Test Title")
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_title(self, service: CompileVisionReportService) -> None:
        html = service.convert_to_html("# Hello", "GTC 2026 Keynote")
        assert "GTC 2026 Keynote" in html

    def test_title_in_title_tag(self, service: CompileVisionReportService) -> None:
        html = service.convert_to_html("# Hello", "My Report")
        assert "<title>My Report</title>" in html

    def test_markdown_content_rendered(
        self, service: CompileVisionReportService
    ) -> None:
        html = service.convert_to_html("# Section Header\n\nSome paragraph text.", "Test")
        assert "<h1" in html
        assert "Section Header" in html
        assert "Some paragraph text." in html

    def test_contains_body_tag(self, service: CompileVisionReportService) -> None:
        html = service.convert_to_html("Content", "Title")
        assert "<body>" in html
        assert "</body>" in html

    def test_contains_style_block(self, service: CompileVisionReportService) -> None:
        html = service.convert_to_html("Content", "Title")
        assert "<style>" in html

    def test_escapes_special_chars_in_title(
        self, service: CompileVisionReportService
    ) -> None:
        html = service.convert_to_html("Content", "A & B <Test>")
        assert "<title>A &amp; B &lt;Test&gt;</title>" in html

    def test_renders_table(self, service: CompileVisionReportService) -> None:
        md = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        html = service.convert_to_html(md, "Table Test")
        assert "<table" in html
        assert "<td" in html


class TestSafeFilename:
    def test_basic_title(self, service: CompileVisionReportService) -> None:
        result = service.safe_filename("GTC 2026 Keynote")
        assert result == "gtc-2026-keynote"

    def test_lowercase(self, service: CompileVisionReportService) -> None:
        result = service.safe_filename("ALL CAPS TITLE")
        assert result == result.lower()

    def test_spaces_become_hyphens(self, service: CompileVisionReportService) -> None:
        result = service.safe_filename("hello world test")
        assert " " not in result
        assert result == "hello-world-test"

    def test_removes_special_chars(self, service: CompileVisionReportService) -> None:
        result = service.safe_filename("Title: With (Special) Chars!")
        assert "(" not in result
        assert ")" not in result
        assert "!" not in result
        assert ":" not in result

    def test_max_length_80(self, service: CompileVisionReportService) -> None:
        long_title = "A" * 100
        result = service.safe_filename(long_title)
        assert len(result) <= 80

    def test_collapses_consecutive_hyphens(
        self, service: CompileVisionReportService
    ) -> None:
        result = service.safe_filename("title -- with --- dashes")
        assert "--" not in result

    def test_strips_leading_trailing_hyphens(
        self, service: CompileVisionReportService
    ) -> None:
        result = service.safe_filename("!title with leading special!")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_numbers_preserved(self, service: CompileVisionReportService) -> None:
        result = service.safe_filename("GTC 2026")
        assert "2026" in result

    def test_empty_after_sanitization(self, service: CompileVisionReportService) -> None:
        result = service.safe_filename("!!!###")
        assert result == ""

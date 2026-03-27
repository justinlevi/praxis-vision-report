"""Unit tests for VisionAnalyzeBatchService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from praxis_vision_report.tasks.vision_analyze_batch.models import (
    SlideAnalysis,
    VisionAnalyzeBatchConfig,
)
from praxis_vision_report.tasks.vision_analyze_batch.service import (
    VisionAnalyzeBatchService,
)


@pytest.fixture
def service() -> VisionAnalyzeBatchService:
    return VisionAnalyzeBatchService()


# ---------------------------------------------------------------------------
# resolve_images
# ---------------------------------------------------------------------------


class TestResolveImages:
    def test_with_explicit_list(self, service: VisionAnalyzeBatchService, tiny_png: Path) -> None:
        result = service.resolve_images([str(tiny_png)], None)
        assert result == [tiny_png]

    def test_with_directory_sorted(
        self, service: VisionAnalyzeBatchService, images_dir_with_pngs: Path
    ) -> None:
        result = service.resolve_images(None, str(images_dir_with_pngs))
        names = [p.name for p in result]
        assert names == sorted(names)
        assert len(names) == 3

    def test_directory_ignores_non_images(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        img_dir = tmp_path / "mixed"
        img_dir.mkdir()
        (img_dir / "slide.png").write_bytes(b"PNG")
        (img_dir / "notes.txt").write_text("text")
        (img_dir / "data.json").write_text("{}")
        result = service.resolve_images(None, str(img_dir))
        assert len(result) == 1
        assert result[0].name == "slide.png"

    def test_raises_when_neither_provided(self, service: VisionAnalyzeBatchService) -> None:
        with pytest.raises(ValueError, match="Either image_paths or images_dir"):
            service.resolve_images(None, None)

    def test_raises_when_image_file_missing(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            service.resolve_images([str(tmp_path / "nonexistent.png")], None)

    def test_raises_when_directory_missing(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            service.resolve_images(None, str(tmp_path / "nonexistent_dir"))


# ---------------------------------------------------------------------------
# load_slide_manifest
# ---------------------------------------------------------------------------


class TestLoadSlideManifest:
    def test_returns_list_when_manifest_exists_bare_list(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        manifest = [{"filename": "a.png", "timecode_seconds": 0.0}]
        (tmp_path / "slide_manifest.json").write_text(json.dumps(manifest))
        result = service.load_slide_manifest(str(tmp_path))
        assert result is not None
        assert result[0]["filename"] == "a.png"

    def test_returns_list_when_manifest_has_slides_key(
        self, service: VisionAnalyzeBatchService, images_dir_with_manifest: Path
    ) -> None:
        result = service.load_slide_manifest(str(images_dir_with_manifest))
        assert result is not None
        assert len(result) == 3
        assert result[0]["timecode_display"] == "00:00:00"

    def test_returns_none_when_manifest_missing(
        self, service: VisionAnalyzeBatchService, images_dir_with_pngs: Path
    ) -> None:
        result = service.load_slide_manifest(str(images_dir_with_pngs))
        assert result is None

    def test_returns_none_when_images_dir_is_none(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        result = service.load_slide_manifest(None)
        assert result is None


# ---------------------------------------------------------------------------
# load_segments
# ---------------------------------------------------------------------------


class TestLoadSegments:
    def test_parses_correctly(
        self, service: VisionAnalyzeBatchService, sample_segments_json: Path
    ) -> None:
        segments = service.load_segments(str(sample_segments_json))
        assert len(segments) == 5
        assert segments[0]["text"] == "Welcome to the talk."
        assert segments[0]["start"] == 0.0

    def test_raises_on_missing_file(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            service.load_segments(str(tmp_path / "missing.segments.json"))

    def test_raises_on_invalid_json(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json [[[")
        with pytest.raises(json.JSONDecodeError):
            service.load_segments(str(bad))


# ---------------------------------------------------------------------------
# build_accumulated_text
# ---------------------------------------------------------------------------


class TestBuildAccumulatedText:
    def test_empty_when_no_segments_before_timecode(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        segments: list[dict[str, object]] = [
            {"start": 10.0, "end": 20.0, "text": "Later segment"},
        ]
        result = service.build_accumulated_text(segments, timecode_seconds=5.0)
        assert result == ""

    def test_accumulates_all_segments_up_to_timecode(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        segments: list[dict[str, object]] = [
            {"start": 0.0, "end": 10.0, "text": "First."},
            {"start": 10.0, "end": 20.0, "text": "Second."},
            {"start": 20.0, "end": 30.0, "text": "Third."},
        ]
        result = service.build_accumulated_text(segments, timecode_seconds=20.0)
        assert "First." in result
        assert "Second." in result
        assert "Third." in result

    def test_excludes_future_segments(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        segments: list[dict[str, object]] = [
            {"start": 0.0, "end": 10.0, "text": "Early."},
            {"start": 50.0, "end": 60.0, "text": "Late."},
        ]
        result = service.build_accumulated_text(segments, timecode_seconds=30.0)
        assert "Early." in result
        assert "Late." not in result

    def test_empty_segments_list(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        result = service.build_accumulated_text([], timecode_seconds=100.0)
        assert result == ""


# ---------------------------------------------------------------------------
# distribute_transcript_evenly
# ---------------------------------------------------------------------------


class TestDistributeTranscriptEvenly:
    def test_splits_into_n_chunks(self, service: VisionAnalyzeBatchService) -> None:
        transcript = " ".join([f"word{i}" for i in range(100)])
        chunks = service.distribute_transcript_evenly(transcript, 5)
        assert len(chunks) == 5
        # Reconstruct — all words should be present
        all_words = " ".join(chunks).split()
        assert len(all_words) == 100

    def test_n_equals_1(self, service: VisionAnalyzeBatchService) -> None:
        transcript = "Hello world this is one chunk."
        chunks = service.distribute_transcript_evenly(transcript, 1)
        assert chunks == [transcript]

    def test_n_equals_0_returns_empty(self, service: VisionAnalyzeBatchService) -> None:
        chunks = service.distribute_transcript_evenly("some text", 0)
        assert chunks == []

    def test_empty_transcript(self, service: VisionAnalyzeBatchService) -> None:
        chunks = service.distribute_transcript_evenly("", 3)
        assert len(chunks) == 3
        assert all(c == "" for c in chunks)

    def test_fewer_words_than_n(self, service: VisionAnalyzeBatchService) -> None:
        transcript = "only three words"
        chunks = service.distribute_transcript_evenly(transcript, 5)
        assert len(chunks) == 5
        # All words should appear across chunks
        combined = " ".join(c for c in chunks if c)
        assert "only" in combined
        assert "three" in combined
        assert "words" in combined


# ---------------------------------------------------------------------------
# encode_image_base64
# ---------------------------------------------------------------------------


class TestEncodeImageBase64:
    def test_returns_base64_string(
        self, service: VisionAnalyzeBatchService, tiny_png: Path
    ) -> None:
        import base64

        result = service.encode_image_base64(tiny_png)
        # Should be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert decoded == tiny_png.read_bytes()

    def test_matches_file_content(
        self, service: VisionAnalyzeBatchService, tmp_path: Path
    ) -> None:
        import base64

        img = tmp_path / "img.bin"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = service.encode_image_base64(img)
        assert base64.b64decode(result) == img.read_bytes()


# ---------------------------------------------------------------------------
# parse_key_points
# ---------------------------------------------------------------------------


class TestParseKeyPoints:
    def test_extracts_bullet_points(self, service: VisionAnalyzeBatchService) -> None:
        text = (
            "This slide shows transformer architecture.\n\n"
            "KEY POINTS:\n"
            "- Self-attention mechanism\n"
            "- Positional encoding\n"
            "- Multi-head attention\n"
        )
        points = service.parse_key_points(text)
        assert points == [
            "Self-attention mechanism",
            "Positional encoding",
            "Multi-head attention",
        ]

    def test_returns_empty_when_no_key_points_section(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        text = "This slide has no key points section."
        points = service.parse_key_points(text)
        assert points == []

    def test_case_insensitive_marker(self, service: VisionAnalyzeBatchService) -> None:
        text = "Analysis.\n\nkey points:\n- First point\n- Second point\n"
        points = service.parse_key_points(text)
        assert len(points) == 2

    def test_handles_empty_string(self, service: VisionAnalyzeBatchService) -> None:
        points = service.parse_key_points("")
        assert points == []

    def test_ignores_non_bullet_lines_after_key_points(
        self, service: VisionAnalyzeBatchService
    ) -> None:
        text = "KEY POINTS:\n- Valid point\nNot a bullet line\n- Another valid\n"
        points = service.parse_key_points(text)
        assert "Valid point" in points
        assert "Another valid" in points
        assert "Not a bullet line" not in points


# ---------------------------------------------------------------------------
# analyze_image (mocked OpenAI client)
# ---------------------------------------------------------------------------


class TestAnalyzeImage:
    @pytest.mark.asyncio
    async def test_returns_slide_analysis(
        self, service: VisionAnalyzeBatchService, tiny_png: Path
    ) -> None:
        analysis_text = (
            "This slide introduces transformers.\n\n"
            "KEY POINTS:\n- Attention is all you need\n- Encoder/decoder structure\n"
        )

        # Build mock OpenAI response
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 100
        mock_usage.total_tokens = 150

        mock_message = MagicMock()
        mock_message.content = analysis_text

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        config = VisionAnalyzeBatchConfig()
        result = await service.analyze_image(
            client=mock_client,
            image_path=tiny_png,
            context_text="Early transcript context.",
            system_prompt="Analyze this slide.",
            global_context="GTC 2025 keynote.",
            config=config,
            timecode="00:01:00",
            timecode_seconds=60.0,
        )

        assert isinstance(result, SlideAnalysis)
        assert result.image_path == str(tiny_png)
        assert result.timecode == "00:01:00"
        assert result.timecode_seconds == 60.0
        assert "transformers" in result.analysis
        assert "Attention is all you need" in result.key_points
        assert result.token_usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_passes_global_context_to_system_message(
        self, service: VisionAnalyzeBatchService, tiny_png: Path
    ) -> None:
        """Verify that global_context is appended to the system message."""
        captured_messages: list[object] = []

        async def capture_create(**kwargs: object) -> MagicMock:
            captured_messages.append(kwargs.get("messages"))
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30
            mock_msg = MagicMock()
            mock_msg.content = "Analysis.\nKEY POINTS:\n- Point"
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = mock_usage
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = capture_create

        config = VisionAnalyzeBatchConfig()
        await service.analyze_image(
            client=mock_client,
            image_path=tiny_png,
            context_text="",
            system_prompt="Be concise.",
            global_context="This is a keynote.",
            config=config,
        )

        assert len(captured_messages) == 1
        messages = captured_messages[0]
        assert messages is not None
        system_msg = messages[0]  # type: ignore[index]
        assert "This is a keynote." in system_msg["content"]

    @pytest.mark.asyncio
    async def test_handles_none_usage(
        self, service: VisionAnalyzeBatchService, tiny_png: Path
    ) -> None:
        """Gracefully handle None usage object (some API mocks don't set it)."""
        mock_msg = MagicMock()
        mock_msg.content = "Analysis text.\nKEY POINTS:\n- Key"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        config = VisionAnalyzeBatchConfig()
        result = await service.analyze_image(
            client=mock_client,
            image_path=tiny_png,
            context_text="",
            system_prompt="Analyze.",
            global_context=None,
            config=config,
        )
        assert result.token_usage == {}

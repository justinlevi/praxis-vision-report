"""Unit tests for vision_analyze_batch models."""

from __future__ import annotations

import pytest
from praxis_vision_report.tasks.vision_analyze_batch.models import (
    SlideAnalysis,
    VisionAnalyzeBatchConfig,
    VisionAnalyzeBatchInput,
    VisionAnalyzeBatchOutput,
)
from pydantic import ValidationError


class TestVisionAnalyzeBatchInput:
    def test_valid_with_image_paths(self) -> None:
        inp = VisionAnalyzeBatchInput(
            image_paths=["/some/image.png"],
            system_prompt="Analyze this slide.",
        )
        assert inp.image_paths == ["/some/image.png"]
        assert inp.images_dir is None

    def test_valid_with_images_dir(self) -> None:
        inp = VisionAnalyzeBatchInput(
            images_dir="/some/dir",
            system_prompt="Analyze this slide.",
        )
        assert inp.images_dir == "/some/dir"
        assert inp.image_paths is None

    def test_raises_if_neither_image_source_provided(self) -> None:
        with pytest.raises(ValidationError, match="image_paths or images_dir"):
            VisionAnalyzeBatchInput(system_prompt="Analyze this.")

    def test_valid_with_both_sources(self) -> None:
        # Providing both is allowed — task will prefer image_paths
        inp = VisionAnalyzeBatchInput(
            image_paths=["/a.png"],
            images_dir="/some/dir",
            system_prompt="Analyze.",
        )
        assert inp.image_paths is not None
        assert inp.images_dir is not None

    def test_optional_transcript_fields(self) -> None:
        inp = VisionAnalyzeBatchInput(
            images_dir="/dir",
            system_prompt="Analyze.",
            segments_path="/path/to/file.segments.json",
            transcript="Some text here.",
            global_context="This is a GTC talk.",
        )
        assert inp.segments_path == "/path/to/file.segments.json"
        assert inp.transcript == "Some text here."
        assert inp.global_context == "This is a GTC talk."

    def test_system_prompt_required(self) -> None:
        with pytest.raises(ValidationError):
            VisionAnalyzeBatchInput(images_dir="/dir")  # type: ignore[call-arg]

    def test_system_prompt_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            VisionAnalyzeBatchInput(images_dir="/dir", system_prompt="")


class TestVisionAnalyzeBatchConfig:
    def test_defaults(self) -> None:
        config = VisionAnalyzeBatchConfig()
        assert config.model == "gpt-5.4"
        assert config.detail == "high"
        assert config.max_tokens == 2048
        assert config.concurrency == 5

    def test_custom_values(self) -> None:
        config = VisionAnalyzeBatchConfig(
            model="gpt-4o",
            detail="low",
            max_tokens=1024,
            concurrency=3,
        )
        assert config.model == "gpt-4o"
        assert config.detail == "low"
        assert config.max_tokens == 1024
        assert config.concurrency == 3

    def test_concurrency_bounds(self) -> None:
        with pytest.raises(ValidationError):
            VisionAnalyzeBatchConfig(concurrency=0)
        with pytest.raises(ValidationError):
            VisionAnalyzeBatchConfig(concurrency=21)

    def test_max_tokens_bounds(self) -> None:
        with pytest.raises(ValidationError):
            VisionAnalyzeBatchConfig(max_tokens=100)  # below 256
        with pytest.raises(ValidationError):
            VisionAnalyzeBatchConfig(max_tokens=99999)  # above 16384


class TestSlideAnalysis:
    def test_construction_minimal(self) -> None:
        slide = SlideAnalysis(
            image_path="/some/image.png",
            analysis="This slide shows a neural network architecture.",
        )
        assert slide.image_path == "/some/image.png"
        assert slide.timecode is None
        assert slide.timecode_seconds is None
        assert slide.key_points == []
        assert slide.token_usage == {}

    def test_construction_full(self) -> None:
        slide = SlideAnalysis(
            image_path="/some/image.png",
            timecode="00:03:42",
            timecode_seconds=222.0,
            analysis="Deep analysis text.",
            key_points=["Point A", "Point B"],
            token_usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        )
        assert slide.timecode == "00:03:42"
        assert slide.timecode_seconds == 222.0
        assert len(slide.key_points) == 2
        assert slide.token_usage["total_tokens"] == 300


class TestVisionAnalyzeBatchOutput:
    def test_empty_analyses(self) -> None:
        out = VisionAnalyzeBatchOutput(
            analyses=[],
            total_slides=0,
            total_tokens=0,
            model_used="gpt-5.4",
        )
        assert out.analyses == []
        assert out.total_slides == 0
        assert out.total_tokens == 0
        assert out.status == "success"
        assert out.metadata == {}

    def test_with_analyses(self) -> None:
        slide = SlideAnalysis(image_path="/img.png", analysis="Some analysis.")
        out = VisionAnalyzeBatchOutput(
            analyses=[slide],
            total_slides=1,
            total_tokens=500,
            model_used="gpt-4o",
            status="success",
            metadata={"concurrency": 5},
        )
        assert len(out.analyses) == 1
        assert out.total_tokens == 500
        assert out.metadata["concurrency"] == 5

    def test_serialization_round_trip(self) -> None:
        slide = SlideAnalysis(
            image_path="/img.png",
            timecode="00:01:00",
            timecode_seconds=60.0,
            analysis="Analysis text.",
            key_points=["Key point one"],
            token_usage={"total_tokens": 150},
        )
        out = VisionAnalyzeBatchOutput(
            analyses=[slide],
            total_slides=1,
            total_tokens=150,
            model_used="gpt-5.4",
        )
        dumped = out.model_dump()
        restored = VisionAnalyzeBatchOutput.model_validate(dumped)
        assert restored.analyses[0].image_path == "/img.png"
        assert restored.analyses[0].timecode == "00:01:00"
        assert restored.total_tokens == 150

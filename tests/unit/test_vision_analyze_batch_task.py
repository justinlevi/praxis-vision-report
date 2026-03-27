"""Unit tests for vision_analyze_batch task run() function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from praxis_vision_report.tasks.vision_analyze_batch.models import (
    SlideAnalysis,
    VisionAnalyzeBatchConfig,
    VisionAnalyzeBatchInput,
    VisionAnalyzeBatchOutput,
)
from praxis_vision_report.tasks.vision_analyze_batch.task import run


def _make_slide_analysis(image_path: str, tokens: int = 100) -> SlideAnalysis:
    return SlideAnalysis(
        image_path=image_path,
        timecode=None,
        timecode_seconds=None,
        analysis="Mock analysis text.\nKEY POINTS:\n- Mock point",
        key_points=["Mock point"],
        token_usage={"total_tokens": tokens},
    )


class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_returns_output_with_analyses(
        self, images_dir_with_pngs: Path
    ) -> None:
        """Happy path: images_dir with 3 PNGs produces 3 analyses."""
        inp = VisionAnalyzeBatchInput(
            images_dir=str(images_dir_with_pngs),
            system_prompt="Analyze each slide.",
        )
        config = VisionAnalyzeBatchConfig()

        fake_analyses = [
            _make_slide_analysis(str(images_dir_with_pngs / f"slide_{i:04d}.png"), tokens=100)
            for i in range(1, 4)
        ]

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.VisionAnalyzeBatchService"
        ) as MockService, patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            svc = MockService.return_value
            svc.resolve_images.return_value = [
                images_dir_with_pngs / f"slide_{i:04d}.png" for i in range(1, 4)
            ]
            svc.load_slide_manifest.return_value = None
            svc.load_segments.return_value = []
            svc.distribute_transcript_evenly.return_value = []
            svc.build_accumulated_text.return_value = ""
            svc.analyze_image = AsyncMock(side_effect=fake_analyses)

            result = await run(inp, config)

        assert isinstance(result, VisionAnalyzeBatchOutput)
        assert result.status == "success"
        assert result.total_slides == 3
        assert len(result.analyses) == 3

    @pytest.mark.asyncio
    async def test_total_tokens_summed(self, images_dir_with_pngs: Path) -> None:
        """total_tokens is the sum across all SlideAnalysis.token_usage."""
        inp = VisionAnalyzeBatchInput(
            images_dir=str(images_dir_with_pngs),
            system_prompt="Analyze.",
        )
        config = VisionAnalyzeBatchConfig()

        fake_analyses = [
            _make_slide_analysis("img.png", tokens=200),
            _make_slide_analysis("img2.png", tokens=300),
            _make_slide_analysis("img3.png", tokens=100),
        ]

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.VisionAnalyzeBatchService"
        ) as MockService, patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            svc = MockService.return_value
            svc.resolve_images.return_value = [
                images_dir_with_pngs / f"slide_{i:04d}.png" for i in range(1, 4)
            ]
            svc.load_slide_manifest.return_value = None
            svc.load_segments.return_value = []
            svc.distribute_transcript_evenly.return_value = []
            svc.build_accumulated_text.return_value = ""
            svc.analyze_image = AsyncMock(side_effect=fake_analyses)

            result = await run(inp, config)

        assert result.total_tokens == 600

    @pytest.mark.asyncio
    async def test_model_used_from_config(self, images_dir_with_pngs: Path) -> None:
        inp = VisionAnalyzeBatchInput(
            images_dir=str(images_dir_with_pngs),
            system_prompt="Analyze.",
        )
        config = VisionAnalyzeBatchConfig(model="gpt-4o")

        fake_analyses = [_make_slide_analysis("img.png")]

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.VisionAnalyzeBatchService"
        ) as MockService, patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            svc = MockService.return_value
            svc.resolve_images.return_value = [images_dir_with_pngs / "slide_0001.png"]
            svc.load_slide_manifest.return_value = None
            svc.load_segments.return_value = []
            svc.distribute_transcript_evenly.return_value = []
            svc.build_accumulated_text.return_value = ""
            svc.analyze_image = AsyncMock(side_effect=fake_analyses)

            result = await run(inp, config)

        assert result.model_used == "gpt-4o"

    @pytest.mark.asyncio
    async def test_with_segments_path(
        self, images_dir_with_manifest: Path, sample_segments_json: Path
    ) -> None:
        """When segments_path is provided, load_segments is called."""
        inp = VisionAnalyzeBatchInput(
            images_dir=str(images_dir_with_manifest),
            system_prompt="Analyze.",
            segments_path=str(sample_segments_json),
        )
        config = VisionAnalyzeBatchConfig()

        fake_segments = [
            {"start": 0.0, "end": 30.0, "text": "Hello."},
        ]
        fake_analyses = [
            _make_slide_analysis(str(images_dir_with_manifest / f"slide_{i:04d}.png"))
            for i in range(1, 4)
        ]

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.VisionAnalyzeBatchService"
        ) as MockService, patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            svc = MockService.return_value
            svc.resolve_images.return_value = [
                images_dir_with_manifest / f"slide_{i:04d}.png" for i in range(1, 4)
            ]
            svc.load_slide_manifest.return_value = [
                {"filename": f"slide_{i:04d}.png", "timecode_seconds": float(i * 60), "timecode_display": f"00:0{i}:00"}
                for i in range(1, 4)
            ]
            svc.load_segments.return_value = fake_segments
            svc.build_accumulated_text.return_value = "Hello."
            svc.analyze_image = AsyncMock(side_effect=fake_analyses)

            result = await run(inp, config)
            # load_segments should have been called
            svc.load_segments.assert_called_once_with(str(sample_segments_json))

        assert result.total_slides == 3

    @pytest.mark.asyncio
    async def test_metadata_populated(self, images_dir_with_pngs: Path) -> None:
        inp = VisionAnalyzeBatchInput(
            images_dir=str(images_dir_with_pngs),
            system_prompt="Analyze.",
        )
        config = VisionAnalyzeBatchConfig(concurrency=3, detail="low")

        fake_analyses = [_make_slide_analysis("img.png")]

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.VisionAnalyzeBatchService"
        ) as MockService, patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            svc = MockService.return_value
            svc.resolve_images.return_value = [images_dir_with_pngs / "slide_0001.png"]
            svc.load_slide_manifest.return_value = None
            svc.load_segments.return_value = []
            svc.distribute_transcript_evenly.return_value = []
            svc.build_accumulated_text.return_value = ""
            svc.analyze_image = AsyncMock(side_effect=fake_analyses)

            result = await run(inp, config)

        assert result.metadata["concurrency"] == 3
        assert result.metadata["detail"] == "low"

    @pytest.mark.asyncio
    async def test_all_analyses_run_concurrently(self, tmp_path: Path) -> None:
        """All analyses are submitted — asyncio.gather runs them all."""
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        n = 8
        for i in range(n):
            (img_dir / f"img_{i:04d}.png").write_bytes(b"PNG")

        inp = VisionAnalyzeBatchInput(
            images_dir=str(img_dir),
            system_prompt="Analyze.",
        )
        config = VisionAnalyzeBatchConfig(concurrency=4)

        fake_analyses = [_make_slide_analysis(f"img_{i}.png", tokens=50) for i in range(n)]

        call_count = 0

        async def count_calls(**_: object) -> SlideAnalysis:
            nonlocal call_count
            call_count += 1
            return fake_analyses[call_count - 1]

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.VisionAnalyzeBatchService"
        ) as MockService, patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            svc = MockService.return_value
            svc.resolve_images.return_value = [img_dir / f"img_{i:04d}.png" for i in range(n)]
            svc.load_slide_manifest.return_value = None
            svc.load_segments.return_value = []
            svc.distribute_transcript_evenly.return_value = []
            svc.build_accumulated_text.return_value = ""
            svc.analyze_image = AsyncMock(side_effect=fake_analyses)

            result = await run(inp, config)

        assert result.total_slides == n
        assert len(result.analyses) == n


class TestRunErrorCases:
    @pytest.mark.asyncio
    async def test_resolve_images_error_propagates(self) -> None:
        """ValueError from resolve_images bubbles up."""
        inp = VisionAnalyzeBatchInput(
            images_dir="/nonexistent_dir",
            system_prompt="Analyze.",
        )
        config = VisionAnalyzeBatchConfig()

        with patch(
            "praxis_vision_report.tasks.vision_analyze_batch.task.AsyncOpenAI"
        ):
            with pytest.raises((FileNotFoundError, ValueError)):
                await run(inp, config)

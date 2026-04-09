"""Vision batch analysis task."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from praxis_vision_report.tasks.vision_analyze_batch.models import (
    SlideAnalysis,
    VisionAnalyzeBatchConfig,
    VisionAnalyzeBatchInput,
    VisionAnalyzeBatchOutput,
)
from praxis_vision_report.tasks.vision_analyze_batch.service import (
    VisionAnalyzeBatchService,
)

logger = logging.getLogger(__name__)


async def run(
    input: VisionAnalyzeBatchInput,
    config: VisionAnalyzeBatchConfig,
) -> VisionAnalyzeBatchOutput:
    """Execute batch vision analysis.

    Resolves images, aligns transcript context to each slide,
    and runs concurrent vision API calls.
    """

    logger.info(
        "VisionAnalyzeBatch: Starting",
        extra={
            "images_dir": input.images_dir,
            "image_paths_count": len(input.image_paths) if input.image_paths else 0,
            "has_segments": input.segments_path is not None,
            "has_transcript": input.transcript is not None,
            "model": config.model,
        },
    )

    service = VisionAnalyzeBatchService()

    # 1. Resolve images
    images = service.resolve_images(input.image_paths, input.images_dir)
    n = len(images)

    logger.info(f"VisionAnalyzeBatch: Resolved {n} images")

    # 2. Load slide manifest (for timecode info)
    manifest = service.load_slide_manifest(input.images_dir)

    # 2a. Apply pre-analysis filters (require manifest with timecode data)
    if manifest:
        if config.filter_stage_shots and input.images_dir:
            manifest = service.filter_stage_shots(manifest, Path(input.images_dir))
        if config.min_slide_interval_seconds > 0.0:
            manifest = service.filter_by_min_interval(manifest, config.min_slide_interval_seconds)

        # Re-filter images to only those still in the manifest
        allowed_filenames = {str(e.get("filename", "")) for e in manifest}
        images = [img for img in images if img.name in allowed_filenames]
        n = len(images)
        logger.info(f"VisionAnalyzeBatch: After filtering — {n} images remaining")

    manifest_by_filename: dict[str, dict[str, object]] = {}
    if manifest:
        for entry in manifest:
            filename = str(entry.get("filename", ""))
            manifest_by_filename[filename] = entry

    # 3. Load segments if provided
    segments: list[dict[str, object]] = []
    if input.segments_path:
        segments = service.load_segments(input.segments_path)

    # 4. Build per-image context text
    distributed: list[str] = []
    if not segments and input.transcript:
        distributed = service.distribute_transcript_evenly(input.transcript, n)

    # 5. Build per-image work items
    #    Each: (image_path, context_text, timecode, timecode_seconds)
    work_items: list[tuple[Path, str, str | None, float | None]] = []
    for image in images:
        # Determine timecode from manifest entry
        entry = manifest_by_filename.get(image.name, {})
        timecode_seconds: float | None = None
        timecode: str | None = None

        if entry:
            raw_ts = entry.get("timecode_seconds")
            if raw_ts is not None:
                timecode_seconds = float(raw_ts)  # type: ignore[arg-type]
            raw_td = entry.get("timecode_display")
            if raw_td is not None:
                timecode = str(raw_td)

        # Build context text
        if segments and timecode_seconds is not None:
            context_text = service.build_accumulated_text(segments, timecode_seconds)
        elif distributed:
            idx = len(work_items)
            context_text = distributed[idx] if idx < len(distributed) else ""
        else:
            context_text = ""

        work_items.append((image, context_text, timecode, timecode_seconds))

    # 6. Run analyses concurrently with semaphore
    semaphore = asyncio.Semaphore(config.concurrency)

    batch_size = config.batch_size
    batches: list[list[tuple[Path, str, str | None, float | None]]] = [
        work_items[i : i + batch_size] for i in range(0, len(work_items), batch_size)
    ]

    async def analyze_batch(
        batch: list[tuple[Path, str, str | None, float | None]],
    ) -> list[SlideAnalysis]:
        async with semaphore:
            if len(batch) == 1:
                # Single-image path — use the original per-image method
                item = batch[0]
                image_path, context_text, timecode, timecode_seconds = item
                result = await service.analyze_image(
                    image_path=image_path,
                    context_text=context_text,
                    system_prompt=input.system_prompt,
                    global_context=input.global_context,
                    config=config,
                    timecode=timecode,
                    timecode_seconds=timecode_seconds,
                )
                return [result]
            return await service.analyze_image_batch(
                items=batch,
                system_prompt=input.system_prompt,
                global_context=input.global_context,
                config=config,
            )

    batch_results = await asyncio.gather(*[analyze_batch(b) for b in batches])
    analyses = [slide for batch in batch_results for slide in batch]

    total_tokens = sum(a.token_usage.get("total_tokens", 0) for a in analyses)

    logger.info(f"VisionAnalyzeBatch: Completed — {n} slides, {total_tokens} total tokens")

    return VisionAnalyzeBatchOutput(
        analyses=list(analyses),
        total_slides=n,
        total_tokens=total_tokens,
        model_used=config.model,
        status="success",
        metadata={
            "concurrency": config.concurrency,
            "batch_size": config.batch_size,
            "batches": len(batches),
            "had_manifest": bool(manifest),
            "had_segments": bool(segments),
            "had_transcript": bool(input.transcript),
        },
    )

# praxis-vision-report

Vision analysis and report generation tasks for [Praxis](https://github.com/justinlevi/praxis) pipelines.

Analyzes batches of images (e.g. extracted video slides) with transcript context using vision LLMs, then synthesizes the per-slide analyses into cohesive blog-post-style reports in Markdown and HTML.

## What It Does

1. **Extract slides** — delegates to the `video_slide_extract` task (from `praxis-audio-video-plugins`) to pull unique frames from an MP4.
2. **Analyze slides** — the `vision_analyze_batch` task sends each slide image to a vision LLM alongside the nearest transcript segment, producing structured per-slide analyses.
3. **Compile report** — the `compile_vision_report` task synthesizes all slide analyses into a cohesive narrative report, outputting both Markdown and HTML with embedded slide images.

## Tasks

| Task | Description |
|------|-------------|
| `vision_analyze_batch` | Analyze a directory of images with optional transcript context using a vision LLM. Produces structured per-image analyses. |
| `compile_vision_report` | Compile a list of per-slide analyses into a formatted blog-post-style report (Markdown + HTML). |

## Pipeline

### `gtc_session_writeup`

End-to-end pipeline for generating a comprehensive writeup from a GTC conference session video.

**Steps:** `ExtractSlides` → `AnalyzeSlides` → `CompileReport`

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `video_path` | Yes | Absolute path to the session MP4 file |
| `session_title` | Yes | Session title (e.g. `"Agentic AI 101"`) |
| `segments_path` | Yes | Absolute path to the `.segments.json` transcript file |
| `session_description` | No | Optional session abstract for additional context |

## Installation

```bash
pip install -e .
```

Or with `uv`:

```bash
uv pip install -e .
```

This package registers itself with Praxis via the `praxis.extensions` entry point. Once installed, the tasks and pipeline are available automatically.

## Usage

### Run the pipeline via Praxis CLI

```bash
praxis pipeline run gtc_session_writeup \
  --param video_path=/path/to/session.mp4 \
  --param session_title="Agentic AI 101" \
  --param segments_path=/path/to/session.segments.json \
  --param session_description="An introduction to building agentic AI systems at scale."
```

### Use tasks individually

```bash
# Analyze a directory of slide images with a transcript
praxis task run vision_analyze_batch \
  --param images_dir=/path/to/slides \
  --param segments_path=/path/to/session.segments.json

# Compile a report from existing analyses
praxis task run compile_vision_report \
  --param analyses_path=/path/to/analyses.json \
  --param title="My Session Title"
```

## Configuration

Key config options for the `AnalyzeSlides` step (set via pipeline YAML `config:` block or environment):

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `gpt-5.4` | Vision LLM model to use |
| `detail` | `high` | Vision detail level (`low` or `high`) |
| `max_tokens` | `2048` | Max tokens per slide analysis |
| `concurrency` | `5` | Number of parallel vision API calls |

Key config options for the `CompileReport` step:

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `gpt-5.4` | LLM model for report synthesis |
| `output_formats` | `["markdown", "html"]` | Output formats to generate |
| `copy_images` | `true` | Copy slide images into the report output directory |
| `images_subdir` | `slides` | Subdirectory name for copied images |

## Requirements

- Python 3.11+
- [Praxis](https://github.com/justinlevi/praxis) >= 0.1.0
- `OPENAI_API_KEY` environment variable set

```bash
export OPENAI_API_KEY=sk-...
```

The `video_slide_extract` task (used in the `gtc_session_writeup` pipeline) requires the `praxis-audio-video-plugins` package to be installed.

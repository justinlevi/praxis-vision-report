"""Praxis Vision Report Extension.

Vision analysis and report generation tasks for Praxis pipelines.
Analyzes batches of images with transcript context using vision LLMs,
then synthesizes results into cohesive blog-post-style reports.
"""

__version__ = "0.1.0"


def register() -> dict:
    """Register the vision report extension with Praxis."""
    return {
        "name": "Vision Report Extension",
        "version": __version__,
        "description": "Vision analysis and report generation for Praxis pipelines",
        "tasks": {
            "vision_analyze_batch": "praxis_vision_report.tasks.vision_analyze_batch",
            "compile_vision_report": "praxis_vision_report.tasks.compile_vision_report",
        },
        "pipelines": {
            "gtc_session_writeup": "praxis_vision_report.pipelines.gtc_session_writeup",
        },
    }

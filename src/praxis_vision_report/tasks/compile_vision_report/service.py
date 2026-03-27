"""Service layer for compile_vision_report task.

Handles LLM synthesis, image copying, HTML conversion, and filename utilities.
Each internal method is independently testable but not exposed as a separate task.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)


class CompileVisionReportService:
    """Encapsulates report compilation logic."""

    def build_synthesis_prompt(
        self,
        title: str,
        description: str | None,
        analyses: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
    ) -> str:
        """Build the LLM prompt for synthesizing a blog-post article from slide analyses.

        Instructs the model to produce a unified narrative article that flows through
        all key points, embeds images, and does NOT write slide-by-slide.
        """
        lines: list[str] = []
        lines.append(
            "You are a technical writer creating a detailed blog-post article from a conference "
            "session. You have been given structured analyses of each slide from the presentation."
        )
        lines.append("")
        lines.append(f"## Session Title\n{title}")
        lines.append("")

        if metadata:
            lines.append("## Session Metadata")
            for key, value in metadata.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")

        if description:
            lines.append(f"## Session Description\n{description}")
            lines.append("")

        lines.append("## Slide Analyses")
        lines.append(
            "Below are the structured analyses for each slide. "
            "Use ALL of this content in your article."
        )
        lines.append("")

        for i, analysis in enumerate(analyses, start=1):
            lines.append(f"### Slide {i}")
            image_path = analysis.get("image_path") or analysis.get("slide_path", "")
            if image_path:
                filename = Path(str(image_path)).name
                lines.append(f"- **Image**: `{filename}`")
            for key, value in analysis.items():
                if key in ("image_path", "slide_path"):
                    continue
                if isinstance(value, list):
                    lines.append(f"- **{key.replace('_', ' ').title()}**:")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")

        lines.append("## Your Task")
        lines.append(
            "Write a complete, detailed blog-post article about this session. Requirements:"
        )
        lines.append(
            "1. **Narrative flow**: Write as a unified article with proper sections and headers "
            "(use ## for major sections, ### for subsections). Do NOT write slide-by-slide."
        )
        lines.append(
            "2. **Embed images**: For each slide that has an image, embed it inline using "
            "Markdown syntax: `![brief description](slides/filename.png)`. "
            "Place each image near the text that discusses its content."
        )
        lines.append(
            "3. **Complete coverage**: Cover every key point from every slide analysis. "
            "Do not omit or summarize — expand into detailed prose."
        )
        if description:
            lines.append(
                "4. **Intro**: Begin with a brief introduction that incorporates the session "
                "description/abstract above."
            )
        lines.append(
            "5. **Professional tone**: Write for a technical audience. Use concrete details, "
            "explain concepts, and make the article self-contained."
        )
        lines.append(
            "6. **Length**: Aim for thoroughness over brevity. Cover every topic in depth."
        )
        lines.append("")
        lines.append("Begin writing the article now:")

        return "\n".join(lines)

    async def synthesize_markdown(
        self,
        client: openai.AsyncOpenAI,
        prompt: str,
        model: str,
    ) -> str:
        """Call OpenAI to synthesize the full markdown blog post.

        Args:
            client: AsyncOpenAI client instance
            prompt: The synthesis prompt built by build_synthesis_prompt
            model: Model identifier string (e.g. "gpt-5.4")

        Returns:
            Full markdown content as a string.
        """
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned empty content for synthesis prompt")
        return content

    def copy_images_to_artifact(
        self,
        analyses: list[dict[str, Any]],
        artifact_dir: str,
        images_subdir: str,
    ) -> int:
        """Copy all unique image files referenced in analyses to artifact_dir/images_subdir/.

        Args:
            analyses: List of analysis dicts, each may have 'image_path' or 'slide_path'
            artifact_dir: Absolute path to the artifact output directory
            images_subdir: Subdirectory name for images (e.g. "slides")

        Returns:
            Number of images successfully copied.
        """
        dest_dir = Path(artifact_dir) / images_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        copied = 0

        for analysis in analyses:
            image_path = analysis.get("image_path") or analysis.get("slide_path")
            if not image_path:
                continue

            src = Path(str(image_path))
            if str(src) in seen:
                continue
            seen.add(str(src))

            if not src.exists():
                logger.warning(
                    "Image file not found, skipping: %s",
                    src,
                )
                continue

            dest = dest_dir / src.name
            shutil.copy2(str(src), str(dest))
            copied += 1
            logger.debug("Copied image: %s -> %s", src, dest)

        logger.info(
            "Copied %d images to %s/%s",
            copied,
            artifact_dir,
            images_subdir,
        )
        return copied

    def convert_to_html(self, markdown_content: str, title: str) -> str:
        """Convert markdown content to a self-contained HTML document.

        Uses markdown2 with extras for tables, fenced code blocks, and more.

        Args:
            markdown_content: Markdown string to convert
            title: Page title for the <title> tag

        Returns:
            Complete HTML document as a string.
        """
        import markdown2  # type: ignore[import-untyped]

        body_html: str = markdown2.markdown(
            markdown_content,
            extras=[
                "fenced-code-blocks",
                "tables",
                "header-ids",
                "strike",
                "task_list",
            ],
        )

        css = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
            line-height: 1.7;
            color: #1a1a1a;
            max-width: 860px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            background: #fff;
        }
        h1, h2, h3, h4 {
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; border-bottom: 1px solid #e5e5e5; padding-bottom: 0.3rem; }
        h3 { font-size: 1.2rem; }
        p { margin: 0.75rem 0; }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5rem auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }
        code {
            background: #f4f4f4;
            padding: 0.15em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }
        pre code {
            background: none;
            padding: 0;
        }
        pre {
            background: #f4f4f4;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 0.5rem 0.75rem;
            text-align: left;
        }
        th { background: #f4f4f4; }
        blockquote {
            border-left: 3px solid #ccc;
            margin: 1rem 0;
            padding-left: 1rem;
            color: #555;
        }
        """

        escaped_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escaped_title}</title>
<style>
{css}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    def safe_filename(self, title: str) -> str:
        """Convert a title to a safe filename.

        Converts to lowercase, replaces spaces with hyphens, removes special
        characters, and limits to 80 characters.

        Args:
            title: Human-readable title string

        Returns:
            Safe filename string (without extension).
        """
        name = title.lower()
        name = name.replace(" ", "-")
        # Remove any character that isn't alphanumeric, hyphen, or underscore
        name = re.sub(r"[^a-z0-9\-_]", "", name)
        # Collapse consecutive hyphens
        name = re.sub(r"-{2,}", "-", name)
        # Strip leading/trailing hyphens
        name = name.strip("-")
        # Limit length
        return name[:80]

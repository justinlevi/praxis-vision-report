"""Service layer for compile_vision_report task.

Handles LLM synthesis via Claude Code CLI, image copying, HTML conversion,
and filename utilities. Each internal method is independently testable but
not exposed as a separate task.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

from praxis.task.core.transform.claude_inference.service import ClaudeInferenceService

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
            "You are a technology journalist and analyst writing for a technical publication. "
            "Write a detailed analytical article about this session. Your voice is authoritative, "
            "declarative, and third-party — you report on content as an informed analyst, not as "
            "a conference attendee."
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
            "1. **Voice**: Write as a technology journalist covering a significant technical "
            "development — authoritative, declarative, editorial. The voice is that of an informed "
            "analyst reporting findings, not a conference attendee describing their experience.\n\n"
            "   NEVER use first-person attendee language. Do not write 'I', 'me', 'my' in the "
            "author-as-participant sense. Forbidden constructions include: 'I walked away with...', "
            "'what struck me was...', 'I came to understand...', 'I kept thinking about...', "
            "'this session changed how I think...', 'I found this surprising...', "
            "'as I listened to...', 'my takeaway was...'\n\n"
            "   Write like technical trade journalism — MIT Technology Review, IEEE Spectrum, "
            "Ars Technica. State findings directly. Use 'you' when addressing the reader "
            "editorially. Attribute claims to organizations and people, not to the author's "
            "experience. NEVER write 'the talk argues', 'the slide shows', 'the screenshot "
            "depicts', 'the speaker said', or any phrase referencing the source artifact.\n\n"
            "   CORRECT: 'The session makes a compelling case that AI infrastructure is best "
            "treated as a factory operation, not a managed service.'\n"
            "   CORRECT: 'Mistral's decision to use only upstream CNCF components is not "
            "philosophical preference — it is a specific competitive positioning against "
            "hyperscaler lock-in.'\n"
            "   CORRECT: 'If you are building GPU infrastructure at scale, the cluster crash "
            "story is the most practically useful lesson in this session.'\n\n"
            "   WRONG: 'I walked away convinced that AI infrastructure needs to be treated "
            "like a factory.'\n"
            "   WRONG: 'What struck me was how open Mistral was about their architectural "
            "choices.'\n"
            "   WRONG: 'This got me thinking about how our own infrastructure decisions compare.'"
        )
        lines.append(
            "2. **Images as evidence**: When you embed an image, the surrounding prose must make "
            "a technical point that the image supports. The image illustrates your argument -- it "
            "is never the subject of the sentence. "
            "CORRECT: 'Cursor's agents run in full cloud VMs with filesystem access and sandboxed "
            "network policies. [image below]' "
            "WRONG: 'The screenshot shows a remote desktop environment used by the agent.'"
        )
        lines.append(
            "3. **Narrative flow**: Write as a unified article with proper sections and headers "
            "(use ## for major sections, ### for subsections). Do NOT write slide-by-slide."
        )
        lines.append(
            "4. **Embed images**: For each slide that has an image, embed it inline using "
            "Markdown syntax: `![brief description](slides/filename.png)`. "
            "Place each image near the text that discusses its content."
        )
        lines.append(
            "5. **Complete coverage**: Cover every key point from every slide analysis. "
            "Do not omit or summarize -- expand into detailed prose."
        )
        if description:
            lines.append(
                "6. **Intro**: Begin with a brief introduction that incorporates the session "
                "description/abstract above."
            )
        lines.append(
            "7. **Professional tone**: Write for a technical audience. Use concrete details, "
            "explain concepts, and make the article self-contained."
        )
        lines.append(
            "8. **Length**: Aim for thoroughness over brevity. Cover every topic in depth."
        )
        lines.append("")
        lines.append("Begin writing the article now:")

        return "\n".join(lines)

    async def synthesize_markdown(
        self,
        prompt: str,
        model: str,
        slide_count: int = 0,
    ) -> str:
        """Call Claude Code CLI to synthesize the full markdown blog post.

        Args:
            prompt: The synthesis prompt built by build_synthesis_prompt
            model: Claude model identifier string
            slide_count: Number of slides (used to scale timeout)

        Returns:
            Full markdown content as a string.
        """
        claude_svc = ClaudeInferenceService()

        full_prompt = claude_svc.build_full_prompt(
            prompt=prompt,
            system_prompt=None,
            image_paths=None,
            file_paths=None,
            context=None,
        )

        raw, _attempts = await claude_svc.execute_with_retry(
            full_prompt=full_prompt,
            model=model,
            output_format="text",
            timeout=max(300, 30 * slide_count),
            max_retries=2,
            retry_delay=2.0,
        )

        if not raw:
            raise ValueError("Claude returned empty content for synthesis prompt")
        return raw

    def inject_subtitle(self, markdown: str, subtitle: str) -> str:
        """Insert an italic subtitle line after the first H1 heading."""
        import re

        return re.sub(
            r"^(# .+)$",
            rf"\1\n\n*{subtitle}*",
            markdown,
            count=1,
            flags=re.MULTILINE,
        )

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
            width: 100%;
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5em 0;
            border-radius: 4px;
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

    def build_chunk_prompt(
        self,
        analyses_chunk: list[dict[str, Any]],
        chunk_idx: int,
        total_chunks: int,
        title: str,
        description: str | None,
    ) -> str:
        """Build the prompt for a single map-phase chunk."""
        lines: list[str] = []
        lines.append(
            f"You are a technology journalist and analyst writing section {chunk_idx + 1} of "
            f'{total_chunks} of an analytical article about "{title}". Your voice is '
            f"authoritative, declarative, and third-party — report findings as an informed "
            f"analyst, not as a conference attendee."
        )
        lines.append("")

        if description:
            lines.append(f"## Session Description\n{description}")
            lines.append("")

        lines.append("## Slide Analyses for This Section")
        lines.append("")

        for analysis in analyses_chunk:
            image_path = analysis.get("image_path") or analysis.get("slide_path", "")
            filename = Path(str(image_path)).name if image_path else ""
            timecode = analysis.get("timecode", "")
            analysis_text = str(analysis.get("analysis", ""))
            key_points = analysis.get("key_points", [])

            if filename:
                lines.append(f"- **Image**: `{filename}`")
            if timecode:
                lines.append(f"- **Timecode**: {timecode}")
            if analysis_text:
                lines.append(f"- **Analysis**: {analysis_text}")
            if key_points:
                lines.append("- **Key Points**:")
                for point in key_points:
                    lines.append(f"  - {point}")
            lines.append("")

        lines.append("## Instructions")
        lines.append(
            "- NEVER use first-person attendee language: 'I walked away...', 'what struck me...', "
            "'I came to understand...', 'I kept thinking about...', 'my takeaway was...', "
            "'I found this surprising...'. Write in the authoritative, declarative voice of "
            "technical journalism. Attribute claims to organizations and people, not to the author."
        )
        lines.append(
            "- NEVER write 'the talk argues', 'the slide shows', 'the screenshot depicts', "
            "'the speaker said', or any phrase referencing the talk or slides as artifacts. "
            "State all technical claims directly. Credit the company or person, not the artifact."
        )
        lines.append(
            "- When embedding an image, write the technical argument first -- the image reinforces "
            "your point, it is never the subject of the sentence. "
            "CORRECT: 'Cursor's agents run in full cloud VMs. [image]' "
            "WRONG: 'The screenshot shows a cloud VM environment.'"
        )
        lines.append(
            "- Where a slide image adds value, embed it inline: "
            "`![description](slides/filename.png)`"
        )
        lines.append(
            "- Do NOT write an introduction or conclusion -- this section will be merged with others."
        )
        lines.append("- Output only markdown section content -- no preamble.")
        lines.append("")

        return "\n".join(lines)

    def build_reduce_prompt(
        self,
        title: str,
        description: str | None,
        section_drafts: list[str],
        metadata: dict[str, Any] | None,
    ) -> str:
        """Build the reduce prompt to synthesize section drafts into a final article."""
        lines: list[str] = []
        lines.append(
            f"You are a technology journalist and analyst writing the final version of an "
            f'analytical article about "{title}". You have {len(section_drafts)} draft sections '
            f"to synthesize. Your voice is authoritative, declarative, and third-party — write "
            f"as an informed analyst, not as a conference attendee."
        )
        lines.append("")

        if description:
            lines.append(f"## Session Description\n{description}")
            lines.append("")

        if metadata:
            lines.append("## Session Metadata")
            for key, value in metadata.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")

        lines.append("## Section Drafts")
        lines.append("")

        for i, draft in enumerate(section_drafts, start=1):
            lines.append(f"### Section {i}")
            lines.append(draft)
            lines.append("")

        lines.append("## Instructions")
        lines.append(
            "- NEVER use first-person attendee language: 'I walked away...', 'what struck me...', "
            "'I came to understand...', 'I kept thinking about...', 'my takeaway was...'. "
            "Write in the authoritative, declarative voice of technical journalism "
            "(MIT Technology Review, IEEE Spectrum, Ars Technica). Attribute to organizations "
            "and people, not to the author's experience."
        )
        lines.append(
            "- NEVER write 'the talk argues', 'the slide shows', 'the screenshot depicts', "
            "'the speaker said', or any phrase referencing the source material as an artifact. "
            "State all technical claims directly. Credit the company or person, not the artifact."
        )
        lines.append(
            "- When an image appears, the surrounding prose must make a technical argument the "
            "image supports -- the image is evidence, never the subject of the sentence."
        )
        lines.append("- Write a compelling introduction and a strong conclusion.")
        lines.append("- Create smooth transitions between sections.")
        lines.append("- Preserve ALL image references (`![...](...)`). Do not remove any.")
        lines.append("- Preserve ALL technical detail from the section drafts.")
        lines.append("- Remove redundancy between sections and ensure a consistent voice.")
        lines.append(f"- Output ONLY the complete markdown starting with `# {title}`.")
        lines.append("")

        return "\n".join(lines)

    async def synthesize_map_reduce(
        self,
        analyses: list[dict[str, Any]],
        title: str,
        description: str | None,
        metadata: dict[str, Any] | None,
        model: str,
        chunk_size: int,
    ) -> str:
        """Orchestrate map-reduce synthesis via Claude Code CLI."""
        import asyncio as _asyncio

        claude_svc = ClaudeInferenceService()

        # Split analyses into chunks
        chunks: list[list[dict[str, Any]]] = [
            analyses[i : i + chunk_size] for i in range(0, len(analyses), chunk_size)
        ]
        total_chunks = len(chunks)

        logger.info(
            "Map-reduce: %d chunks of ~%d slides, model=%s",
            total_chunks,
            chunk_size,
            model,
        )

        # Map phase: compile each chunk concurrently via Claude
        map_semaphore = _asyncio.Semaphore(4)

        async def compile_chunk(
            analyses_chunk: list[dict[str, Any]],
            chunk_idx: int,
        ) -> str:
            async with map_semaphore:
                prompt = self.build_chunk_prompt(
                    analyses_chunk=analyses_chunk,
                    chunk_idx=chunk_idx,
                    total_chunks=total_chunks,
                    title=title,
                    description=description,
                )
                full_prompt = claude_svc.build_full_prompt(
                    prompt=prompt,
                    system_prompt=None,
                    image_paths=None,
                    file_paths=None,
                    context=None,
                )
                raw, _attempts = await claude_svc.execute_with_retry(
                    full_prompt=full_prompt,
                    model=model,
                    output_format="text",
                    timeout=300,
                    max_retries=2,
                    retry_delay=2.0,
                )
                return raw

        section_drafts = await _asyncio.gather(
            *(compile_chunk(chunk, idx) for idx, chunk in enumerate(chunks))
        )

        # Reduce phase: synthesize all sections into one article
        reduce_prompt = self.build_reduce_prompt(
            title=title,
            description=description,
            section_drafts=list(section_drafts),
            metadata=metadata,
        )
        if len(reduce_prompt) > 80_000:
            logger.warning(
                "Reduce prompt is very large (%d chars) — may approach context limits",
                len(reduce_prompt),
            )
        reduce_full = claude_svc.build_full_prompt(
            prompt=reduce_prompt,
            system_prompt=None,
            image_paths=None,
            file_paths=None,
            context=None,
        )
        raw_reduce, _attempts = await claude_svc.execute_with_retry(
            full_prompt=reduce_full,
            model=model,
            output_format="text",
            timeout=max(300, min(60 * len(section_drafts), 900)),  # cap at 15 min
            max_retries=2,
            retry_delay=2.0,
        )

        if not raw_reduce:
            raise ValueError("Claude returned empty content for reduce prompt")

        # Strip accidental code block wrappers
        content = raw_reduce.strip()
        if content.startswith("```markdown"):
            content = content[len("```markdown") :].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        if not content:
            raise ValueError(
                "Claude returned empty content after stripping code fences in reduce phase"
            )

        return content

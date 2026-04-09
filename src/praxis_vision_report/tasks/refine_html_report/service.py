"""Service for refine_html_report task."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from praxis.task.core.transform.claude_inference.service import ClaudeInferenceService

# [OpenAI]
# from openai import AsyncOpenAI
# [/OpenAI]

logger = structlog.get_logger(__name__)

REFINE_SYSTEM = """You are a senior engineer rewriting a conference blog post for publication. You attended this session and are explaining what you learned to a peer. You write for an audience of ML engineers and AI researchers who value precision, depth, and no fluff. You are the author -- you own these ideas."""

REFINE_INSTRUCTIONS = """Rewrite this technical conference blog post from scratch. You are not editing -- you are writing the DEFINITIVE version using this as raw material.

RULES:
1. **Voice**: Write as the author, not as a reviewer of the source material. NEVER write "the talk argues", "the slide shows", "the screenshot depicts", "the speaker said", "the presentation covers", or any phrase that references the talk or slides as artifacts. State every technical claim directly and confidently. If attribution is needed, credit the company or person -- not the slide or talk.
   CORRECT: "Cursor's async cloud agents now produce roughly 30% of internal merged PRs."
   WRONG: "The talk claims that 30% of internal PRs are now agent-generated."

2. **Images as evidence, not subjects**: When you include an image, the surrounding prose must make a technical point that the image supports. Write the argument first, then let the image reinforce it. The image is never the subject of the sentence.
   CORRECT: "Cursor's agents run in full cloud VMs with filesystem access, terminals, and sandboxed network policies.\n\n![Cloud VM environment](slides/frame_0042.png)"
   WRONG: "The screenshot shows a remote desktop environment used by the agent."

3. Include an image ONLY if it genuinely illustrates a technical concept discussed in the surrounding paragraph.
4. OMIT: opening title slides, speaker intro slides, "agenda" slides, "thank you" slides, and any slide that is purely decorative or redundant.
5. Dense, precise, technically grounded. Every key technical point from every slide must appear somewhere in the prose -- don't lose detail.
6. Structure with genuine informational headings -- not "Introduction" but something like "The Core Problem: Why In-Process Inference Fails at Game Scale".
7. Apply every item in PRIORITY EDITS.
8. Output ONLY the complete markdown starting with the # heading. No preamble, no explanation, no ```markdown wrapper."""


class RefineHtmlReportService:
    def __init__(self) -> None:
        # [OpenAI]
        # self._client: AsyncOpenAI | None = None
        # [/OpenAI]
        self._claude_svc = ClaudeInferenceService()

    # [OpenAI]
    # @property
    # def client(self) -> AsyncOpenAI:
    #     if self._client is None:
    #         self._client = AsyncOpenAI()
    #     return self._client
    # [/OpenAI]

    def build_analyses_context(self, analyses: list[dict[str, Any]]) -> str:
        lines = []
        for a in analyses:
            img = Path(a.get("image_path", "")).name
            tc = a.get("timecode", "")
            analysis_text = a.get("analysis", "")[:300]  # cap per-slide context
            kp = "; ".join(a.get("key_points", [])[:3])
            lines.append(f"[{img}] {tc}: {analysis_text} | KEY: {kp}")
        return "\n".join(lines)

    def format_critique_for_prompt(self, critique: dict[str, Any]) -> str:
        summary = critique.get("executive_summary", "")
        priority = "\n".join(
            f"  {i + 1}. {e}" for i, e in enumerate(critique.get("priority_edits", []))
        )
        removes = [
            d["image_filename"]
            for d in critique.get("image_decisions", [])
            if d.get("action") == "remove"
        ]
        remove_str = ", ".join(removes) if removes else "none"
        issues = "\n".join(
            f"  - [{i.get('section_heading', '')}]: {i.get('problem', '')} -> {i.get('suggestion', '')}"
            for i in critique.get("section_issues", [])
        )
        return f"""EDITORIAL CRITIQUE SUMMARY:
{summary}

PRIORITY EDITS (apply all of these):
{priority}

IMAGES TO REMOVE: {remove_str}

SECTION ISSUES TO FIX:
{issues}"""

    def safe_filename(self, title: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
        return slug[:80]

    def count_words(self, text: str) -> int:
        return len(text.split())

    def inject_subtitle(self, markdown: str, subtitle: str) -> str:
        """Insert an italic subtitle line after the first H1 heading."""
        return re.sub(
            r"^(# .+)$",
            rf"\1\n\n*{subtitle}*",
            markdown,
            count=1,
            flags=re.MULTILINE,
        )

    async def rewrite(
        self,
        markdown_content: str,
        analyses: list[dict[str, Any]],
        critique: dict[str, Any],
        title: str,
        description: str | None,
        model: str,
    ) -> tuple[str, str, dict[str, Any]]:
        """Rewrite the markdown via Claude Code CLI. Returns (refined_markdown, changes_summary, token_usage)."""
        desc_section = f"\n\nSESSION ABSTRACT:\n{description}" if description else ""
        analyses_context = self.build_analyses_context(analyses)
        critique_text = self.format_critique_for_prompt(critique)

        prompt = f"""{REFINE_INSTRUCTIONS}

TITLE: {title}{desc_section}

COMPLETE SLIDE ANALYSES (what every image contains -- use this to decide what to include):
{analyses_context}

{critique_text}

ORIGINAL DRAFT (rewrite this):
{markdown_content}"""

        logger.debug(
            "rewrite_request",
            original_words=self.count_words(markdown_content),
        )

        claude_model = "claude-sonnet-4-6"
        full_prompt = self._claude_svc.build_full_prompt(
            prompt=prompt,
            system_prompt=REFINE_SYSTEM,
            image_paths=None,
            file_paths=None,
            context=None,
        )

        raw, _attempts = await self._claude_svc.execute_with_retry(
            full_prompt=full_prompt,
            model=claude_model,
            output_format="text",
            timeout=300,
            max_retries=2,
            retry_delay=2.0,
        )

        refined = raw.strip()
        # Strip any accidental code block wrapper
        refined = re.sub(r"^```(?:markdown)?\s*", "", refined)
        refined = re.sub(r"\s*```$", "", refined).strip()

        token_usage: dict[str, Any] = {}

        changes = (
            f"Full rewrite applying {len(critique.get('priority_edits', []))} priority edits. "
            f"Removed {sum(1 for d in critique.get('image_decisions', []) if d.get('action') == 'remove')} decorative images."
        )
        logger.debug("rewrite_complete", refined_words=self.count_words(refined))
        return refined, changes, token_usage

        # [OpenAI]
        # messages = [
        #     {"role": "system", "content": REFINE_SYSTEM},
        #     {"role": "user", "content": prompt},
        # ]
        #
        # import asyncio
        #
        # last_exc: Exception | None = None
        # for attempt in range(3):
        #     try:
        #         response = await self.client.chat.completions.create(
        #             model=model,
        #             messages=messages,  # type: ignore[arg-type]
        #         )
        #         break
        #     except Exception as exc:
        #         last_exc = exc
        #         if attempt < 2:
        #             wait = 2 ** (attempt + 1)
        #             logger.debug("rewrite_retry", attempt=attempt + 1, wait=wait, error=str(exc))
        #             await asyncio.sleep(wait)
        # else:
        #     raise RuntimeError(f"Rewrite API failed after 3 attempts: {last_exc}") from last_exc
        #
        # refined = (response.choices[0].message.content or "").strip()
        # # Strip any accidental code block wrapper
        # refined = re.sub(r"^```(?:markdown)?\s*", "", refined)
        # refined = re.sub(r"\s*```$", "", refined).strip()
        #
        # usage = response.usage
        # token_usage: dict[str, Any] = {}
        # if usage is not None:
        #     token_usage = {
        #         "input_tokens": usage.prompt_tokens,
        #         "output_tokens": usage.completion_tokens,
        #     }
        #
        # changes = (
        #     f"Full rewrite applying {len(critique.get('priority_edits', []))} priority edits. "
        #     f"Removed {sum(1 for d in critique.get('image_decisions', []) if d.get('action') == 'remove')} decorative images."
        # )
        # logger.debug("rewrite_complete", refined_words=self.count_words(refined))
        # return refined, changes, token_usage
        # [/OpenAI]

    def convert_to_html(self, markdown_content: str, title: str) -> str:
        """Convert markdown to styled HTML. Same CSS as compile_vision_report."""
        import markdown2  # type: ignore[import-untyped]

        body: str = markdown2.markdown(
            markdown_content,
            extras=[
                "fenced-code-blocks",
                "tables",
                "header-ids",
                "strike",
                "task_list",
            ],
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 860px; margin: 0 auto; padding: 2rem; line-height: 1.7; color: #1a1a1a; }}
  h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.4rem; font-weight: 600; margin-top: 2.5rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.4rem; }}
  h3 {{ font-size: 1.15rem; font-weight: 600; margin-top: 1.8rem; }}
  p {{ margin: 1rem 0; }}
  img {{ width: 100%; max-width: 100%; height: auto; display: block; margin: 1.5em 0; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
  pre {{ background: #f4f4f5; border-radius: 6px; padding: 1rem; overflow-x: auto; }}
  code {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.88em; }}
  blockquote {{ border-left: 4px solid #6366f1; margin: 1.5rem 0; padding: 0.5rem 1rem; background: #f8f8ff; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.5rem 0; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 0.6rem 0.9rem; text-align: left; }}
  th {{ background: #f9fafb; font-weight: 600; }}
  .refined-badge {{ display: inline-block; background: #6366f1; color: white; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px; margin-left: 0.5rem; vertical-align: middle; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

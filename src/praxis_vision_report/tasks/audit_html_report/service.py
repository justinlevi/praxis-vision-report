"""Service for audit_html_report task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from praxis.task.core.transform.claude_inference.service import ClaudeInferenceService
from praxis_vision_report.tasks.audit_html_report.models import (
    AuditCritique,
    ImageDecision,
    SectionIssue,
)

logger = structlog.get_logger(__name__)

AUDIT_PERSONA = """You are Sarah Chen, Senior Technical Content Director with 15 years of experience editing ML engineering conference writeups and AI thought leadership content. You've shaped content strategy for The Batch, distill.pub-style explainers, and engineering blogs at top AI companies.

You evaluate technical blog posts through three lenses:

1. EDITORIAL QUALITY: Is the narrative compelling? Does it build momentum? Are headings genuinely informative? Does each section earn its place?

2. TECHNICAL CREDIBILITY: Would a senior ML engineer trust this? Are specific architectures, benchmarks, and trade-offs cited? No hand-waving or marketing speak.

3. VISUAL INTELLIGENCE (your sharpest instinct): For every embedded image you ask — does this image EARN its space?
   - Opening title slides with just a logo = dead weight → REMOVE
   - Architecture diagrams, benchmark tables, code examples = high value → KEEP
   - Speaker intro/bio slides = never appropriate for a blog post → REMOVE
   - Transitional slides ("agenda", "thank you") = decorative noise → REMOVE
   - Diagrams that the surrounding text describes and extends = essential → KEEP

   An image is orphaned if the nearby text doesn't reference what's shown. Flag every orphaned image.

Your critique is specific and ruthless. You cite exact headings, image filenames, and paragraph text. You don't soften feedback."""


class AuditHtmlReportService:
    def build_analyses_summary(self, analyses: list[dict[str, Any]]) -> str:
        """Build a compact summary of all slide analyses for context."""
        lines = []
        for a in analyses:
            img = Path(a.get("image_path", "")).name
            tc = a.get("timecode", "")
            kp = a.get("key_points", [])
            lines.append(f"- {img} [{tc}]: {'; '.join(kp[:3]) if kp else 'no key points'}")
        return "\n".join(lines)

    def build_prompt(
        self,
        markdown_content: str,
        analyses: list[dict[str, Any]],
        title: str,
        description: str | None,
    ) -> str:
        analyses_summary = self.build_analyses_summary(analyses)
        desc_section = f"\n\nSESSION ABSTRACT:\n{description}" if description else ""
        return f"""Review this technical conference blog post and provide a rigorous editorial critique.

TITLE: {title}{desc_section}

SLIDE ANALYSES (what each embedded image actually contains):
{analyses_summary}

BLOG POST MARKDOWN:
{markdown_content}

Respond with a JSON object matching this exact schema:
{{
  "overall_score": <1-10>,
  "executive_summary": "<2-3 sentences>",
  "narrative_coherence_score": <1-10>,
  "narrative_issues": ["<issue>", ...],
  "image_decisions": [
    {{"image_filename": "<filename>", "action": "<keep|remove|move>", "reason": "<why>", "text_context": "<what nearby text says or should say>"}},
    ...
  ],
  "section_issues": [
    {{"section_heading": "<heading>", "problem": "<what's wrong>", "suggestion": "<how to fix>"}},
    ...
  ],
  "missing_technical_depth": ["<topic not covered adequately>", ...],
  "redundant_content": ["<section or paragraph to cut>", ...],
  "priority_edits": ["<highest impact change #1>", "<#2>", "<#3>", "<#4>", "<#5>"]
}}

Output ONLY the JSON object. No preamble, no explanation."""

    def parse_critique(self, data: dict[str, Any]) -> AuditCritique:
        """Build AuditCritique from a parsed JSON dict."""
        data["image_decisions"] = [ImageDecision(**d) for d in data.get("image_decisions", [])]
        data["section_issues"] = [SectionIssue(**d) for d in data.get("section_issues", [])]
        return AuditCritique(**data)

    async def audit(
        self,
        screenshot_paths: list[str],
        markdown_content: str,
        analyses: list[dict[str, Any]],
        title: str,
        description: str | None,
        model: str,
        max_screenshots: int,
    ) -> tuple[AuditCritique, str, dict[str, Any]]:
        """Run the editorial audit via Claude Code CLI.

        Passes the report markdown and screenshot images directly to Claude,
        which reads them natively. Returns (critique, model_used, token_usage).
        """
        screenshots_to_use = screenshot_paths[:max_screenshots]

        missing = [p for p in screenshots_to_use if not Path(p).exists()]
        if missing:
            logger.warning(
                "Skipping %d missing screenshot(s): %s",
                len(missing),
                missing[:3],
            )
            screenshots_to_use = [p for p in screenshots_to_use if Path(p).exists()]

        user_text = self.build_prompt(markdown_content, analyses, title, description)

        claude_svc = ClaudeInferenceService()

        full_prompt = claude_svc.build_full_prompt(
            prompt=user_text,
            system_prompt=AUDIT_PERSONA,
            image_paths=screenshots_to_use,
            file_paths=None,
            context=None,
        )

        logger.debug(
            "audit_request",
            model=model,
            screenshots=len(screenshots_to_use),
            slides=len(analyses),
        )

        raw, _attempts = await claude_svc.execute_with_retry(
            full_prompt=full_prompt,
            model=model,
            output_format="text",
            timeout=300,
            max_retries=3,
            retry_delay=2.0,
        )

        parsed = claude_svc.try_parse_response_json(raw)
        if parsed is None:
            raise RuntimeError(f"Audit response was not valid JSON. Raw output:\n{raw[:500]}")

        critique = self.parse_critique(parsed)

        logger.debug(
            "audit_complete",
            overall_score=critique.overall_score,
            image_decisions=len(critique.image_decisions),
        )
        return critique, model, {}

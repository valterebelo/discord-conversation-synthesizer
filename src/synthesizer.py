"""
Synthesizer module - uses Claude to generate structured summaries.

Handles:
- Building prompts from conversation transcripts
- Calling Claude API (or mock)
- Parsing structured output
- Cost tracking
- Error handling and retries
"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol
from dataclasses import dataclass

from .models import Conversation, SynthesizedNote

logger = logging.getLogger(__name__)


# Cost per million tokens (as of knowledge cutoff)
OPUS_INPUT_COST_PER_M = 15.0   # $15 per million input tokens
OPUS_OUTPUT_COST_PER_M = 75.0  # $75 per million output tokens


@dataclass
class SynthesisResult:
    """Result of synthesizing a single conversation."""
    success: bool
    note: Optional[SynthesizedNote] = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Synthesizer:
    """
    Synthesizes conversations into structured notes using Claude.

    Supports both real Claude API and mock client for testing.
    """

    def __init__(
        self,
        client,  # Claude client (real or mock)
        model: str = "claude-opus-4-5-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the synthesizer.

        Args:
            client: Claude API client (or mock)
            model: Model identifier
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            prompt_template: Custom prompt template (or use default)
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_template = prompt_template or self._default_prompt_template()

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.synthesis_count = 0

    def _default_prompt_template(self) -> str:
        """Return the default synthesis prompt template."""
        return '''You are a knowledge synthesizer transforming Discord conversations into structured understanding. Your explanatory style follows Grant Sanderson (3Blue1Brown): build intuition first, use analogies generously, and always illuminate the "why" behind ideas.

## Your Task

Given a conversation transcript from a quantitative finance Discord, produce a Markdown note that:

1. **Captures the essential insight** — What would someone learn from reading this?

2. **Builds understanding progressively** — Start with the accessible framing, then add complexity. A reader should be able to stop at any paragraph and have gained something.

3. **Preserves attribution** — When someone contributed a key idea, insight, or objection, credit them inline: "As [username] pointed out, ..." or "[username] raised an important objection here: ..."

4. **Identifies tension points** — Where did participants disagree? What remained unresolved? These are often the most valuable parts.

5. **Extracts actionable concepts** — What tags/topics does this connect to? What other conversations might this link to?

## Output Format

You MUST output in this exact format with YAML frontmatter:

```
---
title: "[Descriptive title capturing the core topic]"
date: "[Date of conversation in YYYY-MM-DD format]"
participants: ["username1", "username2", ...]
channel: "[Channel name]"
tags: ["tag1", "tag2", ...]
related: ["[[Related Topic 1]]", "[[Related Topic 2]]"]
---

## Summary

[2-3 sentence overview of what this conversation achieved]

## The Core Idea

[Main explanatory section — this is where the 3B1B style shines. Build the concept from the ground up. Use analogies. Make it clear why this matters. This should be 2-4 paragraphs.]

## Key Contributions

[Bulleted list attributing who brought what to the discussion]

## Points of Tension

[Where did people disagree? What's unresolved? Why is this interesting?]

## Connections

[How does this relate to other concepts? What should the reader explore next?]
```

## Guidelines

- If the conversation is shallow (just greetings, simple Q&A with no depth), produce a very brief note and mark tags as ["shallow", "skip-review"].

- If you detect multiple distinct conversations interleaved, note this in the Summary and suggest they be processed separately.

- Mathematical content should be preserved precisely. Use LaTeX notation where appropriate: $inline$ or $$display$$.

- Code snippets should be preserved in fenced blocks with language tags.

- Quant finance jargon is expected — don't over-explain basics like "alpha" or "Sharpe ratio" unless the conversation itself explains them.

- Keep the Summary to 2-3 sentences. Keep The Core Idea to 2-4 substantial paragraphs.

## Conversation Transcript

{transcript}'''

    async def synthesize(
        self,
        conversation: Conversation,
        retry_count: int = 3,
        retry_delay: float = 2.0,
    ) -> SynthesisResult:
        """
        Synthesize a single conversation into a note.

        Args:
            conversation: The conversation to synthesize
            retry_count: Number of retries on failure
            retry_delay: Seconds between retries

        Returns:
            SynthesisResult with note or error
        """
        logger.info(
            f"Synthesizing conversation: {conversation.id} "
            f"({conversation.message_count} messages)"
        )

        # Build the prompt
        transcript = conversation.to_transcript()
        prompt = self.prompt_template.format(transcript=transcript)

        # Estimate input tokens (rough approximation)
        estimated_input_tokens = len(prompt.split()) * 1.3

        if estimated_input_tokens > 100000:
            logger.warning(
                f"Conversation {conversation.id} is very long "
                f"(~{estimated_input_tokens:.0f} tokens), may need chunking"
            )

        # Call Claude with retries
        last_error = None
        for attempt in range(retry_count):
            try:
                response = await self._call_claude(prompt)

                # Parse the response
                note = self._parse_response(
                    response_text=response["text"],
                    conversation=conversation,
                    input_tokens=response["input_tokens"],
                    output_tokens=response["output_tokens"],
                )

                # Calculate cost
                cost = self._calculate_cost(
                    response["input_tokens"],
                    response["output_tokens"]
                )

                # Update totals
                self.total_input_tokens += response["input_tokens"]
                self.total_output_tokens += response["output_tokens"]
                self.total_cost_usd += cost
                self.synthesis_count += 1

                logger.info(
                    f"Synthesis complete: {note.title} "
                    f"(cost: ${cost:.4f})"
                )

                return SynthesisResult(
                    success=True,
                    note=note,
                    input_tokens=response["input_tokens"],
                    output_tokens=response["output_tokens"],
                    cost_usd=cost,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Synthesis attempt {attempt + 1}/{retry_count} failed: {e}"
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        logger.error(f"Synthesis failed after {retry_count} attempts: {last_error}")
        return SynthesisResult(
            success=False,
            error=last_error,
        )

    async def _call_claude(self, prompt: str) -> dict:
        """
        Call the Claude API.

        Returns dict with text, input_tokens, output_tokens.
        """
        # Check if this is a mock client or real client
        if hasattr(self.client, 'messages'):
            # Real Anthropic client or mock with messages interface
            client_to_use = self.client.messages if hasattr(self.client, 'messages') else self.client

            response = await client_to_use.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "text": response.content[0].text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            # Assume it's our mock client directly
            response = await self.client.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                "text": response.content[0].text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

    def _parse_response(
        self,
        response_text: str,
        conversation: Conversation,
        input_tokens: int,
        output_tokens: int,
    ) -> SynthesizedNote:
        """
        Parse Claude's response into a SynthesizedNote.

        Handles both well-formed and malformed responses.
        """
        # Try to extract YAML frontmatter
        frontmatter = {}
        body = response_text

        # Look for YAML frontmatter between --- markers
        yaml_match = re.search(
            r'^---\s*\n(.*?)\n---\s*\n',
            response_text,
            re.DOTALL | re.MULTILINE
        )

        if yaml_match:
            yaml_content = yaml_match.group(1)
            body = response_text[yaml_match.end():]

            # Parse YAML manually (avoid dependency)
            frontmatter = self._parse_yaml_simple(yaml_content)

        # Extract sections from body
        sections = self._extract_sections(body)

        # Build the note
        note = SynthesizedNote(
            conversation_id=conversation.id,
            title=frontmatter.get("title", self._generate_title(conversation)),
            date=frontmatter.get("date", conversation.timestamp_start.strftime("%Y-%m-%d")),
            participants=frontmatter.get("participants", conversation.participants),
            channel=frontmatter.get("channel", conversation.channel_name),
            tags=frontmatter.get("tags", self._extract_tags(body)),
            related=frontmatter.get("related", []),
            summary=sections.get("summary", "No summary available."),
            core_idea=sections.get("the core idea", sections.get("core idea", "No core idea extracted.")),
            key_contributions=sections.get("key contributions", "No contributions noted."),
            tension_points=sections.get("points of tension", "No tension points identified."),
            connections=sections.get("connections", "No connections identified."),
            raw_insights=sections.get("raw insights"),
            token_usage=input_tokens + output_tokens,
            cost_usd=self._calculate_cost(input_tokens, output_tokens),
            raw_response=response_text,
        )

        return note

    def _parse_yaml_simple(self, yaml_content: str) -> dict:
        """
        Simple YAML parser for frontmatter.

        Handles basic key: value and key: [list] formats.
        """
        result = {}

        for line in yaml_content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Parse lists
                if value.startswith("[") and value.endswith("]"):
                    # Simple list parsing
                    list_content = value[1:-1]
                    items = []
                    for item in list_content.split(","):
                        item = item.strip()
                        if item.startswith('"') and item.endswith('"'):
                            item = item[1:-1]
                        elif item.startswith("'") and item.endswith("'"):
                            item = item[1:-1]
                        if item:
                            items.append(item)
                    value = items

                result[key] = value

        return result

    def _extract_sections(self, body: str) -> dict[str, str]:
        """
        Extract named sections from the Markdown body.

        Looks for ## headers and captures content until next header.
        """
        sections = {}

        # Split by ## headers
        parts = re.split(r'\n##\s+', body)

        for part in parts[1:]:  # Skip content before first header
            lines = part.split("\n", 1)
            if lines:
                header = lines[0].strip().lower()
                content = lines[1].strip() if len(lines) > 1 else ""
                sections[header] = content

        return sections

    def _generate_title(self, conversation: Conversation) -> str:
        """Generate a title from the conversation if none provided."""
        if conversation.thread_name:
            return conversation.thread_name

        # Use first substantive message
        for msg in conversation.messages:
            if len(msg.content) > 20:
                # Take first sentence or 50 chars
                content = msg.content.split(".")[0][:50]
                return f"Discussion: {content}..."

        return f"Conversation in #{conversation.channel_name}"

    def _extract_tags(self, body: str) -> list[str]:
        """Extract likely tags from the body content."""
        # Simple keyword extraction
        keywords = {
            "risk-management": ["risk", "drawdown", "volatility"],
            "portfolio-allocation": ["portfolio", "allocation", "weight"],
            "mean-reversion": ["mean reversion", "revert", "z-score"],
            "statistics": ["correlation", "covariance", "sharpe"],
            "crypto": ["crypto", "bitcoin", "ethereum"],
        }

        body_lower = body.lower()
        tags = []

        for tag, words in keywords.items():
            if any(word in body_lower for word in words):
                tags.append(tag)

        return tags or ["general"]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on token usage."""
        input_cost = (input_tokens / 1_000_000) * OPUS_INPUT_COST_PER_M
        output_cost = (output_tokens / 1_000_000) * OPUS_OUTPUT_COST_PER_M
        return input_cost + output_cost

    async def synthesize_batch(
        self,
        conversations: list[Conversation],
        concurrency: int = 1,  # Sequential by default to manage costs
    ) -> list[SynthesisResult]:
        """
        Synthesize multiple conversations.

        Args:
            conversations: List of conversations to synthesize
            concurrency: Number of concurrent synthesis operations

        Returns:
            List of SynthesisResults in same order as input
        """
        results: list[SynthesisResult] = []

        for i, conv in enumerate(conversations):
            logger.info(f"Processing {i+1}/{len(conversations)}: {conv.id}")
            result = await self.synthesize(conv)
            results.append(result)

            # Small delay between requests
            if i < len(conversations) - 1:
                await asyncio.sleep(0.5)

        return results

    def get_stats(self) -> dict:
        """Get cumulative statistics."""
        return {
            "synthesis_count": self.synthesis_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def estimate_cost(self, conversations: list[Conversation]) -> dict:
        """
        Estimate cost for synthesizing conversations without actually calling API.

        Args:
            conversations: List of conversations to estimate

        Returns:
            Dict with token and cost estimates
        """
        total_input_tokens = 0

        for conv in conversations:
            transcript = conv.to_transcript()
            prompt = self.prompt_template.format(transcript=transcript)
            # Rough token estimate: 1.3 tokens per word
            total_input_tokens += int(len(prompt.split()) * 1.3)

        # Estimate output at ~1500 tokens per conversation
        estimated_output_tokens = len(conversations) * 1500

        estimated_cost = self._calculate_cost(total_input_tokens, estimated_output_tokens)

        return {
            "conversation_count": len(conversations),
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": total_input_tokens + estimated_output_tokens,
            "estimated_cost_usd": estimated_cost,
        }


async def create_synthesizer(
    use_mock: bool = True,
    model: str = "claude-opus-4-5-20250514",
    api_key: Optional[str] = None,
    prompt_template_path: Optional[str] = None,
) -> Synthesizer:
    """
    Factory function to create a Synthesizer.

    Args:
        use_mock: If True, use mock client
        model: Model identifier
        api_key: Anthropic API key (for real client)
        prompt_template_path: Path to custom prompt template

    Returns:
        Configured Synthesizer instance
    """
    # Load prompt template if provided
    prompt_template = None
    if prompt_template_path:
        prompt_path = Path(prompt_template_path)
        if prompt_path.exists():
            prompt_template = prompt_path.read_text()
            logger.info(f"Loaded prompt template from {prompt_template_path}")

    if use_mock:
        from .mocks.claude_mock import MockClaudeClient
        client = MockClaudeClient()
        logger.info("Using mock Claude client")
    else:
        try:
            import anthropic
            import os
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY required for real Claude client")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            logger.info("Using real Anthropic client")
        except ImportError:
            logger.warning("anthropic package not installed, falling back to mock")
            from .mocks.claude_mock import MockClaudeClient
            client = MockClaudeClient()

    return Synthesizer(
        client=client,
        model=model,
        prompt_template=prompt_template,
    )


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from segmenter import segment_from_fixtures

    logging.basicConfig(level=logging.INFO)

    async def test():
        # Get a conversation to synthesize
        fixture_path = Path(__file__).parent.parent / "tests/fixtures/sample_conversations.json"
        conversations = segment_from_fixtures(str(fixture_path))

        if not conversations:
            print("No conversations found!")
            return

        # Synthesize the first one
        synthesizer = await create_synthesizer(use_mock=True)
        result = await synthesizer.synthesize(conversations[0])

        if result.success:
            print("\n=== Synthesized Note ===\n")
            print(result.note.to_markdown()[:2000])
            print("\n...")
            print(f"\n=== Stats ===")
            print(f"Input tokens: {result.input_tokens}")
            print(f"Output tokens: {result.output_tokens}")
            print(f"Cost: ${result.cost_usd:.4f}")
        else:
            print(f"Synthesis failed: {result.error}")

    asyncio.run(test())

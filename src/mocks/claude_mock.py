"""
Mock Claude client for testing synthesis without hitting the Anthropic API.

Generates plausible synthesized notes based on conversation content.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class MockContentBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockResponse:
    id: str
    content: list[MockContentBlock]
    model: str
    usage: MockUsage
    stop_reason: str = "end_turn"


class MockClaudeClient:
    """
    A mock Claude client that generates synthetic summaries for testing.

    The mock analyzes the conversation content and generates a plausible
    (but not AI-quality) synthesis. This lets us test the full pipeline
    without API costs.

    Usage:
        client = MockClaudeClient()
        response = await client.messages.create(
            model="claude-opus-4-5-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": conversation_transcript}]
        )
    """

    def __init__(self, latency_ms: int = 100):
        self.latency_ms = latency_ms
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.messages = self  # For API compatibility (client.messages.create)

    async def create(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        temperature: float = 0.3,
        **kwargs
    ) -> MockResponse:
        """
        Mock the messages.create() API call.

        Parses the conversation from the prompt and generates a synthetic summary.
        """
        import asyncio
        await asyncio.sleep(self.latency_ms / 1000)

        self.call_count += 1

        # Extract conversation content from the prompt
        user_message = messages[-1].get("content", "") if messages else ""

        # Parse out participants and key content
        participants = self._extract_participants(user_message)
        topics = self._extract_topics(user_message)
        has_code = "```" in user_message
        has_math = "$" in user_message

        # Generate a mock synthesis
        synthesis = self._generate_mock_synthesis(
            participants=participants,
            topics=topics,
            has_code=has_code,
            has_math=has_math,
            raw_content=user_message
        )

        # Estimate token counts (rough approximation)
        input_tokens = len(user_message.split()) * 1.3
        output_tokens = len(synthesis.split()) * 1.3

        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)

        # Generate a deterministic ID based on content
        content_hash = hashlib.md5(user_message.encode()).hexdigest()[:8]

        return MockResponse(
            id=f"mock_msg_{content_hash}",
            content=[MockContentBlock(type="text", text=synthesis)],
            model=model,
            usage=MockUsage(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens)
            )
        )

    def _extract_participants(self, content: str) -> list[str]:
        """Extract usernames from conversation transcript."""
        # Look for patterns like "[username]:" or "username:"
        pattern = r'\[([^\]]+)\]:|^([a-z_]+):'
        matches = re.findall(r'\[(\d{2}:\d{2}:\d{2})\]\s+(\w+):', content)
        if matches:
            return list(set(m[1] for m in matches))

        # Fallback: look for @mentions
        mentions = re.findall(r'@(\w+)', content)
        return list(set(mentions)) if mentions else ["participant_1", "participant_2"]

    def _extract_topics(self, content: str) -> list[str]:
        """Extract likely topics from content using keyword matching."""
        topic_keywords = {
            "risk-management": ["risk", "drawdown", "volatility", "var", "cvar"],
            "portfolio-allocation": ["portfolio", "allocation", "weight", "rebalance", "diversif"],
            "mean-reversion": ["mean reversion", "revert", "z-score", "lookback"],
            "momentum": ["momentum", "trend", "breakout"],
            "machine-learning": ["ml", "machine learning", "neural", "model", "train"],
            "statistics": ["correlation", "covariance", "regression", "p-value", "sharpe"],
            "crypto": ["crypto", "bitcoin", "btc", "eth", "defi"],
            "options": ["option", "vol", "greeks", "delta", "gamma"],
            "market-microstructure": ["spread", "liquidity", "order book", "market making"],
        }

        content_lower = content.lower()
        found_topics = []

        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    found_topics.append(topic)
                    break

        return found_topics if found_topics else ["general-discussion"]

    def _generate_mock_synthesis(
        self,
        participants: list[str],
        topics: list[str],
        has_code: bool,
        has_math: bool,
        raw_content: str
    ) -> str:
        """Generate a mock synthesis in the expected output format."""

        # Create a title from topics
        title = topics[0].replace("-", " ").title() if topics else "Discussion"

        # Build the mock response in the expected YAML + Markdown format
        synthesis = f"""---
title: "{title} - Community Discussion"
date: "2026-01-15"
participants: {participants}
channel: "trading-strategies"
tags: {topics}
related: ["[[portfolio-optimization]]", "[[risk-metrics]]"]
---

## Summary

This conversation explored {title.lower()} with {len(participants)} participants contributing insights. The discussion covered practical implementation challenges and theoretical foundations.

## The Core Idea

[MOCK SYNTHESIS] The participants discussed approaches to {title.lower()}. Think of it like this: when you're trying to {topics[0].replace('-', ' ')}, you need to balance multiple competing objectives.

{f"As {participants[0]} pointed out, the key insight is understanding the tradeoffs involved." if participants else ""}

The fundamental challenge is that theoretical models often assume conditions that don't hold in practice. This creates a gap between what should work and what actually works.

## Key Contributions

"""
        # Add mock contributions
        for i, participant in enumerate(participants[:4]):
            synthesis += f"- **{participant}**: Contributed perspective on {'implementation details' if i % 2 == 0 else 'theoretical foundations'}\n"

        synthesis += """
## Points of Tension

The main disagreement centered on practical implementation:

- Some participants favored simpler approaches that are more robust
- Others argued for more sophisticated methods despite complexity
- The cost-benefit tradeoff of advanced techniques remained unresolved

## Connections

This discussion connects to several related topics:

- [[risk-management]] - Managing downside in implementation
- [[backtesting]] - Validating approaches before deployment
- [[transaction-costs]] - Real-world friction that affects results

"""
        if has_code:
            synthesis += """## Code Discussed

The conversation included code examples demonstrating implementation details. See the original thread for the full code snippets.

"""

        if has_math:
            synthesis += """## Mathematical Framework

The discussion referenced mathematical concepts including statistical measures and optimization criteria.

"""

        synthesis += """## Raw Insights

> "The theory is elegant but practice is messy" — on the gap between models and reality

---
*[This is a MOCK synthesis for testing. Replace with real Claude API for production.]*
"""

        return synthesis

    def get_stats(self) -> dict:
        """Get usage statistics for this mock client."""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": (
                self.total_input_tokens * 0.000015 +  # $15/M input
                self.total_output_tokens * 0.000075   # $75/M output
            )
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# Factory function for easy swapping between mock and real
def create_claude_client(use_mock: bool = True, api_key: Optional[str] = None) -> MockClaudeClient:
    """
    Create a Claude client - either mock or real.

    Args:
        use_mock: If True, return a mock client. If False, return real Anthropic client.
        api_key: API key for real client (ignored if use_mock=True)

    Returns:
        A client with compatible interface
    """
    if use_mock:
        print("[MockClaudeClient] Using mock client - no API calls will be made")
        return MockClaudeClient()
    else:
        # Import and return real client
        try:
            import anthropic
            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            print("[ClaudeClient] Using real Anthropic client")
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("[Warning] anthropic package not installed, falling back to mock")
            return MockClaudeClient()


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test():
        client = MockClaudeClient()

        test_conversation = """
        CHANNEL: #trading-strategies
        THREAD: Risk Parity Discussion
        TIMESPAN: 2026-01-10 14:30 to 2026-01-10 15:50

        [14:30:00] alice_quant: Been thinking about risk parity lately. Does anyone see better risk-adjusted returns?
        [14:35:00] bob_trader: I ran a backtest comparing risk parity vs 60/40. Risk parity had better Sharpe.
        [14:42:00] charlie_dev: The problem is you need leverage to get equity-like returns.
        """

        response = await client.create(
            model="claude-opus-4-5-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": test_conversation}]
        )

        print("=== Mock Response ===")
        print(response.content[0].text[:500])
        print("\n=== Stats ===")
        print(client.get_stats())

    asyncio.run(test())

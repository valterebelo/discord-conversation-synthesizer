"""
Links module - extracts and catalogs URLs from conversations.

Identifies links shared in discussions, categorizes them by domain,
and creates a searchable link database.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .models import Conversation, Message

logger = logging.getLogger(__name__)

# URL pattern that matches most common URL formats
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\'{}|\\^`\[\]]+',
    re.IGNORECASE
)

# Known domain categories
DOMAIN_CATEGORIES = {
    # Academic / Research
    "arxiv.org": "research",
    "papers.ssrn.com": "research",
    "scholar.google.com": "research",
    "semanticscholar.org": "research",
    "nber.org": "research",

    # Code / Tech
    "github.com": "code",
    "gitlab.com": "code",
    "stackoverflow.com": "tech",
    "docs.python.org": "docs",
    "pytorch.org": "docs",
    "numpy.org": "docs",
    "pandas.pydata.org": "docs",

    # Finance
    "quantopian.com": "finance",
    "quantconnect.com": "finance",
    "tradingview.com": "finance",
    "investing.com": "finance",
    "bloomberg.com": "finance",
    "reuters.com": "finance",

    # Content platforms
    "substack.com": "newsletter",
    "medium.com": "blog",
    "twitter.com": "social",
    "x.com": "social",
    "discord.com": "social",
    "youtube.com": "video",
    "youtu.be": "video",

    # Wikipedia
    "wikipedia.org": "reference",
    "en.wikipedia.org": "reference",
}


def extract_links_from_message(message: Message) -> list[dict]:
    """
    Extract all URLs from a message.

    Returns:
        List of dicts with url, domain, category, author, timestamp
    """
    urls = URL_PATTERN.findall(message.content)
    links = []

    for url in urls:
        # Clean URL (remove trailing punctuation that might have been captured)
        url = url.rstrip(".,;:!?)")

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix for cleaner domain
            if domain.startswith("www."):
                domain = domain[4:]

            # Determine category
            category = "other"
            for domain_pattern, cat in DOMAIN_CATEGORIES.items():
                if domain_pattern in domain:
                    category = cat
                    break

            links.append({
                "url": url,
                "domain": domain,
                "category": category,
                "author": message.author_name,
                "timestamp": message.timestamp.isoformat(),
                "date": message.timestamp.strftime("%Y-%m-%d"),
            })

        except Exception as e:
            logger.warning(f"Failed to parse URL {url}: {e}")

    return links


def extract_links_from_conversations(
    conversations: list[Conversation]
) -> dict:
    """
    Extract all links from a list of conversations.

    Returns:
        Dict with:
        - links: list of all link dicts
        - by_domain: domain -> count
        - by_category: category -> count
        - by_author: author -> count
    """
    all_links = []
    by_domain: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    by_author: dict[str, int] = defaultdict(int)

    for conv in conversations:
        for msg in conv.messages:
            links = extract_links_from_message(msg)
            for link in links:
                all_links.append(link)
                by_domain[link["domain"]] += 1
                by_category[link["category"]] += 1
                by_author[link["author"]] += 1

    # Sort aggregates by count
    return {
        "extracted_at": datetime.now().isoformat(),
        "total_links": len(all_links),
        "unique_domains": len(by_domain),
        "links": all_links,
        "by_domain": dict(sorted(by_domain.items(), key=lambda x: x[1], reverse=True)),
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)),
        "by_author": dict(sorted(by_author.items(), key=lambda x: x[1], reverse=True)),
    }


def save_links(links_data: dict, output_path: Path) -> None:
    """Save extracted links to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(links_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {links_data['total_links']} links to {output_path}")


def generate_links_markdown(links_data: dict, output_path: Path) -> Path:
    """
    Generate a browsable markdown file of extracted links.

    Args:
        links_data: Output from extract_links_from_conversations
        output_path: Path to write the markdown file

    Returns:
        Path to created file
    """
    lines = [
        "---",
        "type: link-index",
        f'updated: "{datetime.now().strftime("%Y-%m-%d")}"',
        f"total_links: {links_data['total_links']}",
        "---",
        "",
        "# Shared Links",
        "",
        f"Total links extracted: **{links_data['total_links']}** from **{links_data['unique_domains']}** domains",
        "",
        "## By Category",
        "",
    ]

    # Group links by category
    for category, count in links_data["by_category"].items():
        lines.append(f"- **{category.title()}**: {count} links")

    lines.extend([
        "",
        "## Top Domains",
        "",
        "| Domain | Count |",
        "|--------|-------|",
    ])

    for domain, count in list(links_data["by_domain"].items())[:15]:
        lines.append(f"| {domain} | {count} |")

    lines.extend([
        "",
        "## Recent Links",
        "",
    ])

    # Show most recent links (last 20)
    recent_links = sorted(
        links_data["links"],
        key=lambda x: x["timestamp"],
        reverse=True
    )[:20]

    for link in recent_links:
        # Truncate URL for display
        display_url = link["url"][:60] + "..." if len(link["url"]) > 60 else link["url"]
        lines.append(
            f"- [{display_url}]({link['url']}) "
            f"— *{link['author']}* ({link['date']})"
        )

    lines.extend([
        "",
        "## By Contributor",
        "",
    ])

    for author, count in list(links_data["by_author"].items())[:10]:
        lines.append(f"- **{author}**: {count} links shared")

    lines.extend([
        "",
        "---",
        "",
        "*Auto-generated by Discord Conversation Synthesizer*",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated links markdown: {output_path}")

    return output_path


if __name__ == "__main__":
    # Quick test
    import sys
    from datetime import timezone

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    from .models import ConversationType

    msg1 = Message(
        id="1",
        author_id="u1",
        author_name="alice",
        content="Check out this paper: https://arxiv.org/abs/2301.12345 and also https://github.com/example/repo",
        timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
    )
    msg2 = Message(
        id="2",
        author_id="u2",
        author_name="bob",
        content="Here's a good video on the topic: https://youtube.com/watch?v=abc123",
        timestamp=datetime(2026, 1, 15, 10, 35, tzinfo=timezone.utc),
    )
    msg3 = Message(
        id="3",
        author_id="u1",
        author_name="alice",
        content="And this article from Bloomberg: https://www.bloomberg.com/news/article123",
        timestamp=datetime(2026, 1, 15, 10, 40, tzinfo=timezone.utc),
    )

    conv = Conversation(
        id="test_conv",
        channel_id="123",
        channel_name="trading",
        messages=[msg1, msg2, msg3],
        conversation_type=ConversationType.CHANNEL,
    )

    # Extract links
    links_data = extract_links_from_conversations([conv])
    print(json.dumps(links_data, indent=2))

    # Generate markdown
    output_dir = Path(__file__).parent.parent / "output" / "_meta"
    md_path = generate_links_markdown(links_data, output_dir / "links.md")
    print(f"\nGenerated: {md_path}")

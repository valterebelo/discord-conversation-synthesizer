"""
Statistics module - computes community analytics without LLM calls.

Generates participant stats, activity patterns, and topic analysis
from conversation data.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Optional

from .models import Conversation, Message, SynthesizedNote

logger = logging.getLogger(__name__)


def compute_participant_stats(conversations: list[Conversation]) -> dict[str, dict]:
    """
    Compute per-participant statistics.

    Returns:
        Dict mapping username to stats dict with:
        - messages: total message count
        - avg_length: average message length in characters
        - reply_ratio: fraction of messages that are replies
        - first_seen: earliest message date
        - last_seen: most recent message date
        - conversations: number of conversations participated in
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "messages": 0,
        "total_length": 0,
        "replies": 0,
        "first_seen": None,
        "last_seen": None,
        "conversation_ids": set(),
    })

    for conv in conversations:
        for msg in conv.messages:
            user = msg.author_name
            s = stats[user]

            s["messages"] += 1
            s["total_length"] += len(msg.content)

            if msg.is_reply:
                s["replies"] += 1

            msg_date = msg.timestamp.date().isoformat()
            if s["first_seen"] is None or msg_date < s["first_seen"]:
                s["first_seen"] = msg_date
            if s["last_seen"] is None or msg_date > s["last_seen"]:
                s["last_seen"] = msg_date

            s["conversation_ids"].add(conv.id)

    # Finalize stats
    result = {}
    for user, s in stats.items():
        result[user] = {
            "messages": s["messages"],
            "avg_length": round(s["total_length"] / s["messages"], 1) if s["messages"] > 0 else 0,
            "reply_ratio": round(s["replies"] / s["messages"], 3) if s["messages"] > 0 else 0,
            "first_seen": s["first_seen"],
            "last_seen": s["last_seen"],
            "conversations": len(s["conversation_ids"]),
        }

    # Sort by message count descending
    return dict(sorted(result.items(), key=lambda x: x[1]["messages"], reverse=True))


def compute_activity_patterns(conversations: list[Conversation]) -> dict:
    """
    Compute activity patterns by hour and day of week.

    Returns:
        Dict with:
        - by_hour: message count per hour (0-23)
        - by_day_of_week: message count per day (0=Monday, 6=Sunday)
        - by_month: message count per month
    """
    by_hour: dict[int, int] = defaultdict(int)
    by_day: dict[int, int] = defaultdict(int)
    by_month: dict[str, int] = defaultdict(int)

    for conv in conversations:
        for msg in conv.messages:
            by_hour[msg.timestamp.hour] += 1
            by_day[msg.timestamp.weekday()] += 1
            by_month[msg.timestamp.strftime("%Y-%m")] += 1

    return {
        "by_hour": dict(sorted(by_hour.items())),
        "by_day_of_week": dict(sorted(by_day.items())),
        "by_month": dict(sorted(by_month.items())),
    }


def compute_conversation_stats(conversations: list[Conversation]) -> dict:
    """
    Compute aggregate conversation statistics.

    Returns:
        Dict with:
        - total: number of conversations
        - avg_duration_min: average duration in minutes
        - median_duration_min: median duration
        - avg_participants: average participants per conversation
        - avg_messages: average messages per conversation
    """
    if not conversations:
        return {
            "total": 0,
            "avg_duration_min": 0,
            "median_duration_min": 0,
            "avg_participants": 0,
            "avg_messages": 0,
        }

    durations = [c.duration_minutes for c in conversations]
    participant_counts = [len(c.participants) for c in conversations]
    message_counts = [c.message_count for c in conversations]

    return {
        "total": len(conversations),
        "avg_duration_min": round(mean(durations), 1),
        "median_duration_min": round(median(durations), 1),
        "max_duration_min": round(max(durations), 1),
        "avg_participants": round(mean(participant_counts), 1),
        "avg_messages": round(mean(message_counts), 1),
    }


def compute_topic_stats(notes: list[SynthesizedNote]) -> dict:
    """
    Compute topic frequency and co-occurrence from synthesized notes.

    Returns:
        Dict with:
        - frequency: tag -> count
        - cooccurrence: list of [tag1, tag2, count] sorted by count
    """
    frequency: dict[str, int] = defaultdict(int)
    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)

    for note in notes:
        tags = sorted(note.tags)

        for tag in tags:
            frequency[tag] += 1

        # Count co-occurrences (pairs)
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i + 1:]:
                pair = (tag1, tag2)
                cooccurrence[pair] += 1

    # Sort and format
    freq_sorted = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    cooc_list = [[t1, t2, count] for (t1, t2), count in cooccurrence.items()]
    cooc_sorted = sorted(cooc_list, key=lambda x: x[2], reverse=True)[:20]  # Top 20

    return {
        "frequency": freq_sorted,
        "cooccurrence": cooc_sorted,
    }


def compute_interaction_graph(conversations: list[Conversation]) -> dict[str, dict[str, int]]:
    """
    Compute who replies to whom.

    Returns:
        Dict mapping user -> {replied_to_user -> count}
    """
    interactions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for conv in conversations:
        # Build message map for this conversation
        msg_map = {m.id: m for m in conv.messages}

        for msg in conv.messages:
            if msg.reply_to and msg.reply_to in msg_map:
                replier = msg.author_name
                replied_to = msg_map[msg.reply_to].author_name
                if replier != replied_to:  # Ignore self-replies
                    interactions[replier][replied_to] += 1

    # Convert to regular dict and sort
    result = {}
    for user, replies in interactions.items():
        result[user] = dict(sorted(replies.items(), key=lambda x: x[1], reverse=True))

    return result


def compute_all_stats(
    conversations: list[Conversation],
    notes: Optional[list[SynthesizedNote]] = None
) -> dict:
    """
    Compute all statistics.

    Args:
        conversations: List of conversation objects
        notes: Optional list of synthesized notes (for topic stats)

    Returns:
        Complete statistics dictionary
    """
    logger.info(f"Computing stats for {len(conversations)} conversations")

    stats = {
        "generated_at": datetime.now().isoformat(),
        "participants": compute_participant_stats(conversations),
        "activity": compute_activity_patterns(conversations),
        "conversations": compute_conversation_stats(conversations),
        "interactions": compute_interaction_graph(conversations),
    }

    if notes:
        stats["topics"] = compute_topic_stats(notes)

    return stats


def save_stats(stats: dict, output_path: Path) -> None:
    """Save stats to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved stats to {output_path}")


def load_notes_from_vault(vault_path: Path) -> list[SynthesizedNote]:
    """
    Load existing notes from vault to extract topic information.

    This is a lightweight loader that only extracts tags for stats.
    """
    import re
    import yaml

    notes = []
    conversations_dir = vault_path / "conversations"

    if not conversations_dir.exists():
        return notes

    for file in conversations_dir.glob("*.md"):
        if file.name.startswith("_"):
            continue

        try:
            content = file.read_text(encoding="utf-8")

            # Extract YAML frontmatter
            match = re.match(r"^---\n(.+?)\n---", content, re.DOTALL)
            if match:
                frontmatter = yaml.safe_load(match.group(1))
                tags = frontmatter.get("tags", [])
                if isinstance(tags, list):
                    # Create minimal note for stats
                    note = SynthesizedNote(
                        conversation_id=file.stem,
                        title=frontmatter.get("title", ""),
                        date=frontmatter.get("date", ""),
                        participants=frontmatter.get("participants", []),
                        channel=frontmatter.get("channel", ""),
                        tags=tags,
                        related=frontmatter.get("related", []),
                        summary="",
                        core_idea="",
                        key_contributions="",
                        tension_points="",
                        connections="",
                    )
                    notes.append(note)
        except Exception as e:
            logger.warning(f"Failed to parse {file.name}: {e}")

    logger.info(f"Loaded {len(notes)} notes from vault")
    return notes


if __name__ == "__main__":
    # Quick test with sample data
    import sys

    logging.basicConfig(level=logging.INFO)

    # Create sample conversation
    from datetime import timezone

    msg1 = Message(
        id="1",
        author_id="u1",
        author_name="alice",
        content="Hello, this is a test message about trading strategies",
        timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
    )
    msg2 = Message(
        id="2",
        author_id="u2",
        author_name="bob",
        content="Interesting point about risk management",
        timestamp=datetime(2026, 1, 15, 10, 35, tzinfo=timezone.utc),
        reply_to="1",
    )
    msg3 = Message(
        id="3",
        author_id="u1",
        author_name="alice",
        content="Yes, let me elaborate on the Sharpe ratio calculation",
        timestamp=datetime(2026, 1, 15, 10, 40, tzinfo=timezone.utc),
        reply_to="2",
    )

    from .models import ConversationType

    conv = Conversation(
        id="test_conv",
        channel_id="123",
        channel_name="trading",
        messages=[msg1, msg2, msg3],
        conversation_type=ConversationType.CHANNEL,
    )

    stats = compute_all_stats([conv])
    print(json.dumps(stats, indent=2))

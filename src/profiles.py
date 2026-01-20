"""
Profiles module - generates participant profiles from conversation data.

Creates individual profile pages and a ranking index for community members.
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Conversation, SynthesizedNote
from .stats import compute_participant_stats, compute_interaction_graph

logger = logging.getLogger(__name__)


def generate_participant_profiles(
    conversations: list[Conversation],
    notes: list[SynthesizedNote],
    output_dir: Path,
) -> list[Path]:
    """
    Generate individual profile pages for all participants.

    Args:
        conversations: List of conversations
        notes: List of synthesized notes
        output_dir: Directory to write profiles (usually vault/participants/)

    Returns:
        List of paths to created profile files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute participant stats
    participant_stats = compute_participant_stats(conversations)
    interactions = compute_interaction_graph(conversations)

    # Build participant -> topics mapping from notes
    participant_topics: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    participant_notes: dict[str, list[str]] = defaultdict(list)

    for note in notes:
        for participant in note.participants:
            for tag in note.tags:
                participant_topics[participant][tag] += 1
            # Store note reference (filename without .md)
            filename = note.get_filename().replace(".md", "")
            participant_notes[participant].append(filename)

    # Generate profile for each participant
    created_files = []

    for username, stats in participant_stats.items():
        profile_path = output_dir / f"{_safe_filename(username)}.md"

        # Get top topics for this user
        topics = participant_topics.get(username, {})
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]

        # Get interactions
        user_interactions = interactions.get(username, {})
        top_interactions = sorted(user_interactions.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get recent notes
        user_notes = participant_notes.get(username, [])
        recent_notes = user_notes[-10:]  # Last 10

        # Generate markdown
        content = _generate_profile_markdown(
            username=username,
            stats=stats,
            top_topics=top_topics,
            top_interactions=top_interactions,
            recent_notes=recent_notes,
        )

        profile_path.write_text(content, encoding="utf-8")
        created_files.append(profile_path)

    logger.info(f"Generated {len(created_files)} participant profiles")

    # Generate index
    index_path = generate_participant_index(participant_stats, output_dir)
    created_files.append(index_path)

    return created_files


def _generate_profile_markdown(
    username: str,
    stats: dict,
    top_topics: list[tuple[str, int]],
    top_interactions: list[tuple[str, int]],
    recent_notes: list[str],
) -> str:
    """Generate markdown content for a participant profile."""
    lines = [
        "---",
        "type: participant-profile",
        f'username: "{username}"',
        f"messages: {stats['messages']}",
        f"conversations: {stats['conversations']}",
        f'first_seen: "{stats["first_seen"]}"',
        f'last_seen: "{stats["last_seen"]}"',
        "---",
        "",
        f"# {username}",
        "",
        "## Statistics",
        "",
        f"- **Messages:** {stats['messages']}",
        f"- **Conversations:** {stats['conversations']}",
        f"- **Avg message length:** {stats['avg_length']} characters",
        f"- **Reply ratio:** {stats['reply_ratio']:.1%}",
        f"- **Active since:** {stats['first_seen']}",
        f"- **Last seen:** {stats['last_seen']}",
        "",
    ]

    if top_topics:
        lines.extend([
            "## Frequent Topics",
            "",
        ])
        for tag, count in top_topics:
            lines.append(f"- [[{tag}]] ({count} conversations)")
        lines.append("")

    if top_interactions:
        lines.extend([
            "## Interactions",
            "",
            "*Who this user replies to most often:*",
            "",
        ])
        for other_user, count in top_interactions:
            lines.append(f"- **{other_user}**: {count} replies")
        lines.append("")

    if recent_notes:
        lines.extend([
            "## Recent Conversations",
            "",
        ])
        for note_name in reversed(recent_notes[-5:]):  # Most recent first
            lines.append(f"- [[{note_name}]]")
        lines.append("")

    lines.extend([
        "---",
        "",
        f"*Profile generated: {datetime.now().strftime('%Y-%m-%d')}*",
    ])

    return "\n".join(lines)


def generate_participant_index(
    participant_stats: dict[str, dict],
    output_dir: Path,
) -> Path:
    """
    Generate the main participant index with rankings.

    Args:
        participant_stats: Dict of username -> stats
        output_dir: Directory to write index

    Returns:
        Path to the created index file
    """
    index_path = output_dir / "_index.md"

    # Sort by message count for ranking
    ranked = sorted(
        participant_stats.items(),
        key=lambda x: x[1]["messages"],
        reverse=True
    )

    lines = [
        "---",
        "type: participant-index",
        f'updated: "{datetime.now().strftime("%Y-%m-%d")}"',
        "---",
        "",
        "# Community Contributors",
        "",
        f"Total participants: **{len(ranked)}**",
        "",
        "## Leaderboard",
        "",
        "| Rank | Participant | Messages | Conversations | Reply Ratio |",
        "|------|-------------|----------|---------------|-------------|",
    ]

    for i, (username, stats) in enumerate(ranked[:20], 1):  # Top 20
        safe_name = _safe_filename(username)
        lines.append(
            f"| {i} | [[{safe_name}\\|{username}]] | {stats['messages']} | "
            f"{stats['conversations']} | {stats['reply_ratio']:.1%} |"
        )

    lines.extend([
        "",
        "## All Participants",
        "",
    ])

    for username, stats in ranked:
        safe_name = _safe_filename(username)
        lines.append(f"- [[{safe_name}|{username}]] ({stats['messages']} messages)")

    lines.extend([
        "",
        "---",
        "",
        "*Auto-generated by Discord Conversation Synthesizer*",
    ])

    index_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated participant index: {index_path}")

    return index_path


def _safe_filename(username: str) -> str:
    """Convert username to filesystem-safe filename."""
    # Replace problematic characters
    safe = username.replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_").replace("*", "_")
    safe = safe.replace("?", "_").replace('"', "_")
    safe = safe.replace("<", "_").replace(">", "_")
    safe = safe.replace("|", "_")
    return safe


if __name__ == "__main__":
    # Quick test
    import sys
    from datetime import timezone

    logging.basicConfig(level=logging.INFO)

    # Create sample data
    from .models import Message, ConversationType

    msg1 = Message(
        id="1",
        author_id="u1",
        author_name="alice",
        content="Hello, this is a test about machine learning",
        timestamp=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
    )
    msg2 = Message(
        id="2",
        author_id="u2",
        author_name="bob",
        content="Great point about neural networks",
        timestamp=datetime(2026, 1, 15, 10, 35, tzinfo=timezone.utc),
        reply_to="1",
    )
    msg3 = Message(
        id="3",
        author_id="u1",
        author_name="alice",
        content="Yes, and also consider the risk management aspect",
        timestamp=datetime(2026, 1, 15, 10, 40, tzinfo=timezone.utc),
        reply_to="2",
    )

    conv = Conversation(
        id="test_conv",
        channel_id="123",
        channel_name="trading",
        messages=[msg1, msg2, msg3],
        conversation_type=ConversationType.CHANNEL,
    )

    note = SynthesizedNote(
        conversation_id="test_conv",
        title="Test Conversation",
        date="2026-01-15",
        participants=["alice", "bob"],
        channel="trading",
        tags=["machine-learning", "risk-management"],
        related=[],
        summary="A test conversation",
        core_idea="Test core idea",
        key_contributions="Test contributions",
        tension_points="Test tensions",
        connections="Test connections",
    )

    # Generate profiles
    output_dir = Path(__file__).parent.parent / "output" / "participants"
    files = generate_participant_profiles([conv], [note], output_dir)
    print(f"\nGenerated {len(files)} files:")
    for f in files:
        print(f"  - {f}")

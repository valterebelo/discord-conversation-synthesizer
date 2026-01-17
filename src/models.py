"""
Data models for the Discord Conversation Synthesizer.

These dataclasses define the core data structures used throughout the pipeline:
- Message: A single Discord message
- Conversation: A group of related messages
- SynthesizedNote: The output of Claude synthesis
- ProcessingState: Tracks incremental processing progress
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class ConversationType(Enum):
    """Type of conversation source."""
    THREAD = "thread"
    CHANNEL = "channel"


@dataclass
class Message:
    """
    A single Discord message.

    Attributes:
        id: Discord message ID (snowflake)
        author_id: Discord user ID
        author_name: Display name of the author
        content: Message text content
        timestamp: When the message was sent (UTC)
        reply_to: ID of the message this replies to (if any)
        attachments: List of attachment URLs
        embeds: List of embed data dictionaries
    """
    id: str
    author_id: str
    author_name: str
    content: str
    timestamp: datetime
    reply_to: Optional[str] = None
    attachments: list[str] = field(default_factory=list)
    embeds: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Ensure timestamp is timezone-aware UTC
        if self.timestamp.tzinfo is None:
            from datetime import timezone
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    @property
    def is_reply(self) -> bool:
        """Check if this message is a reply to another message."""
        return self.reply_to is not None

    def to_transcript_line(self) -> str:
        """Format message for inclusion in synthesis prompt."""
        timestamp_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{timestamp_str}] {self.author_name}: {self.content}"


@dataclass
class Conversation:
    """
    A group of related messages forming a coherent discussion.

    A conversation is the atomic unit for synthesis. It can come from:
    - A Discord thread (all messages in the thread)
    - A segment of main channel messages (bounded by time gaps or reply chains)

    Attributes:
        id: Unique identifier (channel_id + start_timestamp or thread_id)
        channel_id: Source channel ID
        channel_name: Human-readable channel name
        messages: List of messages in chronological order
        conversation_type: Whether this is from a thread or channel
        thread_id: Thread ID if from a thread
        thread_name: Thread name if from a thread
        timestamp_start: Earliest message timestamp
        timestamp_end: Latest message timestamp
    """
    id: str
    channel_id: str
    channel_name: str
    messages: list[Message]
    conversation_type: ConversationType
    thread_id: Optional[str] = None
    thread_name: Optional[str] = None

    @property
    def timestamp_start(self) -> datetime:
        """Get the timestamp of the first message."""
        if not self.messages:
            raise ValueError("Conversation has no messages")
        return min(m.timestamp for m in self.messages)

    @property
    def timestamp_end(self) -> datetime:
        """Get the timestamp of the last message."""
        if not self.messages:
            raise ValueError("Conversation has no messages")
        return max(m.timestamp for m in self.messages)

    @property
    def participants(self) -> list[str]:
        """Get unique list of participant usernames."""
        seen = set()
        result = []
        for msg in self.messages:
            if msg.author_name not in seen:
                seen.add(msg.author_name)
                result.append(msg.author_name)
        return result

    @property
    def message_count(self) -> int:
        """Get the number of messages in this conversation."""
        return len(self.messages)

    @property
    def duration_minutes(self) -> float:
        """Get the duration of the conversation in minutes."""
        if len(self.messages) < 2:
            return 0.0
        delta = self.timestamp_end - self.timestamp_start
        return delta.total_seconds() / 60

    def to_transcript(self) -> str:
        """
        Format the conversation as a transcript for the synthesis prompt.

        Returns a formatted string with metadata header and message lines.
        """
        header_parts = [
            f"CHANNEL: #{self.channel_name}",
        ]

        if self.thread_name:
            header_parts.append(f"THREAD: {self.thread_name}")
        else:
            header_parts.append("THREAD: main channel")

        date_start = self.timestamp_start.strftime("%Y-%m-%d %H:%M")
        date_end = self.timestamp_end.strftime("%Y-%m-%d %H:%M")
        header_parts.append(f"TIMESPAN: {date_start} to {date_end}")
        header_parts.append(f"PARTICIPANTS: {', '.join(self.participants)}")
        header_parts.append(f"MESSAGE COUNT: {self.message_count}")

        header = "\n".join(header_parts)

        # Sort messages by timestamp and format
        sorted_messages = sorted(self.messages, key=lambda m: m.timestamp)
        message_lines = [msg.to_transcript_line() for msg in sorted_messages]

        return f"{header}\n\n" + "\n".join(message_lines)

    def get_reply_chains(self) -> list[list[Message]]:
        """
        Extract reply chains from the conversation.

        Returns a list of chains, where each chain is a list of messages
        connected by replies (from root to leaf).
        """
        # Build a map of message_id -> message
        msg_map = {m.id: m for m in self.messages}

        # Find messages that are replies
        reply_to_map: dict[str, list[Message]] = {}
        for msg in self.messages:
            if msg.reply_to and msg.reply_to in msg_map:
                if msg.reply_to not in reply_to_map:
                    reply_to_map[msg.reply_to] = []
                reply_to_map[msg.reply_to].append(msg)

        # Find root messages (not replies, but have replies)
        chains = []
        visited = set()

        def build_chain(msg: Message, chain: list[Message]):
            chain.append(msg)
            visited.add(msg.id)
            if msg.id in reply_to_map:
                for reply in sorted(reply_to_map[msg.id], key=lambda m: m.timestamp):
                    if reply.id not in visited:
                        build_chain(reply, chain)

        for msg in sorted(self.messages, key=lambda m: m.timestamp):
            if msg.id not in visited and not msg.is_reply:
                if msg.id in reply_to_map:  # Has replies
                    chain: list[Message] = []
                    build_chain(msg, chain)
                    if len(chain) > 1:
                        chains.append(chain)

        return chains


@dataclass
class SynthesizedNote:
    """
    The output of Claude synthesis for a conversation.

    Contains the structured note content plus metadata about the synthesis.

    Attributes:
        conversation_id: ID of the source conversation
        title: Generated title for the note
        date: Date of the conversation (YYYY-MM-DD)
        participants: List of participant usernames
        channel: Source channel name
        tags: Extracted topic tags
        related: Suggested related topic links
        summary: 2-3 sentence overview
        core_idea: Main explanatory section
        key_contributions: Attributed contributions
        tension_points: Areas of disagreement
        connections: Related concepts
        raw_insights: Notable direct quotes (optional)
        token_usage: Tokens used for this synthesis
        cost_usd: Estimated cost in USD
        raw_response: The raw response from Claude (for debugging)
    """
    conversation_id: str
    title: str
    date: str
    participants: list[str]
    channel: str
    tags: list[str]
    related: list[str]
    summary: str
    core_idea: str
    key_contributions: str
    tension_points: str
    connections: str
    raw_insights: Optional[str] = None
    token_usage: int = 0
    cost_usd: float = 0.0
    raw_response: Optional[str] = None

    def to_markdown(self) -> str:
        """
        Render the note as Obsidian-compatible Markdown.

        Includes YAML frontmatter and all sections.
        """
        # Build YAML frontmatter
        frontmatter_lines = [
            "---",
            f'title: "{self.title}"',
            f'date: "{self.date}"',
            f'participants: {self.participants}',
            f'channel: "{self.channel}"',
            f'tags: {self.tags}',
            f'related: {self.related}',
            "---",
        ]
        frontmatter = "\n".join(frontmatter_lines)

        # Build body
        body_parts = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## The Core Idea",
            "",
            self.core_idea,
            "",
            "## Key Contributions",
            "",
            self.key_contributions,
            "",
            "## Points of Tension",
            "",
            self.tension_points,
            "",
            "## Connections",
            "",
            self.connections,
        ]

        if self.raw_insights:
            body_parts.extend([
                "",
                "## Raw Insights",
                "",
                self.raw_insights,
            ])

        body = "\n".join(body_parts)

        return f"{frontmatter}\n\n{body}"

    def get_filename(self) -> str:
        """
        Generate the filename for this note.

        Format: YYYY-MM-DD_slugified-title.md
        """
        # Simple slugify implementation (no external dependency)
        import re
        import unicodedata

        # Take first 50 chars of title
        text = self.title[:50]

        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Convert to lowercase and replace spaces/special chars with hyphens
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text).strip('-')

        # Limit length
        if len(text) > 50:
            text = text[:50].rsplit('-', 1)[0]

        return f"{self.date}_{text}.md"


@dataclass
class ChannelState:
    """Tracks processing state for a single channel."""
    channel_id: str
    channel_name: str
    last_processed_message_id: Optional[str] = None
    last_processed_timestamp: Optional[datetime] = None
    total_messages_processed: int = 0
    total_conversations_processed: int = 0


@dataclass
class ProcessingState:
    """
    Tracks incremental processing progress across runs.

    Persisted to _meta/processing-state.json between runs.
    """
    version: str = "1.0"
    last_run: Optional[datetime] = None
    channels: dict[str, ChannelState] = field(default_factory=dict)
    total_conversations_processed: int = 0
    total_messages_processed: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0

    def get_channel_state(self, channel_id: str) -> Optional[ChannelState]:
        """Get state for a specific channel."""
        return self.channels.get(channel_id)

    def update_channel_state(
        self,
        channel_id: str,
        channel_name: str,
        last_message_id: str,
        last_timestamp: datetime,
        messages_processed: int,
        conversations_processed: int
    ) -> None:
        """Update state after processing a channel."""
        if channel_id not in self.channels:
            self.channels[channel_id] = ChannelState(
                channel_id=channel_id,
                channel_name=channel_name
            )

        state = self.channels[channel_id]
        state.last_processed_message_id = last_message_id
        state.last_processed_timestamp = last_timestamp
        state.total_messages_processed += messages_processed
        state.total_conversations_processed += conversations_processed

        self.total_messages_processed += messages_processed
        self.total_conversations_processed += conversations_processed

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "channels": {
                cid: {
                    "channel_id": cs.channel_id,
                    "channel_name": cs.channel_name,
                    "last_processed_message_id": cs.last_processed_message_id,
                    "last_processed_timestamp": cs.last_processed_timestamp.isoformat() if cs.last_processed_timestamp else None,
                    "total_messages_processed": cs.total_messages_processed,
                    "total_conversations_processed": cs.total_conversations_processed,
                }
                for cid, cs in self.channels.items()
            },
            "stats": {
                "total_conversations_processed": self.total_conversations_processed,
                "total_messages_processed": self.total_messages_processed,
                "total_tokens_used": self.total_tokens_used,
                "total_cost_usd": self.total_cost_usd,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        """Deserialize from dictionary."""
        state = cls(
            version=data.get("version", "1.0"),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
        )

        stats = data.get("stats", {})
        state.total_conversations_processed = stats.get("total_conversations_processed", 0)
        state.total_messages_processed = stats.get("total_messages_processed", 0)
        state.total_tokens_used = stats.get("total_tokens_used", 0)
        state.total_cost_usd = stats.get("total_cost_usd", 0.0)

        for cid, cs_data in data.get("channels", {}).items():
            state.channels[cid] = ChannelState(
                channel_id=cs_data["channel_id"],
                channel_name=cs_data["channel_name"],
                last_processed_message_id=cs_data.get("last_processed_message_id"),
                last_processed_timestamp=datetime.fromisoformat(cs_data["last_processed_timestamp"]) if cs_data.get("last_processed_timestamp") else None,
                total_messages_processed=cs_data.get("total_messages_processed", 0),
                total_conversations_processed=cs_data.get("total_conversations_processed", 0),
            )

        return state


@dataclass
class RunResult:
    """Result of a single processing run."""
    started_at: datetime
    completed_at: datetime
    conversations_processed: int
    messages_processed: int
    tokens_used: int
    cost_usd: float
    errors: list[str] = field(default_factory=list)
    notes_created: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "conversations_processed": self.conversations_processed,
            "messages_processed": self.messages_processed,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "errors": self.errors,
            "notes_created": self.notes_created,
            "success": self.success,
        }

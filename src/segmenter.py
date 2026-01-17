"""
Segmenter module - groups messages into coherent conversations.

Implements the segmentation rules from the PRD:
1. Thread Isolation: Each thread = exactly 1 conversation
2. Temporal Gap Breaking: Gap > 24h = conversation boundary
3. Reply Chain Grouping: Connected replies belong together
4. Channel Continuity: Messages within time window form conversations
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from collections import defaultdict

from .models import Message, Conversation, ConversationType

logger = logging.getLogger(__name__)


class Segmenter:
    """
    Segments a stream of messages into discrete conversations.

    The segmenter applies rules in order:
    1. Threads are isolated (1 thread = 1 conversation)
    2. Temporal gaps break conversations
    3. Reply chains are grouped together
    4. Remaining messages form time-windowed conversations
    """

    def __init__(
        self,
        temporal_gap_hours: float = 24.0,
        min_messages: int = 3,
        excluded_user_ids: Optional[set[str]] = None,
        redaction_placeholder: str = "[message redacted]",
    ):
        """
        Initialize the segmenter.

        Args:
            temporal_gap_hours: Hours of silence that breaks a conversation
            min_messages: Minimum messages for a conversation to be kept
            excluded_user_ids: User IDs to redact from conversations
            redaction_placeholder: Text to replace redacted messages with
        """
        self.temporal_gap_hours = temporal_gap_hours
        self.min_messages = min_messages
        self.excluded_user_ids = excluded_user_ids or set()
        self.redaction_placeholder = redaction_placeholder

    def segment_thread(
        self,
        messages: list[Message],
        channel_id: str,
        channel_name: str,
        thread_id: str,
        thread_name: str,
    ) -> Optional[Conversation]:
        """
        Create a conversation from a thread's messages.

        A thread is always exactly one conversation, regardless of
        temporal gaps or topic drift within it.

        Args:
            messages: All messages in the thread
            channel_id: Parent channel ID
            channel_name: Parent channel name
            thread_id: Thread ID
            thread_name: Thread name

        Returns:
            Conversation if enough messages, None otherwise
        """
        if not messages:
            logger.debug(f"Thread {thread_name} has no messages, skipping")
            return None

        # Apply redactions
        processed_messages = self._apply_redactions(messages)

        # Check minimum message threshold
        if len(processed_messages) < self.min_messages:
            logger.debug(
                f"Thread {thread_name} has {len(processed_messages)} messages, "
                f"below threshold of {self.min_messages}, skipping"
            )
            return None

        # Create conversation ID from thread ID
        conv_id = f"thread_{thread_id}"

        conversation = Conversation(
            id=conv_id,
            channel_id=channel_id,
            channel_name=channel_name,
            messages=processed_messages,
            conversation_type=ConversationType.THREAD,
            thread_id=thread_id,
            thread_name=thread_name,
        )

        logger.info(
            f"Created thread conversation: {thread_name} "
            f"({len(processed_messages)} messages, "
            f"{len(conversation.participants)} participants)"
        )

        return conversation

    def segment_channel(
        self,
        messages: list[Message],
        channel_id: str,
        channel_name: str,
    ) -> list[Conversation]:
        """
        Segment channel messages into conversations.

        Applies temporal gap breaking and reply chain grouping.

        Args:
            messages: All messages from the channel (not in threads)
            channel_id: Channel ID
            channel_name: Channel name

        Returns:
            List of conversations extracted from the channel
        """
        if not messages:
            logger.debug(f"Channel #{channel_name} has no messages")
            return []

        # Sort by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        # Apply redactions
        sorted_messages = self._apply_redactions(sorted_messages)

        # Step 1: Build reply chain groups
        reply_groups = self._build_reply_groups(sorted_messages)

        # Step 2: Split by temporal gaps, respecting reply groups
        conversations = self._split_by_temporal_gaps(
            sorted_messages,
            reply_groups,
            channel_id,
            channel_name,
        )

        # Step 3: Filter by minimum message threshold
        filtered = []
        for conv in conversations:
            if len(conv.messages) >= self.min_messages:
                filtered.append(conv)
            else:
                logger.debug(
                    f"Dropping conversation with {len(conv.messages)} messages "
                    f"(below threshold of {self.min_messages})"
                )

        logger.info(
            f"Segmented #{channel_name}: {len(messages)} messages -> "
            f"{len(filtered)} conversations"
        )

        return filtered

    def _apply_redactions(self, messages: list[Message]) -> list[Message]:
        """
        Apply redactions for excluded users.

        Replaces message content but preserves message structure
        to maintain conversation flow.
        """
        if not self.excluded_user_ids:
            return messages

        result = []
        for msg in messages:
            if msg.author_id in self.excluded_user_ids:
                # Create redacted copy
                redacted = Message(
                    id=msg.id,
                    author_id=msg.author_id,
                    author_name="[redacted]",
                    content=self.redaction_placeholder,
                    timestamp=msg.timestamp,
                    reply_to=msg.reply_to,
                    attachments=[],
                    embeds=[],
                )
                result.append(redacted)
                logger.debug(f"Redacted message from user {msg.author_id}")
            else:
                result.append(msg)

        return result

    def _build_reply_groups(
        self,
        messages: list[Message]
    ) -> dict[str, set[str]]:
        """
        Build groups of message IDs connected by replies.

        Returns a dict mapping each message ID to the set of all
        message IDs in its reply group.
        """
        # Build adjacency: message_id -> set of connected message_ids
        connections: dict[str, set[str]] = defaultdict(set)

        msg_ids = {m.id for m in messages}

        for msg in messages:
            if msg.reply_to and msg.reply_to in msg_ids:
                # Bidirectional connection
                connections[msg.id].add(msg.reply_to)
                connections[msg.reply_to].add(msg.id)

        # Find connected components using BFS
        visited: set[str] = set()
        groups: dict[str, set[str]] = {}

        def bfs(start_id: str) -> set[str]:
            component = {start_id}
            queue = [start_id]
            while queue:
                current = queue.pop(0)
                for neighbor in connections[current]:
                    if neighbor not in component:
                        component.add(neighbor)
                        queue.append(neighbor)
            return component

        for msg in messages:
            if msg.id not in visited and msg.id in connections:
                component = bfs(msg.id)
                visited.update(component)
                for mid in component:
                    groups[mid] = component

        return groups

    def _split_by_temporal_gaps(
        self,
        messages: list[Message],
        reply_groups: dict[str, set[str]],
        channel_id: str,
        channel_name: str,
    ) -> list[Conversation]:
        """
        Split messages into conversations based on temporal gaps.

        Respects reply groups: if a message is in a reply group,
        all messages in that group stay in the same conversation.
        """
        if not messages:
            return []

        gap_threshold = timedelta(hours=self.temporal_gap_hours)
        conversations: list[Conversation] = []

        current_messages: list[Message] = []
        current_msg_ids: set[str] = set()

        def finalize_conversation():
            if current_messages:
                # Sort by timestamp
                sorted_msgs = sorted(current_messages, key=lambda m: m.timestamp)
                conv_id = f"channel_{channel_id}_{sorted_msgs[0].timestamp.strftime('%Y%m%d_%H%M%S')}"

                conv = Conversation(
                    id=conv_id,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    messages=sorted_msgs,
                    conversation_type=ConversationType.CHANNEL,
                )
                conversations.append(conv)

        prev_timestamp: Optional[datetime] = None

        for msg in messages:
            # Check for temporal gap
            if prev_timestamp:
                gap = msg.timestamp - prev_timestamp
                if gap > gap_threshold:
                    # Check if this message is connected to current conversation via reply
                    if msg.id in reply_groups:
                        group = reply_groups[msg.id]
                        # If any message in the reply group is already in current conversation
                        if group & current_msg_ids:
                            # Keep in current conversation despite gap
                            logger.debug(
                                f"Keeping message in conversation due to reply chain "
                                f"despite {gap.total_seconds()/3600:.1f}h gap"
                            )
                        else:
                            # Start new conversation
                            finalize_conversation()
                            current_messages = []
                            current_msg_ids = set()
                    else:
                        # No reply connection, start new conversation
                        finalize_conversation()
                        current_messages = []
                        current_msg_ids = set()

            current_messages.append(msg)
            current_msg_ids.add(msg.id)

            # If this message is in a reply group, add all group members
            if msg.id in reply_groups:
                group = reply_groups[msg.id]
                for mid in group:
                    if mid not in current_msg_ids:
                        # Find and add the message
                        for m in messages:
                            if m.id == mid:
                                current_messages.append(m)
                                current_msg_ids.add(mid)
                                break

            prev_timestamp = msg.timestamp

        # Finalize last conversation
        finalize_conversation()

        return conversations

    def segment_all(
        self,
        channel_messages: list[Message],
        thread_data: list[dict],
        channel_id: str,
        channel_name: str,
    ) -> list[Conversation]:
        """
        Segment all messages from a channel including threads.

        Args:
            channel_messages: Messages from the main channel
            thread_data: List of dicts with thread_id, thread_name, messages
            channel_id: Channel ID
            channel_name: Channel name

        Returns:
            All conversations (from threads and main channel)
        """
        all_conversations: list[Conversation] = []

        # Process threads first (each thread = 1 conversation)
        for thread in thread_data:
            conv = self.segment_thread(
                messages=thread["messages"],
                channel_id=channel_id,
                channel_name=channel_name,
                thread_id=thread["thread_id"],
                thread_name=thread["thread_name"],
            )
            if conv:
                all_conversations.append(conv)

        # Process main channel messages
        channel_convs = self.segment_channel(
            messages=channel_messages,
            channel_id=channel_id,
            channel_name=channel_name,
        )
        all_conversations.extend(channel_convs)

        logger.info(
            f"Total conversations from #{channel_name}: {len(all_conversations)} "
            f"({len(thread_data)} threads, {len(channel_convs)} channel segments)"
        )

        return all_conversations


def segment_from_fixtures(fixture_path: str, config: Optional[dict] = None) -> list[Conversation]:
    """
    Convenience function to segment conversations directly from fixtures.

    Useful for testing the segmentation logic.

    Args:
        fixture_path: Path to the fixture JSON file
        config: Optional config dict with segmentation parameters

    Returns:
        List of all conversations extracted
    """
    import json
    from pathlib import Path

    with open(fixture_path) as f:
        data = json.load(f)

    test_config = data.get("test_config", {})
    if config:
        test_config.update(config)

    segmenter = Segmenter(
        temporal_gap_hours=test_config.get("temporal_gap_hours", 24),
        min_messages=test_config.get("min_messages_per_conversation", 3),
        excluded_user_ids=set(test_config.get("excluded_user_ids", [])),
        redaction_placeholder=test_config.get("redaction_placeholder", "[message redacted]"),
    )

    all_conversations: list[Conversation] = []

    for conv_data in data.get("conversations", []):
        # Convert raw messages to Message objects
        messages = []
        for msg_data in conv_data.get("messages", []):
            msg = Message(
                id=msg_data["id"],
                author_id=msg_data["author_id"],
                author_name=msg_data["author_name"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(
                    msg_data["timestamp"].replace("Z", "+00:00")
                ),
                reply_to=msg_data.get("reply_to"),
            )
            messages.append(msg)

        if conv_data.get("type") == "thread":
            conv = segmenter.segment_thread(
                messages=messages,
                channel_id=conv_data["channel_id"],
                channel_name=conv_data["channel_name"],
                thread_id=conv_data["thread_id"],
                thread_name=conv_data["thread_name"],
            )
            if conv:
                all_conversations.append(conv)
        else:
            convs = segmenter.segment_channel(
                messages=messages,
                channel_id=conv_data["channel_id"],
                channel_name=conv_data["channel_name"],
            )
            all_conversations.extend(convs)

    return all_conversations


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    fixture_path = Path(__file__).parent.parent / "tests/fixtures/sample_conversations.json"

    print("=== Testing Segmenter ===\n")

    conversations = segment_from_fixtures(str(fixture_path))

    print(f"\nExtracted {len(conversations)} conversations:\n")

    for conv in conversations:
        print(f"  {conv.id}")
        print(f"    Type: {conv.conversation_type.value}")
        print(f"    Messages: {conv.message_count}")
        print(f"    Participants: {', '.join(conv.participants)}")
        print(f"    Duration: {conv.duration_minutes:.1f} minutes")
        if conv.thread_name:
            print(f"    Thread: {conv.thread_name}")
        print()

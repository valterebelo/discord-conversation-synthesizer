"""
Fetcher module - retrieves messages from Discord.

Handles:
- Connecting to Discord via bot token
- Fetching message history from channels
- Resolving threads and reply chains
- Respecting rate limits
- Incremental fetching (only new messages since last run)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Protocol, AsyncIterator
from dataclasses import dataclass

from .models import Message, ChannelState

logger = logging.getLogger(__name__)


class DiscordClientProtocol(Protocol):
    """Protocol defining the interface we need from a Discord client."""

    async def fetch_channel(self, channel_id: int): ...
    async def fetch_guild(self, guild_id: int): ...
    async def close(self): ...


@dataclass
class FetchResult:
    """Result of fetching messages from a channel."""
    channel_id: str
    channel_name: str
    messages: list[Message]
    threads: list["ThreadInfo"]
    oldest_message_id: Optional[str] = None
    newest_message_id: Optional[str] = None

    @property
    def message_count(self) -> int:
        return len(self.messages)


@dataclass
class ThreadInfo:
    """Information about a Discord thread."""
    id: str
    name: str
    parent_channel_id: str
    message_count: int
    created_at: Optional[datetime] = None


class Fetcher:
    """
    Fetches messages from Discord channels and threads.

    Supports both real Discord client and mock client for testing.
    """

    def __init__(
        self,
        client,  # Discord client (real or mock)
        guild_id: str,
        rate_limit_delay: float = 0.5,  # Seconds between API calls
    ):
        self.client = client
        self.guild_id = guild_id
        self.rate_limit_delay = rate_limit_delay
        self._is_mock = hasattr(client, 'get_conversations_raw')

    async def fetch_channel_messages(
        self,
        channel_id: str,
        channel_name: str,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> FetchResult:
        """
        Fetch messages from a channel.

        Args:
            channel_id: The channel to fetch from
            channel_name: Human-readable channel name
            after: Only fetch messages after this time
            before: Only fetch messages before this time
            limit: Maximum number of messages to fetch

        Returns:
            FetchResult with messages and thread info
        """
        logger.info(f"Fetching messages from #{channel_name} ({channel_id})")

        if self._is_mock:
            return await self._fetch_from_mock(channel_id, channel_name, after, before, limit)
        else:
            return await self._fetch_from_discord(channel_id, channel_name, after, before, limit)

    async def _fetch_from_mock(
        self,
        channel_id: str,
        channel_name: str,
        after: Optional[datetime],
        before: Optional[datetime],
        limit: Optional[int],
    ) -> FetchResult:
        """Fetch from mock client (for testing)."""
        guild = self.client.get_guild(self.guild_id)
        if not guild:
            logger.error(f"Guild {self.guild_id} not found in mock")
            return FetchResult(
                channel_id=channel_id,
                channel_name=channel_name,
                messages=[],
                threads=[]
            )

        channel = guild.get_channel(channel_id)
        if not channel:
            logger.error(f"Channel {channel_id} not found in mock")
            return FetchResult(
                channel_id=channel_id,
                channel_name=channel_name,
                messages=[],
                threads=[]
            )

        # Convert mock messages to our Message model
        messages = []
        for mock_msg in channel._messages:
            # Apply time filters
            if after and mock_msg.created_at <= after:
                continue
            if before and mock_msg.created_at >= before:
                continue

            msg = Message(
                id=mock_msg.id,
                author_id=mock_msg.author.id,
                author_name=mock_msg.author.name,
                content=mock_msg.content,
                timestamp=mock_msg.created_at,
                reply_to=mock_msg.reply_to_id,
            )
            messages.append(msg)

        # Apply limit
        if limit:
            messages = messages[:limit]

        # Get thread info
        threads = [
            ThreadInfo(
                id=t.id,
                name=t.name,
                parent_channel_id=t.parent_id,
                message_count=0,  # Would need to count from messages
            )
            for t in channel.threads
        ]

        logger.info(f"Fetched {len(messages)} messages, {len(threads)} threads from mock")

        oldest_id = min((m.id for m in messages), default=None)
        newest_id = max((m.id for m in messages), default=None)

        return FetchResult(
            channel_id=channel_id,
            channel_name=channel_name,
            messages=messages,
            threads=threads,
            oldest_message_id=oldest_id,
            newest_message_id=newest_id,
        )

    async def _fetch_from_discord(
        self,
        channel_id: str,
        channel_name: str,
        after: Optional[datetime],
        before: Optional[datetime],
        limit: Optional[int],
    ) -> FetchResult:
        """Fetch from real Discord API."""
        try:
            import discord
        except ImportError:
            raise RuntimeError("discord.py is required for real Discord fetching. Install with: pip install discord.py")

        channel = await self.client.fetch_channel(int(channel_id))

        messages = []
        threads = []

        # Fetch messages with pagination
        fetch_limit = limit or 10000  # Discord max is 100 per call
        fetched = 0

        # Convert datetime to discord snowflake if needed
        after_obj = discord.Object(id=int(after.timestamp() * 1000 - 1420070400000) << 22) if after else None

        async for msg in channel.history(limit=fetch_limit, after=after_obj, oldest_first=True):
            if before and msg.created_at >= before:
                break

            message = Message(
                id=str(msg.id),
                author_id=str(msg.author.id),
                author_name=msg.author.display_name,
                content=msg.content,
                timestamp=msg.created_at.replace(tzinfo=timezone.utc),
                reply_to=str(msg.reference.message_id) if msg.reference else None,
                attachments=[a.url for a in msg.attachments],
                embeds=[e.to_dict() for e in msg.embeds],
            )
            messages.append(message)
            fetched += 1

            # Rate limiting
            if fetched % 100 == 0:
                logger.debug(f"Fetched {fetched} messages, pausing for rate limit")
                await asyncio.sleep(self.rate_limit_delay)

        # Fetch threads
        if hasattr(channel, 'threads'):
            for thread in channel.threads:
                threads.append(ThreadInfo(
                    id=str(thread.id),
                    name=thread.name,
                    parent_channel_id=channel_id,
                    message_count=thread.message_count,
                    created_at=thread.created_at,
                ))

        # Also fetch archived threads
        if hasattr(channel, 'archived_threads'):
            async for thread in channel.archived_threads(limit=100):
                threads.append(ThreadInfo(
                    id=str(thread.id),
                    name=thread.name,
                    parent_channel_id=channel_id,
                    message_count=thread.message_count,
                    created_at=thread.created_at,
                ))

        logger.info(f"Fetched {len(messages)} messages, {len(threads)} threads from Discord")

        oldest_id = min((m.id for m in messages), default=None)
        newest_id = max((m.id for m in messages), default=None)

        return FetchResult(
            channel_id=channel_id,
            channel_name=channel_name,
            messages=messages,
            threads=threads,
            oldest_message_id=oldest_id,
            newest_message_id=newest_id,
        )

    async def fetch_thread_messages(
        self,
        thread_id: str,
        thread_name: str,
        parent_channel_id: str,
    ) -> list[Message]:
        """
        Fetch all messages from a thread.

        Args:
            thread_id: The thread ID
            thread_name: Human-readable thread name
            parent_channel_id: The parent channel ID

        Returns:
            List of messages in the thread
        """
        logger.info(f"Fetching thread: {thread_name} ({thread_id})")

        if self._is_mock:
            return await self._fetch_thread_from_mock(thread_id, parent_channel_id)
        else:
            return await self._fetch_thread_from_discord(thread_id)

    async def _fetch_thread_from_mock(
        self,
        thread_id: str,
        parent_channel_id: str,
    ) -> list[Message]:
        """Fetch thread messages from mock."""
        # In our mock, thread messages are stored with the channel
        # and identified by having a matching thread_id in the conversation
        raw_convos = self.client.get_conversations_raw()

        for conv in raw_convos:
            if conv.get("thread_id") == thread_id:
                messages = []
                for msg_data in conv.get("messages", []):
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
                return messages

        return []

    async def _fetch_thread_from_discord(self, thread_id: str) -> list[Message]:
        """Fetch thread messages from real Discord."""
        thread = await self.client.fetch_channel(int(thread_id))

        messages = []
        async for msg in thread.history(limit=10000, oldest_first=True):
            message = Message(
                id=str(msg.id),
                author_id=str(msg.author.id),
                author_name=msg.author.display_name,
                content=msg.content,
                timestamp=msg.created_at.replace(tzinfo=timezone.utc),
                reply_to=str(msg.reference.message_id) if msg.reference else None,
                attachments=[a.url for a in msg.attachments],
                embeds=[e.to_dict() for e in msg.embeds],
            )
            messages.append(message)

        return messages

    async def fetch_all_configured_channels(
        self,
        channel_configs: list[dict],
        state: Optional[dict[str, ChannelState]] = None,
    ) -> list[FetchResult]:
        """
        Fetch messages from all configured channels.

        Args:
            channel_configs: List of channel configs with id, name, enabled
            state: Previous processing state for incremental fetching

        Returns:
            List of FetchResults for each enabled channel
        """
        results = []

        for config in channel_configs:
            if not config.get("enabled", True):
                logger.debug(f"Skipping disabled channel: {config.get('name')}")
                continue

            channel_id = config["id"]
            channel_name = config.get("name", channel_id)

            # Determine 'after' timestamp for incremental fetch
            after = None
            if state and channel_id in state:
                channel_state = state[channel_id]
                after = channel_state.last_processed_timestamp
                logger.info(f"Incremental fetch for #{channel_name} after {after}")

            result = await self.fetch_channel_messages(
                channel_id=channel_id,
                channel_name=channel_name,
                after=after,
            )
            results.append(result)

            # Small delay between channels
            await asyncio.sleep(self.rate_limit_delay)

        return results


async def create_fetcher(
    use_mock: bool = True,
    guild_id: str = "",
    token: Optional[str] = None,
    fixture_path: Optional[str] = None,
) -> Fetcher:
    """
    Factory function to create a Fetcher with appropriate client.

    Args:
        use_mock: If True, use mock client
        guild_id: Discord guild/server ID
        token: Discord bot token (for real client)
        fixture_path: Path to fixture file (for mock client)

    Returns:
        Configured Fetcher instance
    """
    if use_mock:
        from .mocks.discord_mock import create_mock_client
        client = create_mock_client(fixture_path or "tests/fixtures/sample_conversations.json")
        return Fetcher(client, guild_id or "123456789012345678")
    else:
        import discord
        import os

        token = token or os.environ.get("DISCORD_TOKEN")
        if not token:
            raise ValueError("DISCORD_TOKEN environment variable required for real Discord")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        client = discord.Client(intents=intents)

        # Login to Discord (without starting the full event loop)
        await client.login(token)
        logger.info("Discord client logged in successfully")

        return Fetcher(client, guild_id)

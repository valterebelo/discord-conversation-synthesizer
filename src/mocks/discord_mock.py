"""
Mock Discord client for testing without hitting the Discord API.

Loads conversations from fixture files and simulates the discord.py interface.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class MockUser:
    id: str
    name: str

    @property
    def display_name(self) -> str:
        return self.name


@dataclass
class MockMessage:
    id: str
    author: MockUser
    content: str
    created_at: datetime
    reference: Optional["MockMessageReference"] = None

    @property
    def reply_to_id(self) -> Optional[str]:
        return self.reference.message_id if self.reference else None


@dataclass
class MockMessageReference:
    message_id: str


@dataclass
class MockThread:
    id: str
    name: str
    parent_id: str


@dataclass
class MockChannel:
    id: str
    name: str
    guild_id: str
    threads: list[MockThread]

    async def history(self, limit: int = 100, after: Optional[datetime] = None) -> list[MockMessage]:
        """Returns messages from this channel (loaded from fixtures)."""
        # This is populated by MockDiscordClient
        return self._messages

    def __init__(self, id: str, name: str, guild_id: str):
        self.id = id
        self.name = name
        self.guild_id = guild_id
        self.threads = []
        self._messages = []


@dataclass
class MockGuild:
    id: str
    name: str
    channels: list[MockChannel]

    def get_channel(self, channel_id: str) -> Optional[MockChannel]:
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        return None


class MockDiscordClient:
    """
    A mock Discord client that loads test data from fixture files.

    Usage:
        client = MockDiscordClient()
        client.load_fixtures("tests/fixtures/sample_conversations.json")

        # Now use like a real discord client
        guild = client.get_guild("123456789")
        channel = guild.get_channel("987654321")
        messages = await channel.history(limit=100)
    """

    def __init__(self):
        self.guilds: dict[str, MockGuild] = {}
        self._conversations: list[dict] = []
        self._fixture_path: Optional[Path] = None

    def load_fixtures(self, fixture_path: str | Path) -> None:
        """Load test conversations from a JSON fixture file."""
        self._fixture_path = Path(fixture_path)

        with open(self._fixture_path) as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        server_id = metadata.get("server_id", "mock_server")
        server_name = metadata.get("server_name", "Mock Server")

        # Create the guild
        guild = MockGuild(
            id=server_id,
            name=server_name,
            channels=[]
        )

        # Build channels from conversations
        channels_by_id: dict[str, MockChannel] = {}

        for conv in data.get("conversations", []):
            channel_id = conv["channel_id"]
            channel_name = conv["channel_name"]

            if channel_id not in channels_by_id:
                channel = MockChannel(
                    id=channel_id,
                    name=channel_name,
                    guild_id=server_id
                )
                channels_by_id[channel_id] = channel
                guild.channels.append(channel)

            channel = channels_by_id[channel_id]

            # If it's a thread, add thread info
            if conv.get("thread_id"):
                thread = MockThread(
                    id=conv["thread_id"],
                    name=conv.get("thread_name", "Unnamed Thread"),
                    parent_id=channel_id
                )
                channel.threads.append(thread)

            # Convert messages
            for msg_data in conv.get("messages", []):
                user = MockUser(
                    id=msg_data["author_id"],
                    name=msg_data["author_name"]
                )

                reference = None
                if msg_data.get("reply_to"):
                    reference = MockMessageReference(message_id=msg_data["reply_to"])

                message = MockMessage(
                    id=msg_data["id"],
                    author=user,
                    content=msg_data["content"],
                    created_at=datetime.fromisoformat(msg_data["timestamp"].replace("Z", "+00:00")),
                    reference=reference
                )

                channel._messages.append(message)

        self.guilds[server_id] = guild
        self._conversations = data.get("conversations", [])

        print(f"[MockDiscordClient] Loaded {len(self._conversations)} conversations from {fixture_path}")
        print(f"[MockDiscordClient] Server: {server_name} ({server_id})")
        print(f"[MockDiscordClient] Channels: {[c.name for c in guild.channels]}")

    def get_guild(self, guild_id: str) -> Optional[MockGuild]:
        return self.guilds.get(guild_id)

    def get_conversations_raw(self) -> list[dict]:
        """Get the raw conversation data (for testing segmentation directly)."""
        return self._conversations

    async def close(self) -> None:
        """Mock close - does nothing."""
        pass


# Convenience function for quick testing
def create_mock_client(fixture_path: str = "tests/fixtures/sample_conversations.json") -> MockDiscordClient:
    """Create and load a mock client with fixtures."""
    client = MockDiscordClient()
    client.load_fixtures(fixture_path)
    return client


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test():
        client = create_mock_client()

        guild = client.get_guild("123456789012345678")
        print(f"\nGuild: {guild.name}")

        for channel in guild.channels:
            print(f"\nChannel: #{channel.name}")
            print(f"  Threads: {[t.name for t in channel.threads]}")
            print(f"  Messages: {len(channel._messages)}")

    asyncio.run(test())

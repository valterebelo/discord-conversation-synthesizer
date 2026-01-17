"""
Configuration loader for the Discord Conversation Synthesizer.

Handles:
- Loading YAML configuration
- Environment variable substitution
- Default values
- Validation
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""
    id: str
    name: str
    enabled: bool = True


@dataclass
class ServerConfig:
    """Configuration for the Discord server."""
    id: str
    name: str
    channels: list[ChannelConfig] = field(default_factory=list)


@dataclass
class SegmentationConfig:
    """Configuration for conversation segmentation."""
    temporal_gap_hours: float = 24.0
    min_messages_per_conversation: int = 3


@dataclass
class SynthesisConfig:
    """Configuration for Claude synthesis."""
    model: str = "claude-opus-4-5-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3
    prompt_file: Optional[str] = None


@dataclass
class ExportConfig:
    """Configuration for export to Obsidian."""
    vault_path: str = "./output"
    filename_format: str = "{date}_{slug}.md"
    generate_topic_indexes: bool = True
    archive_versions: bool = True


@dataclass
class PrivacyConfig:
    """Configuration for privacy and redaction."""
    excluded_user_ids: list[str] = field(default_factory=list)
    redaction_placeholder: str = "[message redacted]"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class Config:
    """Complete configuration for the synthesizer."""
    discord_token: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    server: ServerConfig = field(default_factory=lambda: ServerConfig(id="", name=""))
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Runtime flags (not from config file)
    use_mock: bool = True
    dry_run: bool = False
    verbose: bool = False

    def get_enabled_channels(self) -> list[ChannelConfig]:
        """Get list of enabled channels."""
        return [c for c in self.server.channels if c.enabled]


def _substitute_env_vars(value: str) -> str:
    """
    Substitute environment variables in a string.

    Supports ${VAR_NAME} syntax.
    """
    pattern = r'\$\{([^}]+)\}'

    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(pattern, replacer, value)


def _parse_yaml(content: str) -> dict:
    """
    Parse YAML content using PyYAML.
    """
    return yaml.safe_load(content) or {}


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Parsed Config object
    """
    config = Config()

    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            content = path.read_text()
            data = _parse_yaml(content)

            # Parse Discord section
            if 'discord' in data:
                token = data['discord'].get('token', '')
                if token:
                    config.discord_token = _substitute_env_vars(token)

            # Parse server section
            if 'server' in data:
                server_data = data['server']
                channels = []

                for ch in server_data.get('channels', []):
                    if isinstance(ch, dict):
                        channels.append(ChannelConfig(
                            id=str(ch.get('id', '')),
                            name=ch.get('name', ''),
                            enabled=ch.get('enabled', True),
                        ))

                config.server = ServerConfig(
                    id=str(server_data.get('id', '')),
                    name=server_data.get('name', ''),
                    channels=channels,
                )

            # Parse segmentation section
            if 'segmentation' in data:
                seg = data['segmentation']
                config.segmentation = SegmentationConfig(
                    temporal_gap_hours=float(seg.get('temporal_gap_hours', 24)),
                    min_messages_per_conversation=int(seg.get('min_messages_per_conversation', 3)),
                )

            # Parse synthesis section
            if 'synthesis' in data:
                syn = data['synthesis']
                config.synthesis = SynthesisConfig(
                    model=syn.get('model', 'claude-opus-4-5-20250514'),
                    max_tokens=int(syn.get('max_tokens', 4096)),
                    temperature=float(syn.get('temperature', 0.3)),
                    prompt_file=syn.get('prompt_file'),
                )

            # Parse export section
            if 'export' in data:
                exp = data['export']
                config.export = ExportConfig(
                    vault_path=exp.get('vault_path', './output'),
                    filename_format=exp.get('filename_format', '{date}_{slug}.md'),
                    generate_topic_indexes=exp.get('generate_topic_indexes', True),
                    archive_versions=exp.get('archive_versions', True),
                )

            # Parse privacy section
            if 'privacy' in data:
                priv = data['privacy']
                config.privacy = PrivacyConfig(
                    excluded_user_ids=priv.get('excluded_user_ids', []),
                    redaction_placeholder=priv.get('redaction_placeholder', '[message redacted]'),
                )

            # Parse logging section
            if 'logging' in data:
                log = data['logging']
                config.logging = LoggingConfig(
                    level=log.get('level', 'INFO'),
                    file=log.get('file'),
                )

            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")

    # Override from environment variables
    if os.environ.get('DISCORD_TOKEN'):
        config.discord_token = os.environ['DISCORD_TOKEN']
    if os.environ.get('ANTHROPIC_API_KEY'):
        config.anthropic_api_key = os.environ['ANTHROPIC_API_KEY']

    return config


def create_default_config(output_path: str | Path) -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Where to write the config file
    """
    default_config = '''# Discord Conversation Synthesizer Configuration

discord:
  token: ${DISCORD_TOKEN}

server:
  id: "YOUR_SERVER_ID"
  name: "Your Server Name"
  channels:
    - id: "CHANNEL_ID_1"
      name: "general-discussion"
      enabled: true
    - id: "CHANNEL_ID_2"
      name: "announcements"
      enabled: false

segmentation:
  temporal_gap_hours: 24
  min_messages_per_conversation: 3

synthesis:
  model: "claude-opus-4-5-20250514"
  max_tokens: 4096
  temperature: 0.3
  prompt_file: "./config/prompts/synthesis.md"

export:
  vault_path: "./output"
  filename_format: "{date}_{slug}.md"
  generate_topic_indexes: true
  archive_versions: true

privacy:
  excluded_user_ids: []
  redaction_placeholder: "[message redacted]"

logging:
  level: "INFO"
  file: "./logs/synthesizer.log"
'''

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_config)
    logger.info(f"Created default configuration at {output_path}")


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)

    # Create a test config
    test_config_content = '''
discord:
  token: ${DISCORD_TOKEN}

server:
  id: "123456789"
  name: "Test Server"
  channels:
    - id: "111"
      name: "general"
      enabled: true
    - id: "222"
      name: "announcements"
      enabled: false

segmentation:
  temporal_gap_hours: 24
  min_messages_per_conversation: 3

synthesis:
  model: "claude-opus-4-5-20250514"
  max_tokens: 4096
  temperature: 0.3

export:
  vault_path: "./output"
  generate_topic_indexes: true
'''

    # Write test config
    test_path = Path("/tmp/test_config.yaml")
    test_path.write_text(test_config_content)

    # Load and print
    config = load_config(test_path)

    print("\n=== Loaded Config ===")
    print(f"Server: {config.server.name} ({config.server.id})")
    print(f"Channels: {[(c.name, c.enabled) for c in config.server.channels]}")
    print(f"Enabled channels: {[c.name for c in config.get_enabled_channels()]}")
    print(f"Segmentation gap: {config.segmentation.temporal_gap_hours}h")
    print(f"Model: {config.synthesis.model}")
    print(f"Vault path: {config.export.vault_path}")

    # Clean up
    test_path.unlink()

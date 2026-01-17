# Discord Conversation Synthesizer

Transform organic Discord community conversations into structured, indexed knowledge stored in an Obsidian vault.

## Features

- **Fetch** messages from Discord channels and threads
- **Segment** conversations using temporal gaps and reply chains
- **Synthesize** with Claude Opus 4.5 using 3Blue1Brown's pedagogical style
- **Export** to Obsidian with YAML frontmatter, wikilinks, and topic indexes
- **Incremental processing** — only processes new messages since last run

## Quick Start

### Prerequisites

1. **Install UV** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a Discord Bot**:
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application
   - Go to Bot → Reset Token → Copy the token
   - Enable **Message Content Intent** under Privileged Gateway Intents
   - Invite to your server with this URL (replace `YOUR_APP_ID`):
     ```
     https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&permissions=66560&scope=bot
     ```

3. **Get Anthropic API Key**:
   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Create an API key

### Running

```bash
# Clone the repository
git clone https://github.com/your-username/discord-conversation-synthesizer.git
cd discord-conversation-synthesizer

# Run the interactive script
./run.sh
```

The script will:
1. Install dependencies automatically (via UV)
2. Ask for your credentials (saved to `.env`)
3. Let you choose: dry-run, process, or test with mocks

### Manual Commands

```bash
# Sync dependencies
uv sync

# Dry-run (preview what would be processed)
uv run python -m src --real --dry-run

# Process conversations (uses Claude API, has cost)
uv run python -m src --real

# Test with mocks (no API keys needed)
uv run python -m src
```

## Configuration

Edit `config/config.yaml`:

```yaml
server:
  id: "YOUR_SERVER_ID"
  name: "Your Server"
  channels:
    - id: "CHANNEL_ID"
      name: "general"
      enabled: true

segmentation:
  temporal_gap_hours: 24        # Split conversations after 24h silence
  min_messages_per_conversation: 3  # Skip shallow exchanges

synthesis:
  model: "claude-opus-4-5-20250514"
  temperature: 0.3

export:
  vault_path: "./output"
  generate_topic_indexes: true
```

## Output Structure

```
output/
├── conversations/           # Synthesized notes
│   ├── 2024-01-15_risk-parity-debate.md
│   └── _versions/           # Archived previous versions
├── topics/                  # Auto-generated topic indexes
│   ├── risk-management.md
│   └── portfolio-allocation.md
├── participants/            # Contributor index
└── _meta/                   # Processing state
    ├── processing-state.json
    └── run-history.json
```

## CLI Options

```
--dry-run, -n     Preview mode — show what would be processed
--real, -r        Use real Discord + Claude APIs
--config, -c      Custom config file path
--verbose, -v     Enable debug logging
--from DATE       Process messages from this date
--to DATE         Process messages until this date
--all-history     Process all history (ignore previous state)
```

## Development

```bash
# Sync with dev dependencies
uv sync --dev

# Run tests
uv run python test_runner.py
```

## Architecture

```
                     Discord Server
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │           FETCHER                     │
        │   discord.py → Message objects        │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │          SEGMENTER                    │
        │   Temporal gaps + Reply chains        │
        │   → Conversation objects              │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │         SYNTHESIZER                   │
        │   Claude Opus 4.5 → 3B1B style        │
        │   → SynthesizedNote objects           │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │          EXPORTER                     │
        │   Obsidian Markdown + Topic indexes   │
        │   → Vault files                       │
        └──────────────────────────────────────┘
```

## License

MIT

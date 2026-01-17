#!/bin/bash
#
# Discord Conversation Synthesizer - Easy Runner
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  Discord Conversation Synthesizer"
echo "=========================================="
echo ""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies (creates .venv if needed)
echo -e "${YELLOW}Syncing dependencies...${NC}"
uv sync --quiet
echo -e "${GREEN}✓${NC} Dependencies ready"

# Check/create .env file
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    echo -e "${GREEN}✓${NC} Credentials loaded from .env"
else
    echo -e "${YELLOW}.env file not found. Let's create one:${NC}"
    echo ""

    read -p "Discord Bot Token: " DISCORD_TOKEN
    read -p "Anthropic API Key: " ANTHROPIC_API_KEY

    echo "DISCORD_TOKEN=$DISCORD_TOKEN" > "$ENV_FILE"
    echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> "$ENV_FILE"

    echo ""
    echo -e "${GREEN}✓${NC} Credentials saved to .env"

    export DISCORD_TOKEN
    export ANTHROPIC_API_KEY
fi

# Validate credentials
if [ -z "$DISCORD_TOKEN" ] || [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${RED}Error: Credentials not configured${NC}"
    exit 1
fi

echo ""
echo "What would you like to do?"
echo ""
echo "  1) Dry-run (preview what would be processed, no cost)"
echo "  2) Process conversations (uses Claude API, has cost)"
echo "  3) Run with mocks (test without real APIs)"
echo ""
read -p "Choose [1/2/3]: " choice

case $choice in
    1)
        echo ""
        echo -e "${YELLOW}Running dry-run...${NC}"
        echo ""
        uv run python -m src --real --dry-run
        ;;
    2)
        echo ""
        echo -e "${YELLOW}Processing conversations...${NC}"
        echo ""
        uv run python -m src --real
        ;;
    3)
        echo ""
        echo -e "${YELLOW}Running with mocks (test)...${NC}"
        echo ""
        uv run python -m src
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

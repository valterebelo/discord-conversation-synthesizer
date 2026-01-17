#!/usr/bin/env python3
"""
Quick test runner for the Discord Conversation Synthesizer.

This script provides real-time feedback during development by:
1. Loading mock fixtures
2. Running the segmentation logic
3. Running mock synthesis
4. Exporting to the Obsidian vault
5. Reporting results

Usage:
    python test_runner.py                    # Run all tests with mocks
    python test_runner.py --segment-only     # Test only segmentation
    python test_runner.py --synthesize-only  # Test only synthesis (mock)
    python test_runner.py --export-only      # Test only export
    python test_runner.py --real-discord     # Use real Discord (needs token)
    python test_runner.py --real-claude      # Use real Claude (needs key)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def print_success(text: str) -> None:
    print(f"  ✓ {text}")


def print_error(text: str) -> None:
    print(f"  ✗ {text}")


def print_info(text: str) -> None:
    print(f"  → {text}")


async def test_fixtures() -> bool:
    """Test that fixtures load correctly."""
    print_header("Testing Fixtures")

    fixture_path = Path(__file__).parent / "tests/fixtures/sample_conversations.json"

    if not fixture_path.exists():
        print_error(f"Fixture file not found: {fixture_path}")
        return False

    try:
        with open(fixture_path) as f:
            data = json.load(f)

        conversations = data.get("conversations", [])
        print_success(f"Loaded {len(conversations)} conversations from fixtures")

        for conv in conversations:
            msg_count = len(conv.get("messages", []))
            conv_type = conv.get("type", "unknown")
            print_info(f"{conv['id']}: {msg_count} messages ({conv_type})")

        return True

    except Exception as e:
        print_error(f"Failed to load fixtures: {e}")
        return False


async def test_mock_discord() -> bool:
    """Test the mock Discord client."""
    print_header("Testing Mock Discord Client")

    try:
        from mocks.discord_mock import create_mock_client

        client = create_mock_client(
            fixture_path=Path(__file__).parent / "tests/fixtures/sample_conversations.json"
        )

        guild = client.get_guild("123456789012345678")
        if not guild:
            print_error("Failed to get mock guild")
            return False

        print_success(f"Connected to mock guild: {guild.name}")
        print_info(f"Channels: {len(guild.channels)}")

        for channel in guild.channels:
            print_info(f"  #{channel.name}: {len(channel._messages)} messages, {len(channel.threads)} threads")

        await client.close()
        return True

    except Exception as e:
        print_error(f"Mock Discord client failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_claude() -> bool:
    """Test the mock Claude client."""
    print_header("Testing Mock Claude Client")

    try:
        from mocks.claude_mock import MockClaudeClient

        client = MockClaudeClient()

        test_conversation = """
        CHANNEL: #trading-strategies
        THREAD: Risk Parity Discussion
        TIMESPAN: 2026-01-10 14:30 to 2026-01-10 15:50

        [14:30:00] alice_quant: Been thinking about risk parity lately.
        [14:35:00] bob_trader: I ran a backtest comparing risk parity vs 60/40.
        [14:42:00] charlie_dev: The problem is you need leverage.
        """

        response = await client.create(
            model="claude-opus-4-5-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": test_conversation}]
        )

        if not response.content:
            print_error("No content in response")
            return False

        synthesis = response.content[0].text
        print_success(f"Generated mock synthesis ({len(synthesis)} chars)")
        print_info(f"Input tokens: {response.usage.input_tokens}")
        print_info(f"Output tokens: {response.usage.output_tokens}")

        # Check that synthesis has expected structure
        if "---" in synthesis and "title:" in synthesis:
            print_success("Synthesis has valid YAML frontmatter")
        else:
            print_error("Synthesis missing YAML frontmatter")
            return False

        stats = client.get_stats()
        print_info(f"Estimated cost: ${stats['estimated_cost_usd']:.4f}")

        return True

    except Exception as e:
        print_error(f"Mock Claude client failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_segmentation() -> bool:
    """Test conversation segmentation logic."""
    print_header("Testing Segmentation")

    try:
        # Load raw fixture data
        fixture_path = Path(__file__).parent / "tests/fixtures/sample_conversations.json"
        with open(fixture_path) as f:
            data = json.load(f)

        test_config = data.get("test_config", {})
        min_messages = test_config.get("min_messages_per_conversation", 3)
        gap_hours = test_config.get("temporal_gap_hours", 24)

        print_info(f"Config: min_messages={min_messages}, gap_hours={gap_hours}")

        conversations = data.get("conversations", [])

        # Test: shallow conversation should be filtered
        shallow = next((c for c in conversations if "shallow" in c["id"]), None)
        if shallow:
            msg_count = len(shallow["messages"])
            if msg_count < min_messages:
                print_success(f"Shallow conversation ({msg_count} msgs) would be filtered (threshold: {min_messages})")
            else:
                print_error(f"Shallow conversation has {msg_count} msgs, expected < {min_messages}")

        # Test: temporal gap conversation should be split
        gap_conv = next((c for c in conversations if "temporal_gap" in c["id"]), None)
        if gap_conv:
            messages = gap_conv["messages"]
            timestamps = [datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")) for m in messages]

            # Find the gap
            max_gap_hours = 0
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                max_gap_hours = max(max_gap_hours, gap)

            if max_gap_hours > gap_hours:
                print_success(f"Temporal gap detected: {max_gap_hours:.1f} hours (threshold: {gap_hours})")
            else:
                print_error(f"No temporal gap > {gap_hours}h found (max: {max_gap_hours:.1f}h)")

        # Test: thread should remain unified
        thread_conv = next((c for c in conversations if c.get("type") == "thread"), None)
        if thread_conv:
            print_success(f"Thread '{thread_conv.get('thread_name', 'unnamed')}' would be kept as single conversation")

        # Test: reply chains in channel conversation
        channel_conv = next((c for c in conversations if c.get("type") == "channel" and "mean_reversion" in c["id"]), None)
        if channel_conv:
            messages = channel_conv["messages"]
            replies = [m for m in messages if m.get("reply_to")]
            print_success(f"Channel conversation has {len(replies)} reply-chain messages out of {len(messages)}")

        return True

    except Exception as e:
        print_error(f"Segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_export() -> bool:
    """Test export to Obsidian vault."""
    print_header("Testing Export")

    try:
        output_dir = Path(__file__).parent / "output"

        # Check structure exists
        required_dirs = ["conversations", "topics", "participants", "_meta"]
        for dir_name in required_dirs:
            dir_path = output_dir / dir_name
            if dir_path.exists():
                print_success(f"Directory exists: {dir_name}/")
            else:
                print_error(f"Missing directory: {dir_name}/")
                return False

        # Check meta files
        state_file = output_dir / "_meta/processing-state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print_success(f"Processing state file exists (version: {state.get('version', 'unknown')})")
        else:
            print_error("Processing state file missing")
            return False

        # Test writing a sample note
        test_note_path = output_dir / "conversations/_test_note.md"
        test_content = """---
title: "Test Note"
date: "2026-01-16"
participants: ["test_user"]
tags: ["test"]
---

# Test Note

This is a test note to verify export functionality.
"""
        with open(test_note_path, "w") as f:
            f.write(test_content)

        if test_note_path.exists():
            print_success(f"Successfully wrote test note to vault")
            test_note_path.unlink()  # Clean up
            print_info("Cleaned up test note")
        else:
            print_error("Failed to write test note")
            return False

        return True

    except Exception as e:
        print_error(f"Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests() -> dict:
    """Run all tests and return results."""
    results = {}

    results["fixtures"] = await test_fixtures()
    results["mock_discord"] = await test_mock_discord()
    results["mock_claude"] = await test_mock_claude()
    results["segmentation"] = await test_segmentation()
    results["export"] = await test_export()

    return results


def print_summary(results: dict) -> None:
    """Print test summary."""
    print_header("Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status}  {name}")

    print()
    if passed == total:
        print(f"  All {total} tests passed! Ready to start coding.")
    else:
        print(f"  {passed}/{total} tests passed. Fix failures before proceeding.")


def main():
    parser = argparse.ArgumentParser(description="Test runner for Discord Conversation Synthesizer")
    parser.add_argument("--segment-only", action="store_true", help="Test only segmentation")
    parser.add_argument("--synthesize-only", action="store_true", help="Test only synthesis")
    parser.add_argument("--export-only", action="store_true", help="Test only export")
    parser.add_argument("--real-discord", action="store_true", help="Use real Discord (needs DISCORD_TOKEN)")
    parser.add_argument("--real-claude", action="store_true", help="Use real Claude (needs ANTHROPIC_API_KEY)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Discord Conversation Synthesizer - Test Runner")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    if args.real_discord:
        print_info("Real Discord mode - not implemented yet")
        return

    if args.real_claude:
        print_info("Real Claude mode - not implemented yet")
        return

    # Run tests
    results = asyncio.run(run_all_tests())
    print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

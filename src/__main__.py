"""
Main entry point for the Discord Conversation Synthesizer.

Usage:
    python -m synthesizer                    # Standard run with mocks
    python -m synthesizer --dry-run          # Preview mode
    python -m synthesizer --config config.yaml
    python -m synthesizer --real             # Use real Discord + Claude APIs
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, Config
from .models import ProcessingState, RunResult, Conversation
from .fetcher import Fetcher
from .segmenter import Segmenter, segment_from_fixtures
from .synthesizer import Synthesizer, SynthesisResult
from .exporter import Exporter
from .stats import compute_all_stats, save_stats, load_notes_from_vault

logger = logging.getLogger("synthesizer")


def setup_logging(config: Config) -> None:
    """Configure logging based on config."""
    level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler if configured
    if config.logging.file:
        log_path = Path(config.logging.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def print_banner() -> None:
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  Discord Conversation Synthesizer")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


def print_summary(result: RunResult) -> None:
    """Print a summary of the run."""
    print("\n" + "=" * 60)
    print("  Run Summary")
    print("=" * 60)
    print(f"  Duration: {result.duration_seconds:.1f} seconds")
    print(f"  Conversations processed: {result.conversations_processed}")
    print(f"  Messages processed: {result.messages_processed}")
    print(f"  Tokens used: {result.tokens_used:,}")
    print(f"  Cost: ${result.cost_usd:.4f}")
    print(f"  Notes created: {len(result.notes_created)}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for error in result.errors[:5]:  # Show first 5
            print(f"    - {error}")
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more")

    if result.success:
        print("\n  ✓ Run completed successfully")
    else:
        print("\n  ✗ Run completed with errors")

    print("=" * 60 + "\n")


async def run_with_mocks(config: Config) -> RunResult:
    """
    Run the full pipeline using mock services.

    Used for testing and development.
    """
    started_at = datetime.now(timezone.utc)
    errors: list[str] = []
    notes_created: list[str] = []
    total_tokens = 0
    total_cost = 0.0

    # Load conversations from fixtures
    fixture_path = Path(__file__).parent.parent / "tests/fixtures/sample_conversations.json"

    logger.info(f"Loading conversations from fixtures: {fixture_path}")
    conversations = segment_from_fixtures(
        str(fixture_path),
        config={
            "temporal_gap_hours": config.segmentation.temporal_gap_hours,
            "min_messages_per_conversation": config.segmentation.min_messages_per_conversation,
            "excluded_user_ids": config.privacy.excluded_user_ids,
            "redaction_placeholder": config.privacy.redaction_placeholder,
        }
    )

    logger.info(f"Segmented into {len(conversations)} conversations")

    if config.dry_run:
        # Dry run - just estimate
        from .synthesizer import create_synthesizer
        synthesizer = await create_synthesizer(use_mock=True)
        estimate = synthesizer.estimate_cost(conversations)

        print("\n" + "=" * 60)
        print("  DRY RUN - Preview")
        print("=" * 60)
        print(f"  Conversations to process: {estimate['conversation_count']}")
        print(f"  Estimated input tokens: {estimate['estimated_input_tokens']:,}")
        print(f"  Estimated output tokens: {estimate['estimated_output_tokens']:,}")
        print(f"  Estimated total cost: ${estimate['estimated_cost_usd']:.4f}")
        print("=" * 60 + "\n")

        return RunResult(
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            conversations_processed=0,
            messages_processed=0,
            tokens_used=0,
            cost_usd=0.0,
        )

    # Create synthesizer
    from .synthesizer import create_synthesizer
    synthesizer = await create_synthesizer(
        use_mock=True,
        model=config.synthesis.model,
    )

    # Create exporter
    exporter = Exporter(
        vault_path=config.export.vault_path,
        archive_versions=config.export.archive_versions,
        generate_topic_indexes=config.export.generate_topic_indexes,
    )

    # Process each conversation
    total_messages = sum(c.message_count for c in conversations)
    successful_notes: list = []

    for i, conv in enumerate(conversations):
        logger.info(f"Processing {i+1}/{len(conversations)}: {conv.id}")

        result = await synthesizer.synthesize(conv)

        if result.success and result.note:
            # Export the note
            try:
                path = exporter.export_note(result.note)
                notes_created.append(str(path.name))
                successful_notes.append(result.note)
                total_tokens += result.input_tokens + result.output_tokens
                total_cost += result.cost_usd
            except Exception as e:
                errors.append(f"Export failed for {conv.id}: {e}")
        else:
            errors.append(f"Synthesis failed for {conv.id}: {result.error}")

    # Update topic indexes with all successful notes
    if successful_notes and config.export.generate_topic_indexes:
        exporter._update_topic_indexes(successful_notes)

    # Save run result
    run_result = RunResult(
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        conversations_processed=len(conversations),
        messages_processed=total_messages,
        tokens_used=total_tokens,
        cost_usd=total_cost,
        errors=errors,
        notes_created=notes_created,
    )

    exporter.save_run_result(run_result)

    return run_result


async def run_stats_only(config: Config) -> None:
    """
    Generate statistics from existing vault data.

    Does not process new conversations or call any APIs.
    """
    vault_path = Path(config.export.vault_path)

    logger.info(f"Generating stats from vault: {vault_path}")

    # Load existing notes for topic stats
    notes = load_notes_from_vault(vault_path)

    if not notes:
        logger.warning("No notes found in vault. Run the synthesizer first.")
        print("\n  No notes found in vault. Run the synthesizer first to generate content.\n")
        return

    # For stats, we need conversations. If we don't have them loaded,
    # we can still generate partial stats from notes
    logger.info(f"Found {len(notes)} notes in vault")

    # Since we don't have conversations loaded, we'll generate stats from notes only
    # This gives us topic stats but not participant/activity stats
    from .stats import compute_topic_stats
    topic_stats = compute_topic_stats(notes)

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes_analyzed": len(notes),
        "topics": topic_stats,
        "note": "Partial stats from notes only. Run full synthesis for complete participant/activity stats."
    }

    # Save stats
    stats_path = vault_path / "_meta" / "stats.json"
    save_stats(stats, stats_path)

    print("\n" + "=" * 60)
    print("  Statistics Generated")
    print("=" * 60)
    print(f"  Notes analyzed: {len(notes)}")
    print(f"  Topics found: {len(topic_stats['frequency'])}")
    print(f"  Top topics:")
    for tag, count in list(topic_stats['frequency'].items())[:5]:
        print(f"    - {tag}: {count} conversations")
    print(f"\n  Stats saved to: {stats_path}")
    print("=" * 60 + "\n")


async def run_with_real_services(config: Config) -> RunResult:
    """
    Run the full pipeline using real Discord and Claude APIs.

    Requires valid API keys.
    """
    started_at = datetime.now(timezone.utc)
    errors: list[str] = []
    notes_created: list[str] = []

    # Validate configuration
    if not config.discord_token:
        raise ValueError("DISCORD_TOKEN is required for real Discord connection")
    if not config.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for real Claude synthesis")
    if not config.server.id:
        raise ValueError("Server ID is required in configuration")

    logger.info(f"Connecting to Discord server: {config.server.name}")

    # Create fetcher with real Discord client
    from .fetcher import create_fetcher
    fetcher = await create_fetcher(
        use_mock=False,
        guild_id=config.server.id,
        token=config.discord_token,
    )

    # Create exporter
    exporter = Exporter(
        vault_path=config.export.vault_path,
        archive_versions=config.export.archive_versions,
        generate_topic_indexes=config.export.generate_topic_indexes,
    )

    # Load previous state for incremental processing
    state = exporter.load_state()

    # Create segmenter
    segmenter = Segmenter(
        temporal_gap_hours=config.segmentation.temporal_gap_hours,
        min_messages=config.segmentation.min_messages_per_conversation,
        excluded_user_ids=set(config.privacy.excluded_user_ids),
        redaction_placeholder=config.privacy.redaction_placeholder,
    )

    # Fetch from all enabled channels
    channel_configs = [
        {"id": c.id, "name": c.name, "enabled": c.enabled}
        for c in config.server.channels
    ]

    fetch_results = await fetcher.fetch_all_configured_channels(
        channel_configs,
        state={cs.channel_id: cs for cs in state.channels.values()},
    )

    # Segment all fetched messages
    all_conversations: list[Conversation] = []

    for fetch_result in fetch_results:
        # For now, treat all messages as main channel (threads handled separately)
        convs = segmenter.segment_channel(
            messages=fetch_result.messages,
            channel_id=fetch_result.channel_id,
            channel_name=fetch_result.channel_name,
        )
        all_conversations.extend(convs)

    logger.info(f"Segmented into {len(all_conversations)} conversations")

    if config.dry_run:
        # Dry run - just estimate
        from .synthesizer import create_synthesizer
        synthesizer = await create_synthesizer(use_mock=True)
        estimate = synthesizer.estimate_cost(all_conversations)

        print("\n" + "=" * 60)
        print("  DRY RUN - Preview (Real Discord Data)")
        print("=" * 60)
        print(f"  Conversations to process: {estimate['conversation_count']}")
        print(f"  Estimated input tokens: {estimate['estimated_input_tokens']:,}")
        print(f"  Estimated output tokens: {estimate['estimated_output_tokens']:,}")
        print(f"  Estimated total cost: ${estimate['estimated_cost_usd']:.4f}")
        print("=" * 60 + "\n")

        return RunResult(
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            conversations_processed=0,
            messages_processed=0,
            tokens_used=0,
            cost_usd=0.0,
        )

    # Create synthesizer with real Claude
    from .synthesizer import create_synthesizer
    synthesizer = await create_synthesizer(
        use_mock=False,
        model=config.synthesis.model,
        api_key=config.anthropic_api_key,
        prompt_template_path=config.synthesis.prompt_file,
    )

    # Process conversations
    total_messages = sum(c.message_count for c in all_conversations)
    total_tokens = 0
    total_cost = 0.0
    successful_notes = []

    for i, conv in enumerate(all_conversations):
        logger.info(f"Processing {i+1}/{len(all_conversations)}: {conv.id}")

        result = await synthesizer.synthesize(conv)

        if result.success and result.note:
            try:
                path = exporter.export_note(result.note)
                notes_created.append(str(path.name))
                successful_notes.append(result.note)
                total_tokens += result.input_tokens + result.output_tokens
                total_cost += result.cost_usd
            except Exception as e:
                errors.append(f"Export failed for {conv.id}: {e}")
        else:
            errors.append(f"Synthesis failed for {conv.id}: {result.error}")

    # Update indexes
    if successful_notes:
        exporter._update_topic_indexes(successful_notes)
        exporter.update_participant_index(successful_notes)

    # Update and save state
    for fetch_result in fetch_results:
        if fetch_result.newest_message_id:
            state.update_channel_state(
                channel_id=fetch_result.channel_id,
                channel_name=fetch_result.channel_name,
                last_message_id=fetch_result.newest_message_id,
                last_timestamp=datetime.now(timezone.utc),
                messages_processed=fetch_result.message_count,
                conversations_processed=len([
                    c for c in all_conversations
                    if c.channel_id == fetch_result.channel_id
                ]),
            )

    state.total_tokens_used += total_tokens
    state.total_cost_usd += total_cost
    exporter.save_state(state)

    # Save run result
    run_result = RunResult(
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        conversations_processed=len(all_conversations),
        messages_processed=total_messages,
        tokens_used=total_tokens,
        cost_usd=total_cost,
        errors=errors,
        notes_created=notes_created,
    )

    exporter.save_run_result(run_result)

    return run_result


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discord Conversation Synthesizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m synthesizer                    # Run with mocks (testing)
    python -m synthesizer --dry-run          # Preview what would be processed
    python -m synthesizer --real             # Use real Discord + Claude
    python -m synthesizer --config my.yaml   # Use custom config file
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview mode - show what would be processed without executing"
    )

    parser.add_argument(
        "--real", "-r",
        action="store_true",
        help="Use real Discord and Claude APIs (requires API keys)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--from",
        type=str,
        dest="from_date",
        help="Process messages from this date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--to",
        type=str,
        dest="to_date",
        help="Process messages until this date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--all-history",
        action="store_true",
        help="Process all message history (ignore previous state)"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Generate statistics from existing vault data without processing new conversations"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.dry_run = args.dry_run
    config.use_mock = not args.real
    config.verbose = args.verbose

    if args.verbose:
        config.logging.level = "DEBUG"

    # Setup logging
    setup_logging(config)

    # Print banner
    print_banner()

    if config.use_mock:
        logger.info("Running with MOCK services (testing mode)")
    else:
        logger.info("Running with REAL services")

    if config.dry_run:
        logger.info("DRY RUN mode - no changes will be made")

    # Handle stats-only mode
    if args.stats_only:
        logger.info("STATS ONLY mode - generating statistics from existing vault")
        try:
            await run_stats_only(config)
            return 0
        except Exception as e:
            logger.exception(f"Stats generation failed: {e}")
            return 1

    try:
        if config.use_mock:
            result = await run_with_mocks(config)
        else:
            result = await run_with_real_services(config)

        print_summary(result)
        return 0 if result.success else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


def cli():
    """CLI entry point."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli()

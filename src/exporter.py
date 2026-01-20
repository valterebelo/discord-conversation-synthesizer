"""
Exporter module - writes synthesized notes to Obsidian vault.

Handles:
- Writing conversation notes to /conversations/
- Updating topic index files in /topics/
- Managing version history in /_versions/
- Updating processing state in /_meta/
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict

from .models import SynthesizedNote, ProcessingState, RunResult, Conversation

logger = logging.getLogger(__name__)


class Exporter:
    """
    Exports synthesized notes to an Obsidian vault structure.

    Directory structure:
        vault_root/
        ├── conversations/     - Synthesized conversation notes
        │   └── _versions/     - Archived previous versions
        ├── topics/            - Auto-generated topic index pages
        ├── participants/      - Participant index
        └── _meta/             - Processing state and run history
    """

    def __init__(
        self,
        vault_path: str | Path,
        archive_versions: bool = True,
        generate_topic_indexes: bool = True,
    ):
        """
        Initialize the exporter.

        Args:
            vault_path: Path to the Obsidian vault root
            archive_versions: Whether to archive old versions on re-export
            generate_topic_indexes: Whether to update topic index files
        """
        self.vault_path = Path(vault_path)
        self.archive_versions = archive_versions
        self.generate_topic_indexes = generate_topic_indexes

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.vault_path / "conversations",
            self.vault_path / "conversations" / "_versions",
            self.vault_path / "topics",
            self.vault_path / "participants",
            self.vault_path / "_meta",
            self.vault_path / "_transcripts",
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def export_note(self, note: SynthesizedNote) -> Path:
        """
        Export a single synthesized note to the vault.

        Args:
            note: The note to export

        Returns:
            Path to the written file
        """
        # Generate filename
        filename = note.get_filename()
        file_path = self.vault_path / "conversations" / filename

        # Archive existing version if needed
        if file_path.exists() and self.archive_versions:
            self._archive_version(file_path)

        # Write the note
        content = note.to_markdown()
        file_path.write_text(content, encoding="utf-8")

        logger.info(f"Exported note: {filename}")
        return file_path

    def _archive_version(self, file_path: Path) -> None:
        """
        Archive an existing file to the _versions directory.

        Adds timestamp suffix to filename.
        """
        versions_dir = self.vault_path / "conversations" / "_versions"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate versioned filename
        stem = file_path.stem
        suffix = file_path.suffix
        versioned_name = f"{stem}_v_{timestamp}{suffix}"
        archive_path = versions_dir / versioned_name

        # Copy to archive
        shutil.copy2(file_path, archive_path)
        logger.info(f"Archived previous version: {versioned_name}")

    def save_transcript(self, conversation: Conversation) -> Path:
        """
        Save the original transcript of a conversation.

        Args:
            conversation: The conversation to save

        Returns:
            Path to the saved transcript file
        """
        transcripts_dir = self.vault_path / "_transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from conversation ID
        # Clean the ID to be filesystem-safe
        safe_id = conversation.id.replace("/", "_").replace(":", "_")
        file_path = transcripts_dir / f"{safe_id}.txt"

        # Write the transcript
        transcript = conversation.to_transcript()
        file_path.write_text(transcript, encoding="utf-8")

        logger.info(f"Saved transcript: {file_path.name}")
        return file_path

    def export_with_transcript(
        self,
        note: SynthesizedNote,
        conversation: Conversation
    ) -> tuple[Path, Path]:
        """
        Export a note and its original transcript.

        Args:
            note: The synthesized note
            conversation: The original conversation

        Returns:
            Tuple of (note_path, transcript_path)
        """
        note_path = self.export_note(note)
        transcript_path = self.save_transcript(conversation)
        return note_path, transcript_path

    def export_batch(self, notes: list[SynthesizedNote]) -> list[Path]:
        """
        Export multiple notes.

        Args:
            notes: List of notes to export

        Returns:
            List of paths to written files
        """
        paths = []
        for note in notes:
            try:
                path = self.export_note(note)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to export note {note.conversation_id}: {e}")

        # Update topic indexes if enabled
        if self.generate_topic_indexes and notes:
            self._update_topic_indexes(notes)

        return paths

    def _update_topic_indexes(self, notes: list[SynthesizedNote]) -> None:
        """
        Update topic index files based on exported notes.

        Creates/updates a file for each unique tag, linking to all
        conversations with that tag.
        """
        # Collect all tags and their notes
        tag_to_notes: dict[str, list[SynthesizedNote]] = defaultdict(list)

        for note in notes:
            for tag in note.tags:
                tag_to_notes[tag].append(note)

        # Update each topic file
        topics_dir = self.vault_path / "topics"

        for tag, tag_notes in tag_to_notes.items():
            topic_file = topics_dir / f"{tag}.md"
            self._update_topic_file(topic_file, tag, tag_notes)

        # Update the main index
        self._update_topics_index(list(tag_to_notes.keys()))

        logger.info(f"Updated {len(tag_to_notes)} topic indexes")

    def _update_topic_file(
        self,
        topic_file: Path,
        tag: str,
        new_notes: list[SynthesizedNote]
    ) -> None:
        """
        Update or create a topic index file.

        If the file exists, append new links. Otherwise, create it.
        """
        # Generate links for new notes
        new_links = []
        for note in new_notes:
            filename = note.get_filename().replace(".md", "")
            link = f"- [[{filename}]] - {note.title} ({note.date})"
            new_links.append(link)

        if topic_file.exists():
            # Read existing content
            content = topic_file.read_text()

            # Find the conversations section and append
            if "## Conversations" in content:
                # Append new links before the last section or at end
                insert_pos = content.find("\n## ", content.find("## Conversations") + 1)
                if insert_pos == -1:
                    # No more sections, append at end
                    content = content.rstrip() + "\n" + "\n".join(new_links) + "\n"
                else:
                    # Insert before next section
                    content = (
                        content[:insert_pos] +
                        "\n".join(new_links) + "\n" +
                        content[insert_pos:]
                    )
            else:
                # Add conversations section
                content += f"\n\n## Conversations\n\n" + "\n".join(new_links) + "\n"

            topic_file.write_text(content)
        else:
            # Create new topic file
            title = tag.replace("-", " ").title()
            content = f"""---
title: "{title}"
type: topic-index
updated: "{datetime.now().strftime('%Y-%m-%d')}"
---

# {title}

This page collects all conversations related to **{tag}**.

## Conversations

{chr(10).join(new_links)}

---

*Auto-generated by Discord Conversation Synthesizer*
"""
            topic_file.write_text(content)

    def _update_topics_index(self, tags: list[str]) -> None:
        """Update the main topics index file."""
        index_file = self.vault_path / "topics" / "index.md"

        if not index_file.exists():
            return

        content = index_file.read_text()

        # Update the "Recently Added" section
        recent_section = "\n## Recently Added\n\n"
        for tag in tags[:10]:  # Last 10 topics
            title = tag.replace("-", " ").title()
            recent_section += f"- [[{tag}|{title}]]\n"

        # Find and replace the Recently Added section
        if "## Recently Added" in content:
            start = content.find("## Recently Added")
            end = content.find("\n---", start)
            if end == -1:
                end = len(content)
            content = content[:start] + recent_section.strip() + "\n\n" + content[end:]
        else:
            # Append before final divider
            content = content.rstrip() + "\n" + recent_section

        # Update timestamp
        if "Last updated:" in content:
            content = content.replace(
                content[content.find("Last updated:"):content.find("*", content.find("Last updated:")) + 1],
                f"Last updated: {datetime.now().strftime('%Y-%m-%d')}*"
            )

        index_file.write_text(content)

    def load_state(self) -> ProcessingState:
        """Load processing state from _meta/processing-state.json."""
        state_file = self.vault_path / "_meta" / "processing-state.json"

        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            return ProcessingState.from_dict(data)

        return ProcessingState()

    def save_state(self, state: ProcessingState) -> None:
        """Save processing state to _meta/processing-state.json."""
        state_file = self.vault_path / "_meta" / "processing-state.json"
        state.last_run = datetime.now(timezone.utc)

        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        logger.info("Saved processing state")

    def save_run_result(self, result: RunResult) -> None:
        """Append run result to _meta/run-history.json."""
        history_file = self.vault_path / "_meta" / "run-history.json"

        # Load existing history
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
        else:
            history = {"runs": []}

        # Append new run
        history["runs"].append(result.to_dict())

        # Keep only last 100 runs
        history["runs"] = history["runs"][-100:]

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("Saved run result to history")

    def update_participant_index(self, notes: list[SynthesizedNote]) -> None:
        """Update the participant index with new contributors."""
        index_file = self.vault_path / "participants" / "index.md"

        if not index_file.exists():
            return

        # Collect all participants from notes
        participant_notes: dict[str, list[str]] = defaultdict(list)
        for note in notes:
            for participant in note.participants:
                filename = note.get_filename().replace(".md", "")
                participant_notes[participant].append(f"[[{filename}]]")

        # Read current index
        content = index_file.read_text()

        # Update contributors section
        if "## Contributors" in content:
            # Find the section
            start = content.find("## Contributors")
            end = content.find("\n##", start + 1)
            if end == -1:
                end = content.find("\n---", start)
            if end == -1:
                end = len(content)

            # Build new contributors list
            contributors_section = "## Contributors\n\n"
            for participant, note_links in sorted(participant_notes.items()):
                contributors_section += f"- **{participant}**: {', '.join(note_links[:5])}"
                if len(note_links) > 5:
                    contributors_section += f" (+{len(note_links) - 5} more)"
                contributors_section += "\n"

            content = content[:start] + contributors_section + "\n" + content[end:]

        # Update timestamp
        content = content.replace(
            "*Last updated:",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}"
        )

        index_file.write_text(content)
        logger.info(f"Updated participant index with {len(participant_notes)} contributors")

    def get_existing_conversations(self) -> list[str]:
        """
        Get list of existing conversation IDs in the vault.

        Useful for detecting re-synthesis of existing conversations.
        """
        conversations_dir = self.vault_path / "conversations"
        existing = []

        for file in conversations_dir.glob("*.md"):
            if file.name.startswith("_"):
                continue
            existing.append(file.stem)

        return existing


def create_exporter(
    vault_path: str | Path,
    archive_versions: bool = True,
    generate_topic_indexes: bool = True,
) -> Exporter:
    """
    Factory function to create an Exporter.

    Args:
        vault_path: Path to the Obsidian vault root
        archive_versions: Whether to archive old versions
        generate_topic_indexes: Whether to update topic indexes

    Returns:
        Configured Exporter instance
    """
    return Exporter(
        vault_path=vault_path,
        archive_versions=archive_versions,
        generate_topic_indexes=generate_topic_indexes,
    )


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    logging.basicConfig(level=logging.INFO)

    # Create a test note
    note = SynthesizedNote(
        conversation_id="test_conv_001",
        title="Test Conversation About Risk Management",
        date="2026-01-16",
        participants=["alice", "bob", "charlie"],
        channel="trading-strategies",
        tags=["risk-management", "portfolio-allocation"],
        related=["[[sharpe-ratio]]", "[[volatility]]"],
        summary="A test conversation about risk management concepts.",
        core_idea="This is a test of the export functionality. The core idea would normally contain a detailed explanation of the concept discussed.",
        key_contributions="- alice: Started the discussion\n- bob: Provided counterargument\n- charlie: Synthesized the views",
        tension_points="There was disagreement about the practical applicability of the discussed approach.",
        connections="This connects to broader portfolio theory concepts.",
    )

    # Export to test vault
    vault_path = Path(__file__).parent.parent / "output"
    exporter = create_exporter(vault_path)

    # Export the note
    path = exporter.export_note(note)
    print(f"\nExported to: {path}")

    # Check the file
    content = path.read_text()
    print("\n=== Exported Content ===")
    print(content[:1000])
    print("...")

    # Clean up test file
    path.unlink()
    print("\nCleaned up test file")

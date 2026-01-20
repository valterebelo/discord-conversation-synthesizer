"""
Open Questions module - extracts and indexes unresolved questions from conversations.

Enables the "contribution loop" by identifying where new members can add value.
"""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OpenQuestion:
    """An unresolved question or problem from a conversation."""
    id: str
    question: str
    context: str  # Brief context about the discussion
    source_conversation: str  # Filename of source note
    source_title: str
    date: str
    tags: list[str]
    participants: list[str]
    difficulty: str = "unknown"  # easy, medium, hard, unknown
    question_type: str = "general"  # technical, conceptual, implementation, data, unknown
    status: str = "open"  # open, answered, stale

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "source_conversation": self.source_conversation,
            "source_title": self.source_title,
            "date": self.date,
            "tags": self.tags,
            "participants": self.participants,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "status": self.status,
        }


def extract_questions_from_note(note_path: Path) -> list[OpenQuestion]:
    """
    Extract open questions from a synthesized note.

    Looks for:
    - "Points of Tension" section for unresolved debates
    - Explicit question patterns in the text
    - "Open Questions" section if present
    """
    content = note_path.read_text(encoding="utf-8")
    questions = []

    # Parse frontmatter
    frontmatter = {}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            import yaml
            try:
                frontmatter = yaml.safe_load(content[3:end])
            except:
                pass

    title = frontmatter.get("title", note_path.stem)
    date = frontmatter.get("date", "unknown")
    tags = frontmatter.get("tags", [])
    participants = frontmatter.get("participants", [])

    # Extract Points of Tension section
    tension_match = re.search(
        r'## Points of Tension\s*\n(.*?)(?=\n## |\n---|\Z)',
        content,
        re.DOTALL
    )

    if tension_match:
        tension_text = tension_match.group(1).strip()

        # Split by paragraphs or bullet points
        segments = re.split(r'\n\n|\n(?=\*\*|\n-\s)', tension_text)

        for i, segment in enumerate(segments):
            segment = segment.strip()
            if len(segment) < 20:
                continue

            # Extract the core question/tension
            question_text = _extract_question_from_segment(segment)
            if question_text:
                q_id = f"{note_path.stem}_tension_{i}"
                questions.append(OpenQuestion(
                    id=q_id,
                    question=question_text,
                    context=segment[:200] + "..." if len(segment) > 200 else segment,
                    source_conversation=note_path.stem,
                    source_title=title,
                    date=date,
                    tags=tags,
                    participants=participants,
                    difficulty=_estimate_difficulty(segment, tags),
                    question_type=_classify_question_type(segment, tags),
                ))

    # Look for explicit "Open Questions" section
    open_match = re.search(
        r'## Open Questions\s*\n(.*?)(?=\n## |\n---|\Z)',
        content,
        re.DOTALL
    )

    if open_match:
        open_text = open_match.group(1).strip()
        # Parse bullet points
        bullets = re.findall(r'[-*]\s+(.+?)(?=\n[-*]|\Z)', open_text, re.DOTALL)

        for i, bullet in enumerate(bullets):
            bullet = bullet.strip()
            if len(bullet) < 10:
                continue

            q_id = f"{note_path.stem}_explicit_{i}"
            questions.append(OpenQuestion(
                id=q_id,
                question=bullet,
                context="Explicitly marked as open question",
                source_conversation=note_path.stem,
                source_title=title,
                date=date,
                tags=tags,
                participants=participants,
                difficulty=_estimate_difficulty(bullet, tags),
                question_type=_classify_question_type(bullet, tags),
            ))

    return questions


def _extract_question_from_segment(segment: str) -> Optional[str]:
    """Extract a question or unresolved issue from a text segment."""
    # Look for explicit question marks
    questions = re.findall(r'[^.!?]*\?', segment)
    if questions:
        # Return the most substantive question
        best = max(questions, key=len)
        return best.strip()

    # Look for "unresolved", "remains", "wasn't addressed", etc.
    unresolved_patterns = [
        r'(?:remains?|left|still)\s+(?:unresolved|open|unclear|unanswered)',
        r"(?:wasn't|was not|weren't|were not)\s+(?:addressed|resolved|answered)",
        r'(?:no\s+)?(?:clear\s+)?(?:consensus|resolution|answer)',
        r'implicit\s+(?:tension|disagreement)',
    ]

    for pattern in unresolved_patterns:
        if re.search(pattern, segment, re.IGNORECASE):
            # Extract the first sentence as the question
            sentences = re.split(r'(?<=[.!?])\s+', segment)
            if sentences:
                return sentences[0].strip()

    return None


def _estimate_difficulty(text: str, tags: list[str]) -> str:
    """Estimate the difficulty level of addressing a question."""
    text_lower = text.lower()

    # Hard indicators
    hard_patterns = [
        'mathematical', 'derivation', 'proof', 'theorem',
        'implementation', 'production', 'scale', 'infrastructure',
        'regime', 'non-stationary', 'time-varying',
    ]
    if any(p in text_lower for p in hard_patterns):
        return "hard"

    # Easy indicators
    easy_patterns = [
        'definition', 'explain', 'what is', 'basic',
        'introductory', 'beginner', 'simple',
    ]
    if any(p in text_lower for p in easy_patterns):
        return "easy"

    # Tag-based estimation
    hard_tags = {'machine-learning', 'market-microstructure', 'statistics'}
    if set(tags) & hard_tags:
        return "medium"

    return "medium"


def _classify_question_type(text: str, tags: list[str]) -> str:
    """Classify the type of question."""
    text_lower = text.lower()

    if any(w in text_lower for w in ['code', 'implement', 'build', 'library', 'api']):
        return "implementation"

    if any(w in text_lower for w in ['data', 'dataset', 'source', 'feed', 'vendor']):
        return "data"

    if any(w in text_lower for w in ['formula', 'equation', 'math', 'derive', 'proof']):
        return "technical"

    if any(w in text_lower for w in ['why', 'intuition', 'understand', 'concept']):
        return "conceptual"

    return "general"


def extract_all_questions(vault_path: Path) -> list[OpenQuestion]:
    """Extract all open questions from a vault."""
    conversations_dir = vault_path / "conversations"
    all_questions = []

    if not conversations_dir.exists():
        logger.warning(f"Conversations directory not found: {conversations_dir}")
        return all_questions

    for note_path in conversations_dir.glob("*.md"):
        if note_path.name.startswith("_"):
            continue

        try:
            questions = extract_questions_from_note(note_path)
            all_questions.extend(questions)
        except Exception as e:
            logger.warning(f"Failed to extract questions from {note_path.name}: {e}")

    logger.info(f"Extracted {len(all_questions)} open questions from vault")
    return all_questions


def generate_questions_index(questions: list[OpenQuestion], output_dir: Path) -> list[Path]:
    """
    Generate index files for open questions.

    Creates:
    - _open/index.md - Main list of all open questions
    - _open/by-topic.md - Grouped by tag
    - _open/by-type.md - Grouped by question type
    """
    open_dir = output_dir / "_open"
    open_dir.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Main index
    index_path = open_dir / "index.md"
    index_content = _generate_main_index(questions)
    index_path.write_text(index_content, encoding="utf-8")
    created_files.append(index_path)

    # By topic
    by_topic_path = open_dir / "by-topic.md"
    by_topic_content = _generate_by_topic_index(questions)
    by_topic_path.write_text(by_topic_content, encoding="utf-8")
    created_files.append(by_topic_path)

    # By type
    by_type_path = open_dir / "by-type.md"
    by_type_content = _generate_by_type_index(questions)
    by_type_path.write_text(by_type_content, encoding="utf-8")
    created_files.append(by_type_path)

    # JSON dump for programmatic access
    json_path = open_dir / "questions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)
    created_files.append(json_path)

    logger.info(f"Generated open questions index with {len(questions)} questions")
    return created_files


def _generate_main_index(questions: list[OpenQuestion]) -> str:
    """Generate the main index markdown."""
    # Sort by date (most recent first)
    sorted_qs = sorted(questions, key=lambda q: q.date, reverse=True)

    lines = [
        "---",
        "title: Open Questions",
        "type: question-index",
        f'updated: "{datetime.now().strftime("%Y-%m-%d")}"',
        f"total_questions: {len(questions)}",
        "---",
        "",
        "# Open Questions",
        "",
        "These are unresolved questions and debates from community discussions.",
        "**Your expertise could help answer them.**",
        "",
        f"Total: **{len(questions)}** open questions",
        "",
        "## Recent Questions",
        "",
    ]

    for q in sorted_qs[:20]:  # Show 20 most recent
        difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
        lines.append(f"### {difficulty_emoji} {q.question[:80]}{'...' if len(q.question) > 80 else ''}")
        lines.append("")
        lines.append(f"*From [[{q.source_conversation}|{q.source_title}]] ({q.date})*")
        lines.append("")
        if q.context and q.context != "Explicitly marked as open question":
            lines.append(f"> {q.context[:150]}...")
        lines.append("")
        lines.append(f"Tags: {', '.join(f'`{t}`' for t in q.tags[:5])}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend([
        "",
        "## Browse by",
        "",
        "- [[by-topic|By Topic]] - Find questions in your area of expertise",
        "- [[by-type|By Type]] - Technical, conceptual, implementation, data",
        "",
        "---",
        "",
        "*Help the community by contributing your knowledge!*",
    ])

    return "\n".join(lines)


def _generate_by_topic_index(questions: list[OpenQuestion]) -> str:
    """Generate index grouped by topic/tag."""
    # Group by tag
    by_tag: dict[str, list[OpenQuestion]] = defaultdict(list)
    for q in questions:
        for tag in q.tags:
            by_tag[tag].append(q)

    # Sort tags by question count
    sorted_tags = sorted(by_tag.items(), key=lambda x: len(x[1]), reverse=True)

    lines = [
        "---",
        "title: Open Questions by Topic",
        "type: question-index",
        "---",
        "",
        "# Open Questions by Topic",
        "",
        "Find questions where your expertise can help.",
        "",
    ]

    for tag, tag_questions in sorted_tags:
        lines.append(f"## {tag.replace('-', ' ').title()} ({len(tag_questions)})")
        lines.append("")

        for q in tag_questions[:5]:  # Show top 5 per tag
            difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
            lines.append(f"- {difficulty_emoji} {q.question[:60]}{'...' if len(q.question) > 60 else ''}")
            lines.append(f"  - *[[{q.source_conversation}|Source]]*")

        if len(tag_questions) > 5:
            lines.append(f"- *...and {len(tag_questions) - 5} more*")

        lines.append("")

    return "\n".join(lines)


def _generate_by_type_index(questions: list[OpenQuestion]) -> str:
    """Generate index grouped by question type."""
    by_type: dict[str, list[OpenQuestion]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)

    type_descriptions = {
        "technical": "Mathematical derivations, formulas, proofs",
        "conceptual": "Understanding why things work, intuition building",
        "implementation": "Code, libraries, system design",
        "data": "Data sources, vendors, pipelines",
        "general": "Other questions",
    }

    lines = [
        "---",
        "title: Open Questions by Type",
        "type: question-index",
        "---",
        "",
        "# Open Questions by Type",
        "",
    ]

    for qtype, description in type_descriptions.items():
        type_questions = by_type.get(qtype, [])
        if not type_questions:
            continue

        lines.append(f"## {qtype.title()} ({len(type_questions)})")
        lines.append(f"*{description}*")
        lines.append("")

        for q in type_questions[:5]:
            difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
            lines.append(f"- {difficulty_emoji} {q.question[:60]}{'...' if len(q.question) > 60 else ''}")

        if len(type_questions) > 5:
            lines.append(f"- *...and {len(type_questions) - 5} more*")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test with existing vault
    vault_path = Path(__file__).parent.parent / "output"

    questions = extract_all_questions(vault_path)
    print(f"\nExtracted {len(questions)} open questions:\n")

    for q in questions[:5]:
        print(f"[{q.difficulty}] {q.question[:70]}...")
        print(f"  Source: {q.source_title}")
        print(f"  Tags: {', '.join(q.tags)}")
        print()

    # Generate index
    files = generate_questions_index(questions, vault_path)
    print(f"\nGenerated files:")
    for f in files:
        print(f"  - {f}")

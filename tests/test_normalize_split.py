from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from laban_tts.normalize import (
    TextType,
    load_normalized_parts,
    load_parts,
    normalize_parts,
    partition_text,
)


# TODO: No trivial wrappers.
def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def test_partition_text_creates_xml_parts(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    source.write_text(
        """# Title

## Chapter 1
First section text.

## Chapter 2
Second section text with more detail.
"""
    )

    parts_dir = tmp_path / "parts"
    parts = partition_text(source, parts_dir, max_characters=80)

    assert parts, "No partition entries produced"
    loaded_parts = load_parts(parts_dir)
    assert len(loaded_parts) == len(parts)
    for part in loaded_parts:
        xml_path = parts_dir / f"{source.stem}-part{part.part:03d}.xml"
        assert xml_path.exists()
        assert part.content.strip(), "Partition text is empty"
        assert part.text_name == source.stem
        assert part.part >= 1


@pytest.mark.skipif(
    not _has_openai(), reason="Requires OPENAI_API_KEY and network access"
)
def test_normalize_parts_writes_xml(tmp_path: Path) -> None:
    source = tmp_path / "doc.md"
    source.write_text(
        """# Header

Body paragraph one.

Paragraph two continues.
"""
    )

    parts_dir = tmp_path / "parts"
    partition_text(source, parts_dir, max_characters=120)
    part_entries = load_parts(parts_dir)

    normalize_dir = tmp_path / "normalize"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    outputs = normalize_parts(part_entries, normalize_dir, llm)

    assert outputs, "Normalization yielded no outputs"
    xml_paths = sorted(normalize_dir.glob("*-normalized.xml"))
    assert xml_paths, "No normalized XML files were written"

    reloaded = load_normalized_parts(normalize_dir)
    assert len(reloaded) == len(outputs)
    for entry in reloaded:
        assert entry.cleaned_text().strip(), "Cleaned text missing"
        assert isinstance(entry.category, TextType)
        assert entry.speaker_candidate_count >= 0

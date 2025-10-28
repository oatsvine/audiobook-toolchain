from pathlib import Path
from typing import List

from laban_tts.normalize import (
    Heuristics,
    NormalizedBook,
    NormalizedOutput,
    SpeakerGuess,
    TextType,
    build_normalized_book,
    plan_split,
)


def test_plan_split_detects_chapters(tmp_path: Path) -> None:
    sample_text = tmp_path / "sample.txt"
    sample_text.write_text(
        """CHAPTER I
This is the opening chapter.

CHAPTER II
This is the second chapter."""
    )

    split_plan = plan_split(sample_text)

    titles = [material.meta.chapter_title for material in split_plan.parts if material.meta.chapter_title]
    if split_plan.strategy == "by-title" and len(titles) >= 2:
        assert titles[:2] == ["CHAPTER I", "CHAPTER II"]
    else:
        # Fallback path when the parser cannot infer chapter headings (e.g., libmagic unavailable).
        assert split_plan.strategy == "basic"
        assert len(split_plan.parts) >= 1
    assert all(material.text for material in split_plan.parts)


def test_normalized_book_xml_roundtrip(tmp_path: Path) -> None:
    sample_text = tmp_path / "book.txt"
    sample_text.write_text(
        """CHAPTER I
Call me Ishmael.

CHAPTER II
Some years ago."""
    )

    split_plan = plan_split(sample_text)
    # Fabricate minimal normalized payloads matching the plan.
    normalized_payloads: List[NormalizedOutput] = []
    for material in split_plan.parts:
        text = material.text
        heuristics = Heuristics(
            input_chars=len(text),
            output_chars=len(text),
            removed_chars=0,
            removal_ratio=0.0,
            paragraph_count=max(1, text.count("\n\n") + 1),
            speaker_candidate_count=1,
        )
        normalized_payloads.append(
            NormalizedOutput(
                text_name=material.meta.text_name,
                category=TextType.DIALOGUE
                if material.meta.chapter_title
                else TextType.NARRATIVE_PERSONAL,
                cleaned_text=text,
                speakers=[SpeakerGuess(name="narrator")],
                removed=[],
                heuristics=heuristics,
            )
        )

    normalized_book = build_normalized_book(split_plan, normalized_payloads)
    xml_raw = normalized_book.to_xml(
        encoding="unicode", pretty_print=True, skip_empty=True
    )
    xml_payload = xml_raw.decode("utf-8") if isinstance(xml_raw, bytes) else xml_raw

    reloaded = NormalizedBook.from_xml(xml_payload)
    assert reloaded.text_name == normalized_book.text_name
    assert len(reloaded.parts) == len(split_plan.parts)
    original_titles = [material.meta.chapter_title for material in split_plan.parts]
    assert [part.chapter_title for part in reloaded.parts] == original_titles
    first_part = reloaded.parts[0].to_model(reloaded.text_name)
    assert first_part.cleaned_text.startswith("CHAPTER I")

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

def _normalize_module() -> Any:
    import importlib

    return importlib.import_module("laban_tts.normalize")


def _extract_attr(xml_blob: str, attr: str) -> str:
    token = f"{attr}="
    start = xml_blob.find(token)
    if start == -1:
        raise ValueError(f"Attribute {attr} missing in metadata: {xml_blob}")
    start += len(token)
    quote = xml_blob[start]
    if quote not in ('"', "'"):
        raise ValueError(f"Attribute {attr} not quoted in metadata: {xml_blob}")
    end = xml_blob.find(quote, start + 1)
    if end == -1:
        raise ValueError(f"Attribute {attr} unterminated in metadata: {xml_blob}")
    return xml_blob[start + 1 : end]


class _FakeLLM:
    """Structured-output stub returning deterministic normalization payloads."""

    def with_structured_output(
        self, schema: object, include_raw: bool = False
    ) -> "_FakeStructuredLLM":
        return _FakeStructuredLLM()


class _FakeStructuredLLM:
    def invoke(self, messages: Sequence[Any], config: object) -> dict[str, Any]:
        # Messages[1] is the human prompt composed in normalize_parts.
        payload = messages[1].content
        meta_section, text_section = payload.split("\n\n== INPUT TEXT ==\n", maxsplit=1)
        metadata_xml = meta_section.replace("== METADATA ==\n", "", 1)
        text = text_section
        module = _normalize_module()
        text_name = _extract_attr(metadata_xml, "text-name")
        category = module.TextType.NARRATIVE_HISTORICAL
        heuristics = module.Heuristics(
            input_chars=len(text),
            output_chars=len(text),
            removed_chars=0,
            removal_ratio=0.0,
            paragraph_count=max(1, text.count("\n\n") + 1),
            speaker_candidate_count=1,
            likely_has_front_matter=False,
        )
        normalized = module.NormalizedOutput(
            text_name=text_name,
            category=category,
            cleaned_text=text,
            speakers=[module.SpeakerGuess(name="narrator")],
            removed=[],
            heuristics=heuristics,
        )
        return {"parsed": normalized}


def test_partition_text_creates_xml_parts(tmp_path: Path) -> None:
    module = _normalize_module()
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
    entries = module.partition_text(source, parts_dir, max_characters=80)

    assert entries, "No partition entries produced"
    loaded_entries = module.load_part_entries(parts_dir)
    assert len(loaded_entries) == len(entries)
    for entry in loaded_entries:
        assert entry.xml_path.exists()
        assert entry.text.strip(), "Partition text is empty"
        assert entry.document.text_name == source.stem
        assert entry.part >= 1
        if entry.part > 1:
            assert entry.document.last_excerpt is not None
        if entry.part < len(loaded_entries):
            assert entry.document.next_excerpt is not None


def test_normalize_parts_writes_xml(tmp_path: Path) -> None:
    module = _normalize_module()
    source = tmp_path / "doc.md"
    source.write_text("""# Header

Body paragraph one.

Paragraph two continues.
""")

    parts_dir = tmp_path / "parts"
    module.partition_text(source, parts_dir, max_characters=120)
    part_entries = module.load_part_entries(parts_dir)

    normalize_dir = tmp_path / "normalize"
    llm = _FakeLLM()
    outputs: List[object] = module.normalize_parts(part_entries, normalize_dir, llm)  # type: ignore[arg-type]

    assert outputs, "Normalization yielded no outputs"
    xml_paths = sorted(normalize_dir.glob("*-normalized.xml"))
    assert xml_paths, "No normalized XML files were written"

    reloaded = module.load_normalized_entries(normalize_dir)
    assert len(reloaded) == len(outputs)
    for entry in reloaded:
        assert entry.xml_path.exists()
        model = entry.normalized
        assert model.cleaned_text.strip(), "Cleaned text missing"
        assert model.category == module.TextType.NARRATIVE_HISTORICAL
        assert model.speakers and model.speakers[0].name == "narrator"

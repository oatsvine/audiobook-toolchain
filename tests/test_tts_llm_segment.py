import os
from pathlib import Path

import pytest

from laban_tts.cues import CuedScript
from laban_tts.normalize import load_normalized_parts, load_parts

# NOTE: kept for runtime execution; pyright ignored for this file.


# TODO: Illegal wrapper
def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


TEXT_MD = """
# The Gospel According to Mary Magdalene (The Gospel of Mary)


## Chapter 4
(Pages 1 to 6 of the manuscript, containing chapters 1 - 3, are lost.  The extant text starts on page 7...)

. . . Will matter then be destroyed or not?

22) The Savior said, All nature, all formations, all creatures exist in and with one another, and they will be resolved again into their own roots.

23) For the nature of matter is resolved into the roots of its own nature alone.

24) He who has ears to hear, let him hear.

25) Peter said to him, Since you have explained everything to us, tell us this also: What is the sin of the world?

26) The Savior said There is no sin, but it is you who make sin when you do the things that are like the nature of adultery, which is called sin.

27) That is why the Good came into your midst, to the essence of every nature in order to restore it to its root.

28) Then He continued and said, That is why you become sick and die, for you are deprived of the one who can heal you.

29) He who has a mind to understand, let him understand.

30) Matter gave birth to a passion that has no equal, which proceeded from something contrary to nature. Then there arises a disturbance in its whole body.

31) That is why I said to you, Be of good courage, and if you are discouraged be encouraged in the presence of the different forms of nature.

32) He who has ears to hear, let him hear.

33) When the Blessed One had said this, He greeted them all,saying, Peace be with you. Receive my peace unto yourselves.

34) Beware that no one lead you astray saying Lo here or lo there! For the Son of Man is within you.

35) Follow after Him!

36) Those who seek Him will find Him.

37) Go then and preach the gospel of the Kingdom.

38) Do not lay down any rules beyond what I appointed you, and do not give a law like the lawgiver lest you be constrained by it.

39) When He said this He departed.
"""


@pytest.mark.skipif(
    not _has_openai(), reason="Requires OPENAI_API_KEY and network access"
)
def test_llm_segmentation_end_to_end(tmp_path: Path):
    """Heuristic validation of LLM segmentation on crafted text.

    Notes:
    - Does not run audio generation; only the segmentation node.
    - Uses thresholds to avoid flakiness while catching regressions.
    """
    from langchain_openai import ChatOpenAI

    pytest.importorskip("chatterbox.tts", reason="Requires chatterbox TTS dependency")

    from laban_tts.workflow import Toolchain

    # Prefer a widely available small model; allow override via env.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Crafted text: includes title, narration, urgent and calming dialogue, and a frustration cue.
    # Uses longer paragraphs to encourage multiple segments and clear emotion mapping hints.

    in_dir = tmp_path / "in"
    text_file = in_dir / "thomas.md"
    in_dir.mkdir(parents=True, exist_ok=True)
    text_file.write_text(TEXT_MD)

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    tts = Toolchain(
        debug=True,
        force=True,
        llm=llm,
        in_dir=in_dir,
        voices_dir=voices_dir,
    )

    work_dir = tts.run(text_file=text_file)
    assert work_dir.is_dir()
    parts_dir = work_dir / "parts"
    assert parts_dir.is_dir(), "Partition stage did not create parts/"
    part_files = list(parts_dir.glob("*.xml"))
    assert part_files, "Partition stage wrote no XML parts"

    tts.normalize(work_dir)

    normalize_dir = work_dir / "normalize"
    assert normalize_dir.is_dir(), "Normalization stage did not produce directory"
    normalized_entries = load_normalized_parts(normalize_dir)
    assert normalized_entries, "No normalized entries reloaded"

    tts.cue(work_dir)

    cue_dir = work_dir / "cues"
    assert cue_dir.is_dir(), "Cue stage did not produce directory"
    cue_xml_path = next(cue_dir.glob("*-cues.xml"), None)
    assert cue_xml_path is not None, "Cue XML output missing"

    script = CuedScript.from_xml(cue_xml_path.read_text())
    assert script.chunks, "Cue script missing chunks"
    assert script.speakers, "Cue script speaker registry empty"

    parts = {(part.text_name, part.part): part for part in load_parts(parts_dir)}

    normalized = normalized_entries[0]
    cleaned_text = normalized.cleaned_text()
    chunk_chars = sum(len(chunk.text) for chunk in script.chunks)
    coverage_ratio = chunk_chars / max(1, len(cleaned_text))
    assert coverage_ratio > 0.5, "Cue chunks cover insufficient normalized text"

    part = parts[(normalized.text_name, normalized.part)]
    measured_input_chars = len(part.content)
    assert measured_input_chars > 0, "Original partition produced empty content"

    removed_from_fragments = sum(
        len(fragment.content)
        for fragment in normalized.fragments
        if fragment.role == "removed"
    )
    expected_removed = max(normalized.input_chars - normalized.output_chars, 0)
    measured_removed_chars = (
        removed_from_fragments if removed_from_fragments > 0 else expected_removed
    )
    if measured_removed_chars == 0:
        assert normalized.removed_chars == 0
        assert normalized.removal_ratio == 0
    else:
        removed_delta = abs(normalized.removed_chars - measured_removed_chars)
        assert (
            removed_delta <= 0.15 * measured_removed_chars
        ), "removed_chars diverges beyond 15% tolerance"

        measured_ratio = measured_removed_chars / max(1, normalized.input_chars)
        ratio_delta = abs(normalized.removal_ratio - measured_ratio)
        assert (
            ratio_delta <= 0.15 * measured_ratio
        ), "removal_ratio diverges beyond 15% tolerance"

    chunk_speakers = {chunk.speaker for chunk in script.chunks}
    assert chunk_speakers.issubset(set(script.speakers)), "Chunk speaker not registered"

    profiles = {chunk.profile for chunk in script.chunks}
    rhetorics = {chunk.rhetoric for chunk in script.chunks}
    assert profiles, "Missing delivery profiles"
    assert rhetorics, "Missing rhetoric tags"

    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    first_chunk = script.chunks[0]
    audio_stub = (
        audio_dir
        / f"{normalized.text_name}_p{normalized.part:03d}_{first_chunk.idx:04d}_{first_chunk.speaker}.wav"
    )
    audio_stub.write_bytes(b"")
    tts.finalize(work_dir)

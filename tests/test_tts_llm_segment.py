import json
import os
from pathlib import Path

import pytest

from audiobook_toolchain.cues import CuedScript
from audiobook_toolchain.normalize import NormalizedOutput


def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_TEST_KEY"))


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
    from audiobook_toolchain.workflow import Toolchain

    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_TEST_KEY"]

    # Prefer a widely available small model; allow override via env.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Crafted text: includes title, narration, urgent and calming dialogue, and a frustration cue.
    # Uses longer paragraphs to encourage multiple segments and clear emotion mapping hints.

    in_dir = tmp_path / "in"
    text_file = in_dir / "gospel_of_mary.md"
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

    work_dir = tts.text(text_file, part=1)
    assert work_dir.is_dir()
    assert (work_dir / "text" / text_file.name).is_file()
    tts.normalize(work_dir)

    normalize_dir = work_dir / "normalize"
    assert normalize_dir.is_dir(), "Normalization stage did not produce directory"
    normalized_json_path = next(normalize_dir.glob("*-normalized.json"), None)
    assert normalized_json_path is not None, "Normalized JSON output missing"
    normalized_text_path = normalized_json_path.with_suffix(".txt")
    assert normalized_text_path.is_file(), "Normalized text output missing"

    normalized_payload = json.loads(normalized_json_path.read_text())
    normalized = NormalizedOutput.model_validate(
        {**normalized_payload, "cleaned_text": normalized_text_path.read_text()}
    )
    assert normalized.cleaned_text.strip(), "Cleaned text is empty"
    heuristics = normalized.heuristics
    assert heuristics.input_chars >= heuristics.output_chars > 0
    assert 0.0 <= heuristics.removal_ratio <= 1.0
    assert heuristics.paragraph_count >= 1

    tts.cue(work_dir)

    cue_dir = work_dir / "cues"
    assert cue_dir.is_dir(), "Cue stage did not produce directory"
    cue_json_path = next(cue_dir.glob("*-cues.json"), None)
    assert cue_json_path is not None, "Cue JSON output missing"
    review_path = cue_json_path.with_name(cue_json_path.stem + "-review.txt")
    assert review_path.is_file(), "Cue review output missing"

    script = CuedScript.model_validate_json(cue_json_path.read_text())
    assert script.chunks, "CuedScript contains no chunks"
    assert script.text_name == normalized.text_name
    chunk_lengths = [len(chunk.text) for chunk in script.chunks]
    assert all(length > 0 for length in chunk_lengths)
    assert all(length <= 600 for length in chunk_lengths)
    indices = [chunk.idx for chunk in script.chunks]
    assert indices == list(range(1, len(indices) + 1))
    chunk_coverage = sum(chunk_lengths) / max(1, len(normalized.cleaned_text))
    assert 0.3 <= chunk_coverage <= 1.2
    for chunk in script.chunks:
        assert chunk.speaker == chunk.speaker.lower()
        assert " " not in chunk.speaker

    assert script.speakers, "CuedScript speakers list is empty"
    assert set(script.speakers).issuperset({chunk.speaker for chunk in script.chunks})

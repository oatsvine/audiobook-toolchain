from pathlib import Path
from typing import List


def test_basic_chapter_split(tmp_path: Path) -> None:
    sample_text = tmp_path / "sample.txt"
    sample_text.write_text(
        """CHAPTER I
This is the opening chapter.

CHAPTER II
This is the second chapter."""
    )

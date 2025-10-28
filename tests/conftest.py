import sys
from pathlib import Path
from typing import List

import pytest


@pytest.fixture()
def corpus_texts(request: pytest.FixtureRequest) -> List[Path]:
    corpus_dir = Path(__file__).resolve().parent.parent / "corpus"
    if not corpus_dir.exists():
        pytest.skip("Corpus directory missing")
    files = sorted(path for path in corpus_dir.iterdir() if path.is_file())
    if not files:
        pytest.skip("Corpus directory empty")
    return files

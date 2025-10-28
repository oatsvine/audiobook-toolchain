from pathlib import Path
from typing import List
import pytest


@pytest.fixture()
def corpus_texts(request: pytest.FixtureRequest) -> List[Path]:

    # TODO: Return each file in ../corpus as path to use with all tests needing text inputs.
    pass

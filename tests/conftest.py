import pytest
from core.corpus import Corpus

SAMPLE_FILE = "tests/sample/corpus-1.txt"

@pytest.fixture
def sample_corpus():
    return Corpus(SAMPLE_FILE)
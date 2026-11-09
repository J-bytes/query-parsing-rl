from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import settings

# CI runners are slower and noisier than a laptop: no per-example deadline there.
settings.register_profile("ci", deadline=None, max_examples=100)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_model_path() -> str:
    from qprl.tiny_model import build_tiny_model

    return str(build_tiny_model())


@pytest.fixture(scope="session")
def fixture_queries() -> Path:
    return FIXTURES / "queries.jsonl"

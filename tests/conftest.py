from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_model_path() -> str:
    from qprl.tiny_model import build_tiny_model

    return str(build_tiny_model())


@pytest.fixture(scope="session")
def fixture_queries() -> Path:
    return FIXTURES / "queries.jsonl"

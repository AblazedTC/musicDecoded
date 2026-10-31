import importlib.util
from pathlib import Path

import pytest

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"


@pytest.fixture
def validate():
    """contracts/validate.py check(doc) -> list of errors (skips without jsonschema)."""
    pytest.importorskip("jsonschema")
    spec = importlib.util.spec_from_file_location("contracts_validate", CONTRACTS / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check

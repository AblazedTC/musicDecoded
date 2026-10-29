"""Smoke test: the shared analysis fixture validates with contracts/validate.py.

Runs without the ML stack (no torch/librosa/model checkpoints needed).
"""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"


def _load_validate():
    spec = importlib.util.spec_from_file_location("contracts_validate", CONTRACTS / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    return json.loads((CONTRACTS / "fixtures" / "sample.analysis.json").read_text(encoding="utf-8"))


def test_sample_fixture_validates():
    assert _load_validate().check(_fixture()) == []


def test_check_catches_unsorted_beats():
    doc = _fixture()
    doc["beats"] = list(reversed(doc["beats"]))
    assert "beats not sorted by t" in _load_validate().check(doc)

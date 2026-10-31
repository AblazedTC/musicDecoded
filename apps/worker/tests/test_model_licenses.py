"""Guard for PLAN.md R-7: madmom (CC BY-NC-SA models) must not come back."""
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1]


def test_madmom_not_a_dependency():
    reqs = (WORKER / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "madmom" not in reqs


def test_worker_code_does_not_import_madmom():
    # services/ plus our own wrappers in models/*.py (not the upstream submodules)
    for path in [*(WORKER / "services").glob("*.py"), *(WORKER / "models").glob("*.py")]:
        source = path.read_text(encoding="utf-8")
        assert "import madmom" not in source and "from madmom" not in source, path.name

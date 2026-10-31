"""Validate analysis files against analysis.schema.json.

Usage: python contracts/validate.py [files...]   (defaults to contracts/fixtures/*.analysis.json)
Requires: pip install jsonschema
"""
import json, sys
from pathlib import Path
from jsonschema import Draft202012Validator

here = Path(__file__).parent
validator = Draft202012Validator(json.loads((here / "analysis.schema.json").read_text()))


def check(doc):
    """Schema errors plus the ordering invariants JSON Schema can't express."""
    errors = [e.message for e in validator.iter_errors(doc)]
    if errors:
        return errors
    ts = [b["t"] for b in doc["beats"]]
    if ts != sorted(ts):
        errors.append("beats not sorted by t")
    for a, b in zip(doc["chords"], doc["chords"][1:]):
        if b["start"] < a["end"] - 1e-3:
            errors.append(f"chords overlap at {b['start']}")
        elif b["start"] > a["end"] + 1e-3:
            errors.append(f"gap between chords at {a['end']}")  # the schema says contiguous; gaps are 'N' segments
    if any(c["end"] < c["start"] for c in doc["chords"]):
        errors.append("chord with end < start")
    return errors


if __name__ == "__main__":
    files = [Path(f) for f in sys.argv[1:]] or sorted((here / "fixtures").glob("*.analysis.json"))
    failed = False
    for f in files:
        errs = check(json.loads(f.read_text()))
        print(f"{'FAIL' if errs else 'ok  '} {f}")
        for e in errs[:10]:
            print(f"      {e}")
        failed |= bool(errs)
    sys.exit(1 if failed else 0)

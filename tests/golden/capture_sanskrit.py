"""Capture tests/golden/sanskrit_golden.json from the PRISTINE base commit (run once, before any src/ edit).
Usage: .venv/bin/python tests/golden/capture_sanskrit.py"""
import os, sys, json, subprocess

BASE = "c18927a"
TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS)
import sanskrit_snapshot as S  # noqa: E402

if subprocess.run(["git", "diff", "--quiet", BASE, "--", "src", "demo"], cwd=S.REPO).returncode != 0:
    sys.exit(f"capture_sanskrit: src/ or demo/ differ from {BASE}; goldens must come from pristine code")
snap = S.compute(os.path.join(S.REPO, ".pytest_tmp", "capture"))
snap["_base"] = BASE
snap["_env"] = S.env()
with open(S.GOLDEN, "w", encoding="utf-8") as f:
    f.write(json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
print(f"wrote {S.GOLDEN}")

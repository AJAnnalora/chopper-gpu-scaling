#!/usr/bin/env python3
"""Run every test in this repository. No GPU, no network, no arguments, no dependencies.

    python3 selftest.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKS = ["scripts/gpu_devices.py", "scripts/chopper_launch.py"]

passed, failed = [], []
for rel in CHECKS:
    print("=" * 72)
    print(rel)
    print("=" * 72)
    rc = subprocess.run([sys.executable, str(ROOT / rel), "--selftest"]).returncode
    (passed if rc == 0 else failed).append(rel)
    print()

print("-" * 72)
print("%d passed, %d failed" % (len(passed), len(failed)))
for rel in failed:
    print("  FAILED  " + rel)
print("PASS" if not failed else "FAIL")
raise SystemExit(1 if failed else 0)

#!/usr/bin/env python3
"""Run every test in this repository. No GPU, no network, no arguments, no dependencies.

    python3 selftest.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# (module, args). The benchmark has no test suite of its own -- it is a measurement tool --
# so its --dry-run stands in: it proves the script imports, that it can build the real
# invocation, and that it does so through gpu_devices rather than its own copy of the rules.
CHECKS = [("scripts/gpu_devices.py", ["--selftest"]),
          ("scripts/chopper_launch.py", ["--selftest"]),
          ("scripts/scaling_benchmark.py", ["--dry-run", "--devices", "1,8"])]

passed, failed = [], []
for rel, extra in CHECKS:
    print("=" * 72)
    print(rel + " " + " ".join(extra))
    print("=" * 72)
    rc = subprocess.run([sys.executable, str(ROOT / rel)] + extra).returncode
    (passed if rc == 0 else failed).append(rel)
    print()

print("-" * 72)
print("%d passed, %d failed" % (len(passed), len(failed)))
for rel in failed:
    print("  FAILED  " + rel)
print("PASS" if not failed else "FAIL")
raise SystemExit(1 if failed else 0)

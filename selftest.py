#!/usr/bin/env python3
"""Run every test in this repository. No GPU, no network, no arguments, no dependencies.

    python3 selftest.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

rc = subprocess.run([sys.executable, str(ROOT / "gpu_devices.py"), "--selftest"]).returncode
print()
print("PASS" if rc == 0 else "FAIL")
raise SystemExit(rc)

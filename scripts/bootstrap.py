"""Fetch pinned third-party source and build optional native adapters."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGA_COMMIT = "e4a8f1f9ec9e31f79f3fbd648717dfbb9fe113fc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nga", action="store_true")
    parser.add_argument("--mcsplit", action="store_true")
    args = parser.parse_args()
    if not args.nga and not args.mcsplit:
        args.nga = args.mcsplit = True
    (ROOT / "artifacts").mkdir(exist_ok=True)
    if args.nga:
        target = ROOT / "vendor/nga-official"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            subprocess.run(["git", "clone", "https://github.com/LOGO-CUHKSZ/NGA.git", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", NGA_COMMIT], check=True)
        revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if revision != NGA_COMMIT:
            raise RuntimeError("Existing NGA checkout is not at the pinned revision")
        if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip():
            raise RuntimeError("Existing NGA checkout has local changes")
    if args.mcsplit:
        for name in ("build_mcsplit_followup.py", "build_mcsplit_deadline.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)


if __name__ == "__main__":
    main()

"""Refresh checksums after intentional release edits; no scientific validation is implied."""
from pathlib import Path
import argparse
import hashlib
import sys
sys.dont_write_bytecode = True
from verify_manifest import released_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    rows = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n" for p in released_files(root)]
    (root / "MANIFEST_SHA256.txt").write_text("".join(rows))
    print(f"Wrote SHA-256 checksums for {len(rows)} released files.")

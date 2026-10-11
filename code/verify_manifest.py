"""Verify the complete released file inventory and its SHA-256 checksums."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

IGNORED_PARTS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "reproduced"}
BUILD_SUFFIXES = {".log", ".fls", ".fdb_latexmk", ".out", ".synctex.gz"}

def released_files(root):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or set(rel.parts) & IGNORED_PARTS:
            continue
        if rel.as_posix() == "MANIFEST_SHA256.txt" or path.suffix in BUILD_SUFFIXES:
            continue
        yield path

def verify(root):
    root = Path(root).resolve()
    errors, listed = [], set()
    manifest = root / "MANIFEST_SHA256.txt"
    if not manifest.is_file():
        return {"passed": False, "verified_files": 0, "errors": ["Missing MANIFEST_SHA256.txt"]}
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
            rel = Path(relative)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid SHA-256")
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("unsafe path")
            if relative in listed:
                raise ValueError("duplicate entry")
            listed.add(relative)
            path = root / rel
            if not path.is_file():
                errors.append(f"Missing: {relative}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f"Checksum mismatch: {relative}")
        except ValueError as exc:
            errors.append(f"Manifest line {number}: {exc}")
    present = {p.relative_to(root).as_posix() for p in released_files(root)}
    errors += [f"Unlisted released file: {p}" for p in sorted(present - listed)]
    return {"passed": not errors, "verified_files": len(listed), "errors": errors}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = verify(args.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

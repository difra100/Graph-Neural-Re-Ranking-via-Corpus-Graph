#!/usr/bin/env python3
"""Aggregate split MS MARCO download folders into one data/msmarco_data tree."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge repeated */msmarco_data folders from a split download into a "
            "single destination tree."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("msmarco dataset/ms-marco data"),
        help="Directory containing the split download folders.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/msmarco_data"),
        help="Destination merged msmarco_data directory.",
    )
    parser.add_argument(
        "--part-glob",
        default="msmarco_data-*/msmarco_data",
        help="Glob, relative to source root, that selects each split msmarco_data folder.",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink", "move"),
        default="hardlink",
        help="How to materialize files in the destination.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting destination files instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned aggregation without writing files.",
    )
    return parser.parse_args()


def file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def same_file_contents(left: Path, right: Path) -> bool:
    try:
        if os.path.samefile(left, right):
            return True
    except FileNotFoundError:
        return False

    if left.stat().st_size != right.stat().st_size:
        return False
    return file_digest(left) == file_digest(right)


def discover_parts(source_root: Path, part_glob: str) -> list[Path]:
    parts = sorted(path for path in source_root.glob(part_glob) if path.is_dir())
    if not parts:
        raise SystemExit(f"No source parts found with glob: {source_root / part_glob}")
    return parts


def iter_files(parts: list[Path]) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for part in parts:
        for source in sorted(path for path in part.rglob("*") if path.is_file()):
            files.append((part, source))
    return files


def materialize_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "move":
        shutil.move(str(source), str(destination))
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def aggregate(args: argparse.Namespace) -> None:
    source_root = args.source_root
    destination_root = args.destination
    parts = discover_parts(source_root, args.part_glob)
    source_files = iter_files(parts)

    created = 0
    identical_existing = 0
    overwritten = 0
    conflicts: list[tuple[Path, Path]] = []

    print(f"Source root: {source_root}")
    print(f"Destination: {destination_root}")
    print(f"Parts found: {len(parts)}")
    print(f"Files found: {len(source_files)}")
    print(f"Mode: {args.mode}")
    if args.dry_run:
        print("Dry run: no files will be written")

    for part, source in source_files:
        relative_path = source.relative_to(part)
        destination = destination_root / relative_path

        if destination.exists() or destination.is_symlink():
            if args.overwrite:
                if not args.dry_run:
                    destination.unlink()
                overwritten += 1
            elif same_file_contents(source, destination):
                identical_existing += 1
                continue
            else:
                conflicts.append((source, destination))
                continue

        if not args.dry_run:
            materialize_file(source, destination, args.mode)
        created += 1

    print(f"Created: {created}")
    print(f"Identical existing skipped: {identical_existing}")
    print(f"Overwritten: {overwritten}")

    if conflicts:
        print("Conflicting destination files:")
        for source, destination in conflicts[:20]:
            print(f"  {source} -> {destination}")
        if len(conflicts) > 20:
            print(f"  ... {len(conflicts) - 20} more")
        raise SystemExit(1)


def main() -> None:
    aggregate(parse_args())


if __name__ == "__main__":
    main()

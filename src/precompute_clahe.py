#!/usr/bin/env python3
"""Precompute CLAHE-enhanced NIH ChestX-ray14 224x224 PNG images.

Input:  data/images-224/*.png
Output: data/images-224-clahe/*.png

The original images are never modified. The script is resumable: existing
valid outputs are skipped when rerun with the same CLAHE settings.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "images-224"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "images-224-clahe"
EXPECTED_IMAGES = 112_120
_CLAHE = None


def _init_worker(clip_limit: float, grid_x: int, grid_y: int) -> None:
    global _CLAHE
    cv2.setNumThreads(1)
    _CLAHE = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_x, grid_y),
    )


def _process_one(item: tuple[str, str]) -> tuple[str, str]:
    src_name, dst_name = item
    src = Path(src_name)
    dst = Path(dst_name)

    # Resume safely when an already-created output can be decoded and has
    # the same dimensions as the source.
    if dst.is_file():
        existing = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
        source = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if existing is not None and source is not None and existing.shape == source.shape:
            return (src.name, "skipped")

    image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return (src.name, "ERROR: cannot read source image")

    enhanced = _CLAHE.apply(image)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_dst = dst.with_name(dst.stem + ".tmp.png")
    ok = cv2.imwrite(
        str(temp_dst), enhanced,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        return (src.name, "ERROR: cv2.imwrite failed")
    os.replace(temp_dst, dst)
    return (src.name, "created")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--grid-y", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--expected-count", type=int, default=EXPECTED_IMAGES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.clip_limit <= 0 or args.grid_x < 1 or args.grid_y < 1:
        raise ValueError("clip-limit and CLAHE grid dimensions must be positive")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    sources = sorted(input_dir.glob("*.png"))
    if len(sources) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count:,} PNG files, found {len(sources):,} in {input_dir}. "
            "Check the input directory before proceeding."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "clahe_manifest.json"
    settings = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_images": len(sources),
        "clip_limit": args.clip_limit,
        "tile_grid_size": [args.grid_x, args.grid_y],
        "input_mode": "grayscale",
        "output_format": "PNG grayscale",
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("input_dir", "output_dir", "num_images", "clip_limit", "tile_grid_size"):
            if previous.get(key) != settings.get(key):
                raise RuntimeError(
                    f"Existing manifest setting '{key}' differs. Do not mix CLAHE settings; "
                    "choose a new output directory or inspect it first."
                )

    print("=" * 72)
    print("NIH ChestX-ray14 CLAHE precomputation")
    print("=" * 72)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Images:      {len(sources):,}")
    print(f"CLAHE:       clipLimit={args.clip_limit}, grid=({args.grid_x}, {args.grid_y})")
    print(f"Workers:     {args.workers}")

    jobs: Iterable[tuple[str, str]] = (
        (str(src), str(output_dir / src.name)) for src in sources
    )
    created = skipped = 0
    errors: list[tuple[str, str]] = []
    start = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.clip_limit, args.grid_x, args.grid_y),
    ) as executor:
        for idx, (name, status) in enumerate(executor.map(_process_one, jobs, chunksize=32), start=1):
            if status == "created":
                created += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors.append((name, status))
            if idx % 2000 == 0 or idx == len(sources):
                elapsed = time.time() - start
                print(
                    f"Progress: {idx:,}/{len(sources):,} | created={created:,} "
                    f"skipped={skipped:,} errors={len(errors):,} | elapsed={elapsed/60:.1f} min",
                    flush=True,
                )

    if errors:
        error_path = output_dir / "clahe_errors.csv"
        with error_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Error"])
            writer.writerows(errors)
        print(f"ERROR: {len(errors)} image(s) failed. See {error_path}", file=sys.stderr)
        return 1

    outputs = list(output_dir.glob("*.png"))
    if len(outputs) != len(sources):
        raise RuntimeError(
            f"Output count mismatch: {len(outputs):,} outputs for {len(sources):,} inputs"
        )

    # Verify sample files decode and retain 224x224 dimensions.
    sample_names = [sources[0].name, sources[len(sources)//2].name, sources[-1].name]
    for name in sample_names:
        image = cv2.imread(str(output_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (224, 224):
            raise RuntimeError(f"Output verification failed for {name}: {None if image is None else image.shape}")

    manifest = {
        **settings,
        "completed": True,
        "created": created,
        "skipped_existing_valid": skipped,
        "errors": 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nCLAHE precomputation completed successfully.")
    print(f"Output images: {len(outputs):,}")
    print(f"Created now:   {created:,}")
    print(f"Skipped valid: {skipped:,}")
    print(f"Manifest:      {manifest_path}")
    print(f"Elapsed:       {(time.time()-start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

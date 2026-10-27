#!/usr/bin/env python
"""Fetch/verify the large regression artifact during a Docker build or locally.

The 295 MB compressed artifact is intentionally not committed to git (it exceeds
GitHub's per-file limit). It is hosted on Hugging Face and verified by SHA-256.
If the file is already present and valid, no network request is made, which
keeps local Docker builds fast and offline-friendly.

Usage:
    python scripts/fetch_model.py \
        --url "https://huggingface.co/.../best_regression_model_compressed.pkl?download=true" \
        --sha256 a321... \
        --size 295141016 \
        --dest models/best_regression_model_compressed.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, part: Path, expected_size: int | None) -> tuple[str, int]:
    digest = hashlib.sha256()
    downloaded = 0
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "emi-predict-ai-docker-build/1.0",
            "Accept": "*/*",
        },
    )
    started = time.time()
    last_report = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else expected_size
        print(f"Downloading {url}", flush=True)
        if total:
            print(f"Expected size: {total:,} bytes", flush=True)
        with part.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 25 * 1024 * 1024:
                    last_report = downloaded
                    elapsed = max(time.time() - started, 0.001)
                    rate = downloaded / 1024 / 1024 / elapsed
                    if total:
                        print(
                            f"  {downloaded / 1024 / 1024:,.1f}/"
                            f"{total / 1024 / 1024:,.1f} MiB ({rate:.1f} MiB/s)",
                            flush=True,
                        )
                    else:
                        print(
                            f"  {downloaded / 1024 / 1024:,.1f} MiB "
                            f"({rate:.1f} MiB/s)",
                            flush=True,
                        )
    return digest.hexdigest(), downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("EMI_REGRESSION_MODEL_URL", ""))
    parser.add_argument(
        "--sha256",
        default=os.environ.get("EMI_REGRESSION_MODEL_SHA256", ""),
        help="Expected SHA-256 of the artifact.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=int(os.environ.get("EMI_REGRESSION_MODEL_SIZE", "0") or 0),
        help="Expected size in bytes (optional but recommended).",
    )
    parser.add_argument(
        "--dest",
        default="models/best_regression_model_compressed.pkl",
        help="Destination path.",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    expected_hash = args.sha256.strip().lower()
    expected_size = args.size or None

    if dest.is_file() and dest.stat().st_size > 0:
        actual_size = dest.stat().st_size
        print(f"Found existing artifact: {dest} ({actual_size:,} bytes)")
        if expected_size and actual_size != expected_size:
            print(
                f"ERROR: existing artifact size {actual_size:,} does not match "
                f"expected {expected_size:,}.",
                file=sys.stderr,
            )
            return 1
        if expected_hash:
            actual_hash = sha256_file(dest)
            print(f"SHA-256: {actual_hash}")
            if actual_hash != expected_hash:
                print(
                    "ERROR: existing artifact SHA-256 does not match the expected "
                    "value. Delete it and re-run this script.",
                    file=sys.stderr,
                )
                return 1
            print("Existing artifact verified; no download needed.")
        return 0

    if not args.url:
        print(
            "ERROR: artifact is missing and no --url / EMI_REGRESSION_MODEL_URL "
            "was provided.\n"
            f"Place the artifact at {dest} or provide a download URL. See README.md.",
            file=sys.stderr,
        )
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()

    if args.url.startswith("file://"):
        source = Path(urllib.request.url2pathname(args.url[7:]))
        print(f"Copying local artifact {source} -> {dest}")
        shutil.copyfile(source, part)
        actual_hash = sha256_file(part)
        actual_size = part.stat().st_size
    else:
        last_error: Exception | None = None
        actual_hash = ""
        actual_size = 0
        for attempt in range(1, 4):
            try:
                actual_hash, actual_size = _download(args.url, part, expected_size)
                last_error = None
                break
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
                print(
                    f"Download attempt {attempt}/3 failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if part.exists():
                    part.unlink()
                if attempt < 3:
                    time.sleep(2 * attempt)
        if last_error is not None:
            print(
                "ERROR: could not download the regression artifact. "
                "Check outbound network access or provide the file locally.",
                file=sys.stderr,
            )
            return 1

    print(f"Downloaded {actual_size:,} bytes; SHA-256: {actual_hash}")
    if expected_size and actual_size != expected_size:
        print(
            f"ERROR: downloaded size {actual_size:,} does not match expected "
            f"{expected_size:,}. Removing partial file.",
            file=sys.stderr,
        )
        part.unlink(missing_ok=True)
        return 1
    if expected_hash and actual_hash != expected_hash:
        print(
            "ERROR: downloaded SHA-256 does not match the expected value. "
            "Removing file.",
            file=sys.stderr,
        )
        part.unlink(missing_ok=True)
        return 1

    os.replace(part, dest)
    print(f"Artifact ready: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

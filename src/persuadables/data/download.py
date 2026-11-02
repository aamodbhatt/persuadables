"""Fetch the raw dataset from its pinned URL and refuse anything whose sha256 does not match.

The MineThatData file is only served over plain HTTP, so the checksum is the integrity guarantee:
a tampered or truncated download can never reach the pipeline.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

_CHUNK = 1 << 16


class DataIntegrityError(RuntimeError):
    """The file on disk or on the wire is not the pinned dataset."""


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_dataset(
    url: str,
    destination: Path,
    sha256: str,
    *,
    attempts: int = 3,
    timeout_s: float = 120.0,
) -> Path:
    """Download ``url`` to ``destination`` unless a verified copy is already there.

    Raises:
        DataIntegrityError: the downloaded bytes do not hash to ``sha256``.
        urllib.error.URLError: every attempt failed at the network level.
    """
    if destination.is_file():
        actual = sha256_of(destination)
        if actual == sha256:
            return destination
        msg = (
            f"{destination} exists but its sha256 is {actual}, not the pinned {sha256}. "
            "Delete the file to download a fresh copy."
        )
        raise DataIntegrityError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: urllib.error.URLError | None = None
    for attempt in range(1, attempts + 1):
        try:
            _stream_to(url, partial, timeout_s)
            break
        except urllib.error.URLError as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2.0 * attempt)
    else:
        assert last_error is not None
        raise last_error

    actual = sha256_of(partial)
    if actual != sha256:
        partial.unlink()
        msg = (
            f"download from {url} hashed to {actual}, expected {sha256}. The source changed or "
            "the transfer was corrupted; the file was discarded."
        )
        raise DataIntegrityError(msg)
    partial.replace(destination)
    return destination


def _stream_to(url: str, path: Path, timeout_s: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "persuadables-data/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response, path.open("wb") as out:
        while chunk := response.read(_CHUNK):
            out.write(chunk)

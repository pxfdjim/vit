"""Immutable score and manifest artifact helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np


def publish_bytes_no_replace(path: Path, payload: bytes) -> bool:
    """Atomically publish a file without replacing an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def publish_json_no_replace(path: Path, value: Mapping[str, Any]) -> bool:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return publish_bytes_no_replace(path, text.encode("utf-8"))


def publish_npz_no_replace(path: Path, **arrays: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def read_score(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return stored["scores"].astype(np.float64), stored["timestamps"]


def write_score(path: Path, scores: np.ndarray, timestamps: np.ndarray,
                metadata: dict, elapsed: float) -> None:
    publish_npz_no_replace(
        path,
        scores=np.asarray(scores, np.float64),
        timestamps=np.asarray(timestamps),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
        timing_json=np.asarray([json.dumps({"inference_seconds": elapsed}, sort_keys=True)]),
    )

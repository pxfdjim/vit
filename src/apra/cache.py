"""Immutable visual-cache contract used by APRA inference."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


WINDOW_SIZE = 224
STEP_SIZE = 56
PATCH_COUNT = 196
TOKEN_DIMENSION = 768
BLOCK_INDEX = 5
BLOCK_NUMBER = BLOCK_INDEX + 1
CACHE_VERSION = 1


def input_digest(raw_values: Any) -> str:
    """Return the digest used by the immutable CLIP visual caches."""

    array = np.ascontiguousarray(raw_values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(array.view(np.uint8))
    digest.update(b"VFG-ViT4TS-cache-v1-CLIP-ViT-B-16-224-56")
    return digest.hexdigest()


def safe_component(value: Any) -> str:
    text = str(value)
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in text
    )


def final_cache_path(
    root: str | os.PathLike[str], context: Mapping[str, Any], digest: str
) -> Path:
    return (
        Path(root)
        / safe_component(context.get("dataset", "unknown_dataset"))
        / safe_component(context.get("signal", "unknown_signal"))
        / f"visual_{digest[:20]}.pt"
    )


def intermediate_cache_path(
    root: str | os.PathLike[str], context: Mapping[str, Any], digest: str
) -> Path:
    return (
        Path(root)
        / safe_component(context.get("dataset", "unknown_dataset"))
        / safe_component(context.get("signal", "unknown_signal"))
        / f"curve_block{BLOCK_NUMBER:02d}_{digest[:20]}.pt"
    )


def expected_window_count(length: int) -> int:
    if length < WINDOW_SIZE:
        return 0
    return 1 + (length - WINDOW_SIZE) // STEP_SIZE


def validate_final_cache(
    cached: Mapping[str, Any], *, digest: str, expected_windows: int, path: Path
) -> torch.Tensor:
    if int(cached.get("cache_version", -1)) != 1:
        raise RuntimeError(f"unsupported final visual cache version: {path}")
    if cached.get("input_digest") != digest:
        raise RuntimeError(f"final visual cache digest mismatch: {path}")
    patch = cached.get("patch")
    if not isinstance(patch, torch.Tensor) or not patch.is_floating_point():
        raise RuntimeError(f"invalid final patch tokens: {path}")
    if patch.ndim == 4 and patch.shape[-2] == 1:
        patch = patch.squeeze(-2)
    if tuple(patch.shape) != (expected_windows, PATCH_COUNT, TOKEN_DIMENSION):
        raise RuntimeError(f"invalid final patch shape {tuple(patch.shape)}: {path}")
    if not torch.isfinite(patch).all():
        raise RuntimeError(f"non-finite final patch tokens: {path}")
    return patch


def load_final_tokens(
    roots: Sequence[str | os.PathLike[str]],
    context: Mapping[str, Any],
    digest: str,
    expected_windows: int,
) -> tuple[torch.Tensor, Path]:
    paths = [final_cache_path(root, context, digest) for root in roots]
    present = [path for path in paths if path.is_file()]
    if not present:
        raise FileNotFoundError(
            "missing reusable final visual cache; checked " + ", ".join(map(str, paths))
        )
    cached = torch.load(present[0], map_location="cpu", weights_only=True)
    return (
        validate_final_cache(
            cached, digest=digest, expected_windows=expected_windows, path=present[0]
        ),
        present[0],
    )


def validate_intermediate_cache(
    cached: Mapping[str, Any], *, digest: str, expected_windows: int, path: Path
) -> None:
    expected_scalars = {
        "cache_version": CACHE_VERSION,
        "input_digest": digest,
        "intermediate_block_index": BLOCK_INDEX,
        "intermediate_block_number": BLOCK_NUMBER,
        "token_normalization": "visual.ln_post",
        "token_projection": "none",
    }
    for name, expected in expected_scalars.items():
        if cached.get(name) != expected:
            raise RuntimeError(f"invalid {name} in APRA cache: {path}")
    shapes = {
        "intermediate_patch": (expected_windows, PATCH_COUNT, TOKEN_DIMENSION),
        "curve_pixel_xy": (expected_windows, 14, 2),
        "curve_grid_xy": (expected_windows, 14, 2),
    }
    for name, shape in shapes.items():
        tensor = cached.get(name)
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise RuntimeError(f"invalid {name} in APRA cache: {path}")
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise RuntimeError(f"invalid {name} shape/content in APRA cache: {path}")

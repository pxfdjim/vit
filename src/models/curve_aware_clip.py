"""Frozen CLIP feature extraction and immutable caches for curve aware ViT4TS.

The existing ``visual_*.pt`` artifact remains the source of final-layer ViT
tokens.  This module only adds a separate block-6 artifact.  Block numbering is
one based in reports: block 6 is ``resblocks[5]`` in OpenCLIP.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


WINDOW_SIZE = 224
STEP_SIZE = 56
IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
GRID_SIZE = (14, 14)
INTERMEDIATE_BLOCK_INDEX = 5
INTERMEDIATE_BLOCK_NUMBER = INTERMEDIATE_BLOCK_INDEX + 1
INTERMEDIATE_DIMENSION = 768
CURVE_CACHE_VERSION = 1


def input_digest(raw_values: Any) -> str:
    """Return the exact digest used by ``VFGViT4TSBackend`` visual caches."""

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
        / f"curve_block{INTERMEDIATE_BLOCK_NUMBER:02d}_{digest[:20]}.pt"
    )


def publish_torch_no_replace(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish a torch artifact and never replace an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def expected_window_count(length: int) -> int:
    if length < WINDOW_SIZE:
        return 0
    return 1 + (length - WINDOW_SIZE) // STEP_SIZE


def visual_windows(visual_values: Any) -> np.ndarray:
    """Create the exact 224/56 windows used by the image renderer."""

    values = np.asarray(visual_values, dtype=np.float64).reshape(-1)
    count = expected_window_count(values.size)
    if not count:
        raise ValueError("visual series must contain at least 224 points")
    return np.stack(
        [values[start : start + WINDOW_SIZE] for start in range(0, count * STEP_SIZE, STEP_SIZE)]
    )


def curve_coordinates_from_plot_transform(windows: Any) -> dict[str, torch.Tensor]:
    """Map rendered curve vertices to the 14 CLIP column centers.

    The coordinate transform reproduces ``draw_image``: a 224x224, 100 dpi
    Matplotlib figure; axes filling the full canvas; zero margins; x limits
    0..223; and fixed y limits 0..1. ``pixel_xy`` uses image-edge coordinates
    (origin at the top left, valid extent 0..224). ``grid_xy`` maps those
    coordinates to the continuous 14x14 token-center system, where token
    centers have integer coordinates 0..13.
    """

    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != WINDOW_SIZE:
        raise ValueError("windows must have shape [N, 224]")
    if not np.isfinite(values).all():
        raise ValueError("curve values must be finite")

    # Keep this tied to the plotting library rather than duplicating its axis
    # arithmetic.  One transform serves every window because draw_image fixes
    # both limits after global visual normalization.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(IMAGE_SIZE[1] / 100, IMAGE_SIZE[0] / 100), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.plot(
        np.arange(WINDOW_SIZE),
        values[0],
        linestyle="-",
        linewidth=1,
        marker="*",
        markersize=0.1,
        color="black",
    )
    # ``draw_image`` obtains these limits from the plotted 0..223 x values
    # plus ``margins(0, 0)``.  Set them explicitly before using transData:
    # without a canvas draw, the lazy autoscale view can still be 0..1.
    axis.set_xlim((0, WINDOW_SIZE - 1))
    axis.set_ylim((0, 1))
    axis.set_xticks([])
    axis.set_yticks([])
    figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    axis.margins(0, 0)
    figure.canvas.draw()

    patch_center_x = (np.arange(GRID_SIZE[1], dtype=np.float64) + 0.5) * PATCH_SIZE
    data_x = axis.transData.inverted().transform(
        np.column_stack((patch_center_x, np.zeros_like(patch_center_x)))
    )[:, 0]
    sampled_y = np.stack(
        [np.interp(data_x, np.arange(WINDOW_SIZE), window) for window in values]
    )
    # The Matplotlib display system has a bottom-left origin.  Raster arrays
    # have a top-left origin, hence height - display_y in edge coordinates.
    display_y = axis.transData.transform(
        np.column_stack((np.zeros(sampled_y.size), sampled_y.reshape(-1)))
    )[:, 1].reshape(sampled_y.shape)
    pixel_y = IMAGE_SIZE[0] - display_y
    pixel_x = np.broadcast_to(patch_center_x, sampled_y.shape)
    pixel_xy = np.stack((pixel_x, pixel_y), axis=-1).astype(np.float32)
    grid_xy = pixel_xy.copy()
    grid_xy[..., 0] = grid_xy[..., 0] / PATCH_SIZE - 0.5
    grid_xy[..., 1] = grid_xy[..., 1] / PATCH_SIZE - 0.5
    return {
        "curve_pixel_xy": torch.from_numpy(pixel_xy),
        "curve_grid_xy": torch.from_numpy(grid_xy),
    }


def validate_final_visual_cache(
    cached: Mapping[str, Any], *, digest: str, expected_windows: int, path: Path
) -> torch.Tensor:
    """Validate and return existing final ``ln_post`` patch tokens."""

    if int(cached.get("cache_version", -1)) != 1:
        raise RuntimeError(f"unsupported final visual cache version: {path}")
    if cached.get("input_digest") != digest:
        raise RuntimeError(f"final visual cache digest mismatch: {path}")
    patch = cached.get("patch")
    if not isinstance(patch, torch.Tensor) or not patch.is_floating_point():
        raise RuntimeError(f"invalid final patch tokens: {path}")
    if patch.ndim == 4 and patch.shape[-2] == 1:
        patch = patch.squeeze(-2)
    if tuple(patch.shape) != (expected_windows, 196, INTERMEDIATE_DIMENSION):
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
    """Load final tokens from old caches without writing to their roots."""

    paths = [final_cache_path(root, context, digest) for root in roots]
    present = [path for path in paths if path.is_file()]
    if not present:
        raise FileNotFoundError(
            "missing reusable final visual cache; checked " + ", ".join(map(str, paths))
        )
    cached = torch.load(present[0], map_location="cpu", weights_only=True)
    return (
        validate_final_visual_cache(
            cached, digest=digest, expected_windows=expected_windows, path=present[0]
        ),
        present[0],
    )


class FrozenCLIPBlock6Encoder(nn.Module):
    """Extract normalized spatial output after OpenCLIP ``resblocks[5]``.

    Applying ``visual.ln_post`` makes the intermediate and existing final
    tokens share normalization semantics.  Neither token set receives the
    768-to-512 global CLIP projection, which applies only to pooled features.
    """

    def __init__(self, clip_ad: nn.Module):
        super().__init__()
        self.clip_ad = clip_ad
        self.visual = clip_ad.model.visual
        blocks = getattr(getattr(self.visual, "transformer", None), "resblocks", None)
        if blocks is None or len(blocks) != 12:
            raise ValueError("expected OpenAI CLIP ViT-B/16 with 12 visual blocks")
        self.eval()
        for parameter in self.clip_ad.parameters():
            parameter.requires_grad_(False)
        # ``visual`` is normally registered below ``clip_ad.model``.  Freeze
        # it explicitly as well so wrappers/proxies cannot evade the contract.
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenCLIPBlock6Encoder":
        super().train(False)
        self.clip_ad.eval()
        self.visual.eval()
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.visual.forward_intermediates(
            images,
            indices=[INTERMEDIATE_BLOCK_INDEX],
            stop_early=True,
            normalize_intermediates=True,
            intermediates_only=True,
            output_fmt="NLC",
            output_extra_tokens=False,
        )
        items = output.get("image_intermediates")
        if not isinstance(items, list) or len(items) != 1:
            raise RuntimeError("OpenCLIP did not return the requested intermediate")
        tokens = items[0]
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (196, INTERMEDIATE_DIMENSION):
            raise RuntimeError(f"unexpected block-6 token shape: {tuple(tokens.shape)}")
        return tokens


class _CLIPContainer(nn.Module):
    """Minimal registered container matching the repository CLIP_AD surface."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


def create_openai_block6_encoder(
    checkpoint: str | os.PathLike[str], device: torch.device | str
) -> FrozenCLIPBlock6Encoder:
    """Load the same verified OpenAI ViT-B/16 weights without downloading."""

    checkpoint = Path(checkpoint)
    expected = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for CLIP checkpoint: {checkpoint}")

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=str(checkpoint),
        vision_cfg={"output_tokens": True},
        force_quick_gelu=True,
        weights_only=False,
    )
    return FrozenCLIPBlock6Encoder(_CLIPContainer(model)).to(device)


def encode_block6(
    encoder: FrozenCLIPBlock6Encoder,
    images: torch.Tensor,
    *,
    device: torch.device | str,
    batch_size: int = 20,
) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, *IMAGE_SIZE):
        raise ValueError("images must have shape [N, 3, 224, 224]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pieces = []
    for start in range(0, images.shape[0], batch_size):
        pieces.append(encoder(images[start : start + batch_size].to(device)).cpu())
    return torch.cat(pieces, dim=0)


def validate_intermediate_cache(
    cached: Mapping[str, Any], *, digest: str, expected_windows: int, path: Path
) -> None:
    expected_scalars = {
        "cache_version": CURVE_CACHE_VERSION,
        "input_digest": digest,
        "intermediate_block_index": INTERMEDIATE_BLOCK_INDEX,
        "intermediate_block_number": INTERMEDIATE_BLOCK_NUMBER,
        "token_normalization": "visual.ln_post",
        "token_projection": "none",
    }
    for name, expected in expected_scalars.items():
        if cached.get(name) != expected:
            raise RuntimeError(f"invalid {name} in curve cache: {path}")
    shapes = {
        "intermediate_patch": (expected_windows, 196, INTERMEDIATE_DIMENSION),
        "curve_pixel_xy": (expected_windows, 14, 2),
        "curve_grid_xy": (expected_windows, 14, 2),
    }
    for name, shape in shapes.items():
        tensor = cached.get(name)
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise RuntimeError(f"invalid {name} in curve cache: {path}")
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise RuntimeError(f"invalid {name} shape/content in curve cache: {path}")


def build_intermediate_payload(
    tokens: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor],
    *,
    digest: str,
    source_final_cache: Path,
) -> dict[str, Any]:
    return {
        "cache_version": CURVE_CACHE_VERSION,
        "input_digest": digest,
        "intermediate_block_index": INTERMEDIATE_BLOCK_INDEX,
        "intermediate_block_number": INTERMEDIATE_BLOCK_NUMBER,
        "token_normalization": "visual.ln_post",
        "token_projection": "none",
        "source_final_cache_name": source_final_cache.name,
        "intermediate_patch": tokens.detach().cpu(),
        "curve_pixel_xy": coordinates["curve_pixel_xy"].cpu(),
        "curve_grid_xy": coordinates["curve_grid_xy"].cpu(),
    }

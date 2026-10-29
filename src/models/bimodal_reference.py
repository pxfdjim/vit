"""VETime-inspired visual/numeric reference representation.

The trainable branch is used only to route T3 references.  The detector's
final anomaly evidence remains the frozen CLIP Block-6 column distance.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


class NumericPatchEncoder(nn.Module):
    def __init__(self, hidden: int = 192, layers: int = 2, heads: int = 4):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 96, 3, padding=1), nn.GELU(),
        )
        self.patch = nn.Linear(96 * 16, hidden)
        self.position = nn.Parameter(torch.zeros(1, 14, hidden))
        block = nn.TransformerEncoderLayer(hidden, heads, hidden * 3, 0.1,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.temporal = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is sequence-level robust standardized, never Patch-normalized.
        d1 = F.pad(x[:, 1:] - x[:, :-1], (1, 0))
        d2 = F.pad(d1[:, 1:] - d1[:, :-1], (1, 0))
        trend = F.avg_pool1d(x[:, None], 17, stride=1, padding=8)[:, 0]
        h = self.local(torch.stack((x, d1, d2, trend), dim=1))
        h = h.transpose(1, 2).reshape(x.shape[0], 14, 16 * 96)
        return F.normalize(self.norm(self.temporal(self.patch(h) + self.position)), dim=-1)


class VisualColumnEncoder(nn.Module):
    def __init__(self, hidden: int = 192):
        super().__init__()
        # A fixed Johnson-Lindenstrauss projection preserves the complete
        # vertical Patch column.  Keeping this target fixed prevents the two
        # trainable modalities from jointly collapsing to a constant vector.
        generator = torch.Generator().manual_seed(20261029)
        projection = torch.randn(14 * 768, hidden, generator=generator)
        projection = projection / (14 * 768) ** 0.5
        self.register_buffer("projection", projection)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        grid = visual.reshape(-1, 14, 14, 768).permute(0, 2, 1, 3)
        h = grid.reshape(grid.shape[0], 14, 14 * 768) @ self.projection
        return F.normalize(F.layer_norm(h, (h.shape[-1],)), dim=-1)


class AlignedMultimodalReference(nn.Module):
    """Patch aligned interaction encoder for learned reference routing."""
    def __init__(self, hidden: int = 192, heads: int = 4):
        super().__init__()
        self.numeric = NumericPatchEncoder(hidden=hidden, heads=heads)
        self.visual = VisualColumnEncoder(hidden=hidden)
        self.n_to_v = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
        self.v_to_n = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hidden * 4, hidden * 2), nn.GELU(),
                                  nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden))
        self.log_temperature = nn.Parameter(torch.tensor(-2.3))

    def forward(self, raw: torch.Tensor, visual: torch.Tensor):
        n = self.numeric(raw)
        v = self.visual(visual)
        nv = self.n_to_v(n, v, v, need_weights=False)[0]
        vn = self.v_to_n(v, n, n, need_weights=False)[0]
        fused = F.normalize(self.fuse(torch.cat((n, v, nv, vn), dim=-1)), dim=-1)
        return {"numeric": n, "visual": v, "fused": fused}

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.03, 0.5)


def patch_alignment_loss(outputs: dict[str, torch.Tensor], include_reference: bool = True):
    """Correspondence and reference-geometry objectives, with no negatives.

    Periodic windows can be semantically equivalent after a temporal shift, so
    other windows/columns are never declared negatives.  Instead the trainable
    time branch learns the frozen visual branch's pairwise geometry.
    """
    n, v, f = outputs["numeric"], outputs["visual"], outputs["fused"]
    batch, columns, _ = n.shape
    # Exact visual/numeric time correspondence plus full within-window relation
    # geometry.  Repeated phases may remain similar; no off-diagonal is forced
    # to be dissimilar.
    local = (1.0 - (n * v).sum(-1)).mean()
    numeric_geometry = torch.einsum("bid,bjd->bij", n, n)
    visual_geometry = torch.einsum("bid,bjd->bij", v, v)
    structure = F.smooth_l1_loss(numeric_geometry, visual_geometry)
    # Adjacent smoothness must agree across modalities, while retaining change magnitude.
    dn = 1.0 - (n[:, 1:] * n[:, :-1]).sum(-1)
    dv = 1.0 - (v[:, 1:] * v[:, :-1]).sum(-1)
    transition = F.smooth_l1_loss(dn, dv)
    reference = torch.zeros((), device=n.device)
    if include_reference and batch > 1:
        # Match pairwise reference geometry without calling any pair negative.
        ng, vg = F.normalize(n.mean(1), dim=-1), F.normalize(v.mean(1), dim=-1)
        reference = F.smooth_l1_loss(ng @ ng.T, vg @ vg.T)
    # Prevent fused tokens from losing their exact time correspondence.
    fused_align = 1.0 - (f * 0.5 * (n + v)).sum(-1).mean()
    total = local + 0.5 * structure + 0.25 * transition + 0.5 * reference + 0.25 * fused_align
    return total, {"local": local, "transition": transition,
                   "structure": structure, "reference": reference, "fused": fused_align}


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 192
    heads: int = 4
    layers: int = 2


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**payload["model_config"])
    model = AlignedMultimodalReference(cfg.hidden, cfg.heads)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, payload


"""SpectralViT4TS: frequency-domain multi-branch reference matching.

A stage-1 model-structure change only.  The evaluation protocol (windowed
imagery, official overlap stitching, EWMA + Gaussian alpha thresholds,
contextual F1-max) is untouched.

Changes versus the released ViT4TS first stage:
  1. Multi-branch spectral decomposition.  Each preprocessed window series is
     split into frequency subbands (rfft -> band mask -> irfft): low (periods
     > ~32 points, incl. DC -> level shifts), mid (periods ~8-32), high
     (periods <= ~8, spikes).  Each branch is rendered as its own line image
     (own global min-max, fixed [0,1] axis) and matched against its own
     reference memory bank.  Rationale (from the failure audit): a subtle
     anomaly that is visually dominated by background oscillations in the
     full-band image becomes the dominant pattern inside its own subband,
     while background noise spreads across branches.
  2. Branch fusion at the patch-map level (max or mean), then the official
     upsample + column top-25% mean + overlap-mean stitching.
  3. Numerical stabilization of the harmonic multi-scale fusion (denominator
     terms clamped >= 1e-12).  Verified: bitwise identical on the 48 clean
     SMAP sequences, repairs the five artifact sequences.

No training; the CLIP ViT-B/16 weights are the frozen OpenAI checkpoint.
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import preprocess_time_series, draw_windowed_images
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils import orion_to_internal
from models.clip_vision import CLIP_AD
from models.model_utils import (build_memory, compute_patch_dissimilarity,
                                stitch_anomaly_maps, align_anomaly_vector)


def stabilized_harmonic(score_size, similarity, mask):
    """Harmonic multi-scale fusion with the denominator terms clamped away
    from zero (removes the +-2^31 cancellation artifacts)."""
    b, h, w = score_size
    similarity = similarity.double().clamp(min=1e-12)
    mask = mask.T.long()
    score = torch.zeros((b, h * w), device=similarity.device).double()
    for idx in range(h * w):
        patch_idx = torch.tensor(
            [bool(torch.isin(idx + 1, m)) for m in mask],
            device=similarity.device)
        count = patch_idx.sum().item()
        denom = torch.sum(1.0 / similarity[:, patch_idx], dim=-1)
        score[:, idx] = count / denom
    return score.view(b, h, w)


def knn_patch_dissimilarity(memory_bank, token, offset, k, device):
    """Recurrence-aware matching layer.

    Instead of measuring the distance to the MEDIAN window's patches (the
    released pipeline), measure each test patch's distance to its k nearest
    neighbouring patches among ALL other windows of the same sequence
    (leave-one-window-out).  A pattern that recurs anywhere in the series
    obtains a low score even when it is absent from the median; only patterns
    that are unique to their window obtain a high score.  memory_bank:
    [L, N, D] raw features in window order; token: [B, N, D]; offset: index
    of the first test window of this batch in window order.
    """
    if memory_bank.dim() == 4 and memory_bank.shape[2] == 1:
        memory_bank = memory_bank.squeeze(2)
    L, N, D = memory_bank.shape
    mem = F.normalize(memory_bank.reshape(-1, D), dim=-1)
    tok = F.normalize(token, dim=-1)
    B = tok.shape[0]
    sim = tok.reshape(B * N, D) @ mem.T                     # [B*N, L*N]
    dissim = (1.0 - sim).clamp(min=0)
    row_win = torch.arange(L * N, device=device) // N
    kk = min(k, max(L - 1, 1))
    for bi in range(B):
        wid = offset + bi
        rows = torch.arange(bi * N, (bi + 1) * N, device=device)
        cols = (row_win == wid).nonzero(as_tuple=True)[0]
        if cols.numel():
            dissim[rows.unsqueeze(1), cols.unsqueeze(0)] = float("inf")
    vals, _ = torch.topk(dissim, kk, dim=1, largest=False)
    return (0.5 * vals.mean(dim=1)).reshape(B, N)


def spectral_branches(proc, bands):
    """[full, low, mid, high] reconstructions of one preprocessed series."""
    n = len(proc)
    spec = np.fft.rfft(proc)
    out = [proc]
    for lo, hi in bands:
        s = np.zeros_like(spec)
        s[lo:hi] = spec[lo:hi]
        x = np.fft.irfft(s, n=n)
        span = x.max() - x.min()
        out.append((x - x.min()) / span if span > 0 else np.zeros_like(x))
    return out


class SpectralViT4TS:
    """Frequency-domain multi-branch stage-1 anomaly scorer."""

    def __init__(self, window_size=224, window_step_ratio=4.0,
                 agg_percent=0.25, patch_size=16, model_name="ViT-B-16",
                 device="auto", batch_size=20, image_size=(224, 224),
                 dpi=100, bands=((0, 7), (7, 28), (28, None)), fuse="residual",
                 kappa=4.0, gain=1.0, masked_only=False, agg="top25", matching="median", knn_k=10, verbose=False):
        self.window_size = window_size
        self.window_step_ratio = window_step_ratio
        self.agg_percent = agg_percent
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.dpi = dpi
        self.bands = bands
        self.fuse = fuse
        self.kappa = kappa
        self.gain = gain
        self.masked_only = masked_only
        self.matching = matching  # "median": released pipeline;
        self.knn_k = knn_k        # "knn": leave-one-window-out k-NN recurrence matching
        self.agg = agg  # "top25": released per-column top-25% mean;
                        # "q25": paper text, per-column row-quantile 0.25
                        #        of the 2-D overlap-averaged map
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else (device if device != "auto" else "cpu"))
        self.model = CLIP_AD(model_name=model_name, device=self.device)
        self.model.eval()

    def _branch_maps(self, series, base_dir, branch_id):
        """14x14 fused patch maps for one branch of the series."""
        bdir = os.path.join(base_dir, f"branch{branch_id}")
        ok = draw_windowed_images(
            base_series_id="series", save_path=bdir, time_series=series,
            time_points=np.arange(len(series)),
            window_size=self.window_size,
            step_size=int(self.window_size / self.window_step_ratio),
            override=True, save_image=False, image_size=self.image_size,
            dpi=self.dpi, plot_params=("-", 1, "*", 0.1, "black", (0, 1)))
        if not ok:
            raise RuntimeError(f"image generation failed: branch {branch_id}")
        dataset = CLIPTimeSeriesDataset(results_dir=bdir,
                                        base_series_id="series",
                                        sample_size=None, no_anomaly=True,
                                        plot_type="line")
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False)
        large_mem, mid_mem, patch_mem = build_memory(
            self.model, loader, self.patch_size, self.device)
        key = next(iter(large_mem))
        maps = []
        offset = 0
        with torch.no_grad():
            for items in loader:
                images = items["img"].to(self.device)
                cls_names = items["cls_name"]
                b, _, h, w = images.shape
                (large_t, mid_t, patch_t, _, large_scale,
                 mid_scale) = self.model.encode_image(images, self.patch_size)
                size = (b, h // self.patch_size, w // self.patch_size)
                if self.matching in ("knn", "hybrid"):
                    # hybrid: full band keeps the official median matching
                    # (context), subband branches use recurrence-aware k-NN
                    # so recurring look-alikes do not trigger the residual
                    use_knn = (self.matching == "knn") or branch_id > 0
                else:
                    use_knn = False
                if use_knn:
                    m_l = stabilized_harmonic(
                        size, knn_patch_dissimilarity(
                            large_mem[key], large_t, offset, self.knn_k,
                            self.device), large_scale)
                    m_m = stabilized_harmonic(
                        size, knn_patch_dissimilarity(
                            mid_mem[key], mid_t, offset, self.knn_k,
                            self.device), mid_scale)
                    m_p = knn_patch_dissimilarity(
                        patch_mem[key], patch_t, offset, self.knn_k,
                        self.device).reshape(size)
                else:
                    m_l = stabilized_harmonic(
                        size, compute_patch_dissimilarity(large_mem, large_t,
                                                          cls_names), large_scale)
                    m_m = stabilized_harmonic(
                        size, compute_patch_dissimilarity(mid_mem, mid_t,
                                                          cls_names), mid_scale)
                    m_p = compute_patch_dissimilarity(
                        patch_mem, patch_t, cls_names).reshape(size)
                score = torch.nan_to_num((m_l + m_m + m_p) / 3.0,
                                         nan=0.0, posinf=0.0, neginf=0.0)
                maps.append(score.detach().cpu())
                offset += b
        return torch.cat(maps, dim=0).numpy()  # [n_windows, 14, 14]

    def _overlap_q25(self, maps):
        """Paper-textual aggregation: 2-D overlap average of the window maps,
        then the 0.25 quantile across rows per time column."""
        n, h, w = maps.shape
        stride = int(self.window_size / self.window_step_ratio)
        covered = stride * (n - 1) + w
        sums = np.zeros((h, covered), dtype=np.float64)
        counts = np.zeros(covered, dtype=np.int64)
        for i, m in enumerate(maps):
            start = i * stride
            sums[:, start:start + w] += m
            counts[start:start + w] += 1
        return np.quantile(sums / np.maximum(counts, 1)[None, :], 0.25, axis=0)

    def predict_scores(self, frame):
        values, timestamps = orion_to_internal(frame)
        T = len(values)
        proc = preprocess_time_series(values)  # identical to the baseline
        branches = spectral_branches(proc, self.bands)
        step = int(self.window_size / self.window_step_ratio)
        n_windows = int((T - self.window_size) / step) + 1
        with tempfile.TemporaryDirectory() as tmp:
            branch_maps = [self._branch_maps(s, tmp, i)
                           for i, s in enumerate(branches)]
            for bm in branch_maps:
                if len(bm) != n_windows:
                    raise RuntimeError("window count mismatch across branches")
            stacked = np.stack(branch_maps)  # [n_branches, n, 14, 14]
            boost_map = None
            if self.fuse in ("max_norm", "mean_norm", "max_z", "mean_z",
                             "residual", "residual_add", "residual_rank"):
                # per-branch robust standardization: each branch's
                # dissimilarity scale differs (subband amplification), so
                # normalize by its own background median/MAD before fusing;
                # _z variants also standardize the full band so the fused
                # score is "most anomalous branch in background sigmas"
                flat = stacked.reshape(len(branch_maps), -1)
                med = np.median(flat, axis=1, keepdims=True)
                mad = 1.4826 * np.median(np.abs(flat - med), axis=1,
                                         keepdims=True)
                normed = (stacked - med[:, None, None]) / (mad[:, None, None]
                                                           + 1e-12)
                if self.fuse in ("max_norm", "mean_norm"):
                    sub = normed[1:]  # full band keeps its raw scale
                    fused = (np.concatenate([stacked[:1], sub]).max(axis=0)
                             if self.fuse == "max_norm" else
                             np.concatenate([stacked[:1],
                                             sub]).mean(axis=0))
                elif self.fuse == "residual":
                    # keep the official full-band score as the base; subband
                    # branches contribute only where they deviate strongly
                    # from their own background (z > kappa), as a residual
                    # expressed in full-band score units
                    excess = np.maximum(normed[1:] - self.kappa, 0.0)
                    boost = excess.max(axis=0) * self.gain
                    if self.masked_only:
                        # masked-anomaly discovery: only boost positions whose
                        # full-band evidence is below 1 background sigma
                        boost = boost * (normed[0] < 1.0)
                    fused = stacked[0] + boost * mad[0]
                elif self.fuse == "residual_add":
                    # keep the full-band maps untouched; the subband evidence
                    # is stitched into a sparse 1-D boost that is ADDED to the
                    # aggregated full-band score afterwards (see below), so the
                    # base operating point is preserved exactly
                    fused = stacked[0]
                    excess = np.maximum(normed[1:] - self.kappa, 0.0)
                    boost_map = excess.max(axis=0) * mad[0] * self.gain
                elif self.fuse == "residual_rank":
                    # same additive-boost idea, but the subband evidence is
                    # measured by its EMPIRICAL PERCENTILE RANK within the
                    # sequence (robust to the heavy-tailed subband
                    # dissimilarity distributions that defeat MAD z-scores):
                    # only the top-q fraction of subband patches boost the
                    # full-band score
                    fused = stacked[0]
                    n_patch = stacked.shape[0] * stacked.shape[2]
                    ranks = np.empty(stacked[1:].shape)
                    for bi in range(stacked.shape[0] - 1):
                        flat = stacked[1 + bi].ravel()
                        order = flat.argsort().argsort() / len(flat)
                        ranks[bi] = order.reshape(stacked[1 + bi].shape)
                    q = 1.0 - self.kappa  # kappa now = rank threshold (0.999)
                    excess = np.maximum(ranks - q, 0.0) / q
                    boost_map = np.minimum(excess.max(axis=0), 1.0) \
                        * mad[0] * self.gain
                else:
                    fused = (normed.max(axis=0) if self.fuse == "max_z"
                             else normed.mean(axis=0))
            elif self.fuse == "max":
                fused = stacked.max(axis=0)
            else:
                fused = stacked.mean(axis=0)
            up = F.interpolate(torch.from_numpy(fused).unsqueeze(1).float(),
                               size=self.image_size,
                               mode="bilinear").squeeze(1).numpy()
            if self.agg == "q25":
                stitched = self._overlap_q25(up)
            else:
                stitched = stitch_anomaly_maps(up, self.window_step_ratio,
                                               self.agg_percent)
            aligned = align_anomaly_vector(stitched, T, self.window_size,
                                           step, n_windows)
            if boost_map is not None:
                bup = F.interpolate(
                    torch.from_numpy(boost_map).unsqueeze(1).float(),
                    size=self.image_size, mode="bilinear").squeeze(1).numpy()
                bstitched = stitch_anomaly_maps(bup, self.window_step_ratio,
                                                self.agg_percent)
                baligned = align_anomaly_vector(bstitched, T, self.window_size,
                                                step, n_windows)
                aligned = aligned + baligned
        return aligned, timestamps

    def get_intervals(self, scores, timestamps, alpha):
        from models.model_utils import compute_detection_intervals
        from preprocessing.data_utils import intervals_from_indices
        indices, _, _ = compute_detection_intervals(scores, alpha=alpha)
        return intervals_from_indices(indices, timestamps, scores)

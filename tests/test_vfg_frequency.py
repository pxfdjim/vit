import tempfile
import types
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.models.vfg_vit4ts import (
    FrequencyEncoding,
    FrozenCLIPVisualEncoder,
    VFGViT4TSBackend,
    VisualFrequencyMatcher,
    build_independent_median_reference,
    build_paired_reference,
    fuse_point_scores,
    full_window_fft_encoding,
    harmonic_aggregation_original,
    harmonic_aggregation_stable,
    linear_detrend_series,
    local_stft_encoding,
    robust_scale_calibration,
    _publish_torch_no_replace,
)


def _frequency_from_log(log_magnitude: torch.Tensor) -> FrequencyEncoding:
    amplitude = torch.expm1(log_magnitude.clamp_min(0))
    shape = torch.nn.functional.normalize(log_magnitude, dim=-1, eps=1e-12)
    chunks = torch.tensor_split(amplitude, 3, dim=-1)
    bands = torch.stack([x.square().sum(dim=-1) for x in chunks], dim=-1)
    energy = torch.cat((torch.log1p(bands), torch.ones_like(bands[..., :1])), dim=-1)
    return FrequencyEncoding(log_magnitude, shape, energy)


def _random_inputs(windows: int = 3, patches: int = 196, dim: int = 12):
    generator = torch.Generator().manual_seed(17)
    visual = torch.randn(windows, patches, dim, generator=generator)
    numeric = torch.randn(windows, 224, generator=generator)
    frequency = local_stft_encoding(numeric)
    return visual, frequency


def test_linear_detrend_removes_slope_without_minmax():
    time = torch.arange(224, dtype=torch.float64)
    values = 9.0 + 2.5 * time
    detrended = linear_detrend_series(values)
    assert detrended.dtype == torch.float64
    assert detrended.abs().max() < 1e-10

    nonlinear = 4.0 * torch.sin(2 * torch.pi * time / 32)
    output = linear_detrend_series(nonlinear)
    assert output.max() - output.min() > 1.0  # no hidden min-max scaling


def test_local_stft_has_14_aligned_frames_and_bins_1_through_32():
    generator = torch.Generator().manual_seed(3)
    window = torch.randn(224, generator=generator)
    encoding = local_stft_encoding(window)
    assert encoding.log_magnitude.shape == (14, 32)

    # The first visual column is centered at sample 7.5: its 64-point frame
    # spans original indices -24..39 after reflection padding.
    padded = torch.nn.functional.pad(window[None], (24, 24), mode="reflect")[0]
    frame = padded[:64] * torch.hann_window(64, periodic=True)
    expected = torch.log1p(torch.fft.rfft(frame).abs()[1:33])
    torch.testing.assert_close(encoding.log_magnitude[0], expected)

    last_frame = padded[-64:] * torch.hann_window(64, periodic=True)
    expected_last = torch.log1p(torch.fft.rfft(last_frame).abs()[1:33])
    torch.testing.assert_close(encoding.log_magnitude[-1], expected_last)


def test_spectral_shape_and_energy_are_both_available():
    time = torch.arange(224, dtype=torch.float64)
    base = torch.sin(2 * torch.pi * time / 16)
    low = local_stft_encoding(base)
    high = local_stft_encoding(8 * base)
    assert low.feature("shape").shape == (14, 32)
    assert low.feature("energy").shape == (14, 4)
    assert low.feature("both").shape == (14, 36)
    assert not torch.allclose(low.energy, high.energy)

    low_energy = torch.nn.functional.normalize(low.energy, dim=-1)
    high_energy = torch.nn.functional.normalize(high.energy, dim=-1)
    energy_distance = 1 - (low_energy * high_energy).sum(dim=-1)
    assert torch.all(energy_distance > 0)

    moved = low.to("cpu", torch.float32)
    assert moved.log_magnitude.device.type == "cpu"
    assert moved.shape.dtype == torch.float32
    assert moved.energy.dtype == torch.float32


def test_full_window_fft_removes_time_localization():
    values = torch.arange(224, dtype=torch.float32)
    encoding = full_window_fft_encoding(values)
    assert encoding.log_magnitude.shape == (14, 112)
    for column in range(1, 14):
        torch.testing.assert_close(
            encoding.log_magnitude[0], encoding.log_magnitude[column]
        )


def test_stable_harmonic_is_finite_and_keeps_legacy_control():
    similarity = torch.tensor([[0.0, 0.2]], dtype=torch.float32)
    # [members per group, groups]; pixel target 3 is uncovered under legacy +1.
    mask = torch.tensor([[0.0, 1.0], [1.0, 2.0]])
    original = harmonic_aggregation_original((1, 1, 3), similarity, mask)
    stable = harmonic_aggregation_stable((1, 1, 3), similarity, mask)
    assert not torch.isfinite(original).all()
    assert torch.isfinite(stable).all()
    assert stable[0, 0, 2] == 0


def test_independent_median_candidate_columns_are_reference_patch_mod_14():
    visual, frequency = _random_inputs(windows=3, dim=256)
    # Make every reference patch unique and query exact median-reference rows.
    base = torch.eye(196, 256)
    visual = torch.stack((base - 0.01, base, base + 0.01), dim=0)
    bank = build_independent_median_reference(visual, frequency)
    query = bank.visual[[0, 13, 14, 57, 195]].unsqueeze(0)
    matcher = VisualFrequencyMatcher(bank, top_k=5)
    candidates = matcher.visual_topk_candidates(query)

    expected_indices = torch.tensor([0, 13, 14, 57, 195])
    torch.testing.assert_close(candidates.reference_indices[0, :, 0], expected_indices)
    torch.testing.assert_close(
        candidates.reference_columns,
        candidates.reference_indices % 14,
    )
    torch.testing.assert_close(
        candidates.reference_frequency_indices,
        candidates.reference_columns,
    )


def test_paired_reference_keeps_window_and_column_correspondence():
    visual, frequency = _random_inputs(windows=2, dim=400)
    visual = torch.zeros_like(visual)
    # Identity-like tokens make flattened reference indices deterministic.
    visual[0, :, :196] = torch.eye(196)
    visual[1, :, 196:392] = torch.eye(196)
    bank = build_paired_reference(visual, frequency)
    selected = torch.tensor([0, 13, 14, 195, 196, 211, 391])
    query = bank.visual[selected].unsqueeze(0)
    candidates = VisualFrequencyMatcher(bank, top_k=1).visual_topk_candidates(query)
    indices = candidates.reference_indices[0, :, 0]
    torch.testing.assert_close(indices, selected)
    torch.testing.assert_close(candidates.reference_columns[0, :, 0], indices % 196 % 14)
    torch.testing.assert_close(candidates.reference_window_ids[0, :, 0], indices // 196)
    expected_frequency = (indices // 196) * 14 + (indices % 196) % 14
    torch.testing.assert_close(
        candidates.reference_frequency_indices[0, :, 0], expected_frequency
    )


def test_lambda_zero_exactly_recovers_stable_visual_patch_score():
    visual, frequency = _random_inputs(windows=4)
    bank = build_independent_median_reference(
        visual[:3], _slice_frequency(frequency, slice(0, 3))
    )
    matcher = VisualFrequencyMatcher(bank, top_k=5, frequency_weight=0.3)
    baseline = matcher.stable_visual_score(visual[3:])
    guided_zero = matcher.guided_score(
        visual[3:], _slice_frequency(frequency, slice(3, 4)), frequency_weight=0.0
    )
    assert torch.equal(baseline, guided_zero)


def _slice_frequency(
    encoding: FrequencyEncoding, index: int | slice
) -> FrequencyEncoding:
    return FrequencyEncoding(
        encoding.log_magnitude[index], encoding.shape[index], encoding.energy[index]
    )


def test_guided_reranking_uses_selected_candidate_frequency_not_just_column():
    # Paired reference windows have nearly tied visual matches at patch zero,
    # but opposite frequency vectors in the corresponding first column.
    visual = torch.zeros(2, 196, 2)
    visual[..., 1] = 1.0
    visual[0, 0] = torch.tensor([1.0, 0.0])
    visual[1, 0] = torch.tensor([0.99, 0.1])

    log = torch.ones(2, 14, 4)
    log[0, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    log[1, 0] = torch.tensor([0.0, 10.0, 0.0, 0.0])
    frequency = _frequency_from_log(log)
    bank = build_paired_reference(visual, frequency)

    query_visual = visual[0:1].clone()
    query_visual[0, 0] = torch.tensor([1.0, 0.02])
    query_log = torch.ones(1, 14, 4)
    query_log[0, 0] = torch.tensor([0.0, 10.0, 0.0, 0.0])
    query_frequency = _frequency_from_log(query_log)
    matcher = VisualFrequencyMatcher(
        bank,
        top_k=2,
        frequency_weight=0.3,
        frequency_mode="shape",
        calibration_factor=1.0,
        vit4ts_output_factor=1.0,
    )
    _, candidates, frequency_distance, _ = matcher.guided_score(
        query_visual, query_frequency, return_candidates=True
    )
    candidate_indices = candidates.reference_indices[0, 0]
    assert set(candidate_indices.tolist()) == {0, 196}
    distance_by_index = {
        int(index): float(distance)
        for index, distance in zip(candidate_indices, frequency_distance[0, 0])
    }
    assert distance_by_index[196] < distance_by_index[0]


def test_scale_calibration_is_median_ratio_and_finite_on_degenerate_input():
    visual = torch.tensor([0.0, 2.0, 4.0])
    frequency = torch.tensor([0.0, 1.0, 2.0])
    assert robust_scale_calibration(visual, frequency) == 2.0
    assert robust_scale_calibration(torch.zeros(3), torch.zeros(3)) == 1.0


def test_score_fusion_uses_complete_aligned_a_and_b_vectors_once():
    visual = torch.tensor([0.0, 2.0, 4.0, 6.0])
    frequency = torch.tensor([0.0, 1.0, 2.0, 3.0])
    fused, scale = fuse_point_scores(visual, frequency, frequency_weight=0.3)
    assert scale == 2.0
    torch.testing.assert_close(fused, visual + 0.3 * 2.0 * frequency)


def test_feature_cache_publication_never_replaces_existing_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "features.pt"
        assert _publish_torch_no_replace(path, {"value": torch.tensor([1])})
        assert not _publish_torch_no_replace(path, {"value": torch.tensor([2])})
        stored = torch.load(path, map_location="cpu", weights_only=True)
        assert stored["value"].item() == 1


def test_all_abcd_and_e3_dispatches_have_patch_shape():
    visual, frequency = _random_inputs(windows=4)
    bank = build_independent_median_reference(
        visual[:3], _slice_frequency(frequency, slice(0, 3))
    )
    matcher = VisualFrequencyMatcher(bank)
    query_visual = visual[3:]
    query_frequency = _slice_frequency(frequency, slice(3, 4))
    for method in (
        "stable_visual",
        "frequency_only",
        "score_fusion",
        "guided",
        "direct",
    ):
        score = matcher.score(method, query_visual, query_frequency)
        assert score.shape == (1, 196)
        assert torch.isfinite(score).all()


class _DummyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.called_patch_size = None

    def encode_image(self, image, patch_size):
        self.called_patch_size = patch_size
        return image * self.weight


def test_frozen_clip_interface_disables_gradients_and_training_mode():
    clip = _DummyCLIP()
    encoder = FrozenCLIPVisualEncoder(clip)
    encoder.train(True)
    result = encoder.encode_image(torch.ones(1), 16)
    assert result.item() == 1.0
    assert clip.called_patch_size == 16
    assert not clip.training
    assert not clip.weight.requires_grad
    assert not result.requires_grad


def test_cpu_backend_path_returns_all_variant_point_scores_without_clip():
    backend = object.__new__(VFGViT4TSBackend)
    backend.config = {}
    backend.model_config = {}
    backend.device = torch.device("cpu")
    backend.window_size = 224
    backend.step_size = 56
    backend.patch_size = 16
    backend.image_size = (224, 224)

    generator = torch.Generator().manual_seed(29)
    windows = 2
    features = {
        "cache_version": 1,
        "large": torch.randn(windows, 196, 8, generator=generator),
        "mid": torch.randn(windows, 196, 8, generator=generator),
        "patch": torch.randn(windows, 196, 1, 8, generator=generator),
        "large_mask": torch.arange(196, dtype=torch.float32).reshape(1, 196),
        "mid_mask": torch.arange(196, dtype=torch.float32).reshape(1, 196),
    }

    def fake_features(self, visual_values, context, digest):
        return features, True

    backend._load_or_extract_visual = types.MethodType(fake_features, backend)
    time = torch.arange(280, dtype=torch.float64)
    frame = pd.DataFrame(
        {
            "timestamp": time.numpy(),
            "value": (torch.sin(2 * torch.pi * time / 32) + 0.01 * time).numpy(),
        }
    )
    variants = {
        "A_stable_vit4ts": {},
        "B_frequency_only": {},
        "C_score_fusion": {"lambda": 0.3},
        "D_frequency_guided": {"lambda": 0.3, "top_k": 5},
        "E1_global_fft": {"frequency": "whole_window_fft_shape_and_energy"},
        "E2_shape_only": {"frequency": "local_stft_shape_only"},
        "E2_energy_only": {"frequency": "local_stft_energy_only"},
        "E3_direct_distance": {
            "base_patch_score": "all_reference_visual_frequency_joint_min"
        },
        "E4_paired_reference": {
            "frequency_reference": "paired_real_window_library",
            "reference_library_size": 2,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        backend.cache_dir = directory
        predictions = backend.predict_variants(
            frame, variants, {"dataset": "synthetic", "signal": "cpu"}
        )
    assert set(predictions) == set(variants)
    for name, prediction in predictions.items():
        scores = torch.from_numpy(prediction["scores"])
        assert scores.shape == (280,), name
        assert torch.isfinite(scores).all(), name
    assert predictions["C_score_fusion"]["metadata"]["matching"] == "aligned_point_score_fusion"
    assert "calibration_factor" in predictions["D_frequency_guided"]["metadata"]
